# The MIT License (MIT)
# © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# fmt: off

import io
import os
import uuid
import time
import wandb
import boto3
import torch
import tempfile
import argparse
import traceback
import numpy as np
from tqdm import tqdm
import bittensor as bt
import concurrent.futures  
import torch.optim as optim
from typing import List, Tuple
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import LlamaForCausalLM 
from torch.optim.lr_scheduler import CosineAnnealingLR

from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

# Enable cuDNN benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Instantiate the AWS S3 client.
env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables.
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')  # AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')  # AWS secret access key.
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def main(config):
    # Print the configuration settings.
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)
    
    # Init Bittensor objects.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)    
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')    
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)    
    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')  
    
    # Init my bucket information by submitting it to the chain.  
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed or mismatch, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'{config.name}', config=config)
        
    # Init training state.
    print('\n', '-' * 40, 'Hparams', '-' * 40)
    hparams = load_hparams()
    print ( hparams ) 
    model = None
    upload_history = []  
    last_mask_sync = 0 
    last_master_sync = 0
    n_steps = 0
    while True:
        try:   
            print('\n', '-' * 40, f'Step: {n_steps}', '-' * 40) 
            # Start timing for the entire step
            global_step_start_time = time.time()
            n_steps += 1
            
            # Load hparams.
            print ('\nLoading hparams ...')
            start_time = time.time()
            new_hparams = load_hparams()
            hparams_changed = any(getattr(new_hparams, key) != getattr(hparams, key) for key in set(vars(new_hparams)) | set(vars(hparams)))
            hparams = new_hparams
            print(f'\tLoading hparams completed in {time.time() - start_time} seconds') 

            # Sync the current chain state and hparams.
            print ('\nLoading chain state ...')
            start_time = time.time()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(f'\tLoading chain state completed in {time.time() - start_time} seconds') 
            
            # Sync the full model state every hparams.epoch_length
            if model == None or subtensor.block - last_master_sync > hparams.epoch_length:
                print(f'\nLoading master state ...') 
                start_time = time.time() 
                try:
                    # master_uid = int(metagraph.S.argmax())
                    master_uid = 26
                    master_bucket = subtensor.get_commitment( config.netuid, master_uid )
                    master_hotkey = metagraph.hotkeys[ master_uid ]
                    master_filename = f'master-{master_hotkey}.pt'
                    unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                    CLIENT.download_file( master_bucket, master_filename, unique_temp_file )
                    master_state_dict = torch.load( unique_temp_file, map_location='cpu', weights_only = True )
                    model = LlamaForCausalLM( config = hparams.model_config )
                    model.load_state_dict( master_state_dict )
                    model.to(config.device)
                    model.train()
                    last_master_sync = subtensor.block 
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print (f'No master:{e} Waiting ...')
                    time.sleep(12)
                    continue
                print(f'\tCLoading master state completed in {time.time() - start_time} seconds') 
            
            if 'optimizer' not in locals() or optimizer == None or hparams_changed:
                print(f'\nResetting optimizer ...') 
                start_time = time.time()
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr = hparams.learning_rate,  # Peak learning rate
                    betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
                    weight_decay = hparams.optimizer_weight_decay,  # Weight decay
                    foreach = True,  # more memory usage, but faster
                )
                scheduler = CosineAnnealingLR( optimizer, T_max = hparams.cosine_epoch_length, eta_min=hparams.eta_min, last_epoch=-1 )
                scaler = torch.cuda.amp.GradScaler()
                print(f'\tResetting optimizercompleted in {time.time() - start_time} seconds') 


            print(f'\nGetting blocks and buckets ...')
            start_time = time.time()  # Start timing
            # Function which maps from block to mask_wid such that multiple blocks share the same mask window id.
            # This is used to ensure that the model is not updated too frequently and that the mask is shared.
            # for multiple updates which fall across multiple blocks.
            def block_to_mask_window_id(block: int) -> int:
                return int(block / hparams.mask_window_length)
            # This fast forwards us to the block which shares no ids with the previous block.
            # If we don't do this fast forward, then we will be downloading the same masks multiple times.
            # TODO (const) consider if we should just remember the last mask id and download all masks for that id.
            # Or if we should just redownload and apply the same masks.
            block = subtensor.block
            all_sync_blocks = list(range(last_mask_sync - 2, block + 1))            
            last_mask_sync = (block_to_mask_window_id(block) + 1) * hparams.mask_window_length
            # Get buckets per uid if needs update.
            if 'buckets' not in locals() or len(buckets) != len(metagraph.uids):
                buckets = []
                # for uid in metagraph.uids:
                for uid in [27, 36, 37]:
                    try:
                        buckets.append(subtensor.get_commitment(config.netuid, uid))
                    except:
                        buckets.append(None)
            print(f'\tGetting block completed in {time.time() - start_time} seconds')

            # For each bucket, get all files that need to be synced.
            print(f'\nGetting masks names for blocks: {all_sync_blocks} and buckets: {set(buckets)}')
            num_valid_masks = 0
            
            start_time = time.time()
            mask_filenames_per_mask_wid = {block_to_mask_window_id(blk): [] for blk in all_sync_blocks}
            for bucket in list(set(buckets)):
                if bucket is None:
                    continue
                paginator = CLIENT.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(Bucket=bucket, Prefix='mask-')
                try:
                    for page in page_iterator:
                        for obj in page.get('Contents', []):
                            hotkey, blk = obj['Key'].split('-')[1], obj['Key'].split('-')[2].split('.')[0]
                            if hotkey == wallet.hotkey.ss58_address:
                                continue
                            if int(blk) in all_sync_blocks:
                                mask_wid = block_to_mask_window_id(int(blk))
                                mask_info = SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=obj['Key'], uid=metagraph.hotkeys.index(hotkey), block=int(blk), mask_wid=mask_wid)
                                mask_filenames_per_mask_wid[mask_wid].append(mask_info)
                                num_valid_masks += 1
                except:
                    continue
            print(f'\tGetting masks names for blocks: {all_sync_blocks} completed in {time.time() - start_time} seconds')

            # Get the mask for mask_wids.
            print(f'\nDownloading {num_valid_masks} masks for: {all_sync_blocks}')
            full_sync_start_time = time.time()
            masks_per_id_per_uid = {}
            mask_count_per_id = {}
            compression_per_uid = {}

            for mask_wid in mask_filenames_per_mask_wid.keys():
                masks_per_id_per_uid[mask_wid] = {}
                # Get the number of masks for this step.
                num_masks_for_mask_wid = len(mask_filenames_per_mask_wid[mask_wid])
                if num_masks_for_mask_wid == 0:
                    continue

                # Download the masks from all valid files
                print(f'\n\tDownloading {num_masks_for_mask_wid} mask for mask_wid: {mask_wid} ... ')
                start_time = time.time()
                temp_files = []
                n_downloaded = 0
                def download_file(mask_info):
                    try:
                        temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                        CLIENT.download_file(mask_info.bucket, mask_info.filename, temp_file)
                        mask_info = SimpleNamespace(**vars(mask_info), temp_file=temp_file)
                        return mask_info
                    except:
                        return None

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(download_file, mask_info) for mask_info in mask_filenames_per_mask_wid[mask_wid]]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            temp_files.append(result)
                            n_downloaded += 1
                print(f'\t\tDownloading {n_downloaded} masks completed in {time.time() - start_time} seconds')

                # Break the loop when there is nothing to download.
                if n_downloaded == 0:
                    continue

                # Initialize masks_per_uid to store loaded masks
                masks_per_uid = {}

                # Collect unique compression rates and per-miner compression, and load masks
                compressions_used = set()
                compression_per_uid = {}
                for info in temp_files:
                    # Load the mask only once
                    mask = torch.load(info.temp_file, map_location='cpu')
                    masks_per_uid[info.uid] = mask
                    compression = mask['compression']
                    compressions_used.add(compression)
                    compression_per_uid[info.uid] = compression
                    print(f"Loaded mask from UID {info.uid} (Hotkey: {info.hotkey}) with compression rate: {compression}")


                # Generate mask indices per compression
                mask_indices_per_compression = {}
                for compression in compressions_used:
                    torch.manual_seed(mask_wid)  # Ensure consistent seed
                    mask_indices = {}
                    for name, param in model.named_parameters():
                        param_shape = param.shape
                        # Generate mask using the specific compression rate
                        next_mask = (torch.rand(param_shape, device=config.device) < (1 / compression)).float()
                        indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                        mask_indices[name] = indices
                    mask_indices_per_compression[compression] = mask_indices
                    
                    sample_param_name = list(model.named_parameters())[0][0]  # Get the first parameter name
                    num_indices = len(mask_indices[sample_param_name])
                    print(f"Generated mask indices for compression rate {compression}: {num_indices} indices in parameter '{sample_param_name}'")

                print(f'\t\tGenerated mask indices in {time.time() - start_time} seconds')

                # Load all masks as state dicts.
                print(f'\n\tLoading state dicts for mask_wid: {mask_wid} ...')
                start_time = time.time()
                mask_count = 0
                masks_dicts_values = {}
                for info in temp_files:
                    masks_per_id_per_uid[info.mask_wid][info.uid] = {}
                    mask = masks_per_uid[info.uid]  # Retrieve the already loaded mask
                    compression = compression_per_uid[info.uid]
                    mask_count += 1
                    torch.manual_seed(info.mask_wid)
                    mask_indices = mask_indices_per_compression[compression]
                    for name in mask.keys():
                        if name == 'compression':
                            continue
                        mask_values = mask[name]['values']
                        if torch.isnan(mask_values).any():
                            continue
                        indices = mask_indices[name]
                        if len(indices) != len(mask_values):
                            print(f"Length mismatch for parameter {name}: indices length {len(indices)}, mask values length {len(mask_values)}")
                            continue
                        # Instead of decompressing, store indices and values for efficient aggregation
                        masks_per_id_per_uid[info.mask_wid][info.uid][name] = (indices, mask_values)
                mask_count_per_id[mask_wid] = mask_count
                print(f'\t\tLoading state dicts completed in {time.time() - start_time} seconds')

                # Average the masks before applying.
                print(f'Averaging {mask_count} masks for mask_wid: {mask_wid} ...')
                start_time = time.time()
                
                param_sums = {}
                param_counts = {}
                for name in model.state_dict().keys():
                    param_sums[name] = None
                    param_counts[name] = None
                    
                # Aggregate masks
                for uid in masks_per_uid:
                    mask = masks_per_uid[uid]
                    compression = compression_per_uid[uid]
                    mask_indices = mask_indices_per_compression[compression]
                    hotkey = metagraph.hotkeys[uid]
                    print(f"Aggregating mask from UID {uid} (Hotkey: {hotkey}) with compression rate: {compression}")
    
                    for param_name in mask.keys():
                        if param_name == 'compression':
                            continue  # Skip non-parameter keys
                        mask_values = mask[param_name]['values']
                        if torch.isnan(mask_values).any():
                            continue
                        indices = mask_indices[param_name]
                        if len(indices) != len(mask_values):
                            print(f"Length mismatch for parameter {param_name}: indices length {len(indices)}, mask values length {len(mask_values)}")
                            continue
                        if param_sums[param_name] is None:
                            param_size = model.state_dict()[param_name].numel()
                            param_sums[param_name] = torch.zeros(param_size, dtype=torch.float32, device=config.device)
                            param_counts[param_name] = torch.zeros(param_size, dtype=torch.float32, device=config.device)
                        
                        param_sums[param_name].index_add_(0, indices.to(config.device), mask_values.to(config.device))
                        param_counts[param_name].index_add_(0, indices.to(config.device), torch.ones_like(mask_values, device=config.device))

                # Include own params
                for param_name in param_sums.keys():
                    if param_sums[param_name] is not None:
                        counts = param_counts[param_name]
                        mask = counts > 0 
                        if mask.any():
                            param_values = model.state_dict()[param_name].data.flatten()
                            param_sums[param_name][mask] += param_values[mask]
                            param_counts[param_name][mask] += 1
                
                # Compute average masks
                average_masks = {}
                for param_name in param_sums.keys():
                    if param_sums[param_name] is not None:
                        counts = param_counts[param_name]
                        mask = counts > 0
                        avg_values = torch.zeros_like(counts)
                        avg_values[mask] = param_sums[param_name][mask] / counts[mask]
                        avg_values = avg_values.view(model.state_dict()[param_name].shape)
                        average_masks[param_name] = avg_values

                print(f'Averaged masks in {time.time() - start_time} seconds')

                # Set the average into the model.
                print(f'Applying {mask_count} masks for mask_wid: {mask_wid} ...')
                start_time = time.time()  # Start timing
                for name, param in model.named_parameters():
                    if name in average_masks:
                        avg_values = average_masks[name].to(param.device)
                        counts_mask = (param_counts[name].view(param.shape).to(param.device) > 0)
                        param.data[counts_mask] = avg_values[counts_mask]
                    else:
                        print(f"Name not in average_masks")
                for key in masks_dicts_values.keys():
                    masks_dicts_values[key] = masks_dicts_values[key].cpu()
                for key in mask_indices.keys():
                    mask_indices[key] = mask_indices[key].cpu()
                del mask_indices, masks_dicts_values
                print(f'Applying {mask_count} masks completed in {time.time() - start_time} seconds')
                
                # Delete files and clean up.
                print(f'\n\tDeleting files for mask_wid: {mask_wid} ...')
                start_time = time.time()
                for info in temp_files:
                    os.remove(info.temp_file)
                for key in param_sums.keys():
                    param_sums[key] = None
                    param_counts[key] = None
                print(f'\t\tDeleting files completed in {time.time() - start_time} seconds')

            # Log the average number of masks applied per mask_wid
            avg_masks_per_mask_wid = sum(mask_count_per_id.values()) / len(mask_count_per_id) if mask_count_per_id else 0
            if config.use_wandb: wandb.log({"avg_masks_per_mask_wid": avg_masks_per_mask_wid})

            # Print completion
            print(f'\tDownloading masks for blocks: {all_sync_blocks} and mask_wids: {mask_filenames_per_mask_wid.keys()} in {time.time() - full_sync_start_time} seconds')
            del mask_filenames_per_mask_wid
            torch.cuda.empty_cache()
            # Get the pages for this block and my_uid.
            # This is global and deterministic
            n_pages = max(1, int(hparams.desired_batch_size * 0.01))
            print (f'\nLoading {n_pages} pages ...')
            start_time = time.time()  # Start timing
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset = subtensor.block + hparams.pages_window_speed,
                n_pages = n_pages,
                seed = my_uid 
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size = config.actual_batch_size,
                sequence_length = hparams.sequence_length,
                pages_info = pages,
                tokenizer = hparams.tokenizer
            )
            # TODO: see if wrapping dataloader is faster, with multiple workers and pin_memory=True
            # dataset = torch.utils.data.DataLoader( dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True )
            print(f'\n\tLoading {n_pages} pages completed in {time.time() - start_time} seconds')
            
            # Train my model on the current page.
            print (f'\nTraining {n_pages} pages ...')
            torch.cuda.empty_cache() # Empty cache going into the training step.
            optimizer.zero_grad() # Clear any lingering grads.
            start_time = time.time()  # Start timing
            total_loss = 0.0
            total_steps = hparams.desired_batch_size // config.actual_batch_size
            progress_bar = tqdm(total=total_steps, desc="Training:")
            for idx, batch in enumerate(dataset):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.amp.autocast( device_type = model.device.type, dtype = torch.bfloat16 ):  # Enable autocasting for mixed precision
                    outputs = model(input_ids = input_ids, labels=labels)
                total_loss += outputs.loss.item()
                loss = outputs.loss / total_steps
                scaler.scale( loss ).backward()
                progress_bar.update(1)  # Update the progress bar
                if idx >= total_steps - 1:
                    break
            progress_bar.close()  # Close the progress bar
            
            # Try step with error handling.
            try:
                # grad norm clipping
                if hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
                scaler.step(optimizer)  # Unscale the gradients and step the optimizer
                scaler.update()  # Update the scaler for next iteration
                scheduler.step()  # Update the learning rate.
                optimizer.zero_grad()
            except AssertionError as e:
                print(f"An error occurred during the optimizer step: {e}")
            
            # Clean lingering objects
            del input_ids, labels, outputs
            torch.cuda.empty_cache() # Empty cache at end of step.
            
            # Calculate, print and logg average loss
            average_loss = total_loss / total_steps
            total_time = time.time() - start_time
            steps_per_second = total_steps / total_time
            batches_per_second = config.actual_batch_size * total_steps / total_time
            tokens_per_second = hparams.sequence_length * config.actual_batch_size * total_steps / total_time
            if config.use_wandb:
                wandb.log({
                    "step_loss": average_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    f"incentive{my_uid}": float(metagraph.I[my_uid]),
                    "steps_per_second": steps_per_second,
                    "batches_per_second": batches_per_second,
                    "tokens_per_second": tokens_per_second
                })
            print('\tloss:', average_loss, 'learning_rate:', scheduler.get_last_lr()[0])
            print(f'\tTraining completed in {total_time} seconds, Steps per second: {steps_per_second}, Batches per second: {batches_per_second}, Tokens per second: {tokens_per_second}')
            
            # Select the block to produce a mask for.
            next_upload_block = subtensor.block
            
            # Get the proper mask for my upload block + page.
            print(f'\nCreating upload mask ...')
            start_time = time.time()  # Start timing
            upload_mask = {}
            torch.manual_seed( block_to_mask_window_id(next_upload_block) )  # Seed torch's random generator with the upload mask for its mask_wid.
            torch.cuda.manual_seed_all( block_to_mask_window_id(next_upload_block) )
            for name, param in model.named_parameters():
                next_mask = (torch.rand(param.shape, device=config.device) < (1 / config.compression)).float()
                upload_mask[name] = next_mask
            print(f'\tCreating upload mask_wid mask completed in {time.time() - start_time} seconds')
            
            # Mask the model values given the mask and produce a state dict.                
            print('\nApply upload mask to model ...')
            model_state_dict = model.state_dict()
            model_state_dict['compression'] = config.compression
            for name, param in model.named_parameters():
                param_mask = upload_mask[name].to(param.device)
                param_flat = param.flatten()
                mask_flat = param_mask.flatten()
                unmasked_indices = mask_flat.nonzero(as_tuple=False).flatten()
                unmasked_params = param_flat[unmasked_indices]
                model_state_dict[name] = {'values': unmasked_params.to('cpu')}
                del unmasked_indices
            del upload_mask
            print(f'\tApplied mask to model completed in: {time.time() - start_time} seconds')

            # Upload the state dict of my masked weights.
            print('\nUploading mask ...')
            start_time = time.time()
            upload_filename = f'mask-{wallet.hotkey.ss58_address}-{next_upload_block}.pt'
            with io.BytesIO() as module_buffer:
                torch.save(model_state_dict, module_buffer)
                module_buffer.seek(0)  # Reset the buffer's position to the beginning.
                CLIENT.upload_fileobj(module_buffer, config.bucket, upload_filename)
            CLIENT.put_object_acl(
                Bucket=config.bucket,
                Key=upload_filename,
                GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
                GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
            )
            upload_history.append(upload_filename)
            print(f'\tUploading mask to: {upload_filename} completed in {time.time() - start_time} seconds')

            # Delete old mask files and clean.
            print('\nDeleting history ...')
            start_time = time.time()
            if len(upload_history) > hparams.max_history:
                to_delete = upload_history.pop(0)
                CLIENT.delete_object(Bucket=config.bucket, Key=to_delete)
            print(f'\tDeleting history completed in {time.time() - start_time} seconds')
            
            # Calculate and log global steps per second
            global_step_total_time = time.time() - global_step_start_time
            global_steps_per_second = 1 / global_step_total_time
            if config.use_wandb:
                wandb.log({
                    "global_steps_per_second": global_steps_per_second
                })
                 
        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            # Clean up by deleting the model from S3 if it exists.
            print("Training interrupted. Exiting gracefully.")
            break
    
        # Handle any other exceptions, log the error, and continue after a short delay.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')    
    parser.add_argument('--name', type=str, default=None, help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network UID.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')    
    parser.add_argument('--compression', type=int, default=300)    
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)    
    config = bt.config(parser)    
    config.subtensor.network = 'test'
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'    
    main(config)
