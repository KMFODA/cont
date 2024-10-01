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

                # Load all masks as state dicts.
                print(f'\n\tLoading mask indices for mask_wid: {mask_wid} ...')
                start_time = time.time()
                mask_count = 0
                aggregated_updates = {}

                # For each downloaded mask file (from temp_files)
                for info in temp_files:
                    # Load the mask directly
                    mask = torch.load(info.temp_file, map_location='cpu')
                    mask_count += 1
                    # Iterate over each parameter in the mask
                    for name in mask.keys():
                        if name == 'compression':
                            continue  # Skip if compression is included
                        indices = mask[name]['indices']
                        values = mask[name]['values']
                        # Ensure indices and values are tensors
                        indices = indices.to(config.device)
                        values = values.to(config.device)
                        # Initialize the aggregated_updates entry if not present
                        if name not in aggregated_updates:
                            aggregated_updates[name] = {}
                            aggregated_updates[name]['indices'] = []
                            aggregated_updates[name]['values'] = []
                        # Append the indices and values to the lists
                        aggregated_updates[name]['indices'].append(indices)
                        aggregated_updates[name]['values'].append(values)
                print(f'\t\tLoading values and indices completed in {time.time() - start_time} seconds')
                
                # Include own params
                for name in masked_updates:
                    indices = masked_updates[name]['indices'].to(config.device)
                    values = masked_updates[name]['values'].to(config.device)
                    if name not in aggregated_updates:
                        aggregated_updates[name] = {}
                        aggregated_updates[name]['indices'] = []
                        aggregated_updates[name]['values'] = []
                    aggregated_updates[name]['indices'].append(indices)
                    aggregated_updates[name]['values'].append(values)
                
                # Average the masks before applying.
                print(f'Averaging {mask_count} masks for mask_wid: {mask_wid} ...')
                start_time = time.time()
                
                param_sums = {}
                param_counts = {}
                for name, param in model.named_parameters():
                    param_size = param.numel()
                    param_sums[name] = torch.zeros(param_size, dtype=torch.float32, device=config.device)
                    param_counts[name] = torch.zeros(param_size, dtype=torch.float32, device=config.device)

                # Aggregate masks
                for name in aggregated_updates:
                    all_indices = torch.cat(aggregated_updates[name]['indices'])
                    all_values = torch.cat(aggregated_updates[name]['values'])
                    # Handle overlapping indices
                    # Use index_add_ to sum values at the same indices
                    param_sums[name].index_add_(0, all_indices, all_values)
                    # Increment counts
                    param_counts[name].index_add_(0, all_indices, torch.ones_like(all_values))
                
                # Compute average masks
                for name in param_sums:
                    counts = param_counts[name]
                    # Avoid division by zero
                    counts[counts == 0] = 1.0
                    # Compute average updates
                    avg_updates = param_sums[name] / counts
                    # Reshape to parameter shape
                    avg_updates = avg_updates.view_as(model.state_dict()[name])
                    # Store the average updates
                    param_sums[name] = avg_updates

                print(f'Averaged masks in {time.time() - start_time} seconds')

                # Set the average into the model.
                print(f'Applying {mask_count} masks for mask_wid: {mask_wid} ...')
                start_time = time.time()  # Start timing
                for name, param in model.named_parameters():
                    if name in param_sums:
                        update = param_sums[name]
                        # Apply the update
                        param.data += update.to(param.device)
                    else:
                        print(f"Name not in average_masks")
                
                print(f'Applying {mask_count} masks completed in {time.time() - start_time} seconds')
                
                # Delete files and clean up.
                print(f'\n\tDeleting files for mask_wid: {mask_wid} ...')
                start_time = time.time()
                for info in temp_files:
                    os.remove(info.temp_file)
                del param_sums
                del param_counts
                del aggregated_updates
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
            
            # Unscale the gradients before clipping and accessing them
            scaler.unscale_(optimizer)
            
            # Grad norm clipping
            if hparams.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
            
            # Collect the gradients for mask creation
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.detach().clone()
                else:
                    gradients[name] = torch.zeros_like(param.data)

            
            # Try step with error handling.
            try:
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
            masked_updates = {}
            learning_rate = hparams.learning_rate

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = gradients[name]
                    grad_flat = grad.view(-1)
                    grad_abs = grad_flat.abs()

                    # Determine K based on compression ratio
                    K = max(1, int(grad_flat.numel() // config.compression))

                    # Get top-K indices
                    topk_values, topk_indices = torch.topk(grad_abs, K)

                    # Get the update values at the top-K indices
                    update_values = -learning_rate * grad_flat[topk_indices]

                    # Store the indices and values
                    masked_updates[name] = {
                        'indices': topk_indices.cpu(),
                        'values': update_values.cpu(),
                    }
                else:
                    # If no gradient, skip
                    pass
            del gradients
            print(f'\tCreating upload mask_wid mask completed in {time.time() - start_time} seconds')
            

            # Upload the state dict of my masked weights.
            print('\nUploading mask ...')
            start_time = time.time()
            upload_filename = f'mask-{wallet.hotkey.ss58_address}-{next_upload_block}.pt'
            with io.BytesIO() as module_buffer:
                torch.save(masked_updates, module_buffer)
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
