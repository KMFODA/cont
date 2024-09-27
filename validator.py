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

import argparse
import concurrent.futures
import io
import os
import random
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import boto3
import numpy as np
import torch
import wandb
from dotenv import dotenv_values
from tqdm import tqdm
from transformers import LlamaForCausalLM

from dataset import SubsetFineWebEdu2Loader
from hparams import load_hparams

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

# Main function that runs the validator script.
def main(config):
    # Print the configuration for debugging.
    print('\n', '=' * 40, 'Config', '=' * 40)
    print(config)

    # Initialize Bittensor wallet, subtensor, and metagraph.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print('\n', '=' * 40, 'Objects', '=' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')

    # Assert the chain commitment to ensure the validator's bucket is committed on the chain.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'{config.name}', config=config)
        
    # Init the master model
    hparams = load_hparams()
    model = LlamaForCausalLM(config=hparams.model_config) 
    if not config.restart: 
        try:
            # Load the last master from my bucket.
            print('Loading master state ...')
            start_time = time.time()
            master_filename = f'master-{wallet.hotkey.ss58_address}.pt'
            unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
            CLIENT.download_file(config.bucket, master_filename, unique_temp_file)
            master_state_dict = torch.load(unique_temp_file, map_location='cpu', weights_only=True)
            model.load_state_dict(master_state_dict)
            print(f'Loading master state completed in {time.time() - start_time} seconds.') 
        except Exception as e:
            raise ValueError("There is no master to continue from. Run with --restart")
    model.to(config.device)
        
    # Start.
    last_n = int(metagraph.n)
    scores = torch.zeros( last_n, dtype = torch.float32 )
    last_mask_sync = subtensor.block
    while True:
        try:
            # Load the latest chain state and reset my connection.
            print('Loading chain state ...')
            start_time = time.time()
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(f'Loading chain state completed in {time.time() - start_time} seconds.') 
            
            # Function which maps from block to a mask window id such that multiple blocks share the same mask wid.
            # This is used to ensure that the model is not updated too frequently and that the mask is shared.
            # for multiple updates which fall across multiple blocks.
            def block_to_mask_window_id(block: int) -> int:
                return int(block / hparams.mask_window_length)

            print(f'Getting block state ...')
            start_time = time.time()  # Start timing
            block = subtensor.block
            all_sync_blocks = list(range(last_mask_sync - 2, block + 1))
            # This fast forwards us to the block which shares no ids with the previous block.
            # If we don't do this fast forward, then we will be downloading the same masks multiple times.
            # TODO (const) consider if we should just remember the last mask id and download all masks for that id.
            # Or if we should just redownload an apply the same masks.
            last_mask_sync = (block_to_mask_window_id(block) + 1) * hparams.mask_window_length
            print(f'Getting block completed in {time.time() - start_time} seconds')
            # Get buckets per uid if needs update.
            if 'buckets' not in locals() or len(buckets) != len(metagraph.uids):
                buckets = []
                for uid in metagraph.uids:
                    try:
                        buckets.append(subtensor.get_commitment(config.netuid, uid))
                    except:
                        buckets.append(None)

            # For each bucket, get all files that need to be synced.
            print(f'Getting masks names for blocks: {all_sync_blocks} and buckets: {set(buckets)}')
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
                        for obj in page['Contents']:
                            hotkey, blk = obj['Key'].split('-')[1], obj['Key'].split('-')[2].split('.')[0]
                            if int(blk) in all_sync_blocks:
                                mask_wid = block_to_mask_window_id(int(blk))
                                mask_info = SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=obj['Key'], uid=metagraph.hotkeys.index(hotkey), block=int(blk), mask_wid = mask_wid )
                                mask_filenames_per_mask_wid[mask_wid].append(mask_info)
                                num_valid_masks += 1
                except:
                    continue
            print(f'Getting masks names for blocks: {all_sync_blocks} completed in {time.time() - start_time} seconds')

            # Get the mask for mask_wids.
            print(f'Downloading {num_valid_masks} masks for: {all_sync_blocks}')
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
                print(f'Downloading {num_masks_for_mask_wid} mask for mask_wid: {mask_wid} ... ')
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
                print(f'Downloading {n_downloaded} masks completed in {time.time() - start_time} seconds')

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
                    masks_per_uid[info.uid] = mask  # Store the loaded mask
                    compression = mask['compression']
                    compressions_used.add(compression)
                    compression_per_uid[info.uid] = compression
                    print(f"Loaded mask from UID {info.uid} (Hotkey: {info.hotkey}) with compression rate: {compression}")


                # Generate mask indices per compression
                mask_indices_per_compression = {}
                for compression in compressions_used:
                    torch.manual_seed(mask_wid)
                    torch.cuda.manual_seed_all(mask_wid)
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
                    torch.cuda.manual_seed_all(mask_wid)
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
                
                print(f'Averaging {mask_count} masks for mask_wid: {mask_wid} ...')
                start_time = time.time()
                
                param_sums = {}
                param_counts = {}
                for param_name in model.state_dict().keys():
                    param_sums[param_name] = None
                    param_counts[param_name] = None
                    
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
                del mask_indices
                print(f'Applying {mask_count} masks completed in {time.time() - start_time} seconds')
                
                # Delete files and clean up.
                print(f'Deleting files for mask_wid: {mask_wid} ...')
                start_time = time.time()
                for info in temp_files:
                    os.remove( info.temp_file )
                for key in param_sums.keys():
                    param_sums[key] = None
                    param_counts[key] = None
                print(f'Deleting files completed in {time.time() - start_time} seconds')

            # Print completion
            torch.cuda.empty_cache()
            print(f'Downloading masks for blocks: {all_sync_blocks} and mask_wids: {mask_filenames_per_mask_wid.keys()} in {time.time() - full_sync_start_time} seconds')
            
            # Upload a full copy of the model weights to master
            print('Uploading master ...')
            start_time = time.time()
            model_state_dict = model.state_dict()
            upload_filename = f'master-{wallet.hotkey.ss58_address}.pt'
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
            print(f'Uploading master completed in {time.time() - start_time} seconds.')
        
            id_to_eval = None
            if len(list(masks_per_id_per_uid.keys())) == 0:
                print ('No masks to eval. Continue ...')
                continue # Nothing to do.
            id_to_eval = max(list(masks_per_id_per_uid.keys()))
            if len(list(masks_per_id_per_uid[id_to_eval].keys())) == 0:
                print ('No masks to eval. Continue ...')
                continue # Nothing to do.
            
            # Eval this UID.
            uid_to_eval = random.choice(list(masks_per_id_per_uid[id_to_eval].keys()))
            mask_count = mask_count_per_id[id_to_eval]
            mask = masks_per_id_per_uid[id_to_eval][uid_to_eval]
            compression = compression_per_uid[uid_to_eval]
            mask_indices = mask_indices_per_compression[compression]

            del masks_per_id_per_uid
            
            # Get the pages for this block and my_uid.
            # This is global and deterministic
            print('Loading page for eval ...')
            start_time = time.time()  # Start timing
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset = id_to_eval * hparams.mask_window_length,
                n_pages = 3,
                seed = uid_to_eval
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size = config.batch_size,
                sequence_length = hparams.sequence_length,
                pages_info = pages,
                tokenizer = hparams.tokenizer
            )
            print(f'Loading page for eval completed in {time.time() - start_time} seconds.')
            
            # Remove the mask from the model.
            print('Removing the mask ...')
            start_time = time.time()
            for name, param in model.named_parameters():
                if name in mask:
                    indices, mask_values = mask[name]
                    mask_values = mask_values.to(param.device)
                    indices = indices.to(param.device)
                    counts = param_counts[name][indices].to(param.device)
                    counts[counts == 0] = 1
                    adjusted_mask = mask_values / counts
                    param.data[indices] -= adjusted_mask
            print(f'Removing the mask completed in {time.time() - start_time} seconds.')
            
            # Evaluate my model on the current page.
            print('Evaluating without mask ...')
            start_time = time.time()
            model.eval()
            without_avg_loss = 0
            for idx, batch in enumerate(tqdm(dataset)):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.no_grad(): # Turn of grad calculation.
                    with torch.cuda.amp.autocast():  # Enable autocasting for mixed precision
                        outputs = model(input_ids=input_ids, labels=labels)
                without_avg_loss += outputs.loss.item()
            without_avg_loss /= (idx + 1)
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
            print(f'Evaluating without mask completed in {time.time() - start_time} seconds.')
            
            # Add the mask back to the model.
            print('Adding the mask ...')
            start_time = time.time()
            for name, param in model.named_parameters():
                if name in mask:
                    param.data.add_(adjusted_mask)
            del masks_dicts_values
            print(f'Adding the mask completed in {time.time() - start_time} seconds.')
            
            
            # Evaluate my model on the current page.
            print('Evaluating with mask ...')
            start_time = time.time()
            model.eval()
            with_avg_loss = 0
            for idx, batch in enumerate(tqdm(dataset)):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.no_grad(): # Turn of grad calculation.
                    with torch.cuda.amp.autocast():  # Enable autocasting for mixed precision
                        outputs = model(input_ids=input_ids, labels=labels)
                with_avg_loss += outputs.loss.item()
            with_avg_loss /= (idx + 1)
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
            print(f'Evaluating with mask completed in {time.time() - start_time} seconds.')
            
            # Compute the miner score for their mask.
            print('Computing weights ...')
            start_time = time.time()
            if len(scores) < int(metagraph.n):
                scores = torch.concat([scores, torch.zeros( int(metagraph.n) - len(scores), dtype=torch.float32)])
            scores[uid_to_eval] = 0.1 * (with_avg_loss - without_avg_loss) + 0.9 * scores[uid_to_eval]
            # Compute the weights
            non_zero_indices = scores.nonzero(as_tuple=False).flatten()
            weights = torch.zeros_like(scores, dtype=torch.float32)
            weights[non_zero_indices] = torch.softmax(scores[non_zero_indices], dim=0)
            print ('scores', scores.tolist())
            print ('weights', weights.tolist())
            print(f'Computing weights completed in {time.time() - start_time} seconds.')

            # Set weights on chain based on moving scores.
            print('Settting weights on chain ...')
            start_time = time.time()
            subtensor.set_weights(
                wallet=wallet,
                netuid=config.netuid,
                uids=[27, 36, 37],
                weights=weights.tolist(),
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            print(f'Settting weights on chain completed in {time.time() - start_time} seconds.')
                 
        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            break

        # Handle any other exceptions, log the error, clean up, and continue.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)  # Wait for a short period before retrying.
            continue

# Entry point of the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validator script')
    parser.add_argument('--name', type=str, default=None, help='Optional name')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for eval.')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--restart', action='store_true', help='Restart all evaluation history')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
    main(config)
