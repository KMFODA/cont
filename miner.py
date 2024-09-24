# The MIT License (MIT)
# © 2024 Chakana.tech

import argparse
import concurrent.futures
import io
import os
import tempfile
import time
import traceback
import uuid
from typing import List, Tuple

import bittensor as bt
import boto3
import numpy as np
import torch
import torch.optim as optim
import wandb
from dotenv import dotenv_values
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import LlamaForCausalLM

from dataset import SubsetFineWebEdu2Loader
from hparams import load_hparams

# Performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# AWS S3 client setup
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_config.get("AWS_SECRET_ACCESS_KEY")
CLIENT: boto3.client = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def initialize_bittensor(config):
    """Initialize Bittensor objects and validate wallet."""
    print(config)
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"Wallet {wallet} is not registered on subnet: {metagraph.netuid}"
        )

    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print(
        f"Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}"
    )

    return wallet, subtensor, metagraph, my_uid


def initialize_wandb(config, my_uid):
    """Initialize Weights and Biases for experiment tracking."""
    if config.use_wandb:
        return wandb.init(
            project="cont", resume="allow", name=f"M{my_uid}", config=config
        )
    return None


def sync_chain_state(config, subtensor, my_uid, wallet):
    """Sync the current chain state and commit bucket if necessary."""
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f"Chain commitment does not match: {config.bucket}")
    except Exception:
        subtensor.commit(wallet, config.netuid, config.bucket)
    print("Bucket:", config.bucket)


def sync_model(config, subtensor, metagraph, hparams):
    """Sync the full model state if necessary."""
    model = None
    master_uid = int(metagraph.S.argmax())
    master_uid = 1
    master_bucket = subtensor.get_commitment(config.netuid, master_uid)
    master_hotkey = metagraph.hotkeys[master_uid]
    master_filename = f"master-{master_hotkey}.pt"
    unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")

    CLIENT.download_file(master_bucket, master_filename, unique_temp_file)
    master_state_dict = torch.load(
        unique_temp_file, map_location="cpu", weights_only=True
    )

    model = LlamaForCausalLM(config=hparams.model_config)
    model.load_state_dict(master_state_dict)
    model.to(config.device)
    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        weight_decay=config.optimizer_weight_decay,
        foreach=True,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=hparams.epoch_length, eta_min=4e-5, last_epoch=-1
    )

    return model, optimizer, scheduler


def download_masks(config, subtensor, metagraph, model, last_mask_sync, hparams):
    """Download and apply masks from other miners."""
    block = subtensor.block
    all_sync_blocks = list(range(last_mask_sync - 2, block + 1))
    last_mask_sync = block

    # Get buckets per uid if needs update.
    if "buckets" not in locals() or len(buckets) != len(metagraph.uids):
        buckets = []
        for uid in metagraph.uids:
            try:
                buckets.append(subtensor.get_commitment(config.netuid, uid))
            except:
                buckets.append(None)

    # Get the mask for all sync blocks.
    print(f"Downloading masks for blocks: {all_sync_blocks}")
    for blk in all_sync_blocks:
        mask_filenames = [
            f"mask-{str(metagraph.hotkeys[uid])}-{blk}.pt" for uid in metagraph.uids
        ]
        temp_files = download_mask_files(buckets, mask_filenames)

        if not temp_files:
            continue

        mask_indices = create_sync_mask(model, blk, config.device, hparams.compression)
        masks_dicts_values, mask_count = load_and_average_masks(
            temp_files, mask_indices, model, blk
        )
        apply_masks_to_model(model, masks_dicts_values, mask_indices, mask_count, blk)

        print(f"Deleting files for block: {blk} ...")
        start_time = time.time()
        for file in temp_files:
            os.remove(file)
        print(f"Deleting files completed in {time.time() - start_time} seconds")

    return last_mask_sync


def download_file(bucket, filename):
    try:
        if bucket is None:
            return None
        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
        CLIENT.download_file(bucket, filename, unique_temp_file)
        return unique_temp_file
    except:
        return None


def download_mask_files(buckets, mask_filenames):
    """Download mask files from S3 buckets."""
    temp_files = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_file, bucket, filename)
            for bucket, filename in zip(buckets, mask_filenames)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                temp_files.append(result)

    return temp_files


def create_sync_mask(model, block, device, compression):
    """Create a synchronization mask for the given block."""
    mask_indices = {}
    torch.manual_seed(block)
    for name, param in model.named_parameters():
        param = param.to(device)
        next_mask = (torch.rand(param.shape, device=device) < (1 / compression)).float()
        indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
        mask_indices[name] = indices
    return mask_indices


def load_and_average_masks(temp_files, mask_indices, model, blk):
    """Load and average masks from temporary files."""
    masks_dicts_values = {}
    mask_count = 0
    for file in temp_files:
        mask = torch.load(file, map_location="cpu", weights_only=True)
        mask_count += 1
        for name in mask.keys():
            mask_values = mask[name]["values"]
            if torch.isnan(mask_values).any():
                continue
            param_shape = model.get_parameter(name).shape
            indices = mask_indices[name]
            decompressed = torch.zeros(param_shape, device="cpu").flatten()
            decompressed[indices] = mask_values
            if name not in masks_dicts_values:
                masks_dicts_values[name] = decompressed.view(param_shape)
            else:
                masks_dicts_values[name] += decompressed.view(param_shape)

    # Average the masks before applying.
    print(f"Averaging {mask_count} masks for block: {blk} ...")
    start_time = time.time()
    for key in masks_dicts_values.keys():
        masks_dicts_values[key] /= mask_count
    print(f"Averaged state dicts in {time.time() - start_time} seconds")

    return masks_dicts_values, mask_count


def apply_masks_to_model(model, masks_dicts_values, mask_indices, mask_count, blk):
    """Apply averaged masks to the model."""
    print(f"Applying {mask_count} masks for block: {blk} ...")
    start_time = time.time()  # Start timing
    for name, param in model.named_parameters():
        indices = mask_indices[name]
        if name in masks_dicts_values:
            if masks_dicts_values[name].shape == param.shape:
                on_device = masks_dicts_values[name].to(model.device).flatten()
                param_flat = param.data.flatten()
                param_flat[indices] = on_device[indices]
                param.data.copy_(param_flat.view(param.shape))
                del on_device, param_flat
            else:
                print(
                    f"Shape mismatch for {name}: expected {param.shape}, got {masks_dicts_values[name].shape}"
                )

    for key in masks_dicts_values.keys():
        masks_dicts_values[key].cpu()
    for key in mask_indices.keys():
        mask_indices[key] = mask_indices[key].cpu()
    del mask_indices, masks_dicts_values
    print(
        f"Applying {mask_count} masks completed in {time.time() - start_time} seconds"
    )


def create_upload_mask(model, block, device, compression):
    """Create an upload mask for the current model state."""
    upload_mask = {}
    torch.manual_seed(block)
    for name, param in model.named_parameters():
        param = param.to(device)
        next_mask = (torch.rand(param.shape, device=device) < (1 / compression)).float()
        upload_mask[name] = next_mask.to("cpu")
    return upload_mask


def apply_mask_to_model(model, upload_mask):
    """Apply the upload mask to the model and produce a state dict."""
    model_state_dict = model.state_dict()
    for name, param in model.named_parameters():
        param_mask = upload_mask[name].to(param.device)
        param_flat = param.flatten()
        mask_flat = param_mask.flatten()
        unmasked_indices = mask_flat.nonzero(as_tuple=False).flatten()
        unmasked_params = param_flat[unmasked_indices]
        model_state_dict[name] = {"values": unmasked_params.to("cpu")}
    return model_state_dict


def upload_model_to_s3(model_state_dict, bucket, filename):
    """Upload the model state dict to S3."""
    print("Uploading mask ...")
    start_time = time.time()
    with io.BytesIO() as module_buffer:
        torch.save(model_state_dict, module_buffer)
        module_buffer.seek(0)
        CLIENT.upload_fileobj(module_buffer, bucket, filename)
    CLIENT.put_object_acl(
        Bucket=bucket,
        Key=filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
    )
    print(
        f"Uploading mask to: {filename} completed in {time.time() - start_time} seconds"
    )


def train_model(config, model, optimizer, scheduler, dataset, hparams):
    """Train the model on the current page."""
    total_loss = 0.0
    total_steps = config.desired_batch_size // config.actual_batch_size
    progress_bar = tqdm(total=total_steps, desc="Training:")

    for idx, batch in enumerate(dataset):
        input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
        labels = torch.where(
            input_ids == hparams.tokenizer.pad_token_id, -100, input_ids.clone()
        )

        with torch.amp.autocast(device_type=model.device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)

        total_loss += outputs.loss.item()
        outputs.loss.backward()
        progress_bar.update(1)

        if idx >= total_steps - 1:
            break

    progress_bar.close()

    if config.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    average_loss = total_loss / total_steps
    print(f"Loss: {average_loss}, Learning Rate: {scheduler.get_last_lr()[0]}")

    return average_loss


def create_and_upload_mask(config, model, wallet, subtensor, hparams):
    """Create and upload a mask for the current model state."""
    next_upload_block = subtensor.block
    upload_mask = create_upload_mask(
        model, next_upload_block, config.device, hparams.compression
    )
    model_state_dict = apply_mask_to_model(model, upload_mask)

    upload_filename = f"mask-{wallet.hotkey.ss58_address}-{next_upload_block}.pt"
    upload_model_to_s3(model_state_dict, config.bucket, upload_filename)

    return upload_filename


def main(config):
    wallet, subtensor, metagraph, my_uid = initialize_bittensor(config)
    run = initialize_wandb(config, my_uid)
    sync_chain_state(config, subtensor, my_uid, wallet)

    hparams = load_hparams()
    model = None
    upload_history = []
    last_mask_sync = 0
    last_master_sync = 0

    while True:
        try:
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)

            # Sync the full model state every hparams.epoch_length
            if (
                model is None
                or subtensor.block - last_master_sync > hparams.epoch_length
            ):
                model, optimizer, scheduler = sync_model(
                    config, subtensor, metagraph, hparams
                )
                last_master_sync = subtensor.block
                last_mask_sync = last_master_sync

            last_mask_sync = download_masks(
                config, subtensor, metagraph, model, last_mask_sync, hparams
            )

            n_pages = max(1, int(config.desired_batch_size * 0.01))
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset=subtensor.block + n_pages, n_pages=n_pages, seed=my_uid
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size=config.actual_batch_size,
                sequence_length=hparams.sequence_length,
                pages_info=pages,
                tokenizer=hparams.tokenizer,
            )

            average_loss = train_model(
                config, model, optimizer, scheduler, dataset, hparams
            )

            if config.use_wandb:
                wandb.log(
                    {
                        "step_loss": average_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        f"incentive{my_uid}": float(metagraph.I[my_uid]),
                    }
                )

            upload_filename = create_and_upload_mask(
                config, model, wallet, subtensor, hparams
            )
            upload_history.append(upload_filename)

            # Delete old mask files and clean.
            print("Deleting history ...")
            start_time = time.time()
            if len(upload_history) > 5:
                to_delete = upload_history.pop(0)
                CLIENT.delete_object(Bucket=config.bucket, Key=to_delete)
            print(f"Deleting history completed in {time.time() - start_time} seconds")

        except (KeyboardInterrupt, SystemExit):
            print("Training interrupted. Exiting gracefully.")
            break
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Miner script")
    parser.add_argument("--name", type=str, default=None, help="Optional miner name")
    parser.add_argument(
        "--netuid", type=int, default=212, help="Bittensor network UID."
    )
    parser.add_argument("--bucket", type=str, default="decis", help="S3 bucket name")
    parser.add_argument(
        "--desired_batch_size",
        type=int,
        default=512,
        help="Training batch size per step",
    )
    parser.add_argument(
        "--actual_batch_size",
        type=int,
        default=9,
        help="Training batch size per accumulation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--optimizer_beta1", type=float, default=0.9, help="Beta1 for the optimizer"
    )
    parser.add_argument(
        "--optimizer_beta2", type=float, default=0.95, help="Beta2 for the optimizer"
    )
    parser.add_argument(
        "--optimizer_weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., cpu or cuda)",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights and Biases for logging"
    )
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = "test"
    config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"
    main(config)
