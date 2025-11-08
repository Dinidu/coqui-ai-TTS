#!/usr/bin/env python3
"""
Multi-GPU Training Script using PyTorch Distributed Data Parallel
Optimized for multiple H100 GPUs
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train_vits_h100 import *

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_distributed(rank, world_size):
    """Training function for each GPU"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Modify configuration for distributed training
    config.batch_size = config.batch_size // world_size  # Split batch across GPUs
    config.num_loader_workers = config.num_loader_workers // world_size
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    
    # Initialize model
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    # Load pretrained weights if needed (only on rank 0)
    if rank == 0 and USE_PRETRAINED and PRETRAINED_MODEL_PATH.exists():
        print(f"Loading pretrained weights from {PRETRAINED_MODEL_PATH}...")
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    # Move model to GPU and wrap with DDP
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True  # VITS may have unused parameters
    )
    
    # Create distributed sampler for data loading
    from torch.utils.data import DistributedSampler
    from TTS.tts.datasets.dataset import TTSDataset
    
    # Create dataset
    train_dataset = TTSDataset(
        model=model.module if hasattr(model, 'module') else model,
        samples=train_samples,
        ap=ap,
        tokenizer=tokenizer,
        compute_f0=False,
        compute_input_seq_cache=config.compute_input_seq_cache
    )
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create data loader with distributed sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_loader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )
    
    # Training loop (simplified for demonstration)
    # In production, use the full Trainer class with distributed support
    
    if rank == 0:
        print(f"\nDistributed training on {world_size} GPUs")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"Total effective batch size: {config.batch_size * world_size}")
    
    # Initialize trainer with distributed settings
    trainer_args = TrainerArgs(
        continue_path=None,
        skip_train_epoch=False,
        use_accelerate=False,
        gpu=rank,  # Use specific GPU rank
    )
    
    # Only save checkpoints and logs on rank 0
    if rank == 0:
        output_path = str(OUTPUT_DIR)
    else:
        output_path = None
    
    trainer = Trainer(
        trainer_args,
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples if rank == 0 else None,  # Only evaluate on rank 0
    )
    
    try:
        trainer.fit()
    finally:
        cleanup_distributed()

def main():
    """Main function to launch distributed training"""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for distributed training!")
        sys.exit(1)
    
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print(f"Only {world_size} GPU(s) available. For single GPU, use train_vits_h100.py")
        print("Running single GPU training...")
        # Fall back to single GPU training
        import train_vits_h100
        sys.exit(0)
    
    print(f"Starting distributed training on {world_size} GPUs")
    print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(world_size)]}")
    
    # Launch distributed training
    mp.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()