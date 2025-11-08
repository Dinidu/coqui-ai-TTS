#!/usr/bin/env python3
"""
Production Training Script for Sinhala VITS TTS on H100 GPU
Optimized for NVIDIA H100 with mixed precision and large batch sizes
"""

import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Import and register custom formatter
from custom_formatter import sinhala_formatter
from TTS.tts.datasets import formatters
formatters.sinhala_formatter = sinhala_formatter

# Configuration paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "pretrained"
DATASETS_DIR = BASE_DIR / "datasets" / "sinhala-production"  # Your production dataset
OUTPUT_DIR = BASE_DIR / "output" / "production" / f"vits_sinhala_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Pretrained model paths (optional - for fine-tuning)
USE_PRETRAINED = True  # Set to False to train from scratch
PRETRAINED_MODEL_PATH = MODELS_DIR / "model_file.pth"
PRETRAINED_CONFIG_PATH = MODELS_DIR / "config.json"

# Check GPU availability
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. This script is optimized for GPU training!")
    device = torch.device("cpu")
else:
    # Use H100 optimizations
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for H100
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Count: {torch.cuda.device_count()}")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Production Training Setup for Sinhala VITS TTS")
print("=" * 80)
print(f"Dataset: {DATASETS_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Device: {device}")
if USE_PRETRAINED and PRETRAINED_MODEL_PATH.exists():
    print(f"Fine-tuning from: {PRETRAINED_MODEL_PATH}")
else:
    print("Training from scratch")
print("=" * 80)

# Dataset configuration for production
dataset_config = BaseDatasetConfig(
    formatter="sinhala_formatter",  # Custom formatter
    meta_file_train="metadata_train.csv",  # Separate train file
    meta_file_val="metadata_val.csv",  # Separate validation file
    path=str(DATASETS_DIR),
    language="si",  # Sinhala language code
)

# Audio configuration optimized for quality
audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

# Production training configuration
config = VitsConfig(
    audio=audio_config,
    run_name="vits_sinhala_production",
    
    # H100 optimized batch sizes
    batch_size=64,  # Large batch for H100 (adjust based on your GPU memory)
    eval_batch_size=32,
    batch_group_size=5,
    num_loader_workers=8,  # Multi-threaded data loading
    num_eval_loader_workers=4,
    pin_memory=True,  # Pin memory for faster GPU transfer
    
    # Training configuration
    run_eval=True,
    test_delay_epochs=10,  # Start testing after 10 epochs
    epochs=1000,  # Full training
    
    # Learning rate configuration for fine-tuning or training
    lr_gen=0.0002 if not USE_PRETRAINED else 0.0001,  # Lower LR for fine-tuning
    lr_disc=0.0002 if not USE_PRETRAINED else 0.0001,
    lr_scheduler_gen="ExponentialLR",
    lr_scheduler_gen_params={
        "gamma": 0.999875,
        "last_epoch": -1
    },
    lr_scheduler_disc="ExponentialLR",
    lr_scheduler_disc_params={
        "gamma": 0.999875,
        "last_epoch": -1
    },
    
    # Optimizer configuration
    optimizer="AdamW",
    optimizer_params={
        "betas": [0.8, 0.99],
        "eps": 1e-09,
        "weight_decay": 0.01
    },
    
    # Gradient optimization
    grad_clip=1000.0,  # Gradient clipping
    
    # Text processing
    text_cleaner=None,
    use_phonemes=False,  # Set to True if you want to use phonemes
    compute_input_seq_cache=True,
    max_text_len=350,  # Maximum text length
    max_audio_len=15 * 22050,  # Maximum 15 seconds audio
    min_audio_len=0.5 * 22050,  # Minimum 0.5 seconds audio
    add_blank=True,
    
    # Character set for romanized Sinhala
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=" ,.?!-'\"()[]{}:;/\\|@#$%^&*+=~abcdefghijklmnopqrstuvwxyzāēīōūñṅṇṭḍḷṛṝśṣḥṁ",
        punctuations=" ,.?!-'\"()[]{}:;/\\|",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    
    # Test sentences for monitoring progress
    test_sentences=[
        ["kumbura goviyāṭa vī labā gænīmaṭa upakārī vīm vaśayen pihiṭa vannaki", "speaker", None, None],
        ["manuşyayan visin kumbura vaṭinā deyak lesa piḷigannē da ē nisā ma ya", "speaker", None, None],
        ["siyalu duk duru kara gænīmaṭat siyalu sampat læbīmaṭat hētu vana kuśalayeki", "speaker", None, None],
        ["budun vahansē ē saṉdahā karana vyāyāmayak næti namut", "speaker", None, None],
        ["api heta hambawemu suba dawasak wewa", "speaker", None, None],
    ],
    
    # Checkpointing and logging
    print_step=25,
    log_step=50,
    print_eval=True,
    save_step=1000,  # Save checkpoint every 1000 steps
    save_n_checkpoints=5,  # Keep last 5 checkpoints
    save_best_after=10000,  # Start saving best model after 10k steps
    
    # Mixed precision training for H100
    mixed_precision=True,  # Enable AMP for faster training
    
    # Output paths
    output_path=str(OUTPUT_DIR),
    datasets=[dataset_config],
    
    # CUDNN optimizations
    cudnn_enable=True,
    cudnn_deterministic=False,
    cudnn_benchmark=True,  # Let CUDNN find best algorithms
    
    # Evaluation settings
    eval_split_max_size=256,  # Larger eval set for production
    eval_split_size=0.01,  # 1% of data for evaluation
    
    # VITS specific parameters
    use_speaker_embedding=False,  # Set True for multi-speaker
    use_d_vector_file=False,
    d_vector_dim=0,
    
    # Model architecture parameters (can be tuned)
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    use_spectral_norm=False,
    
    # Loss weights
    kl_loss_alpha=1.0,
    disc_loss_alpha=1.0,
    gen_loss_alpha=1.0,
    feat_loss_alpha=1.0,
    mel_loss_alpha=45.0,
    dur_loss_alpha=1.0,
    speaker_encoder_loss_alpha=1.0,
)

# Initialize audio processor
print("\nInitializing audio processor...")
ap = AudioProcessor.init_from_config(config)

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load data samples
print("Loading dataset samples...")
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=sinhala_formatter,
)

print(f"Loaded {len(train_samples)} training samples")
print(f"Loaded {len(eval_samples)} evaluation samples")

# Calculate steps per epoch for logging
steps_per_epoch = len(train_samples) // config.batch_size
print(f"Steps per epoch: {steps_per_epoch}")

# Initialize model
print("\nInitializing VITS model...")
model = Vits(config, ap, tokenizer, speaker_manager=None)

# Load pretrained weights if available and requested
if USE_PRETRAINED and PRETRAINED_MODEL_PATH.exists():
    print(f"\nLoading pretrained weights from {PRETRAINED_MODEL_PATH}...")
    try:
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle architecture differences
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} keys")
            print(f"First 5 missing: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)} keys")
            print(f"First 5 unexpected: {unexpected_keys[:5]}")
        
        print("✓ Successfully loaded pretrained model weights!")
        
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Training from scratch...")

# Move model to GPU
if device.type == "cuda":
    model = model.cuda()
    print(f"Model moved to GPU: {torch.cuda.get_device_name()}")

# Print model size
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# Initialize trainer with production settings
print("\nInitializing trainer...")
trainer_args = TrainerArgs(
    continue_path=None,  # Set to checkpoint path to resume training
    skip_train_epoch=False,
    use_accelerate=False,  # Set to True for multi-GPU with Accelerate
    gpu=0,  # GPU index (use CUDA_VISIBLE_DEVICES for multi-GPU)
)

trainer = Trainer(
    trainer_args,
    config,
    str(OUTPUT_DIR),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# Save configuration for reference
config_path = OUTPUT_DIR / "config.json"
config.save_json(str(config_path))
print(f"\nConfiguration saved to: {config_path}")

# Start training
print("\n" + "=" * 80)
print("Starting Production Training")
print("=" * 80)
print(f"Epochs: {config.epochs}")
print(f"Batch size: {config.batch_size}")
print(f"Learning rate (gen): {config.lr_gen}")
print(f"Learning rate (disc): {config.lr_disc}")
print(f"Mixed precision: {config.mixed_precision}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80 + "\n")

try:
    trainer.fit()
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
except KeyboardInterrupt:
    print("\n" + "=" * 80)
    print("Training interrupted by user")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
except Exception as e:
    print(f"\nError during training: {e}")
    raise