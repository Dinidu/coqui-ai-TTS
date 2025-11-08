#!/usr/bin/env python3
"""
CPU-optimized fine-tuning script for Sinhala TTS model
Designed for local testing with minimal resources
"""

import os
import sys
import torch
import json
from pathlib import Path

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Import and register custom formatter
sys.path.append(str(Path(__file__).parent))
from custom_formatter import sinhala_formatter

# Register the formatter with the formatters module
from TTS.tts.datasets import formatters
formatters.sinhala_formatter = sinhala_formatter

# Force CPU usage
torch.set_num_threads(4)  # Adjust based on your CPU cores
device = torch.device("cpu")
print(f"Using device: {device}")

# Paths configuration
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models" / "pretrained"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "output" / "finetuned"

# Model and dataset paths
BASE_CHECKPOINT_PATH = MODELS_DIR / "model_file.pth"
BASE_CONFIG_PATH = MODELS_DIR / "config.json"
DATASET_PATH = DATASETS_DIR / "sinhala-tts-dataset" / "oshadi"

# Check if paths exist
if not BASE_CHECKPOINT_PATH.exists():
    print(f"ERROR: Model checkpoint not found at {BASE_CHECKPOINT_PATH}")
    print("Please run: python setup_finetune.py")
    sys.exit(1)

if not BASE_CONFIG_PATH.exists():
    print(f"ERROR: Config file not found at {BASE_CONFIG_PATH}")
    print("Please run: python setup_finetune.py")
    sys.exit(1)

if not DATASET_PATH.exists():
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
    print("Please clone: git clone https://github.com/pnfo/sinhala-tts-dataset.git datasets/sinhala-tts-dataset")
    sys.exit(1)

print("=" * 60)
print("Starting CPU-based Fine-tuning for Sinhala TTS")
print("=" * 60)
print(f"Model: {BASE_CHECKPOINT_PATH}")
print(f"Config: {BASE_CONFIG_PATH}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
# Using custom formatter for Sinhala dataset
dataset_config = BaseDatasetConfig(
    formatter="sinhala_formatter",  # Custom formatter name (registered above)
    meta_file_train="metadata.csv",
    path=str(DATASET_PATH)
)

# Audio configuration (matching the pre-trained model)
audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None
)

# CPU-optimized configuration for local testing
config = VitsConfig(
    audio=audio_config,
    run_name="vits_sinhala_cpu_finetune",
    
    # Very small batch sizes for CPU
    batch_size=2,  # Minimal batch size for CPU
    eval_batch_size=1,
    batch_group_size=1,
    num_loader_workers=0,  # No multiprocessing for CPU
    num_eval_loader_workers=0,
    
    # Short training for testing (just 2 epochs)
    run_eval=True,
    test_delay_epochs=-1,
    epochs=2,  # Just 2 epochs for testing
    
    # Learning rates for fine-tuning
    lr_gen=0.00005,  # Very low learning rate for fine-tuning
    lr_disc=0.00005,
    lr_scheduler_gen="ExponentialLR",
    lr_scheduler_gen_params={"gamma": 0.999, "last_epoch": -1},
    lr_scheduler_disc="ExponentialLR", 
    lr_scheduler_disc_params={"gamma": 0.999, "last_epoch": -1},
    
    # Text processing
    text_cleaner=None,
    use_phonemes=False,
    compute_input_seq_cache=True,
    max_audio_len=10 * 22050,  # Shorter audio for CPU testing
    add_blank=True,
    
    # Romanized Sinhala character set (extracted from dataset)
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=" ,.?abdeghijklmnoprstuvy​æāēīōśşūḷṇṉṭ",
        punctuations=" ,.?",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    
    # Test sentences from the dataset (using romanized text)
    test_sentences=[
        ["kumbura goviyāṭa vī labā gænīmaṭa upakārī vīm", "oshadi", None, None],
        ["manuşyayan visin kumbura vaṭinā deyak", "oshadi", None, None],
        ["siyalu duk duru kara gænīmaṭat", "oshadi", None, None],
    ],
    
    # Monitoring and saving
    print_step=10,  # Print every 10 steps
    print_eval=True,
    mixed_precision=False,  # Disable mixed precision for CPU
    output_path=str(OUTPUT_DIR),
    datasets=[dataset_config],
    cudnn_benchmark=False,
    
    # Small evaluation set for quick testing
    eval_split_max_size=10,
    eval_split_size=0.1,
    
    # Save settings
    save_step=100,  # Save frequently for testing
    save_n_checkpoints=1,
    save_best_after=0,
)

# Initialize audio processor
print("\nInitializing audio processor...")
ap = AudioProcessor.init_from_config(config)

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load data samples
print("Loading dataset samples...")
try:
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=sinhala_formatter,  # Pass the formatter function directly
    )
    print(f"Loaded {len(train_samples)} training samples")
    print(f"Loaded {len(eval_samples)} evaluation samples")
    
    # Limit samples for CPU testing
    max_train_samples = 20  # Use only 20 samples for quick testing
    max_eval_samples = 5
    
    if len(train_samples) > max_train_samples:
        train_samples = train_samples[:max_train_samples]
        print(f"Limited training to {len(train_samples)} samples for CPU testing")
    
    if len(eval_samples) > max_eval_samples:
        eval_samples = eval_samples[:max_eval_samples]
        print(f"Limited evaluation to {len(eval_samples)} samples for CPU testing")
        
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check that the dataset path and format are correct")
    sys.exit(1)

# Initialize model
print("\nInitializing VITS model...")
model = Vits(config, ap, tokenizer, speaker_manager=None)

# Load pre-trained weights
print(f"\nLoading pre-trained weights from {BASE_CHECKPOINT_PATH}...")
try:
    checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Load with strict=False to handle minor architecture differences
    model.load_state_dict(state_dict, strict=False)
    print("✓ Successfully loaded pre-trained model weights!")
    
except Exception as e:
    print(f"Warning: Could not fully load checkpoint: {e}")
    print("Proceeding with partial weight loading or training from scratch...")

# Move model to CPU explicitly
model = model.to(device)

# Initialize trainer
print("\nInitializing trainer...")
trainer = Trainer(
    TrainerArgs(
        continue_path=None,  # Start fresh fine-tuning
        skip_train_epoch=False,
    ),
    config,
    str(OUTPUT_DIR),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# Start fine-tuning
print("\n" + "=" * 60)
print("Starting fine-tuning process...")
print("This will take a while on CPU. Press Ctrl+C to stop.")
print("=" * 60 + "\n")

try:
    trainer.fit()
    print("\n" + "=" * 60)
    print("Fine-tuning completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n" + "=" * 60)
    print("Training interrupted by user")
    print(f"Partial model saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
except Exception as e:
    print(f"\nError during training: {e}")
    sys.exit(1)