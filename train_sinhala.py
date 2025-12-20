#!/usr/bin/env python3
"""
Unified Training Script for Sinhala VITS TTS
Works on any GPU (H100, A100, V100, RTX, etc.) or CPU
"""

import os
import sys
import re
import torch
import argparse
import warnings
import signal
from pathlib import Path
from datetime import datetime

# Suppress TF32 deprecation warning from trainer library
# This is temporary until the trainer library updates to new PyTorch API
warnings.filterwarnings("ignore", message=".*allow_tf32.*deprecated.*")

# Global trainer reference for signal handling
_trainer = None

# Number of best models to keep
KEEP_N_BEST_MODELS = 3


def keep_n_best_models(output_dir: Path, n: int = 3):
    """
    Keep only the N best models in the output directory.
    Ranks models by their loss value (lower is better).
    Maintains: best_model.pth (best), second_best_model.pth, third_best_model.pth
    """
    import glob

    # Find all best_model_*.pth files (but not best_model.pth itself or ranked ones)
    pattern = str(output_dir / "best_model_*.pth")
    best_models = glob.glob(pattern)

    if len(best_models) <= n:
        return  # Nothing to cleanup

    # Load each model and extract loss
    model_losses = []
    for model_path in best_models:
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            loss_info = checkpoint.get('model_loss', {})

            # Extract the relevant loss value
            if isinstance(loss_info, dict):
                # Prefer eval_loss if available, otherwise train_loss
                loss = loss_info.get('eval_loss') or loss_info.get('train_loss') or float('inf')
            else:
                loss = loss_info if loss_info is not None else float('inf')

            step = checkpoint.get('step', 0)
            model_losses.append((model_path, loss, step))
        except Exception as e:
            print(f"Warning: Could not read {model_path}: {e}")
            continue

    if not model_losses:
        return

    # Sort by loss (lower is better)
    model_losses.sort(key=lambda x: x[1])

    # Keep top N, delete the rest
    models_to_keep = model_losses[:n]
    models_to_delete = model_losses[n:]

    # Delete excess models
    for model_path, loss, step in models_to_delete:
        try:
            os.remove(model_path)
            print(f"  Removed old best model: {os.path.basename(model_path)} (loss: {loss:.4f})")
        except Exception as e:
            print(f"Warning: Could not delete {model_path}: {e}")

    # Create ranked copies for easy identification
    rank_names = ['best_model.pth', 'second_best_model.pth', 'third_best_model.pth']

    for i, (model_path, loss, step) in enumerate(models_to_keep):
        if i < len(rank_names):
            ranked_path = output_dir / rank_names[i]
            try:
                import shutil
                shutil.copy2(model_path, ranked_path)
            except Exception as e:
                print(f"Warning: Could not create ranked copy {rank_names[i]}: {e}")

    if models_to_delete:
        print(f"  Kept {len(models_to_keep)} best models, removed {len(models_to_delete)} older ones")


def on_epoch_end_callback(trainer):
    """Callback to manage best models after each epoch"""
    output_dir = Path(trainer.output_path)
    keep_n_best_models(output_dir, KEEP_N_BEST_MODELS)


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global _trainer
    print("\n\nReceived interrupt signal. Cleaning up...")
    if _trainer:
        try:
            _trainer.stop_training = True
        except:
            pass
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, VitsArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Import and register custom formatter
from production.custom_formatter import sinhala_formatter
import TTS.tts.datasets as tts_datasets

# Register the formatter as a module attribute so it can be found
setattr(tts_datasets, 'sinhala_formatter', sinhala_formatter)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sinhala VITS TTS Model")
    parser.add_argument("--dataset_path", type=str, default="datasets/sinhala-native",
                        help="Path to the dataset")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--pretrained_model", type=str, default="models/pretrained/model_file.pth",
                        help="Path to pretrained model weights (default: models/pretrained/model_file.pth)")
    parser.add_argument("--pretrained_config", type=str, default="models/pretrained/config.json",
                        help="Path to pretrained model config.json (default: models/pretrained/config.json)")
    parser.add_argument("--mode", type=str, choices=['finetune', 'scratch', 'custom'], default='finetune',
                        help="Training mode: finetune (default), scratch, or custom")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto-detected based on GPU if not set)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="Use CUDA if available")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    return parser.parse_args()

def detect_hardware():
    """Detect available hardware and return optimized settings"""
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "batch_size": 2,
            "mixed_precision": False,
            "num_workers": 2,
            "gpu_name": None
        }
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    # Determine batch size based on GPU
    if "H100" in gpu_name or "A100" in gpu_name:
        batch_size = 64 if gpu_memory > 40 else 32
    elif "V100" in gpu_name or "A10" in gpu_name:
        batch_size = 32 if gpu_memory > 16 else 16
    elif "RTX 4090" in gpu_name or "RTX 3090" in gpu_name:
        batch_size = 32 if gpu_memory > 20 else 16
    elif "RTX" in gpu_name or "GTX" in gpu_name:
        batch_size = 16 if gpu_memory > 8 else 8
    else:
        # Default for unknown GPUs
        batch_size = 16 if gpu_memory > 10 else 8
    
    return {
        "device": "cuda",
        "batch_size": batch_size,
        "mixed_precision": gpu_memory > 8,  # Use mixed precision for GPUs with >8GB
        "num_workers": min(8, os.cpu_count() or 4),
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory
    }

def main():
    args = parse_args()
    
    # Import config merger if needed
    if args.pretrained_config:
        from config_merger import ConfigMerger
    
    # Detect hardware
    hw_config = detect_hardware()
    
    # Set device
    if args.use_cuda and hw_config["device"] == "cuda":
        device = torch.device("cuda")
        # Enable GPU optimizations (using old API for PyTorch 2.0-2.4 compatibility)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count() or 4)
    
    # Use provided batch size or auto-detected
    batch_size = args.batch_size if args.batch_size else hw_config["batch_size"]
    mixed_precision = args.mixed_precision or hw_config["mixed_precision"]
    
    # Safety checks for fine-tuning mode
    if args.mode == 'finetune':
        if batch_size < 8:
            print(f"⚠️  Warning: Batch size {batch_size} is too small for stable fine-tuning!")
            print("   Recommended minimum: 16")
            print("   Risk: Gradient explosion and model corruption")
            if batch_size < 4:
                print("❌ ERROR: Batch size < 4 is dangerous. Setting to minimum of 8.")
                batch_size = 8
        
        if args.lr and args.lr > 0.0001:
            print(f"⚠️  Warning: Learning rate {args.lr} is too high for fine-tuning!")
            print("   Recommended: 1e-5 (0.00001)")
            print("   Risk: Destroying pretrained weights")
            print("   Overriding to safe value: 1e-5")
            args.lr = 1e-5
    
    # Set output directory
    if args.output_path:
        output_dir = Path(args.output_path)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path("output") / f"vits_sinhala_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("=" * 80)
    print("Sinhala VITS TTS Training")
    print("=" * 80)
    print(f"Device: {device}")
    if hw_config["gpu_name"]:
        print(f"GPU: {hw_config['gpu_name']} ({hw_config['gpu_memory']:.1f} GB)")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    
    # Handle training mode
    pretrained_model_path = Path(args.pretrained_model)
    pretrained_config_path = Path(args.pretrained_config)
    
    if args.mode == 'scratch':
        # Force training from scratch
        use_pretrained = False
        use_pretrained_config = False
        print(f"Mode: TRAINING FROM SCRATCH")
    elif args.mode == 'finetune':
        # Use defaults or provided paths for fine-tuning
        use_pretrained = pretrained_model_path.exists()
        use_pretrained_config = pretrained_config_path.exists()
        
        if not use_pretrained_config:
            print(f"⚠️  Warning: Pretrained config not found at {pretrained_config_path}")
            print("   Falling back to training from scratch")
            use_pretrained = False
        else:
            print(f"Mode: FINE-TUNING")
            print(f"  Config: {args.pretrained_config}")
            if use_pretrained:
                print(f"  Model: {args.pretrained_model}")
            else:
                print(f"  ⚠️  Model not found at {pretrained_model_path}, will use random init")
    elif args.mode == 'custom':
        # Use exactly what user provided
        use_pretrained = args.pretrained_model and pretrained_model_path.exists()
        use_pretrained_config = args.pretrained_config and pretrained_config_path.exists()
        print(f"Mode: CUSTOM CONFIGURATION")
        if use_pretrained_config:
            print(f"  Config: {args.pretrained_config}")
        if use_pretrained:
            print(f"  Model: {args.pretrained_model}")
    
    print("=" * 80)
    
    # Load or create configuration
    if use_pretrained_config:
        # Load pretrained config and merge with training parameters
        print("\nLoading and merging pretrained config...")
        merger = ConfigMerger()
        config_dict = merger.merge(
            args.pretrained_config,
            args.dataset_path,
            output_path=str(output_dir),
            custom_params={
                'batch_size': batch_size,
                'eval_batch_size': max(1, batch_size // 2),
                'epochs': args.epochs,
                'lr': args.lr,
                'lr_gen': args.lr,
                'lr_disc': args.lr,
                'mixed_precision': mixed_precision,
                'num_loader_workers': args.num_workers,
            }
        )
        
        # Load config from merged dictionary
        from TTS.config import load_config
        
        # Save merged config to temp file and reload (workaround for config loading)
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
            temp_config_path = f.name
        
        config = load_config(temp_config_path)
        
        # Clean up temp file
        os.unlink(temp_config_path)
        
        # Final safety validation for fine-tuning
        if config.lr > 0.0001 or config.lr_gen > 0.0001 or config.lr_disc > 0.0001:
            print("❌ CRITICAL: Config merger failed to set safe learning rates!")
            print(f"   lr={config.lr}, lr_gen={config.lr_gen}, lr_disc={config.lr_disc}")
            print("   Forcing safe values...")
            config.lr = 1e-5
            config.lr_gen = 1e-5
            config.lr_disc = 1e-5
        
        # Dataset config is already in the merged config
        dataset_config = None  # Will use config.datasets directly
    else:
        # Original configuration for training from scratch
        # Dataset configuration
        dataset_config = BaseDatasetConfig(
            formatter="sinhala_formatter",
            meta_file_train="metadata_train.csv",
            meta_file_val="metadata_val.csv",
            path=str(args.dataset_path),
            language="si",
        )
        
        # Audio configuration
        audio_config = VitsAudioConfig(
            sample_rate=22050,
            win_length=1024,
            hop_length=256,
            num_mels=80,
            mel_fmin=0,
            mel_fmax=None,
        )
        
        # Model configuration
        config = VitsConfig(
            audio=audio_config,
            run_name="vits_sinhala",
            
            # Batch configuration
            batch_size=batch_size,
            eval_batch_size=max(1, batch_size // 2),
            batch_group_size=5 if device.type == "cuda" else 0,
            num_loader_workers=args.num_workers,
            num_eval_loader_workers=2,
            
            # Training configuration
            run_eval=True,
            test_delay_epochs=10,
            epochs=args.epochs,
            
            # Learning rate
            lr_gen=args.lr,
            lr_disc=args.lr,
            lr_scheduler_gen="ExponentialLR",
            lr_scheduler_gen_params={
                "gamma": 0.999875,
            },
            lr_scheduler_disc="ExponentialLR",
            lr_scheduler_disc_params={
                "gamma": 0.999875,
            },
            
            # Optimizer
            optimizer="AdamW",
            optimizer_params={
                "betas": [0.8, 0.99],
                "eps": 1e-09,
                "weight_decay": 0.01
            },
            
            # Gradient clipping (list for multiple optimizers: [generator, discriminator])
            grad_clip=[1000.0, 1000.0],
            
            # Text processing
            text_cleaner=None,
            use_phonemes=False,
            compute_input_seq_cache=True,
            max_text_len=350,
        max_audio_len=15 * 22050,
        min_audio_len=0.5 * 22050,
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
        
        # Test sentences
        test_sentences=[
            ["api heta hambawemu suba dawasak wewa", "speaker", None, None],
        ],
        
        # Logging
        print_step=25,
        print_eval=True,
        
        # Mixed precision
        mixed_precision=mixed_precision,
        
        # Output
        output_path=str(output_dir),
        datasets=[dataset_config],
        
        # CUDA optimizations
        cudnn_enable=device.type == "cuda",
        cudnn_benchmark=device.type == "cuda",
        
        # Evaluation
        eval_split_max_size=256,
        eval_split_size=0.01,

        # Checkpoint settings (to save storage - keep only best models)
        save_step=5000,  # Less frequent regular checkpoints
        save_n_checkpoints=1,  # Keep only 1 regular checkpoint (for resuming)
        save_all_best=True,  # Save all best models (managed by callback to keep N best)
        save_best_after=100,  # Wait 100 steps before saving best models

        # Model settings
        use_speaker_embedding=False,
        use_d_vector_file=False,
        d_vector_dim=0,
        
        # Model architecture (optimized for memory)
        model_args=VitsArgs(
            hidden_channels=192,
            hidden_channels_ffn_text_encoder=768,
            num_heads_text_encoder=2,
            num_layers_text_encoder=6,
            kernel_size_text_encoder=3,
            dropout_p_text_encoder=0.1,
            dropout_p_duration_predictor=0.5,
            kernel_size_posterior_encoder=5,
            dilation_rate_posterior_encoder=1,
            num_layers_posterior_encoder=16,
            kernel_size_flow=5,
            dilation_rate_flow=1,
            num_layers_flow=4,
            resblock_type_decoder="1",
            resblock_kernel_sizes_decoder=[3, 7, 11],
            resblock_dilation_sizes_decoder=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates_decoder=[8, 8, 2, 2],
            upsample_initial_channel_decoder=512,
            upsample_kernel_sizes_decoder=[16, 16, 4, 4],
            use_spectral_norm_disriminator=False,
            spec_segment_size=32,
            use_sdp=True,
        ),
    )
    
    # Initialize audio processor
    print("\nInitializing audio processor...")
    ap = AudioProcessor.init_from_config(config)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # Load data samples
    print("Loading dataset samples...")
    if use_pretrained_config:
        # Use datasets from config
        train_samples, eval_samples = load_tts_samples(
            config.datasets,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
    else:
        # Use custom formatter for new training
        train_samples, eval_samples = load_tts_samples(
            dataset_config,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
            formatter=sinhala_formatter,
        )
    
    print(f"Loaded {len(train_samples)} training samples")
    print(f"Loaded {len(eval_samples)} evaluation samples")
    
    # Initialize model
    print("\nInitializing VITS model...")
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    # Load pretrained weights if available
    if use_pretrained:
        print(f"\nLoading pretrained weights from {pretrained_model_path}...")
        try:
            checkpoint = torch.load(pretrained_model_path, map_location="cpu")
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            print("✓ Successfully loaded pretrained model weights!")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Training from scratch...")
    
    # Move model to device
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Initialize trainer with callback for best model management
    print("\nInitializing trainer...")
    print(f"  Best model retention: keeping top {KEEP_N_BEST_MODELS} best models")
    global _trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,  # Important: prevent argparse conflicts
        callbacks={
            'on_epoch_end': on_epoch_end_callback,  # Manage best models after each epoch
        },
    )
    _trainer = trainer  # Store reference for signal handling
    
    # Save configuration
    config_path = output_dir / "config.json"
    config.save_json(str(config_path))
    print(f"\nConfiguration saved to: {config_path}")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Steps per epoch: {len(train_samples) // batch_size}")
    print(f"Total steps: {(len(train_samples) // batch_size) * args.epochs}")
    print(f"Checkpoints will be saved to: {output_dir}")
    print("=" * 80 + "\n")
    
    try:
        trainer.fit()
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print(f"Checkpoints saved to: {output_dir}")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main()