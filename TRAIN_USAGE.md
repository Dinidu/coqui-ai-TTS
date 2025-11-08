# Training Usage Guide

## Quick Start

```bash
# Default: Fine-tuning with pretrained model (recommended)
./train.sh

# Train from scratch
./train.sh --mode scratch

# Custom configuration
./train.sh --mode custom --pretrained_config path/to/config.json
```

The `train.sh` script automatically handles:
- ✅ Virtual environment activation
- ✅ TensorBoard startup with automatic port selection
- ✅ Training mode selection (fine-tune, scratch, custom)
- ✅ Graceful cleanup on exit

## Training Modes

### 1. Fine-tuning Mode (Default) 🚀

**When to use**: When you have a pretrained Sinhala model and want to improve it with new data.

```bash
# Simple command - uses all defaults
./train.sh

# With custom parameters
./train.sh --epochs 500 --batch_size 32 --lr 0.00001
```

**What it does**:
- Automatically uses `models/pretrained/config.json` (architecture + character set)
- Automatically uses `models/pretrained/model_file.pth` (pretrained weights)
- Uses native Sinhala dataset from `datasets/sinhala-native/`
- Applies lower learning rate for fine-tuning

### 2. Scratch Training Mode 🔧

**When to use**: Training a completely new model without pretrained weights.

```bash
./train.sh --mode scratch --epochs 2000 --batch_size 64
```

**What it does**:
- Ignores pretrained model and config
- Creates new model architecture with random weights
- Uses romanized character set

### 3. Custom Mode 🎨

**When to use**: When you have custom paths or want specific configurations.

```bash
./train.sh --mode custom \
  --pretrained_config path/to/config.json \
  --pretrained_model path/to/model.pth
```

## Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `finetune` | Training mode: finetune, scratch, or custom |
| `--dataset_path` | `datasets/sinhala-native` | Path to dataset |
| `--output_path` | Auto-generated timestamp | Output directory |
| `--pretrained_model` | `models/pretrained/model_file.pth` | Pretrained model path |
| `--pretrained_config` | `models/pretrained/config.json` | Pretrained config path |
| `--batch_size` | Auto-detected | Batch size (based on GPU) |
| `--epochs` | 1000 | Number of training epochs |
| `--lr` | 0.0002 | Learning rate |
| `--num_workers` | 4 | DataLoader workers |
| `--mixed_precision` | False | Enable mixed precision |
| `--use_cuda` | True | Use GPU if available |
| `--resume` | None | Resume from checkpoint |

## Examples

### Quick Test
```bash
# Test with 1 epoch
./train.sh --epochs 1 --batch_size 2
```

### Production Fine-tuning
```bash
# Fine-tune with optimal settings
./train.sh --epochs 500 --batch_size 32 --lr 0.00001 --mixed_precision
```

### New Model Training
```bash
# Train from scratch for 2000 epochs
./train.sh --mode scratch --epochs 2000 --batch_size 64
```

### Resume Training
```bash
# Resume from checkpoint
./train.sh --resume output/vits_sinhala_*/checkpoint_500.pth
```

### Custom Dataset
```bash
# Use different dataset
./train.sh --dataset_path datasets/my_dataset --epochs 2000
```

## Features

### Automatic TensorBoard
- Starts automatically when training begins
- Finds available port if 6006 is busy
- URL displayed in console
- Stops cleanly with Ctrl+C

### Hardware Detection
- Auto-detects GPU/CPU
- Sets optimal batch size
- Configures mixed precision
- Optimizes worker threads

### Progress Monitoring
- Real-time loss values in terminal
- TensorBoard graphs at http://localhost:6006
- Automatic checkpointing
- Best model saved automatically

## Output Structure

```
output/
└── vits_sinhala_YYYYMMDD_HHMMSS/
    ├── config.json                    # Training configuration
    └── vits_sinhala-Date-Time-hash/
        ├── checkpoint_N.pth           # Model checkpoints
        ├── best_model.pth             # Best model so far
        ├── events.out.tfevents.*      # TensorBoard logs
        └── trainer_0_log.txt          # Training logs
```

## Mode Comparison

| Feature | Fine-tune (default) | Scratch | Custom |
|---------|-------------------|---------|---------|
| Pretrained Config | ✅ Auto-load | ❌ Ignore | 🔧 User choice |
| Pretrained Weights | ✅ Auto-load | ❌ Ignore | 🔧 User choice |
| Character Set | Sinhala (native) | Romanized | Depends on config |
| Learning Rate | Lower (1e-5) | Standard (2e-4) | User defined |
| Default Epochs | 500 | 1000 | User defined |

## Tips for Best Results

### Fine-tuning
- Use lower learning rates (1e-5 to 1e-6)
- Fewer epochs needed (200-500)
- Monitor for overfitting
- Keep moderate batch sizes

### Scratch Training
- Use standard learning rates (2e-4)
- More epochs needed (1000-2000)
- Larger batch sizes if GPU allows
- Enable mixed precision for speed

## Monitoring Training

### Terminal Output
Shows real-time:
- Current epoch/step
- Loss values (discriminator, generator, KL, mel, duration)
- Learning rate
- Step timing

### TensorBoard
Open browser to see:
- Loss curves over time
- Audio samples
- Spectrograms
- Model architecture

```bash
# TensorBoard usually at:
http://localhost:6006
```

## Stopping and Resuming

### Stop Training
Press `Ctrl+C` to:
- Save current checkpoint
- Stop TensorBoard
- Clean up resources

### Resume Training
```bash
./train.sh --resume output/vits_sinhala_*/checkpoint_*.pth
```

## Troubleshooting

### Port Already in Use
The script automatically finds the next available port.

### Out of Memory
```bash
# Reduce batch size
./train.sh --batch_size 8

# Enable mixed precision
./train.sh --mixed_precision
```

### Config Not Found Warning
- Check if `models/pretrained/config.json` exists
- Use `--mode scratch` to train without pretrained files
- Download pretrained model files if needed

### Character Mismatch Errors
- Fine-tuning: Dataset must use native Sinhala script
- Scratch: Dataset must use romanized text
- Regenerate dataset with `prepare_sinhala_dataset.py`

### Slow Training
```bash
# Enable mixed precision and increase batch
./train.sh --mixed_precision --batch_size 64
```

## File Requirements

```
coqui-ai-TTS/
├── models/
│   └── pretrained/           # For fine-tuning mode
│       ├── config.json       # Model architecture
│       └── model_file.pth    # Pretrained weights
├── datasets/
│   └── sinhala-native/       # Native Sinhala dataset
│       ├── metadata_train.csv
│       ├── metadata_val.csv
│       └── wavs/
├── train.sh                  # Main training script
└── train_sinhala.py          # Python training script
```