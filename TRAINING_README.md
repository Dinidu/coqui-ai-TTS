# Sinhala TTS Training Guide

Simple, unified setup for training/fine-tuning VITS TTS models for Sinhala language on any GPU.

## Quick Start (3 Steps)

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Prepare Dataset
Place your dataset in `datasets/sinhala-production/` with:
- `wavs/` - Audio files
- `metadata_train.csv` - Training data
- `metadata_val.csv` - Validation data

### 3. Train
```bash
# Default: Fine-tuning with pretrained model (recommended)
./train.sh

# Or train from scratch
./train.sh --mode scratch

# Or use Python directly
source venv/bin/activate
python train_sinhala.py
```

> **Note**: Default mode is fine-tuning. See [TRAINING_MODES.md](TRAINING_MODES.md) for all training modes.

## Key Files

- `train_sinhala.py` - Main training script (auto-detects GPU)
- `setup.sh` - One-click setup script
- `requirements.txt` - All dependencies
- `production/custom_formatter.py` - Sinhala dataset formatter

## Training Options

### Using train.sh (Recommended)
```bash
# Fine-tuning (default - uses pretrained model)
./train.sh

# Train from scratch
./train.sh --mode scratch

# Custom configuration
./train.sh --mode custom --pretrained_config path/to/config.json
```

### Using Python Directly
```bash
# Fine-tuning with defaults
python train_sinhala.py

# Custom settings
python train_sinhala.py \
    --mode finetune \
    --batch_size 32 \
    --epochs 500 \
    --lr 0.00001 \
    --mixed_precision
```

## Hardware Support

| GPU | Batch Size | Training Time (1000 epochs) |
|-----|------------|---------------------------|
| H100 | 64 | 12-24 hours |
| A100 | 32-64 | 24-36 hours |
| V100 | 32 | 36-48 hours |
| RTX 4090 | 32 | 24-36 hours |
| RTX 3090 | 16-32 | 36-48 hours |
| CPU | 2 | Not recommended |

## Monitor Training

```bash
tensorboard --logdir output/
```

## Use Trained Model

```python
from TTS.api import TTS

tts = TTS(
    model_path="output/vits_sinhala_*/best_model.pth",
    config_path="output/vits_sinhala_*/config.json"
)

tts.tts_to_file(text="ඔබට කෙසේද?", file_path="output.wav")
```

## Troubleshooting

**Out of memory:** Reduce batch size with `--batch_size 16`

**Missing deps:** Run `pip install -r requirements.txt`

**No GPU:** Script automatically falls back to CPU (very slow)

## Support

- This fork: https://github.com/pnfo/coqui-ai-TTS
- Dataset: https://github.com/pnfo/sinhala-tts-dataset