# Sinhala TTS Model Fine-tuning Guide

This guide explains how to fine-tune the Sinhala VITS TTS model locally on CPU for testing purposes.

## Quick Start

### 1. Setup Environment

First, run the setup script to download the pre-trained model and prepare the environment:

```bash
python setup_finetune.py
```

This will:
- Download the model from HuggingFace (tharindumihi/tts-si-female-vits-v2)
- Create necessary directories
- Check for the dataset

### 2. Get the Dataset

If not already cloned, get the Sinhala TTS dataset:

```bash
git clone https://github.com/pnfo/sinhala-tts-dataset.git datasets/sinhala-tts-dataset
```

The dataset structure should be:
```
datasets/sinhala-tts-dataset/
├── oshadi/
│   ├── wavs/
│   │   ├── 001.wav
│   │   ├── 002.wav
│   │   └── ...
│   └── metadata.csv
└── ...
```

### 3. Install Dependencies

Install the TTS package if not already installed:

```bash
pip install -e .
```

### 4. Run Fine-tuning (CPU Test)

Run the CPU-optimized fine-tuning script:

```bash
python finetune_cpu_local.py
```

This script is configured for:
- **CPU-only execution** (no GPU required)
- **2 epochs only** (for quick testing)
- **Small batch size** (2 samples)
- **Limited dataset** (20 training samples)
- **Low learning rate** (0.00005)

### 5. Test the Fine-tuned Model

After fine-tuning, test the model:

```bash
python test_finetuned_model.py
```

This will generate sample audio files using your fine-tuned model.

## File Descriptions

- **`setup_finetune.py`**: Downloads model from HuggingFace and sets up environment
- **`finetune_cpu_local.py`**: CPU-optimized fine-tuning script for local testing
- **`test_finetuned_model.py`**: Tests the fine-tuned model with sample sentences
- **`recipes/pathnirvana/finetune_vits_sinhala.py`**: Full GPU fine-tuning script for production

## Directory Structure

```
coqui-ai-TTS/
├── models/
│   └── pretrained/
│       ├── model_file.pth     # Downloaded from HuggingFace
│       └── config.json         # Model configuration
├── datasets/
│   └── sinhala-tts-dataset/   # Cloned dataset
│       └── oshadi/
│           ├── wavs/
│           └── metadata.csv
└── output/
    └── finetuned/              # Fine-tuned model output
        └── vits_sinhala_cpu_finetune-{timestamp}/
            ├── best_model.pth
            └── config.json
```

## For Production Training (GPU)

For actual production fine-tuning with GPU and full dataset, use:

```bash
# Edit paths in the script first
vim recipes/pathnirvana/finetune_vits_sinhala.py

# Run with GPU
CUDA_VISIBLE_DEVICES=0 python recipes/pathnirvana/finetune_vits_sinhala.py
```

The production script (`finetune_vits_sinhala.py`) includes:
- Full dataset usage
- 100 epochs
- Larger batch sizes (16)
- GPU acceleration
- Mixed precision training
- Proper learning rate scheduling

## Configuration Details

### CPU Testing Configuration
- **Batch Size**: 2 (minimal for CPU)
- **Epochs**: 2 (just for testing)
- **Learning Rate**: 0.00005 (very low for fine-tuning)
- **Max Audio Length**: 10 seconds
- **Training Samples**: 20 (limited for quick testing)
- **Eval Samples**: 5

### Production Configuration (GPU)
- **Batch Size**: 16-56 (depending on GPU memory)
- **Epochs**: 100-200
- **Learning Rate**: 0.0001
- **Max Audio Length**: 15 seconds  
- **Full Dataset**: All available samples

## Common Issues

1. **Out of Memory (CPU)**
   - Reduce batch_size to 1
   - Reduce max_audio_len
   - Use fewer training samples

2. **Dataset Not Found**
   - Ensure dataset is cloned to `datasets/sinhala-tts-dataset`
   - Check metadata.csv format matches LJSpeech format

3. **Model Download Fails**
   - Manually download from: https://huggingface.co/tharindumihi/tts-si-female-vits-v2
   - Place files in `models/pretrained/`

4. **Import Errors**
   - Ensure TTS package is installed: `pip install -e .`
   - Install missing dependencies: `pip install torch torchaudio`

## Monitoring Training

For the CPU test, training output will be printed to console showing:
- Loss values
- Learning rate
- Step progress
- Evaluation metrics

For production training with TensorBoard:
```bash
tensorboard --logdir output/finetuned
```

## Next Steps

After successful CPU testing:
1. Prepare your full dataset in the same format
2. Adjust configuration in `finetune_vits_sinhala.py`
3. Run on GPU with more epochs and larger batch size
4. Monitor with TensorBoard
5. Evaluate generated samples regularly

## Support

For issues with:
- Coqui TTS: https://github.com/coqui-ai/TTS
- Dataset: https://github.com/pnfo/sinhala-tts-dataset
- Model: https://huggingface.co/tharindumihi/tts-si-female-vits-v2