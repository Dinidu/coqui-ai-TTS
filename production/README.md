# Production Training Guide for Sinhala VITS TTS on H100

This guide provides a complete setup for training a production-quality Sinhala TTS model on NVIDIA H100 GPUs.

## 📋 Requirements

- **GPU**: NVIDIA H100 (80GB) or similar high-end GPU
- **CUDA**: 11.8 or higher
- **Python**: 3.10+
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for datasets and checkpoints

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3.10 -m venv venv_production
source venv_production/bin/activate

# Install dependencies
pip install -r production/requirements_h100.txt

# Install TTS package
pip install -e .
```

### 2. Prepare Your Dataset

```bash
# Prepare and validate your dataset
python production/prepare_dataset.py \
    /path/to/raw/dataset \
    datasets/sinhala-production \
    --val-ratio 0.1 \
    --text-column 2 \
    --min-duration 0.5 \
    --max-duration 15.0
```

Expected dataset structure:
```
raw_dataset/
├── metadata.csv  # Format: filename|sinhala_text|romanized_text
└── wavs/
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

### 3. Launch Training

```bash
# Make launch script executable
chmod +x production/launch_training.sh

# Single GPU training
CUDA_VISIBLE_DEVICES=0 ./production/launch_training.sh

# Multi-GPU training (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./production/launch_training.sh
```

## 📁 File Structure

```
production/
├── train_vits_h100.py      # Main training script (single GPU)
├── train_multi_gpu.py      # Multi-GPU training with DDP
├── prepare_dataset.py      # Dataset preparation and validation
├── launch_training.sh      # Launch script with monitoring
├── requirements_h100.txt   # Production dependencies
└── README.md              # This file
```

## ⚙️ Configuration

### Key Training Parameters

Edit `train_vits_h100.py` to adjust:

```python
# Batch sizes (H100 optimized)
batch_size=64  # Adjust based on GPU memory
eval_batch_size=32

# Learning rates
lr_gen=0.0002  # Generator learning rate
lr_disc=0.0002  # Discriminator learning rate

# Training duration
epochs=1000
save_step=1000  # Save checkpoint every N steps

# Mixed precision
mixed_precision=True  # Enable for faster training
```

### Character Set Configuration

The script automatically detects characters from your dataset. To manually specify:

```python
characters=CharactersConfig(
    characters=" ,.?!abcdefg...",  # Your character set
    punctuations=" ,.?!",
    # ...
)
```

## 🔥 H100 Optimizations

The scripts include H100-specific optimizations:

1. **TF32 Tensor Cores**: Enabled by default
2. **Mixed Precision Training**: Uses Automatic Mixed Precision (AMP)
3. **CUDNN Benchmark**: Auto-tunes convolution algorithms
4. **Large Batch Sizes**: Optimized for 80GB memory
5. **Multi-GPU Support**: Distributed Data Parallel (DDP)

## 📊 Monitoring

### TensorBoard

Training automatically starts TensorBoard:
```bash
# Access at http://localhost:6006
tensorboard --logdir output/production/tensorboard
```

### Metrics to Monitor

- **Loss Values**: 
  - `loss_gen`: Generator loss (should decrease)
  - `loss_disc`: Discriminator loss (should stabilize)
  - `loss_kl`: KL divergence loss
  - `loss_mel`: Mel spectrogram loss

- **Learning Rates**: Check for proper decay
- **Gradient Norms**: Monitor for stability

### Weights & Biases (Optional)

To use W&B for logging:
```bash
wandb login
# Set use_wandb=True in training script
```

## 🎯 Training Strategies

### From Scratch

```python
USE_PRETRAINED = False  # In train_vits_h100.py
epochs = 1000  # Full training
lr_gen = 0.0002  # Standard learning rate
```

### Fine-tuning

```python
USE_PRETRAINED = True  # Use pretrained model
epochs = 200  # Fewer epochs needed
lr_gen = 0.0001  # Lower learning rate
```

### Multi-Speaker Training

```python
# In configuration
use_speaker_embedding = True
speaker_embedding_dim = 256
# Ensure metadata includes speaker column
```

## 🚄 Performance Tips

### Single H100 Performance

- **Batch Size**: 64-128 (depending on sequence length)
- **Training Speed**: ~3-5 steps/second
- **Memory Usage**: 40-60GB
- **Expected Time**: 24-48 hours for 1000 epochs

### Multi-GPU Scaling

| GPUs | Batch/GPU | Total Batch | Speedup |
|------|-----------|-------------|---------|
| 1    | 64        | 64          | 1.0x    |
| 2    | 32        | 64          | 1.8x    |
| 4    | 16        | 64          | 3.5x    |
| 8    | 8         | 64          | 6.5x    |

### Memory Optimization

If encountering OOM errors:
```python
# Reduce batch size
batch_size = 32

# Enable gradient checkpointing
gradient_checkpoint = True

# Reduce max sequence length
max_audio_len = 10 * 22050  # 10 seconds
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU usage
nvidia-smi -l 1
```

### Slow Training

1. Check I/O bottleneck:
```python
num_loader_workers = 16  # Increase workers
pin_memory = True  # Enable pinned memory
```

2. Enable benchmarking:
```python
cudnn_benchmark = True
```

### NaN Losses

1. Reduce learning rate
2. Enable gradient clipping:
```python
grad_clip = 1.0
```
3. Check for silence in audio files

## 📈 Expected Results

### Training Progress

- **0-100 epochs**: Noisy outputs, learning basic phonemes
- **100-300 epochs**: Intelligible speech, improving quality
- **300-600 epochs**: Natural prosody development
- **600-1000 epochs**: Fine-tuning and quality refinement

### Quality Metrics

- **MOS (Mean Opinion Score)**: Target > 3.5
- **Character Error Rate**: Target < 5%
- **Real-time Factor**: Target < 0.1 (10x faster than real-time)

## 🔄 Checkpoints and Resume

### Resume Training

```bash
# Edit train_vits_h100.py
trainer_args = TrainerArgs(
    continue_path="/path/to/checkpoint.pth",
    # ...
)
```

### Convert to Inference Model

```python
# Extract model for deployment
python -c "
import torch
ckpt = torch.load('best_model.pth')
torch.save(ckpt['model'], 'model_inference.pth')
"
```

## 📦 Deployment

After training, deploy your model:

```bash
# Test inference
tts --text "Your text here" \
    --model_path output/production/best_model.pth \
    --config_path output/production/config.json \
    --out_path test.wav

# Export for production
python export_model.py \
    --checkpoint best_model.pth \
    --output model_production.pt
```

## 📚 Additional Resources

- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [H100 Optimization Guide](https://docs.nvidia.com/deeplearning/performance/index.html)

## 💡 Tips for Production

1. **Data Quality > Quantity**: Clean 10 hours > Noisy 100 hours
2. **Regular Validation**: Monitor validation loss, not just training
3. **Early Stopping**: Stop if validation loss increases for 50+ epochs
4. **Backup Checkpoints**: Save to cloud storage regularly
5. **A/B Testing**: Keep multiple checkpoints for comparison

## 🤝 Support

For issues or questions:
1. Check training logs in `output/production/logs/`
2. Review TensorBoard metrics
3. Validate dataset with `prepare_dataset.py`
4. Test with smaller batch sizes first

---

**Note**: Adjust configurations based on your specific H100 variant (80GB/40GB) and dataset characteristics.