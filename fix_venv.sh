#!/bin/bash

echo "========================================"
echo "Fixing Virtual Environment"
echo "========================================"

# Ensure we're in the right directory
cd ~/workspace/coqui-ai-TTS || exit 1

# Deactivate any existing venv
deactivate 2>/dev/null || true

# Remove existing venv if it exists
if [ -d "venv_tts" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv_tts
fi

# Create fresh virtual environment
echo "Creating fresh virtual environment..."
python3.10 -m venv venv_tts

# Activate it
echo "Activating virtual environment..."
source venv_tts/bin/activate

# Verify we're using the right pip
echo "Verifying environment..."
which python
which pip
python --version

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "Installing core dependencies..."
pip install numpy==1.25.2
pip install "cython>=3.0.0"
pip install "scipy>=1.11.2"

# Install TTS dependencies
echo "Installing TTS dependencies..."
pip install \
    "tensorboard>=2.14.0" \
    "matplotlib>=3.7.0" \
    "librosa>=0.10.1" \
    "soundfile>=0.12.0" \
    "coqui-tts-trainer>=0.1.4,<0.2.0" \
    "trainer>=0.0.36" \
    "coqpit>=0.0.16" \
    "monotonic-alignment-search>=0.1.0" \
    "pyyaml>=6.0" \
    "inflect>=5.6.0" \
    "tqdm>=4.64.1" \
    "anyascii>=0.3.0" \
    "packaging>=23.1" \
    "pysbd>=0.3.4" \
    "fsspec[http]>=2023.6.0"

# Test installation
echo ""
echo "Testing installation..."
python -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
import trainer
print('✓ Trainer module imported')
import tensorboard
print('✓ TensorBoard imported')
"

echo ""
echo "========================================"
echo "Virtual environment fixed!"
echo "========================================"
echo "Now run:"
echo "  source venv_tts/bin/activate"
echo "  ./production/launch_training.sh"