#!/bin/bash

echo "========================================"
echo "TTS Fine-tuning Setup"
echo "========================================"

# Detect CUDA version
detect_cuda_version() {
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        echo "Detected CUDA version: $cuda_version"
        
        if [[ "$cuda_version" == "11."* ]]; then
            echo "Installing PyTorch for CUDA 11.8..."
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
        elif [[ "$cuda_version" == "12."* ]]; then
            echo "Installing PyTorch for CUDA 12.1..."
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            echo "Unknown CUDA version. Installing CPU PyTorch..."
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        echo "CUDA not detected. Installing CPU PyTorch..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
}

# Note: Python version check is handled during venv creation

# Skip system dependencies - Python packages handle audio processing
echo "Skipping system package installation (not required for training)..."
# Note: If you encounter audio loading issues, you may need to install:
# - Linux: sudo apt-get install libsndfile1
# - Mac: brew install libsndfile

# Remove old virtual environments if they exist
if [ -d "venv_tts" ]; then
    echo "Removing old venv_tts..."
    rm -rf venv_tts
fi

if [ -d "venv" ] && [ "$1" == "--fresh" ]; then
    echo "Removing existing venv for fresh install..."
    rm -rf venv
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.10..."
    # Try to find python3.10 explicitly
    if command -v python3.10 &> /dev/null; then
        python3.10 -m venv venv
    elif command -v python3 &> /dev/null && [[ $(python3 --version | cut -d" " -f2 | cut -d"." -f1,2) == "3.10" ]]; then
        python3 -m venv venv
    else
        echo "ERROR: Python 3.10 is required but not found!"
        echo "Please install Python 3.10 first:"
        echo "  Mac: brew install python@3.10"
        echo "  Ubuntu: sudo apt install python3.10 python3.10-venv"
        echo "  Or use pyenv to install Python 3.10"
        exit 1
    fi
else
    echo "Using existing virtual environment (use --fresh to recreate)"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with appropriate CUDA version
detect_cuda_version

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Download spacy model (optional)
python -m spacy download en_core_web_sm 2>/dev/null || echo "Spacy model download skipped"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import torchaudio
import trainer
import coqpit
import TTS
print('✓ All core packages installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Virtual environment: venv/"
echo ""
echo "To train the model:"
echo "  ./train.sh                  # Fine-tuning with defaults"
echo "  ./train.sh --epochs 500     # Custom parameters"
echo "  ./train.sh --mode scratch   # Train from scratch"
echo ""
echo "For all options:"
echo "  ./train.sh --help"
echo ""
echo "Manual training (advanced):"
echo "  source venv/bin/activate"
echo "  python train_sinhala.py --help"
echo ""
echo "To recreate environment from scratch:"
echo "  ./setup.sh --fresh"
echo "========================================"