#!/bin/bash

echo "========================================"
echo "TTS Fine-tuning Setup"
echo "========================================"

# Function to download file from Google Drive
download_from_gdrive() {
    local FILE_ID=$1
    local DEST_PATH=$2
    
    echo "Downloading from Google Drive..."
    
    # Method 1: Try using gdown if available
    if command -v gdown &> /dev/null; then
        gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$DEST_PATH"
        return $?
    fi
    
    # Method 2: Use curl with cookie handling
    echo "Using curl to download..."
    local CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" | \
                    grep -o 'confirm=[^&]*' | sed 's/confirm=//')
    
    if [ ! -z "$CONFIRM" ]; then
        curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "$DEST_PATH"
    else
        curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o "$DEST_PATH"
    fi
    
    rm -f /tmp/gcookie
    return $?
}

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

# Download pretrained model if not exists
echo ""
echo "Checking pretrained model..."
PRETRAINED_DIR="models/pretrained"
MODEL_FILE="$PRETRAINED_DIR/model_file.pth"
CONFIG_FILE="$PRETRAINED_DIR/config.json"

# Create directory if it doesn't exist
mkdir -p "$PRETRAINED_DIR"

# Download model file if not exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Pretrained model not found. Downloading (~950MB)..."
    echo "This may take a few minutes depending on your internet speed..."
    
    # Install gdown if not available (faster than curl for Google Drive)
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown for faster Google Drive downloads..."
        pip install gdown -q
    fi
    
    # Extract file ID from the Google Drive URL
    # URL: https://drive.google.com/file/d/13CA3ZgqBxyKaayLURkQzqp8W5vOzq0rS/view?usp=sharing
    FILE_ID="13CA3ZgqBxyKaayLURkQzqp8W5vOzq0rS"
    
    if download_from_gdrive "$FILE_ID" "$MODEL_FILE"; then
        echo "✓ Pretrained model downloaded successfully!"
        
        # Verify file size
        if [ -f "$MODEL_FILE" ]; then
            SIZE=$(ls -lh "$MODEL_FILE" | awk '{print $5}')
            echo "  Model size: $SIZE"
        fi
    else
        echo "⚠️  Warning: Failed to download pretrained model"
        echo "  You can manually download from:"
        echo "  https://drive.google.com/file/d/13CA3ZgqBxyKaayLURkQzqp8W5vOzq0rS/view"
        echo "  And place it at: $MODEL_FILE"
    fi
else
    SIZE=$(ls -lh "$MODEL_FILE" | awk '{print $5}')
    echo "✓ Pretrained model already exists (size: $SIZE)"
fi

# Check for config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  Warning: config.json not found at $CONFIG_FILE"
    echo "  Make sure to have the pretrained config file for fine-tuning"
else
    echo "✓ Pretrained config found"
fi

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