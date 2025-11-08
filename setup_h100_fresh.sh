#!/bin/bash
set -e

echo "========================================"
echo "Fresh H100 Ubuntu Setup for TTS Training"
echo "========================================"

# Update system
echo "1. Updating system packages..."
apt-get update
apt-get install -y git wget curl build-essential software-properties-common

# Install Python 3.10 if not present
echo "2. Setting up Python 3.10..."
if ! command -v python3.10 &> /dev/null; then
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
fi

# Install pip for Python 3.10
echo "3. Installing pip..."
if ! python3.10 -m pip --version &> /dev/null; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
fi

# Install system dependencies for audio processing
echo "4. Installing system dependencies..."
apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-dev \
    libsox-fmt-all \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    python3-pyaudio

# Clone the repository (if not already cloned)
echo "5. Setting up repository..."
if [ ! -d "coqui-ai-TTS" ]; then
    git clone https://github.com/pnfo/coqui-ai-TTS.git
fi

cd coqui-ai-TTS

# Create virtual environment
echo "6. Creating Python virtual environment..."
python3.10 -m venv venv_tts

# Activate virtual environment
source venv_tts/bin/activate

# Upgrade pip in venv
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 11.8 support (compatible with H100)
echo "7. Installing PyTorch with CUDA support..."
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies first
echo "8. Installing core dependencies..."
pip install numpy==1.25.2
pip install cython>=3.0.0
pip install scipy>=1.11.2

# Install TTS dependencies
echo "9. Installing TTS dependencies..."
pip install \
    tensorboard>=2.14.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.64.1 \
    librosa>=0.10.1 \
    soundfile>=0.12.0 \
    inflect>=5.6.0 \
    anyascii>=0.3.0 \
    pyyaml>=6.0 \
    fsspec[http]>=2023.6.0 \
    packaging>=23.1 \
    pysbd>=0.3.4 \
    num2words>=0.5.11 \
    requests>=2.31.0 \
    Pillow>=10.0.0 \
    pandas>=1.5.0 \
    scikit-learn>=1.3.0 \
    psutil>=5.9.0

# Install Coqui TTS trainer and related packages
echo "10. Installing Coqui TTS trainer..."
pip install \
    coqui-tts-trainer>=0.1.4,<0.2.0 \
    trainer>=0.0.36 \
    coqpit>=0.0.16 \
    monotonic-alignment-search>=0.1.0

# Install text processing dependencies
echo "11. Installing text processing dependencies..."
pip install \
    gruut[de,es,fr]>=2.4.0 \
    spacy>=3,<3.8 \
    einops>=0.6.0 \
    transformers>=4.43.0,<=4.46.2 \
    encodec>=0.1.1

# Download the pretrained model
echo "12. Setting up pretrained model..."
mkdir -p models/pretrained
cd models/pretrained

if [ ! -f "model_file.pth" ]; then
    echo "Downloading pretrained model from HuggingFace..."
    wget https://huggingface.co/tharindumihi/tts-si-female-vits-v2/resolve/main/model_file.pth
fi

if [ ! -f "config.json" ]; then
    wget https://huggingface.co/tharindumihi/tts-si-female-vits-v2/resolve/main/config.json
fi

cd ../..

# Verify CUDA installation
echo "13. Verifying CUDA setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    # Test CUDA
    try:
        x = torch.zeros(1).cuda()
        print('✓ CUDA test successful')
    except Exception as e:
        print(f'✗ CUDA test failed: {e}')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To start training:"
echo "1. Activate the virtual environment:"
echo "   source venv_tts/bin/activate"
echo ""
echo "2. Launch the training:"
echo "   ./production/launch_training.sh"
echo ""
echo "========================================"