#!/bin/bash

# Script to install all required dependencies for H100 production training
# This ensures all necessary packages are installed before training

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "===========================================" 
echo "Installing TTS Training Dependencies"
echo "==========================================="

# Activate virtual environment if it exists
if [ -f "${PROJECT_DIR}/venv_tts/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${PROJECT_DIR}/venv_tts/bin/activate"
else
    echo "Warning: Virtual environment not found at ${PROJECT_DIR}/venv_tts"
    echo "Consider creating one with: python3.10 -m venv venv_tts"
fi

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install production requirements
if [ -f "${SCRIPT_DIR}/requirements_h100.txt" ]; then
    echo "Installing H100 production requirements..."
    pip install -r "${SCRIPT_DIR}/requirements_h100.txt"
else
    echo "Warning: requirements_h100.txt not found"
    echo "Installing essential packages manually..."
    
    # Install essential packages
    pip install tensorboard>=2.14.0
    pip install coqui-tts-trainer>=0.1.4,<0.2.0
    pip install torch>=2.1.0 torchaudio>=2.1.0
    pip install numpy>=1.25.2,<2.0
    pip install scipy>=1.11.2
    pip install librosa>=0.10.1
    pip install matplotlib>=3.7.0
    pip install tqdm>=4.64.1
    pip install coqpit>=0.0.16
    pip install monotonic-alignment-search>=0.1.0
fi

# Verify critical imports
echo ""
echo "Verifying installations..."
python -c "
import sys
failed = []

try:
    import tensorboard
    print('✓ tensorboard installed')
except ImportError:
    failed.append('tensorboard')
    print('✗ tensorboard NOT installed')

try:
    import trainer
    print('✓ coqui-tts-trainer installed')
except ImportError:
    failed.append('coqui-tts-trainer')
    print('✗ coqui-tts-trainer NOT installed')

try:
    import torch
    print(f'✓ PyTorch {torch.__version__} installed')
    if torch.cuda.is_available():
        print(f'  CUDA {torch.version.cuda} available')
        print(f'  GPU count: {torch.cuda.device_count()}')
except ImportError:
    failed.append('torch')
    print('✗ PyTorch NOT installed')

try:
    import TTS
    print('✓ TTS module available')
except ImportError:
    failed.append('TTS')
    print('✗ TTS module NOT available')

if failed:
    print('\nFailed imports:', ', '.join(failed))
    sys.exit(1)
else:
    print('\nAll critical dependencies installed successfully!')
"

echo ""
echo "==========================================="
echo "Dependency installation complete!"
echo "You can now run: ./launch_training.sh"
echo "==========================================="