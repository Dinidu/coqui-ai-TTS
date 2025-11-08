#!/bin/bash

echo "========================================"
echo "Installing ALL Dependencies"
echo "========================================"

# Activate virtual environment
source ~/workspace/coqui-ai-TTS/venv_tts/bin/activate

# Install system dependencies for gruut
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    espeak-ng \
    libespeak-ng1 \
    libespeak-ng-dev \
    festival \
    festvox-kallpc16k \
    cmake \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch if not already installed
python -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch with CUDA 11.8..."
    pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
}

# Install all requirements
echo "Installing all Python packages..."
pip install -r requirements_complete.txt --ignore-installed

# Fix gruut if it fails
pip install gruut --no-deps 2>/dev/null || {
    echo "Installing gruut without dependencies..."
    pip install gruut-ipa gruut-lang-en --no-deps
}

# Test critical imports
echo ""
echo "Testing critical imports..."
python -c "
import sys
print(f'Python: {sys.executable}')

failed = []

try:
    import torch
    print(f'✓ torch {torch.__version__}')
except ImportError as e:
    failed.append(f'torch: {e}')

try:
    import yaml
    print('✓ yaml')
except ImportError as e:
    failed.append(f'yaml: {e}')

try:
    import tqdm
    print('✓ tqdm')
except ImportError as e:
    failed.append(f'tqdm: {e}')

try:
    import librosa
    print('✓ librosa')
except ImportError as e:
    failed.append(f'librosa: {e}')

try:
    import inflect
    print('✓ inflect')
except ImportError as e:
    failed.append(f'inflect: {e}')

try:
    import trainer
    print('✓ trainer')
except ImportError as e:
    failed.append(f'trainer: {e}')

try:
    import coqpit
    print('✓ coqpit')
except ImportError as e:
    failed.append(f'coqpit: {e}')

try:
    import TTS
    print('✓ TTS module')
except ImportError as e:
    failed.append(f'TTS: {e}')

# Try importing gruut but don't fail if it doesn't work
try:
    import gruut
    print('✓ gruut (optional)')
except:
    print('⚠ gruut not available (optional for Sinhala)')

if failed:
    print('\nFailed imports:')
    for f in failed:
        print(f'  ✗ {f}')
    sys.exit(1)
else:
    print('\n✓ All critical dependencies installed!')
"

echo ""
echo "========================================"
echo "Installation complete!"
echo "Now run: ./production/launch_training.sh"
echo "========================================"