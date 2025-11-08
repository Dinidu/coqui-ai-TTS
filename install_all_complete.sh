#!/bin/bash

echo "========================================"
echo "Installing ALL Dependencies for TTS"
echo "========================================"

# Ensure we're in the right directory
cd ~/workspace/coqui-ai-TTS

# Activate virtual environment
source venv_tts/bin/activate

echo "Installing system dependencies for audio and phonemization..."
apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    espeak \
    espeak-ng \
    libespeak-ng1 \
    festival \
    festvox-kallpc16k \
    cmake \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    build-essential

echo "Upgrading pip and setuptools..."
pip install --upgrade pip wheel setuptools

echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "Installing core dependencies..."
pip install \
    numpy==1.25.2 \
    "cython>=3.0.0" \
    "scipy>=1.11.2"

echo "Installing audio processing libraries..."
pip install \
    "librosa>=0.10.1" \
    "soundfile>=0.12.0" \
    "audioread>=3.0.0" \
    "webrtcvad>=2.0.10" \
    "praat-parselmouth>=0.4.3" \
    "pyworld>=0.3.2"

echo "Installing text processing libraries..."
pip install \
    "inflect>=5.6.0" \
    "anyascii>=0.3.0" \
    "pyyaml>=6.0" \
    "pysbd>=0.3.4" \
    "num2words>=0.5.11" \
    "unidecode>=1.3.0" \
    "pypinyin>=0.50.0" \
    "jieba>=0.42.1" \
    "g2p-en>=2.1.0"

echo "Installing training infrastructure..."
pip install \
    "tensorboard>=2.14.0" \
    "matplotlib>=3.7.0" \
    "tqdm>=4.64.1" \
    "packaging>=23.1" \
    "fsspec[http]>=2023.6.0" \
    "requests>=2.31.0" \
    "Pillow>=10.0.0" \
    "psutil>=5.9.0" \
    "pandas>=1.5.0" \
    "scikit-learn>=1.3.0"

echo "Installing Coqui TTS specific packages..."
pip install \
    "coqui-tts-trainer>=0.1.4,<0.2.0" \
    "trainer>=0.0.36" \
    "coqpit>=0.0.16" \
    "monotonic-alignment-search>=0.1.0"

echo "Installing gruut and phonemization tools..."
# Install gruut dependencies first
pip install \
    "dateparser>=1.1.0" \
    "jsonlines>=1.2.0" \
    "networkx>=2.5.1" \
    "python-crfsuite>=0.9.7"

# Install gruut language packages
pip install \
    "gruut-ipa>=0.13.0" \
    "gruut-lang-en>=2.0.0" \
    "gruut>=2.4.0"

# Try additional gruut languages (optional)
pip install gruut[de,es,fr] 2>/dev/null || echo "Additional gruut languages skipped"

echo "Installing phonemizer..."
pip install "phonemizer>=3.2.1"

echo "Installing neural network components..."
pip install \
    "einops>=0.6.0" \
    "encodec>=0.1.1"

echo "Installing NLP libraries..."
pip install "spacy>=3,<3.8"
# Download spacy English model
python -m spacy download en_core_web_sm 2>/dev/null || echo "Spacy model download skipped"

echo "Installing transformers (optional but useful)..."
pip install "transformers>=4.43.0,<=4.46.2"

echo "Installing additional optional dependencies..."
pip install \
    "six>=1.16.0" \
    "mecab-python3>=1.0.5" \
    "jamo>=0.4.1" \
    "pyparsing>=3.0.9"

echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"

python -c "
import sys
print(f'Python: {sys.executable}')

packages = [
    'torch',
    'torchaudio',
    'numpy',
    'scipy',
    'librosa',
    'soundfile',
    'yaml',
    'inflect',
    'tqdm',
    'tensorboard',
    'matplotlib',
    'trainer',
    'coqpit',
    'gruut',
    'gruut_ipa',
    'einops',
    'spacy',
    'transformers',
    'TTS'
]

failed = []
for package in packages:
    try:
        if package == 'yaml':
            __import__('yaml')
        elif package == 'gruut_ipa':
            __import__('gruut_ipa')
        else:
            __import__(package)
        print(f'✓ {package}')
    except ImportError as e:
        failed.append(package)
        print(f'✗ {package}: {e}')

if failed:
    print(f'\n⚠ {len(failed)} packages failed to import: {failed}')
    print('Attempting to fix...')
    import subprocess
    for pkg in failed:
        if pkg == 'yaml':
            subprocess.run(['pip', 'install', 'pyyaml'])
        elif pkg == 'gruut_ipa':
            subprocess.run(['pip', 'install', 'gruut-ipa'])
        else:
            subprocess.run(['pip', 'install', pkg])
else:
    print('\n✓ All packages successfully installed!')

# Test CUDA
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA is available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ CUDA not available')
"

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo "Now run: ./production/launch_training.sh"