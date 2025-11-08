#!/bin/bash

echo "Installing missing dependencies..."

# Activate virtual environment
source ~/workspace/coqui-ai-TTS/venv_tts/bin/activate

# Install all required dependencies
pip install \
    pyyaml>=6.0 \
    inflect>=5.6.0 \
    tqdm>=4.64.1 \
    anyascii>=0.3.0 \
    packaging>=23.1 \
    pysbd>=0.3.4 \
    "fsspec[http]>=2023.6.0" \
    requests>=2.31.0 \
    Pillow>=10.0.0 \
    pandas>=1.5.0 \
    scikit-learn>=1.3.0 \
    psutil>=5.9.0 \
    num2words>=0.5.11 \
    einops>=0.6.0 \
    "spacy>=3,<3.8" \
    "transformers>=4.43.0,<=4.46.2" \
    encodec>=0.1.1

# Test imports
echo ""
echo "Testing imports..."
python -c "
import yaml
print('✓ yaml imported')
import inflect
print('✓ inflect imported')
import tqdm
print('✓ tqdm imported')
import anyascii
print('✓ anyascii imported')
import torch
print('✓ torch imported')
import trainer
print('✓ trainer imported')
"

echo ""
echo "Dependencies installed. Try running training again:"
echo "  ./production/launch_training.sh"