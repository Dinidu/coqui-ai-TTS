#!/bin/bash

# Setup script for creating virtual environment with Python 3.10.13
# and installing all required dependencies for TTS fine-tuning

set -e  # Exit on error

echo "=========================================="
echo "Setting up TTS Fine-tuning Environment"
echo "=========================================="

# Check if Python 3.10 is available
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "✓ Found Python 3.10"
elif command -v python3 &> /dev/null; then
    # Check if python3 is version 3.10
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$PYTHON_VERSION" = "3.10" ]; then
        PYTHON_CMD="python3"
        echo "✓ Found Python 3.10"
    else
        echo "⚠ Warning: Python 3.10 not found. Found Python $PYTHON_VERSION"
        echo "For best compatibility, install Python 3.10.13"
        echo "Using available Python version..."
        PYTHON_CMD="python3"
    fi
else
    echo "❌ Error: Python 3 not found!"
    echo "Please install Python 3.10.13 first:"
    echo "  - macOS: brew install python@3.10"
    echo "  - Ubuntu: sudo apt install python3.10 python3.10-venv"
    echo "  - Or use pyenv: pyenv install 3.10.13"
    exit 1
fi

# Create virtual environment
VENV_DIR="venv_tts"

if [ -d "$VENV_DIR" ]; then
    echo "⚠ Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version first (to avoid downloading CUDA versions)
echo "Installing PyTorch (CPU version)..."
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install requirements
echo "Installing requirements from requirements_finetune.txt..."
pip install -r requirements_finetune.txt

# Install TTS package in editable mode
echo "Installing TTS package in editable mode..."
pip install -e .

# Download spacy model for text processing
echo "Downloading spacy language model..."
python -m spacy download en_core_web_sm || echo "Warning: Could not download spacy model"

# Create activation reminder script
cat > activate_venv.sh << 'EOF'
#!/bin/bash
# Quick activation script for the TTS virtual environment
source venv_tts/bin/activate
echo "✓ TTS virtual environment activated"
echo "Python: $(which python)"
echo "Version: $(python --version)"
EOF
chmod +x activate_venv.sh

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment created at: $VENV_DIR"
echo ""
echo "To activate the environment:"
echo "  source venv_tts/bin/activate"
echo "  OR"
echo "  ./activate_venv.sh"
echo ""
echo "To test the setup:"
echo "  1. python setup_finetune.py      # Download model"
echo "  2. python prepare_test_dataset.py # Prepare dataset"
echo "  3. python finetune_cpu_local.py  # Run fine-tuning"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo "=========================================="