#!/bin/bash

# Production Training Launch Script for H100 GPU(s)
# This script handles environment setup, logging, and monitoring

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="vits_sinhala_${TIMESTAMP}"

# Training parameters (modify as needed)
DATASET_PATH="${PROJECT_DIR}/datasets/sinhala-production"
OUTPUT_BASE="${PROJECT_DIR}/output/production"
LOG_DIR="${OUTPUT_BASE}/logs"
CHECKPOINT_DIR="${OUTPUT_BASE}/checkpoints"

# GPU Configuration
export CUDA_VISIBLE_DEVICES="0"  # Set to "0,1,2,3" for multi-GPU
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# H100 Optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 architecture
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Mixed Precision Settings
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Logging setup
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
TENSORBOARD_LOG="${OUTPUT_BASE}/tensorboard/${EXPERIMENT_NAME}"

echo "==========================================="
echo "Production TTS Training on H100"
echo "==========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_BASE}"
echo "GPUs: ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "Log file: ${LOG_FILE}"
echo "==========================================="

# Check dataset
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    echo "Please prepare your dataset first using:"
    echo "  python production/prepare_dataset.py <input_path> ${DATASET_PATH}"
    exit 1
fi

# Check if metadata files exist
if [ ! -f "${DATASET_PATH}/metadata_train.csv" ] || [ ! -f "${DATASET_PATH}/metadata_val.csv" ]; then
    echo "ERROR: metadata_train.csv or metadata_val.csv not found"
    echo "Running dataset preparation..."
    python "${SCRIPT_DIR}/prepare_dataset.py" \
        "${DATASET_PATH}" \
        "${DATASET_PATH}_prepared" \
        --val-ratio 0.1
    DATASET_PATH="${DATASET_PATH}_prepared"
fi

# Activate virtual environment if it exists
if [ -f "${PROJECT_DIR}/venv_tts/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${PROJECT_DIR}/venv_tts/bin/activate"
fi

# Install missing dependencies if needed
echo "Checking and installing dependencies..."

# Check for tensorboard
if ! command -v tensorboard &> /dev/null; then
    echo "Installing tensorboard..."
    pip install tensorboard>=2.14.0
fi

# Check for trainer module
python -c "import trainer" 2>/dev/null || {
    echo "Installing coqui-tts-trainer..."
    pip install coqui-tts-trainer>=0.1.4,<0.2.0
}

# Install all requirements to ensure compatibility (optional, comment out if slow)
# if [ -f "${SCRIPT_DIR}/requirements_h100.txt" ]; then
#     echo "Installing H100 requirements..."
#     pip install -q -r "${SCRIPT_DIR}/requirements_h100.txt"
# fi

# Check Python and PyTorch installation
echo "Checking environment..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Function to handle interruption
cleanup() {
    echo ""
    echo "Training interrupted. Cleaning up..."
    # Kill tensorboard if running
    if [ ! -z "$TENSORBOARD_PID" ]; then
        kill $TENSORBOARD_PID 2>/dev/null || true
    fi
    exit 1
}

trap cleanup INT TERM

# Start TensorBoard in background
echo ""
echo "Starting TensorBoard..."
tensorboard --logdir="${TENSORBOARD_LOG}" --port=6006 --bind_all &
TENSORBOARD_PID=$!
echo "TensorBoard running at: http://localhost:6006"
echo ""

# Launch training based on GPU count
if [ $NUM_GPUS -gt 1 ]; then
    echo "Starting multi-GPU training on $NUM_GPUS GPUs..."
    
    # Using torchrun for distributed training (recommended)
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=12355 \
        "${SCRIPT_DIR}/train_multi_gpu.py" \
        2>&1 | tee -a "$LOG_FILE"
    
    # Alternative: using python -m torch.distributed.launch (legacy)
    # python -m torch.distributed.launch \
    #     --nproc_per_node=$NUM_GPUS \
    #     --master_port=12355 \
    #     "${SCRIPT_DIR}/train_multi_gpu.py" \
    #     2>&1 | tee -a "$LOG_FILE"
    
else
    echo "Starting single-GPU training..."
    
    python "${SCRIPT_DIR}/train_vits_h100.py" \
        2>&1 | tee -a "$LOG_FILE"
fi

# Training completed
echo ""
echo "==========================================="
echo "Training completed!"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Log file: ${LOG_FILE}"
echo "TensorBoard: http://localhost:6006"
echo "==========================================="

# Keep TensorBoard running
echo "Press Ctrl+C to stop TensorBoard..."
wait $TENSORBOARD_PID