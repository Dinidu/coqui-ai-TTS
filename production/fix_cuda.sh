#!/bin/bash

echo "=========================================="
echo "CUDA Fix Script"
echo "=========================================="

# 1. Check if we're in a container
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
elif [ -f /run/.containerenv ]; then
    echo "Running in Podman/container"
else
    echo "Not in a container (or not detected)"
fi

# 2. Check NVIDIA container runtime
echo ""
echo "Checking NVIDIA container toolkit..."
if command -v nvidia-container-cli &> /dev/null; then
    echo "nvidia-container-cli found"
    nvidia-container-cli info 2>/dev/null || echo "Could not get info"
else
    echo "nvidia-container-cli not found"
fi

# 3. Check device files
echo ""
echo "Checking GPU device files..."
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* files found"
ls -la /dev/dri/ 2>/dev/null || echo "No /dev/dri/ files found"

# 4. Check CUDA libraries
echo ""
echo "Checking CUDA libraries..."
ldconfig -p | grep cuda | head -5

# 5. Try to reload NVIDIA UVM module
echo ""
echo "Trying to reload NVIDIA UVM module..."
if [ -f /sys/module/nvidia_uvm/parameters/uvm_enabled ]; then
    cat /sys/module/nvidia_uvm/parameters/uvm_enabled
else
    echo "NVIDIA UVM module not found"
fi

# 6. Check environment
echo ""
echo "Environment variables:"
env | grep -E "CUDA|NVIDIA|GPU" | sort

# 7. Try nvidia-modprobe
echo ""
echo "Trying nvidia-modprobe..."
if command -v nvidia-modprobe &> /dev/null; then
    sudo nvidia-modprobe -u -c=0 2>/dev/null || echo "nvidia-modprobe failed (may need sudo)"
else
    echo "nvidia-modprobe not found"
fi

# 8. Check PyTorch CUDA
echo ""
echo "Testing PyTorch CUDA with different methods..."

# Method 1: Force reinit
python3 -c "
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;9.0'  # H100 is 9.0

import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('Trying torch.cuda.init()...')
    try:
        torch.cuda.init()
        print('✓ CUDA init successful')
    except Exception as e:
        print(f'✗ CUDA init failed: {e}')
"

echo ""
echo "=========================================="
echo "Suggested fixes:"
echo "=========================================="
echo "1. Restart the container with proper GPU flags:"
echo "   docker run --gpus all --rm -it <image>"
echo ""
echo "2. Or with nvidia-docker:"
echo "   nvidia-docker run --rm -it <image>"
echo ""
echo "3. If using docker-compose, add:"
echo "   deploy:"
echo "     resources:"
echo "       reservations:"
echo "         devices:"
echo "           - driver: nvidia"
echo "             count: 1"
echo "             capabilities: [gpu]"
echo ""
echo "4. Try running outside the container directly on the host"
echo ""
echo "5. Check if GPU is in compute mode:"
echo "   nvidia-smi -q | grep 'Compute Mode'"
echo "=========================================="