#!/usr/bin/env python3
"""
Simple CUDA test script
"""
import os
import torch
import sys

print("=" * 60)
print("CUDA Diagnostic Test")
print("=" * 60)

# Environment variables that might help
print("\nEnvironment variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'not set')}")

# PyTorch info
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    sys.exit(1)

print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Try different ways to access GPU
print("\n" + "=" * 60)
print("Testing different CUDA access methods:")
print("=" * 60)

# Method 1: Direct cuda() call
try:
    print("\n1. Testing direct .cuda() call...")
    x = torch.zeros(1).cuda()
    print(f"   ✓ Success - tensor on {x.device}")
    del x
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Method 2: Using device
try:
    print("\n2. Testing torch.device('cuda')...")
    device = torch.device('cuda')
    x = torch.zeros(1).to(device)
    print(f"   ✓ Success - tensor on {x.device}")
    del x
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Method 3: Using cuda:0 explicitly
try:
    print("\n3. Testing torch.device('cuda:0')...")
    device = torch.device('cuda:0')
    x = torch.zeros(1).to(device)
    print(f"   ✓ Success - tensor on {x.device}")
    del x
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Method 4: Set device first
try:
    print("\n4. Testing torch.cuda.set_device(0)...")
    torch.cuda.set_device(0)
    x = torch.zeros(1).cuda()
    print(f"   ✓ Success - tensor on {x.device}")
    del x
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test memory allocation
print("\n" + "=" * 60)
print("Testing memory allocation:")
print("=" * 60)

try:
    # Small allocation
    print("\nAllocating small tensor (1MB)...")
    x = torch.zeros(256, 1024).cuda()  # ~1MB
    print(f"✓ Small allocation successful")
    del x
    
    # Medium allocation
    print("\nAllocating medium tensor (100MB)...")
    x = torch.zeros(256, 102400).cuda()  # ~100MB
    print(f"✓ Medium allocation successful")
    del x
    
    # Large allocation
    print("\nAllocating large tensor (1GB)...")
    x = torch.zeros(256, 1024000).cuda()  # ~1GB
    print(f"✓ Large allocation successful")
    del x
    
    print("\n✓ All memory allocations successful!")
    
except Exception as e:
    print(f"\n✗ Memory allocation failed: {e}")

# Clean up
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("CUDA test completed")
print("=" * 60)