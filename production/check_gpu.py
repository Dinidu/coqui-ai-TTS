#!/usr/bin/env python3
"""
Script to check GPU availability and usage
"""
import torch
import subprocess
import sys

def check_gpu():
    print("=" * 60)
    print("GPU Status Check")
    print("=" * 60)
    
    # Check PyTorch CUDA availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Try to allocate memory on GPU
            try:
                device = torch.device(f"cuda:{i}")
                x = torch.zeros(1, device=device)
                print(f"  ✓ GPU {i} is accessible")
                
                # Check memory
                mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"  Memory: {mem_alloc:.2f}GB allocated / {mem_reserved:.2f}GB reserved / {mem_total:.2f}GB total")
                
                del x
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ✗ GPU {i} error: {e}")
    
    # Check nvidia-smi
    print("\n" + "=" * 60)
    print("nvidia-smi output:")
    print("=" * 60)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
    
    # Check for other processes using GPU
    print("\n" + "=" * 60)
    print("GPU Processes:")
    print("=" * 60)
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("No processes currently using GPU")
    except Exception as e:
        print(f"Error checking GPU processes: {e}")

def kill_gpu_processes():
    """Kill all processes using the GPU (use with caution!)"""
    print("\n" + "=" * 60)
    print("Attempting to kill GPU processes...")
    print("=" * 60)
    
    try:
        # Get list of processes
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        pids = [p.strip() for p in pids if p.strip()]
        
        if pids:
            print(f"Found {len(pids)} processes using GPU")
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid])
                    print(f"  Killed process {pid}")
                except Exception as e:
                    print(f"  Failed to kill process {pid}: {e}")
        else:
            print("No processes to kill")
            
    except Exception as e:
        print(f"Error: {e}")

def reset_gpu():
    """Try to reset GPU state"""
    print("\n" + "=" * 60)
    print("Resetting GPU state...")
    print("=" * 60)
    
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("✓ Cleared CUDA cache")
            
        # Try nvidia-smi reset
        try:
            subprocess.run(['nvidia-smi', '--gpu-reset'], capture_output=True)
            print("✓ GPU reset attempted (may require elevated privileges)")
        except:
            print("⚠ Could not reset GPU (may need sudo)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_gpu()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--kill":
            kill_gpu_processes()
        elif sys.argv[1] == "--reset":
            reset_gpu()
        else:
            print("\nUsage:")
            print("  python check_gpu.py         # Check GPU status")
            print("  python check_gpu.py --kill   # Kill GPU processes")
            print("  python check_gpu.py --reset  # Reset GPU state")