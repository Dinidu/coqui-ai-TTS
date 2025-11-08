#!/usr/bin/env python3
"""
Setup script to prepare for fine-tuning the Sinhala TTS model locally on CPU
Downloads the model from HuggingFace and prepares the dataset
"""

import os
import sys
import requests
import json
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination"""
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"Progress: {percent:.1f}%", end='\r')
    print(f"\nDownloaded to {destination}")

def setup_environment():
    """Setup the environment for fine-tuning"""
    
    # Create necessary directories
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models" / "pretrained"
    datasets_dir = base_dir / "datasets"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Setting up Sinhala TTS Fine-tuning Environment")
    print("=" * 60)
    
    # Download model and config from HuggingFace
    hf_base_url = "https://huggingface.co/tharindumihi/tts-si-female-vits-v2/resolve/main"
    
    model_files = {
        "model_file.pth": f"{hf_base_url}/tts-si-female-vits-v2_124000.pth",
        "config.json": f"{hf_base_url}/config.json"
    }
    
    print("\n1. Downloading pre-trained model from HuggingFace...")
    for filename, url in model_files.items():
        destination = models_dir / filename
        if destination.exists():
            print(f"   {filename} already exists, skipping...")
        else:
            try:
                download_file(url, str(destination))
            except Exception as e:
                print(f"   Error downloading {filename}: {e}")
                print(f"   Please manually download from: {url}")
    
    # Check if dataset exists
    print("\n2. Checking for Sinhala TTS dataset...")
    dataset_path = datasets_dir / "sinhala-tts-dataset"
    
    if not dataset_path.exists():
        print(f"   Dataset not found. Please clone it manually:")
        print(f"   git clone https://github.com/pnfo/sinhala-tts-dataset.git {dataset_path}")
    else:
        # Check for oshadi subfolder
        oshadi_path = dataset_path / "oshadi"
        if oshadi_path.exists():
            print(f"   ✓ Dataset found at {dataset_path}")
            print(f"   ✓ Oshadi speaker data found")
        else:
            print(f"   Dataset exists but 'oshadi' folder not found")
            print(f"   Please ensure the dataset is properly downloaded")
    
    print("\n3. Setup Summary:")
    print(f"   - Models directory: {models_dir}")
    print(f"   - Dataset directory: {datasets_dir}")
    print(f"   - Model file: {models_dir / 'model_file.pth'}")
    print(f"   - Config file: {models_dir / 'config.json'}")
    
    print("\n" + "=" * 60)
    print("Setup complete! Next steps:")
    print("1. Install dependencies if not already installed:")
    print("   pip install -e .")
    print("2. Run the fine-tuning script:")
    print("   python finetune_cpu_local.py")
    print("=" * 60)

if __name__ == "__main__":
    setup_environment()