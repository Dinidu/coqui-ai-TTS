#!/usr/bin/env python3
"""
Test script for the fine-tuned Sinhala TTS model
"""

import os
import sys
from pathlib import Path
import glob
import torch

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "finetuned"

def find_latest_checkpoint():
    """Find the latest checkpoint in the output directory"""
    pattern = str(OUTPUT_DIR / "vits_sinhala_cpu_finetune-*" / "*.pth")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None, None
    
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    config_path = Path(latest_checkpoint).parent / "config.json"
    
    if config_path.exists():
        return latest_checkpoint, str(config_path)
    
    return latest_checkpoint, None

def test_model():
    """Test the fine-tuned model with sample text"""
    
    print("=" * 60)
    print("Testing Fine-tuned Sinhala TTS Model")
    print("=" * 60)
    
    # Find checkpoint
    checkpoint_path, config_path = find_latest_checkpoint()
    
    if not checkpoint_path:
        print("No fine-tuned model found!")
        print(f"Expected location: {OUTPUT_DIR}")
        print("\nPlease run the fine-tuning script first:")
        print("  python finetune_cpu_local.py")
        return
    
    print(f"Found checkpoint: {checkpoint_path}")
    print(f"Found config: {config_path}")
    
    # Test sentences
    test_sentences = [
        "mama gedara yanawa",
        "oyā koheda yanne", 
        "api heta hambawemu",
        "suba dawasak wewa",
        "ayubowan"
    ]
    
    print("\nGenerating speech for test sentences...")
    print("-" * 40)
    
    for i, text in enumerate(test_sentences, 1):
        output_path = BASE_DIR / f"test_output_{i}.wav"
        
        print(f"\n{i}. Text: '{text}'")
        print(f"   Output: {output_path}")
        
        # Build TTS command
        cmd = f"tts --text \"{text}\" --model_path \"{checkpoint_path}\" --config_path \"{config_path}\" --out_path \"{output_path}\""
        
        print(f"   Command: {cmd}")
        
        # Execute TTS
        result = os.system(cmd)
        
        if result == 0:
            print(f"   ✓ Successfully generated: {output_path}")
        else:
            print(f"   ✗ Failed to generate audio")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("Audio files saved in the current directory as test_output_*.wav")
    print("=" * 60)

if __name__ == "__main__":
    test_model()