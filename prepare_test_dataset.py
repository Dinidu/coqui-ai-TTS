#!/usr/bin/env python3
"""
Prepare a minimal test dataset structure for CPU testing
This creates a mock dataset structure if the real dataset is not available
"""

import os
from pathlib import Path

def create_test_dataset():
    """Create a minimal test dataset structure"""
    
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "datasets" / "sinhala-tts-dataset" / "oshadi"
    wavs_dir = dataset_dir / "wavs"
    
    # Create directories
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating test dataset structure...")
    print(f"Dataset directory: {dataset_dir}")
    
    # Create a sample metadata.csv
    metadata_path = dataset_dir / "metadata.csv"
    
    # Sample metadata entries (you should replace with actual data)
    metadata_content = """001|mama gedara yanawa|oshadi
002|oyā koheda yanne|oshadi
003|api heta hambawemu|oshadi
004|suba dawasak wewa|oshadi
005|ayubowan|oshadi
006|kohomada oyāṭa|oshadi
007|mama hoňdin innawā|oshadi
008|stuti|oshadi
009|eya lassanayi|oshadi
010|ada kāla guṇaya hoňdayi|oshadi"""
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(metadata_content)
    
    print(f"Created metadata.csv with {len(metadata_content.splitlines())} entries")
    
    # Create dummy wav files (empty files for testing structure)
    # In real scenario, these would be actual audio files
    for i in range(1, 11):
        wav_file = wavs_dir / f"{i:03d}.wav"
        wav_file.touch()  # Create empty file
    
    print(f"Created {len(list(wavs_dir.glob('*.wav')))} dummy wav files")
    
    print("\nTest dataset structure created successfully!")
    print("\nNOTE: This is a mock dataset for testing only.")
    print("For actual fine-tuning, you need to:")
    print("1. Clone the real dataset:")
    print("   git clone https://github.com/pnfo/sinhala-tts-dataset.git datasets/sinhala-tts-dataset")
    print("2. Ensure wav files are present in the wavs/ directory")
    
    return str(dataset_dir)

if __name__ == "__main__":
    dataset_path = create_test_dataset()
    print(f"\nDataset prepared at: {dataset_path}")