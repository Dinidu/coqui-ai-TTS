#!/usr/bin/env python3
"""
Prepare Sinhala dataset for TTS fine-tuning
Ensures native Sinhala script is used to match pretrained model
"""

import os
import shutil
import json
from pathlib import Path
import argparse
from collections import Counter

def get_unique_chars(text_list):
    """Extract unique characters from text list"""
    all_chars = ''.join(text_list)
    unique_chars = sorted(set(all_chars))
    return ''.join(unique_chars)

def prepare_dataset(source_dir, target_dir, train_split=0.9):
    """
    Prepare dataset with native Sinhala text
    
    Args:
        source_dir: Path to original-dataset
        target_dir: Path to output directory
        train_split: Percentage for training (rest for validation)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    target_path.mkdir(parents=True, exist_ok=True)
    wavs_path = target_path / "wavs"
    wavs_path.mkdir(exist_ok=True)
    
    print(f"Preparing dataset from {source_path} to {target_path}")
    
    # Read original metadata
    metadata_file = source_path / "metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # Parse metadata
    all_samples = []
    sinhala_texts = []
    romanized_texts = []
    
    print("Reading metadata...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) != 3:
                print(f"Warning: Skipping invalid line: {line}")
                continue
            
            audio_id = parts[0]
            sinhala_text = parts[1].strip()
            romanized_text = parts[2].strip()
            
            # Check if audio exists
            source_wav = source_path / "wavs" / f"{audio_id}.wav"
            if not source_wav.exists():
                print(f"Warning: Audio not found: {source_wav}")
                continue
            
            all_samples.append({
                'id': audio_id,
                'sinhala': sinhala_text,
                'romanized': romanized_text,
                'wav_path': source_wav
            })
            sinhala_texts.append(sinhala_text)
            romanized_texts.append(romanized_text)
    
    print(f"Found {len(all_samples)} valid samples")
    
    # Split into train/val
    split_idx = int(len(all_samples) * train_split)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Copy audio files
    print("Copying audio files...")
    copied = 0
    for sample in all_samples:
        target_wav = wavs_path / f"{sample['id']}.wav"
        if not target_wav.exists():
            shutil.copy2(sample['wav_path'], target_wav)
            copied += 1
            if copied % 100 == 0:
                print(f"  Copied {copied} files...")
    
    print(f"Audio files copied: {copied}")
    
    # Write metadata files
    def write_metadata(samples, filename):
        filepath = target_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Format: audio_id|sinhala_text|romanized_text
                line = f"{sample['id']}|{sample['sinhala']}|{sample['romanized']}\n"
                f.write(line)
        print(f"Wrote {filepath}")
    
    write_metadata(train_samples, "metadata_train.csv")
    write_metadata(val_samples, "metadata_val.csv")
    
    # Extract character sets
    sinhala_chars = get_unique_chars(sinhala_texts)
    romanized_chars = get_unique_chars(romanized_texts)
    
    # Save character set
    char_file = target_path / "character_set_sinhala.txt"
    with open(char_file, 'w', encoding='utf-8') as f:
        f.write(sinhala_chars)
    print(f"Sinhala character set saved: {len(sinhala_chars)} characters")
    
    # Also save romanized for reference
    char_file_rom = target_path / "character_set_romanized.txt"
    with open(char_file_rom, 'w', encoding='utf-8') as f:
        f.write(romanized_chars)
    
    # Create dataset stats
    stats = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'sinhala_chars': len(sinhala_chars),
        'romanized_chars': len(romanized_chars),
        'sinhala_charset': sinhala_chars,
        'avg_sinhala_length': sum(len(s['sinhala']) for s in all_samples) / len(all_samples),
        'avg_romanized_length': sum(len(s['romanized']) for s in all_samples) / len(all_samples),
    }
    
    stats_file = target_path / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Dataset statistics saved to {stats_file}")
    
    # Compare with pretrained model character set
    pretrained_chars = " !'(),-.:;?ංඃඅආඇඈඉඊඋඌඍඑඒඓඔඕඖකඛගඝඞඟචඡජඣඤඥටඨඩඪණඬතථදධනඳපඵබභමඹයරලවශෂසහළෆ්ාැෑිීුූෘෙේෛොෝෞෟෲ\n"
    
    print("\n" + "="*60)
    print("Character Set Comparison:")
    print("="*60)
    
    # Find missing characters
    dataset_set = set(sinhala_chars)
    pretrained_set = set(pretrained_chars)
    
    missing_in_dataset = pretrained_set - dataset_set
    extra_in_dataset = dataset_set - pretrained_set
    
    if missing_in_dataset:
        print(f"⚠️  Characters in pretrained but missing in dataset: {''.join(sorted(missing_in_dataset))}")
    if extra_in_dataset:
        print(f"ℹ️  New characters in dataset: {''.join(sorted(extra_in_dataset))}")
    
    common_chars = dataset_set & pretrained_set
    print(f"✓ Common characters: {len(common_chars)} out of {len(pretrained_set)} pretrained")
    print(f"  Coverage: {len(common_chars)/len(pretrained_set)*100:.1f}%")
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    
    return target_path

def main():
    parser = argparse.ArgumentParser(description="Prepare Sinhala dataset for TTS fine-tuning")
    parser.add_argument("--source", default="datasets/original-dataset", 
                        help="Source dataset directory")
    parser.add_argument("--target", default="datasets/sinhala-native", 
                        help="Target dataset directory")
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="Training split ratio (default: 0.9)")
    args = parser.parse_args()
    
    prepare_dataset(args.source, args.target, args.train_split)

if __name__ == "__main__":
    main()