#!/usr/bin/env python3
"""
Dataset Preparation Script for Production Training
- Validates audio files
- Splits dataset into train/validation
- Computes statistics
- Creates normalized metadata
"""

import os
import sys
import json
import random
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import argparse

def validate_audio_file(audio_path, min_duration=0.5, max_duration=15.0, sample_rate=22050):
    """Validate audio file and return duration"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        # Check duration
        if duration < min_duration or duration > max_duration:
            return False, f"Duration {duration:.2f}s out of range [{min_duration}, {max_duration}]"
        
        # Check sample rate
        if sr != sample_rate:
            # Resample if needed
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            # Save resampled audio
            sf.write(audio_path, audio, sample_rate)
            
        # Check for silence
        if np.max(np.abs(audio)) < 0.01:
            return False, "Audio is silent or too quiet"
            
        return True, duration
        
    except Exception as e:
        return False, str(e)

def analyze_text(text):
    """Analyze text for character statistics"""
    char_counts = Counter(text)
    return {
        'length': len(text),
        'unique_chars': len(char_counts),
        'char_counts': dict(char_counts)
    }

def split_dataset(items, val_ratio=0.1, speaker_balanced=True):
    """Split dataset into train and validation sets"""
    if speaker_balanced:
        # Group by speaker
        speaker_items = {}
        for item in items:
            speaker = item.get('speaker', 'default')
            if speaker not in speaker_items:
                speaker_items[speaker] = []
            speaker_items[speaker].append(item)
        
        train_items = []
        val_items = []
        
        # Split each speaker's data
        for speaker, s_items in speaker_items.items():
            random.shuffle(s_items)
            n_val = max(1, int(len(s_items) * val_ratio))
            val_items.extend(s_items[:n_val])
            train_items.extend(s_items[n_val:])
    else:
        # Simple random split
        random.shuffle(items)
        n_val = int(len(items) * val_ratio)
        val_items = items[:n_val]
        train_items = items[n_val:]
    
    return train_items, val_items

def prepare_dataset(
    input_path,
    output_path,
    metadata_file="metadata.csv",
    val_ratio=0.1,
    min_duration=0.5,
    max_duration=15.0,
    sample_rate=22050,
    text_column=2,  # Which column to use for text (0-indexed)
    speaker_column=None,  # Optional speaker column
):
    """Prepare dataset for production training"""
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Dataset Preparation for Production Training")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Validation ratio: {val_ratio}")
    print(f"Duration range: [{min_duration}, {max_duration}] seconds")
    print(f"Target sample rate: {sample_rate} Hz")
    print("=" * 80)
    
    # Read metadata
    metadata_path = input_path / metadata_file
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    items = []
    invalid_items = []
    text_stats = []
    
    print("\nReading metadata and validating audio files...")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Processing"):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('|')
        if len(parts) < text_column + 1:
            invalid_items.append(('metadata', line, 'Invalid format'))
            continue
        
        # Extract fields
        audio_name = parts[0]
        text = parts[text_column].strip()
        speaker = parts[speaker_column].strip() if speaker_column and len(parts) > speaker_column else "default"
        
        # Check audio file
        audio_path = input_path / "wavs" / f"{audio_name}.wav"
        if not audio_path.exists():
            invalid_items.append((audio_name, text, 'Audio file not found'))
            continue
        
        # Validate audio
        is_valid, info = validate_audio_file(audio_path, min_duration, max_duration, sample_rate)
        if not is_valid:
            invalid_items.append((audio_name, text, info))
            continue
        
        # Copy audio to output directory
        output_audio = wavs_dir / f"{audio_name}.wav"
        if not output_audio.exists():
            shutil.copy2(audio_path, output_audio)
        
        # Analyze text
        text_stat = analyze_text(text)
        text_stats.append(text_stat)
        
        # Add to valid items
        items.append({
            'audio': audio_name,
            'text': text,
            'speaker': speaker,
            'duration': info,  # duration in seconds
            'text_length': text_stat['length']
        })
    
    print(f"\nValid samples: {len(items)}")
    print(f"Invalid samples: {len(invalid_items)}")
    
    if invalid_items:
        # Save invalid items for review
        invalid_path = output_path / "invalid_samples.txt"
        with open(invalid_path, 'w', encoding='utf-8') as f:
            for audio, text, reason in invalid_items[:100]:  # Save first 100
                f.write(f"{audio}|{text}|{reason}\n")
        print(f"Invalid samples saved to: {invalid_path}")
    
    # Compute statistics
    print("\nComputing dataset statistics...")
    
    durations = [item['duration'] for item in items]
    text_lengths = [item['text_length'] for item in items]
    speakers = Counter(item['speaker'] for item in items)
    
    # Collect all unique characters
    all_chars = set()
    for stat in text_stats:
        all_chars.update(stat['char_counts'].keys())
    
    stats = {
        'total_samples': len(items),
        'total_duration_hours': sum(durations) / 3600,
        'avg_duration': np.mean(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'avg_text_length': np.mean(text_lengths),
        'min_text_length': min(text_lengths),
        'max_text_length': max(text_lengths),
        'num_speakers': len(speakers),
        'speakers': dict(speakers),
        'unique_characters': sorted(list(all_chars)),
        'num_unique_characters': len(all_chars),
    }
    
    # Save statistics
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total duration: {stats['total_duration_hours']:.2f} hours")
    print(f"  Average duration: {stats['avg_duration']:.2f} seconds")
    print(f"  Duration range: [{stats['min_duration']:.2f}, {stats['max_duration']:.2f}] seconds")
    print(f"  Average text length: {stats['avg_text_length']:.1f} characters")
    print(f"  Text length range: [{stats['min_text_length']}, {stats['max_text_length']}] characters")
    print(f"  Number of speakers: {stats['num_speakers']}")
    print(f"  Unique characters: {stats['num_unique_characters']}")
    
    # Split dataset
    print(f"\nSplitting dataset (validation ratio: {val_ratio})...")
    train_items, val_items = split_dataset(items, val_ratio, speaker_balanced=(len(speakers) > 1))
    
    print(f"  Training samples: {len(train_items)}")
    print(f"  Validation samples: {len(val_items)}")
    
    # Save metadata files
    train_meta_path = output_path / "metadata_train.csv"
    val_meta_path = output_path / "metadata_val.csv"
    
    with open(train_meta_path, 'w', encoding='utf-8') as f:
        for item in train_items:
            # Reconstruct the original format with all columns
            f.write(f"{item['audio']}|{item['text']}|{item['speaker']}\n")
    
    with open(val_meta_path, 'w', encoding='utf-8') as f:
        for item in val_items:
            f.write(f"{item['audio']}|{item['text']}|{item['speaker']}\n")
    
    print(f"\nMetadata files saved:")
    print(f"  Training: {train_meta_path}")
    print(f"  Validation: {val_meta_path}")
    
    # Create character set file for model configuration
    charset_path = output_path / "character_set.txt"
    with open(charset_path, 'w', encoding='utf-8') as f:
        f.write(''.join(sorted(all_chars)))
    print(f"  Character set: {charset_path}")
    
    # Create a sample configuration snippet
    config_snippet = f"""
# Add this to your training configuration:

characters=CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    characters="{''.join(sorted(all_chars))}",
    punctuations="{''.join([c for c in all_chars if not c.isalnum() and c != ' '])}",
    phonemes=None,
    is_unique=True,
    is_sorted=True,
)
"""
    
    config_path = output_path / "config_snippet.py"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_snippet)
    print(f"  Config snippet: {config_path}")
    
    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print(f"Output directory: {output_path}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for TTS training")
    parser.add_argument("input_path", help="Path to input dataset")
    parser.add_argument("output_path", help="Path to output dataset")
    parser.add_argument("--metadata", default="metadata.csv", help="Metadata filename")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum audio duration")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Maximum audio duration")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--text-column", type=int, default=2, help="Text column index (0-based)")
    parser.add_argument("--speaker-column", type=int, default=None, help="Speaker column index (0-based)")
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.input_path,
        args.output_path,
        args.metadata,
        args.val_ratio,
        args.min_duration,
        args.max_duration,
        args.sample_rate,
        args.text_column,
        args.speaker_column
    )

if __name__ == "__main__":
    main()