"""Custom formatter for Sinhala TTS dataset"""
import os

def sinhala_formatter(root_path, meta_file, **kwargs):
    """
    Formatter for Sinhala dataset with format:
    audio_filename|sinhala_text|romanized_text
    
    Uses the native Sinhala text (2nd column) for training
    to match pretrained model's character set
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "oshadi"  # Default speaker name
    
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            line = line.strip()
            if not line:
                continue
                
            cols = line.split("|")
            if len(cols) != 3:
                print(f"Warning: Skipping line with {len(cols)} columns: {line}")
                continue
            
            # Column 0: audio filename (without extension)
            # Column 1: Sinhala text (native script)
            # Column 2: Romanized text
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            
            # Use native Sinhala text (column 1) for training
            text = cols[1].strip()
            
            if os.path.exists(wav_file):
                items.append({
                    "text": text, 
                    "audio_file": wav_file, 
                    "speaker_name": speaker_name, 
                    "root_path": root_path
                })
            else:
                print(f"Warning: Audio file not found: {wav_file}")
    
    print(f"Loaded {len(items)} samples from {meta_file}")
    return items