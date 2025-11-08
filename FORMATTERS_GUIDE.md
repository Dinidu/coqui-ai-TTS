# TTS Dataset Formatters Guide

This document lists all available dataset formatters in Coqui TTS and their expected metadata formats.

## Available Formatters

### 1. **ljspeech**
- **Format**: `filename|unused|text`
- **Example**: `LJ001-0001|unused|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition`
- **Notes**: Standard LJSpeech format, uses column 3 for text

### 2. **ljspeech_test**
- **Format**: Same as ljspeech but assigns different speaker IDs for testing
- **Notes**: Creates multiple speakers from single speaker dataset

### 3. **thorsten**
- **Format**: `filename|text`
- **Example**: `thorsten_001|Das ist ein Beispieltext`
- **Notes**: German TTS dataset

### 4. **cml_tts**
- **Format**: CSV with headers including `wav_filename`, `transcript`, `client_id`, `emotion_name`
- **Notes**: Multi-speaker emotional TTS dataset

### 5. **coqui**
- **Format**: CSV with `audio_file` and `text` columns
- **Notes**: Simple CSV format

### 6. **tweb**
- **Format**: `filename|text`
- **Notes**: Two column format

### 7. **mozilla**
- **Format**: `filename|text`
- **Notes**: Mozilla Common Voice format

### 8. **mozilla_de**
- **Format**: `filename|text`
- **Notes**: German Mozilla dataset

### 9. **mailabs**
- **Format**: Directory structure based, no metadata file needed
- **Notes**: Multi-speaker dataset organized by folders

### 10. **sam_accenture**
- **Format**: XML file with voice recordings
- **Notes**: Non-binary voice dataset

### 11. **ruslan**
- **Format**: `filename|text`
- **Notes**: Russian TTS dataset

### 12. **css10**
- **Format**: `filepath|text`
- **Notes**: CSS10 multi-language dataset

### 13. **nancy**
- **Format**: `filename|text`
- **Notes**: Nancy corpus

### 14. **common_voice**
- **Format**: TSV with columns: `client_id`, `path`, `sentence`, `up_votes`, `down_votes`, `age`, `gender`, `accent`
- **Notes**: Mozilla Common Voice format

### 15. **libri_tts**
- **Format**: Directory structure based with speaker folders
- **Notes**: LibriTTS dataset

### 16. **custom_turkish**
- **Format**: `filename|text|phoneme`
- **Notes**: Turkish dataset with phonemes

### 17. **brspeech**
- **Format**: `filename|speaker_id|text`
- **Notes**: Brazilian Portuguese dataset

### 18. **vctk**
- **Format**: Directory structure with txt files for transcripts
- **Notes**: Multi-speaker British English dataset

### 19. **vctk_old**
- **Format**: Older VCTK format
- **Notes**: Legacy VCTK support

### 20. **open_bible**
- **Format**: Directory structure with verse files
- **Notes**: Open Bible dataset

### 21. **mls**
- **Format**: Directory structure with transcripts
- **Notes**: Multilingual LibriSpeech

### 22. **emotion**
- **Format**: Directory structure organized by emotion
- **Notes**: Emotional speech dataset

### 23. **kokoro**
- **Format**: `filename|text`
- **Notes**: Japanese TTS dataset

### 24. **kss**
- **Format**: `filename|text|duration`
- **Notes**: Korean Single Speaker dataset

### 25. **bel_tts_formatter**
- **Format**: `filename|text`
- **Notes**: Belarusian TTS dataset

### 26. **pathnirvana**
- **Format**: `filename|sinhala_text|romanized_text|speaker`
- **Example**: `sin_01_00001|සිංහල පාඨය|romanized text|oshadi`
- **Notes**: Uses column 3 (romanized text) for training, column 4 for speaker

### 27. **pathnirvana2**
- **Format**: `filename|text|unused|speaker`
- **Notes**: Uses column 2 (first text field) for training

### 28. **pathnirvana_mettananda**
- **Format**: `filename|text|unused|speaker`
- **Notes**: Filters only "mettananda" speaker

## Your Dataset Format

Your dataset uses a 3-column format:
```
sin_01_00001|කුඹුර ගොවියාට වී ලබා ගැනීමට උපකාරී වීම් වශයෙන් පිහිට වන්නකි.|kumbura goviyāṭa vī labā gænīmaṭa upakārī vīm vaśayen pihiṭa vannaki.
```

### Compatible Formatters:
1. **Custom formatter** (what we created) - Uses column 3 for romanized text
2. **ljspeech** - Would use column 3 but expects different format
3. **custom_turkish** - Similar 3-column format

### Best Choice:
The custom formatter we created is ideal because it:
- Uses the romanized text (column 3) for training
- Properly handles the Sinhala text structure
- Can be easily modified to use either column

## Usage Example

```python
# In your config
dataset_config = BaseDatasetConfig(
    formatter="formatter_name",  # e.g., "ljspeech", "vctk", etc.
    meta_file_train="metadata.csv",
    path="/path/to/dataset"
)
```

## Creating Custom Formatters

To create a custom formatter:

1. Define a function that takes `root_path` and `meta_file`
2. Return a list of dictionaries with keys:
   - `text`: The text to be synthesized
   - `audio_file`: Full path to audio file
   - `speaker_name`: Speaker identifier
   - `root_path`: Dataset root path

Example:
```python
def my_formatter(root_path, meta_file, **kwargs):
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("|")
            items.append({
                "text": cols[1],
                "audio_file": os.path.join(root_path, "wavs", cols[0] + ".wav"),
                "speaker_name": "speaker1",
                "root_path": root_path
            })
    return items
```