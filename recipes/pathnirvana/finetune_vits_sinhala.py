import os
import sys
import torch

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Path to your base checkpoint - UPDATE THIS
BASE_CHECKPOINT_PATH = "/path/to/your/base/checkpoint.pth"  # e.g., "vits_sinhala-March-08-2023/best_model.pth"
BASE_CONFIG_PATH = "/path/to/your/base/config.json"  # e.g., "vits_sinhala-March-08-2023/config.json"

# Path to your new dataset - UPDATE THIS
NEW_DATASET_PATH = "/path/to/your/new/dataset/"  # Should contain wavs/ folder and metadata.csv

output_path = os.path.dirname(os.path.abspath(__file__))

# Dataset configuration for your new data
dataset_config = BaseDatasetConfig(
    formatter="pathnirvana_mettananda",  # Use same formatter as original
    meta_file_train="metadata.csv",  # Your metadata file
    path=NEW_DATASET_PATH
)

# Audio configuration (keep same as original model)
audio_config = VitsAudioConfig(
    sample_rate=22050, 
    win_length=1024, 
    hop_length=256, 
    num_mels=80, 
    mel_fmin=0, 
    mel_fmax=None
)

# Fine-tuning configuration
config = VitsConfig(
    audio=audio_config,
    run_name="vits_sinhala_finetuned",  # New run name
    
    # Reduced batch size for fine-tuning (adjust based on GPU memory)
    batch_size=16,  # Reduced from 56
    eval_batch_size=8,  # Reduced from 32
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    
    # Training parameters for fine-tuning
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,  # Reduced epochs for fine-tuning
    
    # Learning rate schedule for fine-tuning
    lr_gen=0.0001,  # Lower learning rate for fine-tuning
    lr_disc=0.0001,  # Lower learning rate for fine-tuning
    lr_scheduler_gen="ExponentialLR",
    lr_scheduler_gen_params={"gamma": 0.999875, "last_epoch": -1},
    lr_scheduler_disc="ExponentialLR",
    lr_scheduler_disc_params={"gamma": 0.999875, "last_epoch": -1},
    
    # Keep same text processing
    text_cleaner=None,
    use_phonemes=False,
    compute_input_seq_cache=True,
    max_audio_len=15 * 22050,
    add_blank=True,
    
    # Same character set as original
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=" !'(),-.:;=?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ",
        punctuations=" !'(),-.:;=?",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    
    # Test sentences with your speaker - UPDATE THESE
    test_sentences=[
        ["Your test sentence 1 in Sinhala", "your_speaker_name", None, None],
        ["Your test sentence 2 in Sinhala", "your_speaker_name", None, None],
        # Add more test sentences from your dataset
    ],
    
    print_step=25,  # More frequent printing for monitoring
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    eval_split_max_size=50,  # Smaller eval set for fine-tuning
    eval_split_size=0.1,
    
    # Fine-tuning specific
    save_step=500,  # Save checkpoints more frequently
    save_n_checkpoints=2,  # Keep only 2 latest checkpoints to save space
    save_best_after=0,  # Start saving best model immediately
)

# Initialize audio processor
ap = AudioProcessor.init_from_config(config)

# Initialize tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load data samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Initialize model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# IMPORTANT: Load pre-trained weights for fine-tuning
if os.path.exists(BASE_CHECKPOINT_PATH):
    print(f"Loading checkpoint from {BASE_CHECKPOINT_PATH}")
    checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    print("Successfully loaded pre-trained model weights!")
else:
    print(f"WARNING: Checkpoint not found at {BASE_CHECKPOINT_PATH}")
    print("Training from scratch instead of fine-tuning!")

# Initialize trainer with restore_path for continuing training
trainer = Trainer(
    TrainerArgs(
        restore_path=BASE_CHECKPOINT_PATH,  # Path to base checkpoint
        skip_train_epoch=False,  # Don't skip training epochs
    ),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# Start fine-tuning
trainer.fit()