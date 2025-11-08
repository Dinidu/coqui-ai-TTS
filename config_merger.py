#!/usr/bin/env python3
"""
Config merger for fine-tuning pretrained TTS models
Intelligently merges pretrained config with new training parameters
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigMerger:
    """Merge pretrained model config with fine-tuning parameters"""
    
    # Model architecture parameters that MUST be preserved
    PRESERVED_KEYS = {
        # Model architecture
        'model_args.hidden_channels',
        'model_args.hidden_channels_ffn_text_encoder',
        'model_args.num_heads_text_encoder',
        'model_args.num_layers_text_encoder',
        'model_args.kernel_size_text_encoder',
        'model_args.dropout_p_text_encoder',
        'model_args.dropout_p_duration_predictor',
        'model_args.kernel_size_posterior_encoder',
        'model_args.dilation_rate_posterior_encoder',
        'model_args.num_layers_posterior_encoder',
        'model_args.kernel_size_flow',
        'model_args.dilation_rate_flow',
        'model_args.num_layers_flow',
        'model_args.resblock_type_decoder',
        'model_args.resblock_kernel_sizes_decoder',
        'model_args.resblock_dilation_sizes_decoder',
        'model_args.upsample_rates_decoder',
        'model_args.upsample_initial_channel_decoder',
        'model_args.upsample_kernel_sizes_decoder',
        'model_args.use_sdp',
        'model_args.spec_segment_size',
        'model_args.out_channels',
        
        # Audio configuration
        'audio.sample_rate',
        'audio.hop_length',
        'audio.win_length',
        'audio.num_mels',
        'audio.mel_fmin',
        'audio.mel_fmax',
        
        # Character configuration (critical for fine-tuning)
        'characters.characters_class',
        'characters.characters',
        'characters.punctuations',
        'characters.pad',
        'characters.eos',
        'characters.bos',
        'characters.blank',
        'characters.is_unique',
        'characters.is_sorted',
    }
    
    # Parameters that should be adjusted for fine-tuning
    FINETUNE_ADJUSTMENTS = {
        'lr': 1e-5,  # Lower learning rate for fine-tuning
        'lr_gen': 1e-5,
        'lr_disc': 1e-5,
        'epochs': 500,  # Fewer epochs needed
        'save_step': 1000,  # More frequent saves
        'run_eval_steps': 500,
        'test_delay_epochs': 10,
    }
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load config from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str):
        """Save config to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def get_nested_value(config: Dict, key_path: str, default=None):
        """Get value from nested dictionary using dot notation"""
        keys = key_path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    @staticmethod
    def set_nested_value(config: Dict, key_path: str, value):
        """Set value in nested dictionary using dot notation"""
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def merge(self, 
              pretrained_config_path: str,
              dataset_path: str,
              output_path: Optional[str] = None,
              custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge pretrained config with fine-tuning parameters
        
        Args:
            pretrained_config_path: Path to pretrained model config.json
            dataset_path: Path to new dataset
            output_path: Where to save training output
            custom_params: Additional parameters to override
        
        Returns:
            Merged configuration dictionary
        """
        # Load pretrained config
        config = self.load_config(pretrained_config_path)
        
        # Update dataset configuration
        config['datasets'] = [{
            'formatter': 'sinhala_formatter',
            'dataset_name': 'sinhala_finetune',
            'path': dataset_path,
            'meta_file_train': 'metadata_train.csv',
            'meta_file_val': 'metadata_val.csv',
            'language': 'si',
        }]
        
        # Update output path
        if output_path:
            config['output_path'] = output_path
        
        # Apply fine-tuning adjustments
        for key, value in self.FINETUNE_ADJUSTMENTS.items():
            if key in config:
                # For learning rates, use minimum of current and suggested
                if 'lr' in key:
                    config[key] = min(config.get(key, value), value)
                else:
                    config[key] = value
        
        # Ensure gradient clipping is a list for dual optimizers
        if 'grad_clip' in config:
            if not isinstance(config['grad_clip'], list):
                config['grad_clip'] = [config['grad_clip'], config['grad_clip']]
        else:
            config['grad_clip'] = [1000.0, 1000.0]
        
        # Apply custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                # Check if this is a preserved key
                full_key = key
                if full_key not in self.PRESERVED_KEYS:
                    self.set_nested_value(config, key, value)
                else:
                    print(f"Warning: Skipping preserved parameter: {key}")
        
        # Update run name
        config['run_name'] = 'vits_sinhala_finetune'
        
        # Validate character set if dataset exists
        dataset_dir = Path(dataset_path)
        if dataset_dir.exists():
            char_file = dataset_dir / 'character_set_sinhala.txt'
            if char_file.exists():
                with open(char_file, 'r', encoding='utf-8') as f:
                    dataset_chars = f.read()
                
                config_chars = self.get_nested_value(config, 'characters.characters', '')
                
                # Check coverage
                dataset_set = set(dataset_chars)
                config_set = set(config_chars)
                coverage = len(dataset_set & config_set) / len(config_set) * 100 if config_set else 0
                
                print(f"Character set coverage: {coverage:.1f}%")
                if coverage < 90:
                    print(f"⚠️  Warning: Low character coverage! Consider checking dataset.")
        
        return config
    
    def merge_to_file(self,
                      pretrained_config_path: str,
                      dataset_path: str,
                      output_config_path: str,
                      custom_params: Optional[Dict[str, Any]] = None):
        """Merge configs and save to file"""
        merged_config = self.merge(
            pretrained_config_path,
            dataset_path,
            custom_params=custom_params
        )
        self.save_config(merged_config, output_config_path)
        print(f"Merged config saved to: {output_config_path}")
        return merged_config


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge pretrained config with fine-tuning parameters")
    parser.add_argument("--pretrained_config", required=True,
                        help="Path to pretrained model config.json")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset directory")
    parser.add_argument("--output", required=True,
                        help="Path to save merged config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    
    args = parser.parse_args()
    
    # Prepare custom parameters
    custom_params = {}
    if args.batch_size:
        custom_params['batch_size'] = args.batch_size
        custom_params['eval_batch_size'] = args.batch_size * 2
    if args.epochs:
        custom_params['epochs'] = args.epochs
    if args.lr:
        custom_params['lr'] = args.lr
        custom_params['lr_gen'] = args.lr
        custom_params['lr_disc'] = args.lr
    
    # Merge configs
    merger = ConfigMerger()
    merger.merge_to_file(
        args.pretrained_config,
        args.dataset,
        args.output,
        custom_params
    )


if __name__ == "__main__":
    main()