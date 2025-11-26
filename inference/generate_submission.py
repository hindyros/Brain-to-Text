"""
Generate submission.csv file for Kaggle competition.

Supports both CTC-based and end-to-end models.
"""

import torch
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import argparse
import yaml

from data.tokenizer import PhonemeTokenizer, CharTokenizer
from models.ctc_models import RecurrentModel, TransformerEncModel, ConformerModel
from models.e2e_models import EndToEndModel


def load_ctc_model(checkpoint_path, model_type, config, device):
    """Load CTC-based model."""
    from training.train_ctc import create_model
    
    model = create_model(model_type, config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_e2e_model(checkpoint_path, config, device):
    """Load end-to-end model."""
    model_config = config.get('model', {})
    tokenizer = CharTokenizer()
    
    model = EndToEndModel(
        n_channels=model_config.get('n_channels', 512),
        d_model=model_config.get('d_model', 512),
        vocab_size=tokenizer.vocab_size,
        n_encoder_layers=model_config.get('n_encoder_layers', 8),
        n_decoder_layers=model_config.get('n_decoder_layers', 6),
        n_heads=model_config.get('n_heads', 8),
        dropout=0.0  # No dropout in inference
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def ctc_decode(model, neural_features, tokenizer, device):
    """Decode using CTC model."""
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        if neural_features.dim() == 2:
            neural_features = neural_features.unsqueeze(0)
        
        neural_features = torch.tensor(neural_features, dtype=torch.float32).to(device)
        
        # Forward pass
        logits = model(neural_features)  # (1, T, vocab_size)
        
        # Greedy decoding
        pred_indices = torch.argmax(logits[0], dim=-1)
        
        # Decode to phonemes
        phoneme_text = tokenizer.decode(pred_indices, collapse_repeats=True)
        
        # Convert phonemes to text (simplified - should use language model)
        # For now, return phonemes as-is
        return phoneme_text


def e2e_decode(model, neural_features, tokenizer, device, max_len=200, temperature=1.0):
    """Decode using end-to-end model."""
    model.eval()
    with torch.no_grad():
        # Convert to tensor
        if not isinstance(neural_features, torch.Tensor):
            neural_features = torch.tensor(neural_features, dtype=torch.float32)
        
        if neural_features.dim() == 2:
            neural_features = neural_features.unsqueeze(0)
        
        neural_features = neural_features.to(device)
        
        # Generate tokens
        tokens = model.inference(neural_features, max_len=max_len, temperature=temperature)
        
        # Decode to text
        text = tokenizer.decode(tokens)
        
        # Normalize
        text = normalize_text(text)
        
        return text


def normalize_text(text):
    """Normalize text for submission (remove punctuation except apostrophe)."""
    import re
    text = text.lower()
    text = text.replace("'", "'")
    
    # Remove punctuation except apostrophe
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def generate_submission_ctc(
    model_path,
    config_path,
    test_data_dir,
    output_path='submission.csv',
    device='cuda',
    model_type='LSTM'
):
    """Generate submission using CTC model."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading CTC model ({model_type}) from {model_path}...")
    model = load_ctc_model(model_path, model_type, config, device)
    tokenizer = PhonemeTokenizer()
    
    # Find test files
    test_files = sorted(glob.glob(f"{test_data_dir}/t15.*/data_test.hdf5"))
    print(f"Found {len(test_files)} test files.")
    
    all_predictions = []
    
    print("Generating predictions...")
    for file_path in tqdm(test_files, desc="Files"):
        with h5py.File(file_path, 'r') as f:
            keys = sorted(f.keys())
            for key in tqdm(keys, desc=f"Trials in {Path(file_path).parent.name}", leave=False):
                neural_features = f[key]['input_features'][:]
                
                # Decode
                phoneme_text = ctc_decode(model, neural_features, tokenizer, device)
                
                # Convert phonemes to text (simplified - should use language model)
                # For now, use phonemes directly
                text = normalize_text(phoneme_text)
                all_predictions.append(text)
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': range(len(all_predictions)),
        'text': all_predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    
    return submission_df


def generate_submission_e2e(
    model_path,
    config_path,
    test_data_dir,
    output_path='submission.csv',
    device='cuda',
    max_len=200,
    temperature=1.0
):
    """Generate submission using end-to-end model."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading end-to-end model from {model_path}...")
    model = load_e2e_model(model_path, config, device)
    tokenizer = CharTokenizer()
    
    # Find test files
    test_files = sorted(glob.glob(f"{test_data_dir}/t15.*/data_test.hdf5"))
    print(f"Found {len(test_files)} test files.")
    
    all_predictions = []
    
    print("Generating predictions...")
    for file_path in tqdm(test_files, desc="Files"):
        with h5py.File(file_path, 'r') as f:
            keys = sorted(f.keys())
            for key in tqdm(keys, desc=f"Trials in {Path(file_path).parent.name}", leave=False):
                neural_features = f[key]['input_features'][:]
                
                # Decode
                text = e2e_decode(model, neural_features, tokenizer, device, max_len, temperature)
                all_predictions.append(text)
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': range(len(all_predictions)),
        'text': all_predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    
    return submission_df


def main():
    parser = argparse.ArgumentParser(description='Generate submission file')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--test-data-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output CSV path')
    parser.add_argument('--model-type', type=str, choices=['ctc', 'e2e'],
                       required=True, help='Model type')
    parser.add_argument('--ctc-arch', type=str, default='LSTM',
                       choices=['RNN', 'LSTM', 'GRU', 'TRANSFORMER', 'CONFORMER'],
                       help='CTC architecture (if using CTC)')
    parser.add_argument('--max-len', type=int, default=200,
                       help='Max generation length (for e2e)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (for e2e)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.model_type == 'ctc':
        generate_submission_ctc(
            args.model_path,
            args.config,
            args.test_data_dir,
            args.output,
            device,
            args.ctc_arch
        )
    else:
        generate_submission_e2e(
            args.model_path,
            args.config,
            args.test_data_dir,
            args.output,
            device,
            args.max_len,
            args.temperature
        )


if __name__ == '__main__':
    main()

