"""
Dataset loading utilities for Brain-to-Text competition.

This module provides:
- BrainDataset: Dataset class for loading HDF5 neural data
- Data augmentation functions (temporal masking, electrode dropout, etc.)
- Collate functions for batching variable-length sequences
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.utils.rnn as rnn_utils


def temporal_mask(data, mask_percentage=0.05, mask_value=0.0):
    """
    Applies temporal masking to a 2D tensor [Sequence, Features].
    
    Args:
        data: Input tensor of shape (T, F)
        mask_percentage: Percentage of timesteps to mask
        mask_value: Value to use for masking
        
    Returns:
        Masked tensor
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)
        
    seq_len, _ = data.shape
    num_to_mask = int(seq_len * mask_percentage)
    
    if num_to_mask > 0:
        mask_indices = torch.randperm(seq_len)[:num_to_mask]
        data[mask_indices, :] = mask_value
        
    return data


def electrode_dropout(data, dropout_prob=0.1):
    """
    Randomly zero out entire electrode channels.
    
    Args:
        data: Input tensor of shape (T, F)
        dropout_prob: Probability of dropping each electrode
        
    Returns:
        Augmented tensor
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)
    
    num_features = data.shape[1]
    drop_mask = torch.bernoulli(torch.ones(num_features) * (1 - dropout_prob))
    return data * drop_mask


def gaussian_noise(data, std=0.05):
    """
    Add Gaussian noise to neural features.
    
    Args:
        data: Input tensor
        std: Standard deviation of noise
        
    Returns:
        Noisy tensor
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)
    
    return data + torch.randn_like(data) * std


class BrainDataset(Dataset):
    """
    Dataset for loading neural data from HDF5 files.
    
    Supports:
    - Training/validation/test splits
    - Data augmentation (temporal masking, electrode dropout, noise)
    - Variable-length sequences with CTC-ready targets
    - Text labels for end-to-end training
    """
    
    def __init__(
        self, 
        hdf5_file, 
        input_key="input_features", 
        target_key="seq_class_ids", 
        is_test=False, 
        use_augmentation=False,
        augmentation_config=None,
        return_text=False
    ):
        """
        Args:
            hdf5_file: Path to HDF5 file
            input_key: Key for input neural features
            target_key: Key for target sequence (phoneme indices)
            is_test: Whether this is test data (no labels)
            use_augmentation: Whether to apply data augmentation
            augmentation_config: Dict with augmentation parameters
        """
        self.file_path = hdf5_file
        self.input_key = input_key
        self.target_key = target_key
        self.is_test = is_test
        self.use_augmentation = use_augmentation
        self.return_text = return_text  # For end-to-end models
        
        # Default augmentation config
        if augmentation_config is None:
            augmentation_config = {
                'temporal_mask_prob': 0.3,
                'temporal_mask_percentage': 0.1,
                'electrode_dropout_prob': 0.2,
                'electrode_dropout_rate': 0.1,
                'gaussian_noise_prob': 0.25,
                'gaussian_noise_std': 0.05
            }
        self.augmentation_config = augmentation_config
        
        self.file = None  # File handle (lazy loading)
        
        try:
            with h5py.File(self.file_path, "r") as f:
                self.trial_keys = sorted(list(f.keys()))
        except FileNotFoundError:
            print(f"Warning: File not found {self.file_path}, creating empty dataset.")
            self.trial_keys = []

    def __len__(self):
        return len(self.trial_keys)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
            
        trial_key = self.trial_keys[idx]
        trial_group = self.file[trial_key]
        
        # Load neural features
        x_data = trial_group[self.input_key][:]
        x = torch.tensor(x_data, dtype=torch.float32)
        
        # Apply augmentation (only during training)
        if self.use_augmentation and not self.is_test:
            x = self._apply_augmentation(x)
        
        # Load targets (if available)
        if self.target_key in trial_group:
            y_data = trial_group[self.target_key][:]
            y = torch.tensor(y_data, dtype=torch.long)
        else:
            # Empty tensor for test data
            y = torch.tensor([], dtype=torch.long)
        
        # Load text label if needed (for end-to-end models)
        text_label = None
        if self.return_text and 'sentence_label' in trial_group.attrs:
            text_label = trial_group.attrs['sentence_label']
        
        if self.is_test:
            if self.return_text:
                return x, y, text_label, trial_key
            return x, y, trial_key
        else:
            if self.return_text:
                return x, y, text_label
            return x, y
    
    def _apply_augmentation(self, x):
        """Apply random augmentations to input."""
        cfg = self.augmentation_config
        
        # Temporal masking
        if torch.rand(1) < cfg['temporal_mask_prob']:
            x = temporal_mask(x, mask_percentage=cfg['temporal_mask_percentage'])
        
        # Electrode dropout
        if torch.rand(1) < cfg['electrode_dropout_prob']:
            x = electrode_dropout(x, dropout_prob=cfg['electrode_dropout_rate'])
        
        # Gaussian noise
        if torch.rand(1) < cfg['gaussian_noise_prob']:
            x = gaussian_noise(x, std=cfg['gaussian_noise_std'])
        
        return x


def load_datasets(data_dir, use_augmentation=True, return_text=False):
    """
    Load train, validation, and test datasets from all session folders.
    
    Args:
        data_dir: Root directory containing session subfolders
        use_augmentation: Whether to apply augmentation to training set
        return_text: Whether to return text labels (for end-to-end models)
        
    Returns:
        train_dataset, val_dataset, test_dataset: ConcatDataset objects
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []

    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    print(f"Found {len(subfolders)} session folders.")
    
    for subfolder_path in subfolders:
        session_name = os.path.basename(subfolder_path)
        
        train_file = os.path.join(subfolder_path, "data_train.hdf5")
        val_file = os.path.join(subfolder_path, "data_val.hdf5")
        test_file = os.path.join(subfolder_path, "data_test.hdf5")

        # Create datasets (augmentation only for training)
        train_set = BrainDataset(
            train_file, 
            input_key="input_features", 
            target_key="seq_class_ids", 
            is_test=False, 
            use_augmentation=use_augmentation,
            return_text=return_text
        )
        val_set = BrainDataset(
            val_file, 
            input_key="input_features", 
            target_key="seq_class_ids", 
            is_test=False, 
            use_augmentation=False,
            return_text=return_text
        )
        test_set = BrainDataset(
            test_file, 
            input_key="input_features", 
            target_key="seq_class_ids", 
            is_test=True, 
            use_augmentation=False,
            return_text=return_text
        ) 
        
        if len(train_set) > 0:
            train_datasets.append(train_set)
        if len(val_set) > 0:
            val_datasets.append(val_set)
        if len(test_set) > 0:
            test_datasets.append(test_set)
            
    # Combine all datasets
    full_train_dataset = ConcatDataset(train_datasets) if train_datasets else ConcatDataset([])
    full_val_dataset = ConcatDataset(val_datasets) if val_datasets else ConcatDataset([])
    full_test_dataset = ConcatDataset(test_datasets) if test_datasets else ConcatDataset([])
    
    return full_train_dataset, full_val_dataset, full_test_dataset


def ctc_collate_fn(batch):
    """
    Custom collate function for CTC loss (Sequence-to-Sequence).
    
    Pads both inputs and targets, returns lengths for CTC.
    Handles both training (x, y) and test (x, y, key) batches.
    """
    # Check if it's a test batch (has 3 items: x, y, key)
    is_test = len(batch[0]) == 3

    if is_test:
        xs, ys, keys = zip(*batch)
    else:
        xs, ys = zip(*batch)
        
    # Get unpadded lengths (required by CTCLoss)
    x_lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    y_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    
    # Pad inputs
    padded_xs = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0.0)
    
    # Pad targets (padding_value=0 is blank token for CTC)
    padded_ys = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=0)
    
    if is_test:
        return padded_xs, padded_ys, x_lengths, y_lengths, keys
    else:
        return padded_xs, padded_ys, x_lengths, y_lengths


def text_collate_fn(batch, tokenizer=None):
    """
    Custom collate function for text generation (end-to-end).
    
    Pads neural features and text tokens separately.
    
    Args:
        batch: List of (neural, targets, text_label) or (neural, targets, text_label, key) tuples
        tokenizer: CharTokenizer instance for encoding text labels
    """
    # Check if test batch (has 4 items) or train batch (has 3 items)
    is_test = len(batch[0]) == 4
    
    if is_test:
        neurals, targets, text_labels, keys = zip(*batch)
    else:
        neurals, targets, text_labels = zip(*batch)
        keys = None
    
    # Pad neural features
    neural_padded = rnn_utils.pad_sequence(neurals, batch_first=True, padding_value=0.0)
    
    # Encode text labels to tokens if tokenizer provided
    tokens_padded = None
    if tokenizer is not None and text_labels[0] is not None:
        tokens_list = []
        for text in text_labels:
            if text:
                tokens = tokenizer.encode(text)
                tokens_list.append(tokens)
            else:
                tokens_list.append(torch.tensor([], dtype=torch.long))
        
        if tokens_list and len(tokens_list[0]) > 0:
            tokens_padded = rnn_utils.pad_sequence(tokens_list, batch_first=True, padding_value=0)
    
    if is_test:
        return {
            'neural': neural_padded,
            'tokens': tokens_padded,
            'keys': keys,
            'texts': text_labels
        }
    else:
        return {
            'neural': neural_padded,
            'tokens': tokens_padded,
            'texts': text_labels
        }

