"""
Training script for CTC-based models (phoneme prediction).

Supports:
- RNN/LSTM/GRU models
- Transformer encoder models
- Conformer models
- Fine-tuning from pretrained checkpoints
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
from pathlib import Path

from data.dataset import load_datasets, ctc_collate_fn
from data.tokenizer import PhonemeTokenizer, BLANK_ID
from models.ctc_models import RecurrentModel, TransformerEncModel, ConformerModel
import jiwer


class CTCTrainer:
    """Trainer for CTC-based phoneme prediction models."""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Tokenizer
        self.tokenizer = PhonemeTokenizer()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_per = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'per': []}
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for x, y, x_lengths, y_lengths in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            x_lengths = x_lengths.to(self.device)
            y_lengths = y_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(x)  # (B, T, vocab_size)
            y_pred_for_loss = y_pred.permute(1, 0, 2)  # (T, B, vocab_size) for CTC
            
            # CTC loss
            loss = self.criterion(y_pred_for_loss, y, x_lengths, y_lengths)
            
            if torch.isinf(loss) or torch.isnan(loss):
                print("Warning: Skipping batch with inf/nan loss")
                continue
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return running_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validate and compute phoneme error rate."""
        self.model.eval()
        val_loss = 0.0
        all_pred_texts = []
        all_true_texts = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for x, y, x_lengths, y_lengths in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            x_lengths = x_lengths.to(self.device)
            y_lengths = y_lengths.to(self.device)
            
            # Forward pass
            y_pred = self.model(x)
            y_pred_for_loss = y_pred.permute(1, 0, 2)
            
            # Loss
            loss = self.criterion(y_pred_for_loss, y, x_lengths, y_lengths)
            val_loss += loss.item() * x.size(0)
            
            # Decode predictions
            for i in range(x.size(0)):
                pred_logits = y_pred[i, :x_lengths[i], :]
                true_indices = y[i, :y_lengths[i]]
                
                # Greedy decoding
                pred_indices = torch.argmax(pred_logits, dim=-1)
                pred_text = self.tokenizer.decode(pred_indices, collapse_repeats=True)
                true_text = self.tokenizer.decode(true_indices, collapse_repeats=False)
                
                all_pred_texts.append(pred_text)
                all_true_texts.append(true_text)
        
        avg_loss = val_loss / len(val_loader.dataset)
        
        # Compute Phoneme Error Rate (using WER metric)
        per = jiwer.wer(all_true_texts, all_pred_texts)
        
        return avg_loss, per
    
    def save_checkpoint(self, epoch, checkpoint_dir, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_per': self.best_per,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
            print(f"✅ Saved best model (PER: {self.best_per:.4f})")
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir):
        """Main training loop."""
        print("="*50)
        print("Starting CTC Training")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*50)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, per = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['per'].append(per)
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  PER: {per:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for best model
            is_best = per < self.best_per
            if is_best:
                self.best_per = per
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, checkpoint_dir, is_best=is_best)
            
            print("-"*50)
        
        print("\nTraining Complete!")
        print(f"Best PER: {self.best_per:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")


def create_model(model_type, config):
    """Create model based on type."""
    model_config = config.get('model', {})
    
    if model_type in ['RNN', 'LSTM', 'GRU']:
        return RecurrentModel(
            model_type=model_type,
            data_input_size=model_config.get('data_input_size', 512),
            adapter_output_size=model_config.get('adapter_output_size', 256),
            hidden_size=model_config.get('hidden_size', 512),
            output_size=model_config.get('output_size', 41),
            num_layers=model_config.get('num_layers', 1),
            bidirectional=model_config.get('bidirectional', False),
            dropout=model_config.get('dropout', 0.1)
        )
    elif model_type == 'TRANSFORMER':
        return TransformerEncModel(
            data_input_size=model_config.get('data_input_size', 512),
            adapter_output_size=model_config.get('adapter_output_size', 256),
            n_head=model_config.get('n_head', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 2048),
            output_size=model_config.get('output_size', 41),
            dropout=model_config.get('dropout', 0.1)
        )
    elif model_type == 'CONFORMER':
        return ConformerModel(
            data_input_size=model_config.get('data_input_size', 512),
            adapter_output_size=model_config.get('adapter_output_size', 256),
            n_head=model_config.get('n_head', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 2048),
            output_size=model_config.get('output_size', 41),
            dropout=model_config.get('dropout', 0.1),
            conv_kernel_size=model_config.get('conv_kernel_size', 31)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CTC model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--model-type', type=str, required=True, 
                       choices=['RNN', 'LSTM', 'GRU', 'TRANSFORMER', 'CONFORMER'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint (optional)')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, _ = load_datasets(
        args.data_dir, 
        use_augmentation=config.get('use_augmentation', True)
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    model = create_model(args.model_type, config)
    
    # Load pretrained checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Checkpoint loaded.")
    
    # Create trainer
    trainer = CTCTrainer(model, config, device=device)
    
    # Train
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.get('num_epochs', 50),
        checkpoint_dir=checkpoint_dir
    )


if __name__ == '__main__':
    main()

