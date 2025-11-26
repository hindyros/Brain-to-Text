"""
Training script for end-to-end models (direct neural -> text).

Supports:
- End-to-end transformer models
- Character-level tokenization
- Teacher forcing during training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from pathlib import Path

from data.dataset import load_datasets, text_collate_fn
from data.tokenizer import CharTokenizer
from models.e2e_models import EndToEndModel
import jiwer


class EndToEndTrainer:
    """Trainer for end-to-end text generation models."""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Tokenizer
        self.tokenizer = CharTokenizer()
        
        # Loss function (cross-entropy for text generation)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 5e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.98),
            eps=1e-8
        )
        
        # Scheduler
        total_steps = config.get('num_epochs', 50) * 1000  # Approximate
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('learning_rate', 5e-4),
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos'
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_wer = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'wer': []}
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and device == 'cuda' else None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            neural = batch['neural'].to(self.device)
            tokens = batch['tokens'].to(self.device) if batch['tokens'] is not None else None
            
            if tokens is None:
                continue  # Skip batches without labels
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(neural, target_tokens=tokens)  # (B, T, vocab_size)
                
                # Shift for next-token prediction
                logits_flat = logits[:, :-1].reshape(-1, logits.size(-1))
                target_flat = tokens[:, 1:].reshape(-1)
                
                loss = self.criterion(logits_flat, target_flat)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('grad_clip', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip', 1.0)
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            running_loss += loss.item() * neural.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return running_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validate and compute word error rate."""
        self.model.eval()
        val_loss = 0.0
        all_pred_texts = []
        all_true_texts = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            neural = batch['neural'].to(self.device)
            tokens = batch['tokens'].to(self.device) if batch['tokens'] is not None else None
            
            if tokens is None:
                continue
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(neural, target_tokens=tokens)
                
                # Loss
                logits_flat = logits[:, :-1].reshape(-1, logits.size(-1))
                target_flat = tokens[:, 1:].reshape(-1)
                loss = self.criterion(logits_flat, target_flat)
                val_loss += loss.item() * neural.size(0)
            
            # Decode predictions
            pred_tokens = torch.argmax(logits, dim=-1)
            
            for i in range(neural.size(0)):
                # Get predicted tokens (skip BOS)
                pred_seq = pred_tokens[i, 1:]  # Skip BOS
                true_seq = tokens[i, 1:]  # Skip BOS
                
                # Decode to text
                pred_text = self.tokenizer.decode(pred_seq)
                true_text = self.tokenizer.decode(true_seq)
                
                # Normalize for WER
                pred_text = self._normalize_text(pred_text)
                true_text = self._normalize_text(true_text)
                
                all_pred_texts.append(pred_text)
                all_true_texts.append(true_text)
        
        avg_loss = val_loss / len(val_loader.dataset)
        
        # Compute Word Error Rate
        wer = jiwer.wer(all_true_texts, all_pred_texts)
        
        return avg_loss, wer
    
    def _normalize_text(self, text):
        """Normalize text for evaluation (remove punctuation, lowercase)."""
        import re
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
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
            'best_wer': self.best_wer,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
            print(f"✅ Saved best model (WER: {self.best_wer:.4f})")
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir):
        """Main training loop."""
        print("="*50)
        print("Starting End-to-End Training")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*50)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, wer = self.validate(val_loader, epoch)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['wer'].append(wer)
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  WER: {wer:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for best model
            is_best = wer < self.best_wer
            if is_best:
                self.best_wer = wer
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, checkpoint_dir, is_best=is_best)
            
            print("-"*50)
        
        print("\nTraining Complete!")
        print(f"Best WER: {self.best_wer:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train end-to-end model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
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
    
    # Load datasets with text labels
    print("Loading datasets...")
    train_dataset, val_dataset, _ = load_datasets(
        args.data_dir,
        use_augmentation=config.get('use_augmentation', True),
        return_text=True  # Enable text label loading
    )
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
    # Create dataloaders with tokenizer for collate function
    from functools import partial
    collate_fn = partial(text_collate_fn, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    model_config = config.get('model', {})
    tokenizer = CharTokenizer()
    
    model = EndToEndModel(
        n_channels=model_config.get('n_channels', 512),
        d_model=model_config.get('d_model', 512),
        vocab_size=tokenizer.vocab_size,
        n_encoder_layers=model_config.get('n_encoder_layers', 8),
        n_decoder_layers=model_config.get('n_decoder_layers', 6),
        n_heads=model_config.get('n_heads', 8),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Load pretrained checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Checkpoint loaded.")
    
    # Create trainer
    trainer = EndToEndTrainer(model, config, device=device)
    
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

