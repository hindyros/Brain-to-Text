"""
End-to-end models that directly predict text from neural features.

These models bypass phoneme intermediate representation and learn
direct neural -> text mappings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class NeuralEncoder(nn.Module):
    """
    Encodes neural signals (512 channels, T timesteps) → latent z_neural.
    
    Uses convolutional downsampling + Transformer encoder.
    """
    
    def __init__(self, n_channels=512, d_model=512, n_layers=8, n_heads=8, dropout=0.1):
        super().__init__()
        
        # Input projection with downsampling
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=5000)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, 512) neural features
            mask: Optional padding mask
            
        Returns:
            (B, L, d_model) neural latent (L = T//2 due to downsampling)
        """
        B, T, C = x.shape
        
        # Conv projection + downsample 2x
        x = x.transpose(1, 2)  # (B, 512, T)
        x = self.input_proj(x)  # (B, d_model, T//2)
        x = x.transpose(1, 2)  # (B, T//2, d_model)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Transformer encoding
        z_neural = self.transformer(x, src_key_padding_mask=mask)
        z_neural = self.layer_norm(z_neural)
        
        return z_neural


class TextDecoder(nn.Module):
    """
    Decodes from neural latent directly to text tokens.
    
    Uses Transformer decoder with autoregressive generation.
    """
    
    def __init__(self, d_model=512, vocab_size=256, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, z_neural, target_tokens=None, max_len=200):
        """
        Args:
            z_neural: (B, L, d_model) encoded neural features
            target_tokens: (B, T) ground truth tokens (teacher forcing during training)
            max_len: Maximum generation length (for inference)
            
        Returns:
            (B, T, vocab_size) logits
        """
        B = z_neural.size(0)
        
        if target_tokens is not None:
            # Teacher forcing (training)
            tgt_emb = self.token_embedding(target_tokens)
            tgt_emb = self.pos_enc(tgt_emb)
            
            # Create causal mask
            T = target_tokens.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(z_neural.device)
            
            # Decode
            out = self.decoder(tgt_emb, z_neural, tgt_mask=causal_mask)
            logits = self.output_proj(out)
            
            return logits
        else:
            # Autoregressive generation (inference)
            return self.generate(z_neural, max_len)
    
    @torch.no_grad()
    def generate(self, z_neural, max_len=200, temperature=1.0):
        """
        Autoregressive generation.
        
        Args:
            z_neural: (B, L, d_model) encoded neural features
            max_len: Maximum sequence length
            temperature: Sampling temperature
            
        Returns:
            (B, T) generated token IDs
        """
        B = z_neural.size(0)
        device = z_neural.device
        
        # Start with BOS token (assume vocab[1] = BOS)
        generated = torch.ones(B, 1, dtype=torch.long, device=device)  # BOS = 1
        
        for _ in range(max_len):
            # Embed current sequence
            tgt_emb = self.token_embedding(generated)
            tgt_emb = self.pos_enc(tgt_emb)
            
            # Decode
            T = generated.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            out = self.decoder(tgt_emb, z_neural, tgt_mask=causal_mask)
            
            # Get last token logits
            logits = self.output_proj(out[:, -1, :])  # (B, vocab_size)
            
            # Sample next token
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS (assume vocab[2] = EOS)
            if (next_token == 2).all():
                break
        
        return generated


class EndToEndModel(nn.Module):
    """
    Simple end-to-end model: Neural Encoder -> Text Decoder.
    
    Directly maps neural features to text without phoneme intermediate.
    """
    
    def __init__(
        self,
        n_channels=512,
        d_model=512,
        vocab_size=256,
        n_encoder_layers=8,
        n_decoder_layers=6,
        n_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.neural_encoder = NeuralEncoder(
            n_channels=n_channels,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.text_decoder = TextDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
    
    def forward(self, neural_features, target_tokens=None):
        """
        Args:
            neural_features: (B, T, 512) neural features
            target_tokens: (B, T_text) text tokens (for teacher forcing)
            
        Returns:
            (B, T_text, vocab_size) logits
        """
        # Encode neural
        z_neural = self.neural_encoder(neural_features)
        
        # Decode to text
        logits = self.text_decoder(z_neural, target_tokens)
        
        return logits
    
    @torch.no_grad()
    def inference(self, neural_features, max_len=200, temperature=1.0):
        """
        Pure inference: neural -> text.
        
        Args:
            neural_features: (B, T, 512) or (T, 512) neural features
            max_len: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            (B, T) token IDs or (T,) if single sample
        """
        if neural_features.dim() == 2:
            neural_features = neural_features.unsqueeze(0)
        
        # Encode
        z_neural = self.neural_encoder(neural_features)
        
        # Decode
        tokens = self.text_decoder.generate(z_neural, max_len=max_len, temperature=temperature)
        
        if tokens.size(0) == 1:
            return tokens[0]
        return tokens

