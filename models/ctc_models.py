"""
CTC-based models for phoneme sequence prediction.

These models predict phoneme probabilities at each timestep,
which are then decoded using CTC and a language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""
    
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


class RecurrentModel(nn.Module):
    """
    Recurrent model (RNN/LSTM/GRU) for phoneme prediction.
    
    Architecture:
    - Input adapter: 512 -> adapter_output_size
    - RNN/LSTM/GRU layers
    - Output projection: hidden_size -> num_phonemes
    """
    
    def __init__(
        self, 
        model_type,  # 'RNN', 'LSTM', or 'GRU'
        data_input_size=512,
        adapter_output_size=256,
        hidden_size=512,
        output_size=41,  # num_phonemes + blank
        num_layers=1,
        bidirectional=False,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # Input adapter
        self.adapter_layer = nn.Linear(data_input_size, adapter_output_size)
        
        # RNN layer
        rnn_args = {
            'input_size': adapter_output_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_first': True,
            'bidirectional': bidirectional,
            'dropout': dropout if num_layers > 1 else 0
        }
        
        if model_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_args)
        elif model_type == "GRU":
            self.rnn = nn.GRU(**rnn_args)
        elif model_type == "RNN":
            self.rnn = nn.RNN(**rnn_args)
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'RNN', 'LSTM', or 'GRU'")

        # Output projection
        fc_in_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in_features, output_size)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, T, 512) neural features
            
        Returns:
            (B, T, output_size) log probabilities
        """
        x = self.adapter_layer(x)  # (B, T, adapter_output_size)
        out, _ = self.rnn(x)       # (B, T, hidden_size or hidden_size*2)
        out = self.fc(out)         # (B, T, output_size)
        return F.log_softmax(out, dim=2)


class TransformerEncModel(nn.Module):
    """
    Transformer encoder model for phoneme prediction.
    
    Architecture:
    - Input adapter: 512 -> d_model
    - Transformer encoder layers
    - Output projection: d_model -> num_phonemes
    """
    
    def __init__(
        self,
        data_input_size=512,
        adapter_output_size=256,
        n_head=8,
        num_layers=6,
        dim_feedforward=2048,
        output_size=41,  # num_phonemes + blank
        dropout=0.1
    ):
        super().__init__()
        
        # Input adapter
        self.adapter_layer = nn.Linear(data_input_size, adapter_output_size)
        self.d_model = adapter_output_size
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(self.d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.fc = nn.Linear(self.d_model, output_size)

    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: (B, T, 512) neural features
            mask: Optional padding mask
            
        Returns:
            (B, T, output_size) log probabilities
        """
        x = self.adapter_layer(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        out = self.transformer_encoder(x, src_key_padding_mask=mask)  # (B, T, d_model)
        out = self.fc(out)  # (B, T, output_size)
        return F.log_softmax(out, dim=2)


class ConformerModel(nn.Module):
    """
    Conformer-based model (combines CNN and Transformer).
    
    Better for speech-like sequential data with local and global patterns.
    """
    
    def __init__(
        self,
        data_input_size=512,
        adapter_output_size=256,
        n_head=8,
        num_layers=6,
        dim_feedforward=2048,
        output_size=41,
        dropout=0.1,
        conv_kernel_size=31
    ):
        super().__init__()
        
        self.adapter_layer = nn.Linear(data_input_size, adapter_output_size)
        self.d_model = adapter_output_size
        self.pos_enc = PositionalEncoding(self.d_model, dropout)
        
        # Conformer blocks (simplified - full Conformer has more components)
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                d_model=self.d_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size
            ) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(self.d_model, output_size)
    
    def forward(self, x, mask=None):
        x = self.adapter_layer(x)
        x = self.pos_enc(x)
        
        for layer in self.conformer_layers:
            x = layer(x, mask)
        
        out = self.fc(x)
        return F.log_softmax(out, dim=2)


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN -> Multi-head Self-Attention -> Convolution -> FFN."""
    
    def __init__(self, d_model, n_head, dim_feedforward, dropout, conv_kernel_size=31):
        super().__init__()
        
        # Feed-forward module 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.mhsa = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                d_model, d_model, 
                kernel_size=conv_kernel_size, 
                padding=(conv_kernel_size - 1) // 2,
                groups=d_model  # Depthwise convolution
            ),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.conv_norm = nn.LayerNorm(d_model)
        
        # Feed-forward module 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # FFN 1 (half-step residual)
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-head self-attention
        attn_out, _ = self.mhsa(x, x, x, key_padding_mask=mask)
        x = x + self.mhsa_dropout(attn_out)
        x = self.mhsa_norm(x)
        
        # Convolution
        x_conv = x.transpose(1, 2)  # (B, d_model, T)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_model)
        x = x + x_conv
        x = self.conv_norm(x)
        
        # FFN 2 (half-step residual)
        x = x + 0.5 * self.ffn2(x)
        
        return x

