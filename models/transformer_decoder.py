import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomDecoderLayer, self).__init__()

        # Self-attention layer with causal masking
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Feedforward network
        self.FFN = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model))

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, causal_mask=None, is_causal=False):
        # Self-attention with causal mask
        attn_output, _ = self.self_attn(x, x, x, attn_mask=causal_mask, is_causal=is_causal)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward network
        ff_output = self.FFN(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

class QuantizedSequenceTransformer(nn.Module):
    def __init__(self, codebook_size, dim_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1,
                 max_seq_length=512, include_cls_token=False):
        super(QuantizedSequenceTransformer, self).__init__()

        self.d_model = dim_model
        self.codebook_size = codebook_size
        self.include_cls_token = include_cls_token

        self.vocab_size = codebook_size

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, dim_model)

        # Mask for the transformer decoder - NOT NEEDED
        self.register_buffer('mask', self.generate_square_subsequent_mask(max_seq_length))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(dim_model, dropout, max_seq_length)

        # Transformer decoder
        self.transformer_decoder = nn.ModuleList([
            CustomDecoderLayer(dim_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(dim_model, self.vocab_size)


    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask

    def forward(self, src, src_mask=None, labeled=False):
        # src shape: (batch_size, seq_len) -> (seq_len, batch_size)
        src = src.transpose(0, 1)

        if src_mask is None:
            src_mask = self.mask[:src.size(0), :src.size(0)]

        # Embed the input tokens
        src = self.embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through the transformer encoder
        for i, layer in enumerate(self.transformer_decoder):
            src = layer(src, src_mask, is_causal=True)

        output = src

        # Project to vocabulary size
        if not labeled:
            output = self.output_layer(output)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def train_step(model, optimizer, data, criterion):
    optimizer.zero_grad()

    device = data.device

    # data is a tensor of shape (batch_size, seq_len)
    input_seq = data[:, :-1]
    target_seq = data[:, 1:]

    # Forward pass
    output = model(input_seq)

    # Compute loss
    output = output.view(-1, model.module.vocab_size)
    target = target_seq.contiguous().view(-1)

    loss = criterion(output, target)

    return loss
