import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class QuantizedSequenceTransformer(nn.Module):
    def __init__(self, codebook_size, dim_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1,
                 max_seq_length=512, include_cls_token=False, use_quantized=False):
        super(QuantizedSequenceTransformer, self).__init__()

        self.d_model = dim_model
        self.codebook_size = codebook_size
        self.include_cls_token = include_cls_token
        self.use_quantized = use_quantized

        if include_cls_token:
            self.vocab_size = codebook_size + 1  # Include CLS token
            self.cls_token_id = codebook_size  # CLS token index
        else:
            self.vocab_size = codebook_size

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, dim_model)

        # Mask for the transformer encoder
        self.register_buffer('mask', self.generate_square_subsequent_mask(max_seq_length))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(dim_model, dropout, max_seq_length)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(dim_model, nhead, dim_feedforward, dropout, norm_first=True,
                                                 activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Output layer
        self.output_layer = nn.Linear(dim_model, self.vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        if self.include_cls_token:
            # Allow CLS token (position -1) to attend to all positions
            mask[-1, :] = 0
        return mask

    def forward(self, src, src_mask=None, labeled=False):
        # src shape: (seq_len, batch_size)
        if src_mask is None:
            src_mask = self.mask[:src.size(0), :src.size(0)]
            is_causal = True
        else:
            is_causal = False

        # Embed the input tokens
        if not self.use_quantized:
            src = self.embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through the transformer encoder
        embeddings = self.transformer_encoder(src, src_mask, is_causal=is_causal)

        # Project to vocabulary size
        output = self.output_layer(embeddings)

        return output, embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


def train_step(model, optimizer, data, criterion, quantized, use_quantized=False, epoch=200):
    n_pred = 2

    device = data.device

    data = data.transpose(0, 1).contiguous()  # Shape: (seq_len, batch_size)

    # Prepare input and target sequences
    if model.module.include_cls_token:
        cls_tokens = torch.full((1, data.size(1)), model.module.cls_token_id, dtype=data.dtype, device=device)
        input_seq = torch.cat([cls_tokens, data[:-1, :]], dim=0)
        target_seq = data
    else:
        input_seq = data[:-n_pred, :]
        target_seq = data[n_pred:, :]

    if use_quantized:
        # quantized starts as (B, seq_len, dim_quantized) and needs to be (seq_len, B, dim_quantized)
        quantized = quantized.transpose(0, 1).contiguous()
        input_quantized = quantized[:-n_pred, :, :]
        output, embeddings = model(input_quantized)

    # Forward pass
    else:
        output, embeddings = model(input_seq)

    # Compute loss
    output = output.view(-1, model.module.vocab_size)
    target = target_seq.contiguous().view(-1)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, target)
    if epoch > 30:
        top_k_acc = top_k_accuracy(output, target, k=[1, 3, 5])

        # check how many times the sequence is the same
        top_1_baseline_acc = 0
        for i in range(len(input_seq[0])):
            top_1_baseline_acc += sum(1 for a, b in zip(list(input_seq[:, i]), list(target_seq[:, i])) if a == b) / len(input_seq)

        top_1_baseline_acc /= len(input_seq[0])
    else:
        top_k_acc = 0
        top_1_baseline_acc = 0

    return loss, embeddings, top_k_acc, top_1_baseline_acc

def eval_step(model, optimizer, data, criterion, quantized, use_quantized=False):
    data = data.transpose(0, 1).contiguous()  # Shape: (seq_len, batch_size)


    input_seq = data[:-1, :]
    target_seq = data[1:, :]

    if use_quantized:
        # quantized starts as (B, seq_len, dim_quantized) and needs to be (seq_len, B, dim_quantized)
        quantized = quantized.transpose(0, 1).contiguous()
        input_quantized = quantized[:-1, :, :]
        with torch.no_grad():
            output, embeddings = model(input_quantized)

    # Forward pass
    else:
        with torch.no_grad():
            output, embeddings = model(input_seq)

    # Compute loss
    output = output.view(-1, model.vocab_size)
    target = target_seq.contiguous().view(-1)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, target)

    top_k_acc = top_k_accuracy(output, target, k=[1, 3, 5])

    # check how many times the sequence is the same
    top_1_baseline_acc = 0
    for i in range(len(input_seq[0])):
        top_1_baseline_acc += sum(1 for a, b in zip(list(input_seq[:, i]), list(target_seq[:, i])) if a == b) / len(input_seq)
        # accuracy is out of 100
        top_1_baseline_acc *= 100

    top_1_baseline_acc /= len(input_seq[0])

    return loss, embeddings, top_k_acc, top_1_baseline_acc



def train_step_contrastive(model, optimizer, data, ids, criterion, quantized, use_quantized=False):
    """
    Train the model for one step on the given data.
    Use a contrastive loss for the pooled quantized embeddings.
    Posisitve pairs are sequences from the same identity, negative pairs are from different identities.
    :param model:
    :param optimizer:
    :param data:
    :param criterion:
    :param quantized:
    :param use_quantized:
    :return:
    """

    device = data.device

    data = data.transpose(0, 1).contiguous()  # Shape: (seq_len, batch_size)

    # Prepare input and target sequences
    if model.module.include_cls_token:
        cls_tokens = torch.full((1, data.size(1)), model.module.cls_token_id, dtype=data.dtype, device=device)
        input_seq = torch.cat([cls_tokens, data[:-1, :]], dim=0)
        target_seq = data
    else:
        input_seq = data[:-1, :]
        target_seq = data[1:, :]

    if use_quantized:
        # quantized starts as (B, seq_len, dim_quantized) and needs to be (seq_len, B, dim_quantized)
        quantized = quantized.transpose(0, 1).contiguous()
        input_quantized = quantized[:-1, :, :]
        output, embeddings = model(input_quantized)

    # Forward pass
    else:
        output, embeddings = model(input_seq)

    # Compute loss
    output = output.view(-1, model.module.vocab_size)
    target = target_seq.contiguous().view(-1)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, target)

    top_k_acc = top_k_accuracy(output, target, k=[5])

    # move the input sequence by 1 to the left and try that accuracy
    baseline_logits = data[1:, :]
    baseline_acc = top_k_accuracy(baseline_logits, input_seq, k=[1])

    return loss, embeddings, top_k_acc

def get_baseline_accuracy(seq):
    input_seq = seq[:-1, :]  # [[1,2], [1,4]]
    target_seq = seq[1:, :]  # [[1,4], [5,5]]

    top_1_baseline_acc = 0
    for i in range(input_seq.shape[1]):  # iterate through columns
        matches = sum(1 for a, b in zip(input_seq[:, i], target_seq[:, i]) if a == b)
        accuracy = (matches / len(input_seq)) * 100
        top_1_baseline_acc += accuracy

    # Average accuracy across all columns
    top_1_baseline_acc /= input_seq.shape[1]

    return top_1_baseline_acc


def top_k_accuracy(logits, targets, k=None):
    """
    Calculate the top-k accuracy for next-token prediction.

    Parameters:
        logits (torch.Tensor): The model's output logits of shape (batch_size, num_classes).
        targets (torch.Tensor): The ground truth labels of shape (batch_size).
        k list: The 'k' value for top-k accuracy.

    Returns:
        float: The top-k accuracy as a percentage.
    """
    # Get the top-k predictions along the last dimension
    if k is None:
        k = [5]
    top_k_accuracy_dict = {a: 0 for a in k}
    for a in k:
        _, top_k_predictions = torch.topk(logits, a, dim=-1)

        # Check if the true targets are in the top-k predictions
        correct_predictions = top_k_predictions.eq(targets.view(-1, 1).expand_as(top_k_predictions))

        # Calculate the top-k accuracy by averaging the correct predictions
        top_k_accuracy_dict[a] = correct_predictions.any(dim=1).float().mean().item() * 100

    return top_k_accuracy_dict  # Return as a percentage

