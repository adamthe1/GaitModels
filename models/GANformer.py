import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class GenerativeFormer(nn.Module):
    def __init__(self, codebook_size, num_biomarkers=0, biomarker_names=None, dim_model=512, nhead=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.1, max_seq_length=512, use_quantized=False,
                 bin_sizes=None, max_bins=None, smallest_value=None, biomarker_mask=True, predict_biomarkers=False,
                 biomarker_dropout=0.1, biomarker_hidden_dim=256, ignore_index=-100):
        super(GenerativeFormer, self).__init__()

        self.dim_model = dim_model
        self.codebook_size = codebook_size
        self.num_biomarkers = num_biomarkers
        self.use_quantized = use_quantized
        self.predict_biomarkers = predict_biomarkers
        self.ignore_index = ignore_index
        if bin_sizes is None:
            self.bin_sizes = [1.0] * num_biomarkers
        else:
            self.bin_sizes = bin_sizes
        if max_bins is None:
            self.max_bins = [10] * num_biomarkers
        else:
            self.max_bins = max_bins
        if smallest_value is None:
            self.smallest_value = [0.0] * num_biomarkers
        else:
            self.smallest_value = smallest_value

        self.vocab_size = codebook_size

        # Embedding layers
        if not use_quantized:
            self.token_embedding = nn.Embedding(self.vocab_size, dim_model)
        # Total number of unique biomarker tokens + 1 start token
        self.biomarker_vocab = int(np.ceil(sum(self.max_bins))) + 1

        if self.predict_biomarkers:
            self.biomarker_mask_tokens = list(range(self.biomarker_vocab, self.num_biomarkers + self.biomarker_vocab))
            self.biomarker_vocab += self.num_biomarkers  # Add mask token for prediction

        # start token is the last token in the biomarker vocab
        self.start_token = self.biomarker_vocab - 1

        self.biomarker_embedding = nn.Embedding(self.biomarker_vocab, dim_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(dim_model, dropout, max_seq_length)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(dim_model, nhead, dim_feedforward, dropout, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Output layer for token prediction
        self.token_output_layer = nn.Linear(dim_model, self.vocab_size)

        self.biomarker_dropout = nn.Dropout(biomarker_dropout)
        self.biomarker_output_layer = nn.Linear(dim_model, self.biomarker_vocab - 1)

        """ test it later
        self.biomarker_hidden_layer = nn.Linear(dim_model, biomarker_hidden_dim)
        self.predict_biomarkers_layer = nn.Linear(biomarker_hidden_dim, self.biomarker_vocab - 1)
        self.dropout = nn.Dropout(biomarker_dropout)
        self.activation = nn.GELU(approximate='tanh')
        self.biomarker_output_layer = nn.Sequential(self.biomarker_hidden_layer,
                                                    #nn.LayerNorm(biomarker_hidden_dim),
                                                    self.activation,
                                                    self.dropout,
                                                    self.predict_biomarkers_layer)
        """

        # Output layer maps to biomarker vocab sizes
        """ test it later
        self.biomarker_output_layers = nn.ModuleDict({
            f'biomarker_{i}': nn.Linear(biomarker_hidden_dim, vocab_size - 1)  # Exclude start token
            for i, vocab_size in enumerate(max_bins)
        })
        """

        if biomarker_mask:
            self.attention_mask = self.generate_biomarker_seq_mask(num_biomarkers + 1, max_seq_length)
        else:
            self.attention_mask = self.generate_square_subsequent_mask(max_seq_length)

        self._init_weights()

    def _init_weights(self):
        # Xavier Uniform Initialization for Embedding Layer
        nn.init.xavier_uniform_(self.biomarker_embedding.weight)

        # Initialize Token Output Layer
        nn.init.zeros_(self.token_output_layer.bias)  # Biases set to zero
        nn.init.xavier_uniform_(self.token_output_layer.weight)  # Xavier Uniform for weights

        # Initialize Biomarker Output Layer
        #nn.init.zeros_(self.biomarker_hidden_layer.bias)  # Biases set to zero
        #nn.init.xavier_uniform_(self.biomarker_hidden_layer.weight)  # Xavier Uniform for weights

        nn.init.zeros_(self.biomarker_output_layer.bias)  # Biases set to zero
        nn.init.xavier_uniform_(self.biomarker_output_layer.weight)  # Xavier Uniform for weights

        # In _init_weights method
        for layer in self.transformer_encoder.layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

    def tokenize_biomarkers(self, biomarkers):
        """
        Tokenizes the biomarkers by binning with specified bin sizes and predefined max bins,
        assigning unique tokens to each feature.

        Args:
            biomarkers: Tensor of shape (num_biomarkers, batch_size).
            bin_sizes: List of bin sizes for each feature.
            max_bins: List of maximum number of bins for each feature.

        Returns:
            Tensor of shape (num_biomarkers, batch_size) with tokenized biomarkers.
        """
        assert biomarkers.size(0) == len(self.bin_sizes)
        assert biomarkers.size(0) == len(self.max_bins)
        assert biomarkers.size(0) == len(self.smallest_value)

        tokenized = torch.zeros_like(biomarkers, dtype=torch.long)
        offsets = []
        offset = 0

        for i in range(len(self.bin_sizes)):
            bin_size = self.bin_sizes[i]
            max_bin = self.max_bins[i]
            min_value = self.smallest_value[i]
            biomarkers[i, :] -= min_value
            biomarkers[i, :] = torch.clamp(biomarkers[i, :], min=0)  # Clip negative values to 0
            binned_feature = torch.floor(biomarkers[i, :] / bin_size).long()
            binned_feature = torch.clamp(binned_feature, min=0, max=max_bin - 1)  # max_bin not included
            tokenized[i, :] = binned_feature
            offsets.append(offset)
            offset += max_bin # Use predefined max_bin
            
        offsets.append(offset) # Add the last offset, len(offsets) is len(self.bin_sizes) + 1

        for i in range(len(self.bin_sizes)):
            tokenized[i, :] += offsets[i]

        return tokenized, offsets
    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    def generate_biomarker_seq_mask(self, biomarker_seq_len, src_seq_len):
        combined_seq_len = biomarker_seq_len + src_seq_len
        mask = torch.triu(torch.full((combined_seq_len, combined_seq_len), float('-inf')), diagonal=1)
        mask[:biomarker_seq_len, :biomarker_seq_len] = 0
        return mask

    def generate_prediction_mask(self, tokenized_biomarkers, offset=None):
        """
        Switch 15% of the tokens to mask tokens or random tokens for prediction.
        Can only use tokens that are in the same offset bin.

        Args:
            tokenized_biomarkers: tensor of shape (biomarker_size, batch_size)
            self.ignore_index: value to use for tokens that shouldn't be predicted
            offset: list of starting indices for each biomarker's token range

        Returns:
            masked_biomarkers: tensor with masked/random/unchanged tokens
            mlm_labels: tensor indicating which tokens should be predicted
        """

        biomarker_size, batch_size = tokenized_biomarkers.shape
        device = tokenized_biomarkers.device

        # Create a copy of the input tensor
        masked_biomarkers = tokenized_biomarkers.clone()
        # Initialize labels with self.ignore_index
        mlm_labels = torch.full_like(tokenized_biomarkers, self.ignore_index)

        # Calculate number of tokens to mask (15% of total)
        num_tokens = biomarker_size * batch_size
        num_to_mask = int(0.15 * num_tokens)

        # Create a mask for tokens to be predicted
        mask_indices = torch.randperm(num_tokens)[:num_to_mask]

        # Calculate how many tokens for each category (80% mask, 10% random, 10% unchanged)
        num_mask = int(0.8 * num_to_mask)
        num_random = int(0.1 * num_to_mask)
        # The rest will remain unchanged

        # Get biomarker indices and batch indices
        biomarker_idx = mask_indices // batch_size
        batch_idx = mask_indices % batch_size

        # Handle masked tokens (80%)
        mask_tokens = biomarker_idx[:num_mask]
        mask_samples = batch_idx[:num_mask]
        for b, s in zip(mask_tokens, mask_samples):
            masked_biomarkers[b, s] = self.biomarker_mask_tokens[b]
            mlm_labels[b, s] = tokenized_biomarkers[b, s]

        # Handle random tokens (10%)
        random_tokens = biomarker_idx[num_mask:num_mask + num_random]
        random_samples = batch_idx[num_mask:num_mask + num_random]

        for b, s in zip(random_tokens, random_samples):
            # Find the range for this biomarker
            start_offset = offset[b]
            end_offset = offset[b + 1]
            # Generate random token within the valid range
            random_token = torch.randint(start_offset, end_offset, (1,), device=device)
            masked_biomarkers[b, s] = random_token
            mlm_labels[b, s] = tokenized_biomarkers[b, s]

        # Handle unchanged tokens (remaining 10%)
        unchanged_tokens = biomarker_idx[num_mask + num_random:]
        unchanged_samples = batch_idx[num_mask + num_random:]
        for b, s in zip(unchanged_tokens, unchanged_samples):
            mlm_labels[b, s] = tokenized_biomarkers[b, s]

        return masked_biomarkers, mlm_labels
    
    def create_ignore_mask(self, ignore_mask, seq_len, batch_size):
        # create a starting tensor with shape (biomarker_and_seq_len, batch_size)
        mask_tensor = torch.tensor([list(range(seq_len)) for _ in range(batch_size)])
        mask_tensor = mask_tensor.transpose(0, 1)
        
        # mask start token
        # mask_tensor[self.num_biomarkers, :] = self.ignore_index # dont mask the start token because it also predicts a token
        
        # mask the biomarker tokens using ignore_mask
        if ignore_mask is not None:
            mask_tensor[:self.num_biomarkers, :] = ignore_mask
        
        return mask_tensor


    def forward(self, src, biomarkers, src_mask=None):
        """
        Args:
            src: Tensor of shape (seq_len, batch_size) containing token indices.
            biomarkers: Tensor of shape (batch_size, num_biomarkers) containing continuous biomarker values.
            gait_gen: Bool indicating the task.
                      True: Generate gait sequence from biomarkers.
                      False: Predict biomarkers from gait sequence.
            src_mask: Tensor for masking (optional).

        Returns:
            token_logits: Tensor of shape (output_seq_len, batch_size, vocab_size) or placeholder.
            biomarker_predictions: Tensor of shape (num_biomarkers, batch_size) or placeholder.
        """
        device = src.device
        batch_size = src.size(1)
        seq_len = src.size(0)

        if not self.use_quantized:
            src = self.token_embedding(src)

        tokenized_biomarkers, offsets = self.tokenize_biomarkers(biomarkers)
        if self.predict_biomarkers:
            pred_biomarkers_masked, bio_ignore_mask = self.generate_prediction_mask(tokenized_biomarkers, offset=offsets)
            # embed biomarkers after masking
            biomarker_embedding = self.biomarker_embedding(pred_biomarkers_masked)  # (biomarker_vocab, batch_size ,d_model)
        else:
            biomarker_embedding = self.biomarker_embedding(tokenized_biomarkers)
            bio_ignore_mask = None
            
        # Embed start token and concatenate with source tokens
        start_batch = torch.full((1, batch_size), self.start_token, dtype=torch.long, device=device)
        start_embedding = self.biomarker_embedding(start_batch)
        src = torch.cat([start_embedding, src], dim=0)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Concatenate biomarker embedding with source tokens
        biomarker_seq = torch.cat([biomarker_embedding, src], dim=0).to(device)
        
        ignore_mask_full = self.create_ignore_mask(bio_ignore_mask, biomarker_seq.size(0), biomarker_seq.size(1)).to(device)

        if src_mask is None:
           src_mask = self.attention_mask[:biomarker_seq.size(0), :biomarker_seq.size(0)].to(device)

        encoder_output = self.transformer_encoder(biomarker_seq, src_mask)

        # token logits size(0) is biomarker_seq_len + 1 start token + src_seq_len
        token_logits = self.token_output_layer(encoder_output[self.num_biomarkers:, :])  # include prediction for start token
        # take the biomarkers from the encoder output and aggregate the last token to each biomarker
        biomarker_aggregated = encoder_output[:self.num_biomarkers, :] + encoder_output[-1, :]

        # add dropout for generalization
        biomarker_aggregated = self.biomarker_dropout(biomarker_aggregated)
        biomarker_logits = self.biomarker_output_layer(biomarker_aggregated)

        # we predict on token logits[:-1] because we dont loss the last token
        return token_logits, biomarker_logits, encoder_output, tokenized_biomarkers, bio_ignore_mask.to(device)

    def generate(self, biomarkers, vq_model, max_length=100, stop_token=None, temperature=1.0, vq_model_device=None):
        """
        Generates a gait sequence conditioned on the given biomarkers.

        Args:
            biomarkers (torch.Tensor): Tensor of shape (num_biomarkers, batch_size) containing continuous biomarker values.
            max_length (int): Maximum length of the generated gait sequence.
            device (torch.device, optional): Device to perform computation on. If None, uses the model's device.
            stop_token (int, optional): Token ID that signals the end of generation. If None, generation stops at max_length.
            temperature (float): Temperature parameter for scaling logits. Default is 1.0 (no scaling).

        Returns:
            List[List[int]]: Generated sequences for each instance in the batch.
        """

        batch_size = biomarkers.size(1)
        is_finished = [False] * batch_size
        if vq_model_device is None:
            vq_model_device = biomarkers.device

        # Initial step: prepare biomarker embeddings and <gait> token
        with torch.no_grad():
            start_token = torch.full((1, batch_size), self.start_token, dtype=torch.long, device=biomarkers.device)
            # embed and positional encode start token
            generated_emb = self.biomarker_embedding(start_token)
            # Project biomarkers to d_model
            tokenized_biomarkers, offsets = self.tokenize_biomarkers(biomarkers)
            biomarker_embedding = self.biomarker_embedding(tokenized_biomarkers)


        logits = torch.zeros((1, batch_size, self.vocab_size), dtype=torch.long, device=biomarkers.device)
        generated_sequences = torch.zeros((batch_size, 1, 32, 3), dtype=torch.long, device=biomarkers.device)
        for _ in range(max_length):
            with torch.no_grad():

                # positional encode only the tokens from start token to current token
                positional_encoded = self.pos_encoder(generated_emb)
                # Concatenate biomarker embedding with source tokens
                combined_seq = torch.cat([biomarker_embedding, positional_encoded], dim=0)

                # Pass through the transformer encoder dont need a mask for generation of next token
                transformer_output = self.transformer_encoder(combined_seq)

                # Get the last token's hidden state
                last_hidden = transformer_output[-1, :, :]  # (batch_size, d_model)

                # Predict next token logits
                next_token_logits = self.token_output_layer(last_hidden)  # (batch_size, vocab_size)

                if temperature != 1.0:
                    # Apply temperature scaling
                    next_token_logits = next_token_logits / temperature

                logits = torch.cat([logits, next_token_logits.unsqueeze(0)], dim=0)
                # Apply softmax to get probabilities
                next_token_probs = torch.softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)

                # Sample the next tokens
                next_tokens = torch.multinomial(next_token_probs, num_samples=1).to(vq_model_device) # (batch_size, 1)

                # get the quantized from the codebook
                frames, quantized = vq_model.get_frames(next_tokens, permute=False) # frames is (batch_size, 1, 32, 3) and quantized is (batch_size, F, dim_model)
                # Append the quantized to the generated embeddings
                frames = frames.to(biomarkers.device)
                quantized = quantized.to(biomarkers.device)

                quantized = quantized.permute(1, 0, 2)  # (1, B, d_model)
                generated_emb = torch.cat([generated_emb, quantized], dim=0)  # (current_length + 1, batch_size, d_model)

                # Append the generated tokens to the sequences
                generated_sequences = torch.cat([generated_sequences, frames], dim=1)

        return generated_sequences[:, 1:], logits[1:]

    def predict(self, src):
        """
        Predicts biomarkers from the given gait sequence.

        Args:
            src (torch.Tensor): Tensor of shape (seq_len, batch_size) containing token indices.

        Returns:
            torch.Tensor: Predicted biomarkers of shape (num_biomarkers, batch_size).
        """
        device = src.device
        batch_size = src.size(1)

        if not self.use_quantized:
            src = self.token_embedding(src)

        start_batch = torch.full((1, batch_size), self.start_token, dtype=torch.long, device=device)
        start_embedding = self.biomarker_embedding(start_batch)
        src = torch.cat([start_embedding, src], dim=0)

        # Add positional encoding
        src = self.pos_encoder(src)

        biomarker_masK_tensor = torch.tensor([self.biomarker_mask_tokens] * batch_size, dtype=torch.long, device=device) # (batch_size, num_biomarkers)
        biomarker_masK_tensor = biomarker_masK_tensor.transpose(0, 1)  # (num_biomarkers, batch_size)
        biomarker_mask_embedding = self.biomarker_embedding(biomarker_masK_tensor)

        # Concatenate biomarker embedding with source tokens
        biomarker_seq = torch.cat([biomarker_mask_embedding, src], dim=0)

        # create attention mask
        src_mask = self.attention_mask[:biomarker_seq.size(0), :biomarker_seq.size(0)].to(device)

        # Pass through the transformer encoder
        encoder_output = self.transformer_encoder(biomarker_seq, src_mask)

        biomarker_aggregated = encoder_output[:self.num_biomarkers, :] + encoder_output[-1, :]

        # Project to biomarker vocab size
        biomarker_logits = self.biomarker_output_layer(biomarker_aggregated)

        return biomarker_logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encodings added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
