# -*- coding: utf-8 -*-
"""
Conditional LSTM model for genre-aware poetry generation.

A genre-conditional 2-layer unidirectional LSTM language model.
The genre control token is prepended to the input sequence so the model
learns to generate poetry in the specified form.

Input:  [<genre>, SOP, char_1, char_2, ..., char_n, EOP]
Output: next-character probability distribution at each position.

Genre tokens:
    <5JUE> — 五言绝句 (4 lines × 5 characters)
    <7JUE> — 七言绝句 (4 lines × 7 characters)
    <5LV>  — 五言律诗 (8 lines × 5 characters)
    <7LV>  — 七言律诗 (8 lines × 7 characters)

Architecture:
    [<genre>, SOP, chars..., EOP]
        ↓
    Embedding (vocab_size → embedding_dim)
        ↓
    Embedding Dropout
        ↓
    2-layer Unidirectional LSTM (embedding_dim → hidden_dim)
        ↓
    LayerNorm
        ↓
    Output Dropout
        ↓
    Linear (hidden_dim → vocab_size)
        ↓
    Next character logits

This is the minimum viable conditional model. If the causal attention
variant is difficult to debug, this model serves as a reliable fallback.
"""

import torch
import torch.nn as nn


class ConditionalLSTM(nn.Module):
    """Genre-conditional 2-layer unidirectional LSTM language model.

    Identical in architecture to BaselineLSTM, but designed to receive
    a genre control token prepended to the input sequence. The model
    learns to condition its generation on the genre token through the
    standard embedding + LSTM pathway.

    See Also
    --------
    BaselineLSTM : The same architecture without genre conditioning.
    ConditionalAttnLSTM : Adds causal self-attention on top.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        embedding_dropout: float = 0.20,
        lstm_dropout: float = 0.30,
        output_dropout: float = 0.20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(embedding_dropout)

        # 2-layer unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output
        self.out_dropout = nn.Dropout(output_dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden states with zeros."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)

    def forward(self, input: torch.Tensor, hidden=None):
        """Forward pass with genre conditioning.

        Parameters
        ----------
        input : torch.Tensor
            Input token indices with genre token prepended,
            shape ``(batch_size, seq_len)``.
            Expected format: [<genre>, SOP, char_1, ..., EOP]
        hidden : tuple of (h_0, c_0) or None
            Initial LSTM hidden state. If None, initialized to zeros.

        Returns
        -------
        output : torch.Tensor
            Logits for each position, shape ``(batch_size, seq_len, vocab_size)``.
        hidden : tuple of (h_n, c_n)
            Final LSTM hidden state, each of shape
            ``(num_layers, batch_size, hidden_dim)``.
        """
        batch_size, seq_len = input.size()

        if hidden is None:
            hidden = self._init_hidden(batch_size, input.device)

        # Embedding + dropout
        embeds = self.embedding(input)                      # (B, S, E)
        embeds = self.emb_dropout(embeds)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embeds, hidden)    # (B, S, H)

        # LayerNorm
        normed = self.layer_norm(lstm_out)                  # (B, S, H)

        # Output dropout + linear projection
        dropped = self.out_dropout(normed)
        output = self.linear(dropped)                       # (B, S, V)

        return output, (h_n, c_n)
