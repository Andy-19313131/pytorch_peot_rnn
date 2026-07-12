# -*- coding: utf-8 -*-
"""
Baseline LSTM model for poetry generation.

A minimal 2-layer unidirectional LSTM language model.
Input:  [SOP, char_1, char_2, ..., char_n, EOP]
Output: next-character probability distribution at each position.

Architecture:
    Input tokens
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
"""

import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """Baseline 2-layer unidirectional LSTM language model.

    This is the simplest model in the hierarchy. It treats poetry generation
    as a standard next-token prediction task without any genre conditioning.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (including PAD, UNK, SOP, EOP).
    embedding_dim : int, default 256
        Dimension of token embeddings.
    hidden_dim : int, default 256
        Dimension of LSTM hidden states.
    num_layers : int, default 2
        Number of stacked LSTM layers.
    embedding_dropout : float, default 0.20
        Dropout rate applied after the embedding layer.
    lstm_dropout : float, default 0.30
        Dropout rate applied between LSTM layers.
    output_dropout : float, default 0.20
        Dropout rate applied before the output linear projection.
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
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input token indices, shape ``(batch_size, seq_len)``.
        hidden : tuple of (h_0, c_0) or None
            Initial LSTM hidden state. If None, initialized to zeros.

        Returns
        -------
        output : torch.Tensor
            Logits for each position, shape ``(batch_size, seq_len, vocab_size)``.
        hidden : tuple of (h_n, c_n)
            Final LSTM hidden state, each of shape ``(num_layers, batch_size, hidden_dim)``.
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
