# -*- coding: utf-8 -*-
"""
Conditional Attention LSTM — the main model for genre-aware poetry generation.

A genre-conditional 2-layer unidirectional LSTM with causal self-attention,
residual connection, and LayerNorm. This is the flagship model.

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
    Embedding Dropout (p=0.20)
        ↓
    2-layer Unidirectional LSTM (embedding_dim → hidden_dim, dropout=0.30)
        ↓
    ┌─ Causal Self-Attention (4 heads, dropout=0.10)
    │      ↓
    └─ + (Residual Connection)
        ↓
    LayerNorm
        ↓
    Output Dropout (p=0.20)
        ↓
    Linear (hidden_dim → vocab_size)
        ↓
    Next character logits

Generation guidance
--------------------
Since the causal-attention layer needs the full prefix for each prediction,
the recommended generation loop passes the accumulated sequence each step:

    seq = [<genre>, SOP]
    for _ in range(max_len):
        output, _ = model(seq)          # recompute over full prefix
        next_token = sample(output[-1])  # last position's prediction
        seq.append(next_token)
        if next_token == EOP: break

For the LSTM-only models (BaselineLSTM, ConditionalLSTM), the generation
loop can be more efficient by reusing the hidden state.
"""

import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Causal (masked) multi-head self-attention.

    Each position attends only to itself and previous positions,
    making it suitable for autoregressive generation.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model (must be divisible by num_heads).
    num_heads : int
        Number of parallel attention heads.
    dropout : float, default 0.10
        Dropout rate applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.10):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections combined for efficiency
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build a causal attention mask.

        Returns a float mask of shape ``(seq_len, seq_len)`` where masked
        positions are filled with ``-inf`` (so softmax yields ~0).

        Shape
        -----
        For seq_len=4, the mask is::

            [[  0., -inf, -inf, -inf],
             [  0.,   0., -inf, -inf],
             [  0.,   0.,   0., -inf],
             [  0.,   0.,   0.,   0.]]
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape ``(batch_size, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Attended features, same shape as input.
        """
        batch_size, seq_len, embed_dim = x.size()

        # Linear projections
        q = self.q_proj(x)  # (B, S, E)
        k = self.k_proj(x)  # (B, S, E)
        v = self.v_proj(x)  # (B, S, E)

        # Reshape to multi-head: (B, num_heads, S, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, S, S)

        # Apply causal mask
        causal_mask = self._build_causal_mask(seq_len, x.device)  # (S, S)
        attn_scores = attn_scores + causal_mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_out = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)

        # Merge heads: (B, S, E)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Output projection
        attn_out = self.out_proj(attn_out)

        return attn_out


class ConditionalAttnLSTM(nn.Module):
    """Genre-conditional LSTM with causal self-attention.

    The flagship model for poetry generation. A genre control token is
    prepended to the sequence; a 2-layer unidirectional LSTM encodes the
    sequence; causal self-attention refines the representations; a residual
    connection with LayerNorm stabilizes training.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (including PAD, UNK, SOP, EOP, and genre tokens).
    embedding_dim : int, default 256
        Dimension of token embeddings.
    hidden_dim : int, default 256
        Dimension of LSTM hidden states and the attention residual stream.
    num_layers : int, default 2
        Number of stacked LSTM layers.
    embedding_dropout : float, default 0.20
        Dropout rate applied after the embedding layer.
    lstm_dropout : float, default 0.30
        Dropout rate applied between LSTM layers.
    output_dropout : float, default 0.20
        Dropout rate applied before the output linear projection.
    attention_heads : int, default 4
        Number of parallel heads in the causal self-attention layer.
    attention_dropout : float, default 0.10
        Dropout rate applied to attention weights.
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
        attention_heads: int = 4,
        attention_dropout: float = 0.10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Embedding ──
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(embedding_dropout)

        # ── 2-layer unidirectional LSTM ──
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # ── Causal self-attention ──
        self.self_attn = CausalSelfAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        # ── Residual + LayerNorm ──
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # ── Output ──
        self.out_dropout = nn.Dropout(output_dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values for stable training."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden states with zeros."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)

    def forward(self, input: torch.Tensor, hidden=None):
        """Forward pass with causal attention.

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

        Notes
        -----
        During autoregressive generation, pass the **full accumulated
        sequence** each step so the causal attention layer can see all
        previous positions. The LSTM will recompute from scratch, which
        is affordable for poetry-length sequences (≤200 tokens).
        """
        batch_size, seq_len = input.size()

        if hidden is None:
            hidden = self._init_hidden(batch_size, input.device)

        # ── Embedding + dropout ──
        embeds = self.embedding(input)                      # (B, S, E)
        embeds = self.emb_dropout(embeds)

        # ── LSTM ──
        lstm_out, (h_n, c_n) = self.lstm(embeds, hidden)    # (B, S, H)

        # ── Causal self-attention ──
        attn_out = self.self_attn(lstm_out)                 # (B, S, H)

        # ── Residual connection + LayerNorm ──
        residual = self.layer_norm(lstm_out + attn_out)     # (B, S, H)

        # ── Output dropout + linear ──
        dropped = self.out_dropout(residual)
        output = self.linear(dropped)                       # (B, S, V)

        return output, (h_n, c_n)
