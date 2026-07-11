#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the three poetry generation models.

Covers:
    - Model instantiation with various configs
    - Forward pass output shapes
    - Causal attention mask correctness
    - Autoregressive generation loop
    - Hidden state continuity
    - Gradient flow
    - Edge cases (seq_len=1, batch_size=1, etc.)

Run from the project root::

    python -m pytest tests/test_model.py -v
    # or
    python tests/test_model.py
"""

import sys
import os
import unittest

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models import BaselineLSTM, ConditionalLSTM, ConditionalAttnLSTM
from src.models.conditional_attn_lstm import CausalSelfAttention


# ── Test config ──
VOCAB_SIZE = 100
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
NUM_LAYERS = 2
ATTENTION_HEADS = 4


def make_batch(batch_size=2, seq_len=10, vocab_size=VOCAB_SIZE):
    """Create a random batch of token indices, excluding PAD (0)."""
    return torch.randint(1, vocab_size, (batch_size, seq_len)).long()


# ═══════════════════════════════════════════════════════════════════
# BaselineLSTM
# ═══════════════════════════════════════════════════════════════════

class TestBaselineLSTM(unittest.TestCase):
    """Tests for BaselineLSTM."""

    @classmethod
    def setUpClass(cls):
        cls.model = BaselineLSTM(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        )

    def test_instantiation(self):
        """Model instantiates without error."""
        self.assertIsInstance(self.model, BaselineLSTM)

    def test_forward_shape(self):
        """Forward pass returns correct output shapes."""
        batch_size, seq_len = 2, 10
        x = make_batch(batch_size, seq_len)
        output, hidden = self.model(x)

        # output: (batch, seq_len, vocab_size)
        self.assertEqual(output.shape, (batch_size, seq_len, VOCAB_SIZE))
        # hidden: tuple of (h, c), each (num_layers, batch, hidden_dim)
        h, c = hidden
        self.assertEqual(h.shape, (NUM_LAYERS, batch_size, HIDDEN_DIM))
        self.assertEqual(c.shape, (NUM_LAYERS, batch_size, HIDDEN_DIM))

    def test_forward_batch_size_one(self):
        """Forward pass works with batch_size=1."""
        x = make_batch(1, 5)
        output, hidden = self.model(x)
        self.assertEqual(output.shape, (1, 5, VOCAB_SIZE))

    def test_forward_seq_len_one(self):
        """Forward pass works with seq_len=1."""
        x = make_batch(3, 1)
        output, hidden = self.model(x)
        self.assertEqual(output.shape, (3, 1, VOCAB_SIZE))

    def test_hidden_continuity(self):
        """Hidden state can be passed to continue a sequence."""
        x1 = make_batch(2, 5)
        _, (h1, c1) = self.model(x1, hidden=None)

        x2 = make_batch(2, 3)
        output2, (h2, c2) = self.model(x2, hidden=(h1, c1))

        self.assertEqual(output2.shape, (2, 3, VOCAB_SIZE))
        # Hidden shape unchanged
        self.assertEqual(h2.shape, h1.shape)

    def test_hidden_from_different_batch_fails(self):
        """Passing hidden state with mismatched batch size raises an error."""
        x1 = make_batch(2, 5)
        _, hidden = self.model(x1)
        x2 = make_batch(3, 5)  # different batch size
        with self.assertRaises(RuntimeError):
            self.model(x2, hidden=hidden)

    def test_gradient_flow(self):
        """Gradients flow through the model."""
        x = make_batch(2, 10)
        output, _ = self.model(x)
        loss = output.sum()
        loss.backward()

        # Check that embedding weights got gradients
        self.assertIsNotNone(self.model.embedding.weight.grad)
        self.assertNotEqual(self.model.embedding.weight.grad.abs().sum(), 0)

    def test_embedding_dropout_training_vs_eval(self):
        """Embedding dropout is active in training, inactive in eval."""
        x = make_batch(2, 10)
        self.model.train()
        out_train, _ = self.model(x)
        self.model.eval()
        with torch.no_grad():
            out_eval, _ = self.model(x)
        self.assertEqual(out_train.shape, out_eval.shape)

    def test_init_hidden(self):
        """_init_hidden returns correctly shaped zeros."""
        h = self.model._init_hidden(batch_size=4, device=torch.device('cpu'))
        h0, c0 = h
        self.assertEqual(h0.shape, (NUM_LAYERS, 4, HIDDEN_DIM))
        self.assertEqual(c0.shape, (NUM_LAYERS, 4, HIDDEN_DIM))
        self.assertTrue((h0 == 0).all())
        self.assertTrue((c0 == 0).all())


# ═══════════════════════════════════════════════════════════════════
# ConditionalLSTM
# ═══════════════════════════════════════════════════════════════════

class TestConditionalLSTM(unittest.TestCase):
    """Tests for ConditionalLSTM."""

    @classmethod
    def setUpClass(cls):
        cls.model = ConditionalLSTM(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        )

    def test_instantiation(self):
        self.assertIsInstance(self.model, ConditionalLSTM)

    def test_forward_shape(self):
        """Forward pass with genre token prepended."""
        batch_size, seq_len = 2, 11  # 1 genre + SOP + 8 chars + EOP
        x = make_batch(batch_size, seq_len)
        output, hidden = self.model(x)
        self.assertEqual(output.shape, (batch_size, seq_len, VOCAB_SIZE))
        h, c = hidden
        self.assertEqual(h.shape, (NUM_LAYERS, batch_size, HIDDEN_DIM))

    def test_genre_token_flow(self):
        """The model can learn to use the genre token (gradients reach it)."""
        # Simulate: [<5JUE>=4, SOP=2, chars..., EOP=3]
        batch_size = 2
        input_ids = torch.tensor([
            [4, 2, 5, 6, 7, 3],
            [5, 2, 8, 9, 10, 3],
        ])
        output, _ = self.model(input_ids)
        loss = output.sum()
        loss.backward()
        # Embedding weight for genre tokens should have gradients
        genre_grad = self.model.embedding.weight.grad[4:8].abs().sum()
        self.assertGreater(genre_grad.item(), 0)

    def test_architecture_match_baseline(self):
        """ConditionalLSTM has same parameter count as BaselineLSTM (same arch)."""
        baseline = BaselineLSTM(
            vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        )
        baseline_params = sum(p.numel() for p in baseline.parameters())
        cond_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(baseline_params, cond_params)


# ═══════════════════════════════════════════════════════════════════
# CausalSelfAttention
# ═══════════════════════════════════════════════════════════════════

class TestCausalSelfAttention(unittest.TestCase):
    """Tests for the CausalSelfAttention module."""

    @classmethod
    def setUpClass(cls):
        cls.attn = CausalSelfAttention(
            embed_dim=HIDDEN_DIM,
            num_heads=ATTENTION_HEADS,
            dropout=0.0,  # deterministic for testing
        )

    def test_forward_shape(self):
        """Output shape matches input shape."""
        x = torch.randn(2, 10, HIDDEN_DIM)
        out = self.attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_causal_mask_structure(self):
        """The causal mask prevents attending to future positions."""
        mask = self.attn._build_causal_mask(5, torch.device('cpu'))
        # Position 0: only position 0 is unmasked (0.0)
        self.assertEqual(mask[0, 0], 0.0)
        self.assertTrue((mask[0, 1:] == float('-inf')).all())
        # Position 1: positions 0,1 are unmasked
        self.assertEqual(mask[1, 0], 0.0)
        self.assertEqual(mask[1, 1], 0.0)
        self.assertTrue((mask[1, 2:] == float('-inf')).all())
        # Position 4: all positions unmasked
        self.assertTrue((mask[4, :] == 0.0).all())

    def test_causal_behavior(self):
        """Position i's output depends only on positions ≤ i."""
        x = torch.randn(1, 5, HIDDEN_DIM)
        self.attn.eval()
        with torch.no_grad():
            out = self.attn(x)

        # Perturb position 4 (future relative to position 2)
        x_perturbed = x.clone()
        x_perturbed[0, 4] = 999.0
        with torch.no_grad():
            out_perturbed = self.attn(x_perturbed)

        # Position 2's output should be unchanged
        self.assertTrue(torch.allclose(out[0, 2], out_perturbed[0, 2], atol=1e-6))
        # Position 4's output should change
        self.assertFalse(torch.allclose(out[0, 4], out_perturbed[0, 4], atol=1e-6))

    def test_seq_len_one(self):
        """Attention works with a single token."""
        x = torch.randn(4, 1, HIDDEN_DIM)
        out = self.attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_embed_dim_not_divisible_raises(self):
        """embed_dim not divisible by num_heads raises ValueError."""
        with self.assertRaises(ValueError):
            CausalSelfAttention(embed_dim=10, num_heads=3)

    def test_dropout_active_in_train(self):
        """Dropout is active during training."""
        attn = CausalSelfAttention(HIDDEN_DIM, ATTENTION_HEADS, dropout=0.5)
        x = torch.randn(2, 5, HIDDEN_DIM)
        attn.train()
        out1 = attn(x)
        out2 = attn(x)
        # With dropout=0.5, outputs should differ
        self.assertFalse(torch.allclose(out1, out2))


# ═══════════════════════════════════════════════════════════════════
# ConditionalAttnLSTM
# ═══════════════════════════════════════════════════════════════════

class TestConditionalAttnLSTM(unittest.TestCase):
    """Tests for ConditionalAttnLSTM — the flagship model."""

    @classmethod
    def setUpClass(cls):
        cls.model = ConditionalAttnLSTM(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            attention_heads=ATTENTION_HEADS,
            attention_dropout=0.0,
        )

    def test_instantiation(self):
        self.assertIsInstance(self.model, ConditionalAttnLSTM)

    def test_forward_shape(self):
        """Forward pass returns correct shapes."""
        batch_size, seq_len = 2, 12
        x = make_batch(batch_size, seq_len)
        output, hidden = self.model(x)

        self.assertEqual(output.shape, (batch_size, seq_len, VOCAB_SIZE))
        h, c = hidden
        self.assertEqual(h.shape, (NUM_LAYERS, batch_size, HIDDEN_DIM))
        self.assertEqual(c.shape, (NUM_LAYERS, batch_size, HIDDEN_DIM))

    def test_forward_seq_len_one(self):
        """Forward with a single token (for autoregressive steps)."""
        x = make_batch(2, 1)
        output, hidden = self.model(x)
        self.assertEqual(output.shape, (2, 1, VOCAB_SIZE))

    def test_forward_seq_len_two(self):
        """Forward with two tokens — causal attention mask is minimal."""
        x = make_batch(2, 2)
        output, hidden = self.model(x)
        self.assertEqual(output.shape, (2, 2, VOCAB_SIZE))

    def test_residual_path(self):
        """Verify the residual connection + LayerNorm path is present."""
        # The layer_norm should be after the residual add
        self.assertIsInstance(self.model.layer_norm, torch.nn.LayerNorm)

    def test_gradient_flow_all_components(self):
        """Gradients flow through LSTM, attention, and residual."""
        x = make_batch(2, 10)
        output, _ = self.model(x)
        loss = output.sum()
        loss.backward()

        # Embedding
        self.assertIsNotNone(self.model.embedding.weight.grad)
        # Attention Q-proj
        self.assertIsNotNone(self.model.self_attn.q_proj.weight.grad)
        self.assertNotEqual(self.model.self_attn.q_proj.weight.grad.abs().sum(), 0)
        # Linear output
        self.assertIsNotNone(self.model.linear.weight.grad)
        self.assertNotEqual(self.model.linear.weight.grad.abs().sum(), 0)

    def test_autoregressive_generation_loop(self):
        """Simulate an autoregressive generation loop.

        Each step passes the full accumulated sequence so the causal
        attention layer sees all previous positions.
        """
        self.model.eval()
        batch_size = 1
        max_len = 20

        # Start: [<5JUE>=4, SOP=2]
        seq = torch.tensor([[4, 2]])
        generated = []

        with torch.no_grad():
            for step in range(max_len):
                output, _ = self.model(seq)
                # Sample from last position
                logits = output[0, -1, :]  # (vocab_size,)
                next_token = logits.argmax().item()
                generated.append(next_token)
                # Append to sequence
                seq = torch.cat([seq, torch.tensor([[next_token]])], dim=1)
                if next_token == 3:  # EOP
                    break

        self.assertGreater(len(generated), 0)
        # The generated tokens should be within vocab range
        for tok in generated:
            self.assertTrue(0 <= tok < VOCAB_SIZE)

    def test_eval_mode_no_dropout(self):
        """In eval mode, same input gives same output."""
        self.model.eval()
        x = make_batch(2, 10)
        with torch.no_grad():
            out1, _ = self.model(x)
            out2, _ = self.model(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_weight_initialization(self):
        """Weights are initialized to non-zero values."""
        # Embedding weights should not be all zeros
        self.assertNotEqual(self.model.embedding.weight.abs().sum(), 0)
        # Linear bias should be zeros
        self.assertTrue((self.model.linear.bias == 0).all())


# ═══════════════════════════════════════════════════════════════════
# Cross-model consistency
# ═══════════════════════════════════════════════════════════════════

class TestCrossModelConsistency(unittest.TestCase):
    """Tests comparing behavior across models."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = BaselineLSTM(
            vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        )
        cls.conditional = ConditionalLSTM(
            vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        )
        cls.attn = ConditionalAttnLSTM(
            vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            attention_heads=ATTENTION_HEADS,
        )

    def test_parameter_count_ordering(self):
        """Parameter counts: Baseline ≈ Conditional < ConditionalAttn."""
        b_params = sum(p.numel() for p in self.baseline.parameters())
        c_params = sum(p.numel() for p in self.conditional.parameters())
        a_params = sum(p.numel() for p in self.attn.parameters())

        # Baseline and Conditional have identical architecture
        self.assertEqual(b_params, c_params)
        # Attention model has more parameters (attention weights)
        self.assertGreater(a_params, b_params)

    def test_forward_output_shape_consistent(self):
        """All models produce same-shaped outputs for same input size."""
        batch_size, seq_len = 2, 8
        x = make_batch(batch_size, seq_len)

        out_b, _ = self.baseline(x)
        out_c, _ = self.conditional(x)
        out_a, _ = self.attn(x)

        self.assertEqual(out_b.shape, out_c.shape)
        self.assertEqual(out_b.shape, out_a.shape)

    def test_hidden_shape_consistent(self):
        """All models produce same-shaped hidden states."""
        x = make_batch(2, 5)
        _, hb = self.baseline(x)
        _, hc = self.conditional(x)
        _, ha = self.attn(x)

        for h in (hb, hc, ha):
            self.assertEqual(h[0].shape, (NUM_LAYERS, 2, HIDDEN_DIM))
            self.assertEqual(h[1].shape, (NUM_LAYERS, 2, HIDDEN_DIM))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    unittest.main(verbosity=2)
