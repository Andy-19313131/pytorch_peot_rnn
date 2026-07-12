#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Count and compare parameters across all three poetry models.

Usage::

    python scripts/count_parameters.py

or from the project root::

    python -m scripts.count_parameters
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import BaselineLSTM, ConditionalLSTM, ConditionalAttnLSTM


# ── Default config (shared across models) ──
DEFAULT_CONFIG = {
    'vocab_size': 5000,       # typical Tang poetry vocabulary
    'embedding_dim': 256,
    'hidden_dim': 256,
    'num_layers': 2,
    'embedding_dropout': 0.20,
    'lstm_dropout': 0.30,
    'output_dropout': 0.20,
    'attention_heads': 4,
    'attention_dropout': 0.10,
}


def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_by_component(model):
    """Break down parameter count by named top-level children."""
    components = {}
    for name, child in model.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        components[name] = (total, trainable)
    return components


def format_num(n: int) -> str:
    """Format a number with commas."""
    return f"{n:,}"


def print_separator():
    print("-" * 70)


def main():
    config = DEFAULT_CONFIG.copy()
    print_separator()
    print(f"Model Parameter Comparison")
    print(f"  vocab_size = {config['vocab_size']}, "
          f"embedding_dim = {config['embedding_dim']}, "
          f"hidden_dim = {config['hidden_dim']}")
    print(f"  num_layers = {config['num_layers']}, "
          f"attention_heads = {config['attention_heads']}")
    print_separator()

    # ── Build models ──
    baseline = BaselineLSTM(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        embedding_dropout=config['embedding_dropout'],
        lstm_dropout=config['lstm_dropout'],
        output_dropout=config['output_dropout'],
    )

    conditional = ConditionalLSTM(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        embedding_dropout=config['embedding_dropout'],
        lstm_dropout=config['lstm_dropout'],
        output_dropout=config['output_dropout'],
    )

    attn_model = ConditionalAttnLSTM(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        embedding_dropout=config['embedding_dropout'],
        lstm_dropout=config['lstm_dropout'],
        output_dropout=config['output_dropout'],
        attention_heads=config['attention_heads'],
        attention_dropout=config['attention_dropout'],
    )

    models = {
        'BaselineLSTM': baseline,
        'ConditionalLSTM': conditional,
        'ConditionalAttnLSTM': attn_model,
    }

    # ── Totals ──
    print(f"\n{'Model':<25} {'Total':>15} {'Trainable':>15}")
    print_separator()
    for name, model in models.items():
        total, trainable = count_parameters(model)
        print(f"{name:<25} {format_num(total):>15} {format_num(trainable):>15}")

    # ── Per-component breakdown ──
    print(f"\n\n{'='*70}")
    print("Per-Component Breakdown")
    print("=" * 70)

    for model_name, model in models.items():
        print(f"\n── {model_name} ──")
        print(f"{'Component':<25} {'Total':>15} {'Trainable':>15}")
        print_separator()
        components = count_by_component(model)
        comp_total = 0
        comp_trainable = 0
        for comp_name, (total, trainable) in components.items():
            print(f"  {comp_name:<23} {format_num(total):>15} {format_num(trainable):>15}")
            comp_total += total
            comp_trainable += trainable
        print_separator()
        print(f"  {'SUM':<23} {format_num(comp_total):>15} {format_num(comp_trainable):>15}")

    # ── Comparison note ──
    base_total, _ = count_parameters(baseline)
    attn_total, _ = count_parameters(attn_model)
    attn_overhead = attn_total - base_total
    print(f"\n\nAttention overhead vs baseline: +{format_num(attn_overhead)} parameters "
          f"({attn_overhead / base_total * 100:.1f}%)")


if __name__ == '__main__':
    main()
