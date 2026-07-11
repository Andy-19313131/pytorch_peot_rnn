# -*- coding: utf-8 -*-
"""
Poetry generation models.

Three tiers of models for classical Chinese poetry generation:
- BaselineLSTM:        basic 2-layer LSTM language model
- ConditionalLSTM:     genre-conditional 2-layer LSTM
- ConditionalAttnLSTM: genre-conditional LSTM + causal self-attention
"""

from .baseline_lstm import BaselineLSTM
from .conditional_lstm import ConditionalLSTM
from .conditional_attn_lstm import ConditionalAttnLSTM

# Genre control tokens 体裁控制标记
# These are prepended to the input sequence to condition on poetry form.
# Usage: [<5JUE>, SOP, chars..., EOP] for a 五言绝句
GENRE_TOKENS = {
    '<5JUE>': '五言绝句 (Five-character quatrain, 4 lines × 5 chars)',
    '<7JUE>': '七言绝句 (Seven-character quatrain, 4 lines × 7 chars)',
    '<5LV>':  '五言律诗 (Five-character regulated verse, 8 lines × 5 chars)',
    '<7LV>':  '七言律诗 (Seven-character regulated verse, 8 lines × 7 chars)',
}

# Recommended token ID assignment (after PAD=0, UNK=1, SOP=2, EOP=3)
GENRE_TOKEN_IDS = {
    '<5JUE>': 4,
    '<7JUE>': 5,
    '<5LV>': 6,
    '<7LV>': 7,
}

NUM_GENRE_TOKENS = len(GENRE_TOKENS)

__all__ = [
    'BaselineLSTM',
    'ConditionalLSTM',
    'ConditionalAttnLSTM',
    'GENRE_TOKENS',
    'GENRE_TOKEN_IDS',
    'NUM_GENRE_TOKENS',
]
