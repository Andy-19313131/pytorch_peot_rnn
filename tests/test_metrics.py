#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    aggregate_generation_metrics,
    distinct_n,
    format_scores,
    rhyme_consistency,
    split_poem_lines,
)


class TestPoetryMetrics(unittest.TestCase):
    def test_split_lines(self):
        poem = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        self.assertEqual(split_poem_lines(poem), ["春眠不觉晓", "处处闻啼鸟", "夜来风雨声", "花落知多少"])

    def test_format_correctness(self):
        poem = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        scores = format_scores(poem, form_token="<5JUE>")
        self.assertEqual(scores["format_correctness"], 1.0)
        self.assertEqual(scores["line_length_accuracy"], 1.0)

    def test_distinct(self):
        self.assertGreater(distinct_n(["春眠春眠"], 1), 0.0)
        self.assertLessEqual(distinct_n(["春眠春眠"], 1), 1.0)

    def test_rhyme_consistency_returns_ratio(self):
        poem = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        score = rhyme_consistency(poem, form_token="<5JUE>")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_aggregate_metrics(self):
        samples = [
            {
                "poem": "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
                "form_token": "<5JUE>",
                "strategy": "constrained",
            }
        ]
        metrics = aggregate_generation_metrics(samples, train_texts=[])
        self.assertIn("format_correctness", metrics)
        self.assertEqual(metrics["unique_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

