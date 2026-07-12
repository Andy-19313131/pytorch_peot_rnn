# -*- coding: utf-8 -*-
"""Automatic metrics for generated classical Chinese poems."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .generate import FORM_ID_TO_TOKEN, FORM_SPECS


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_RE = re.compile(r"[，。！？；,.!?;]+")
SPECIAL_RE = re.compile(r"<(?:PAD|UNK|SOP|EOP|5JUE|7JUE|5LV|7LV)>|PAD|UNK|SOP|EOP")


def normalize_poem(text: str) -> str:
    """Remove whitespace and special tokens while keeping Chinese punctuation."""
    text = SPECIAL_RE.sub("", text or "")
    return re.sub(r"\s+", "", text)


def chinese_chars(text: str) -> List[str]:
    return CHINESE_RE.findall(normalize_poem(text))


def split_poem_lines(text: str) -> List[str]:
    """Split a poem into punctuation-delimited lines and remove punctuation."""
    text = normalize_poem(text)
    lines: List[str] = []
    buf: List[str] = []
    for ch in text:
        if CHINESE_RE.match(ch):
            buf.append(ch)
        elif ch in "，。！？；,.!?;":
            if buf:
                lines.append("".join(buf))
                buf = []
    if buf:
        lines.append("".join(buf))
    return lines


def form_spec(form_id: Optional[int] = None, form_token: Optional[str] = None) -> Optional[Tuple[int, int]]:
    token = form_token
    if token is None and form_id is not None:
        token = FORM_ID_TO_TOKEN.get(int(form_id))
    if token is None:
        return None
    return FORM_SPECS.get(token)


def format_scores(text: str, form_id: Optional[int] = None, form_token: Optional[str] = None) -> Dict[str, float]:
    """Return strict poem-level and softer line-level format scores."""
    spec = form_spec(form_id=form_id, form_token=form_token)
    lines = split_poem_lines(text)
    if spec is None:
        return {
            "format_correctness": 0.0,
            "line_length_accuracy": 0.0,
            "line_count_accuracy": 0.0,
        }

    expected_lines, expected_len = spec
    line_count_accuracy = 1.0 if len(lines) == expected_lines else 0.0
    if not lines:
        line_length_accuracy = 0.0
    else:
        correct_lines = sum(1 for line in lines[:expected_lines] if len(line) == expected_len)
        line_length_accuracy = correct_lines / expected_lines
    strict = 1.0 if line_count_accuracy and line_length_accuracy == 1.0 else 0.0
    return {
        "format_correctness": strict,
        "line_length_accuracy": line_length_accuracy,
        "line_count_accuracy": line_count_accuracy,
    }


def distinct_n(texts: Sequence[str], n: int) -> float:
    units: List[Tuple[str, ...]] = []
    for text in texts:
        chars = chinese_chars(text)
        units.extend(tuple(chars[i:i + n]) for i in range(0, max(len(chars) - n + 1, 0)))
    if not units:
        return 0.0
    return len(set(units)) / len(units)


def repeat_bigram_ratio(texts: Sequence[str]) -> float:
    bigrams: List[Tuple[str, str]] = []
    for text in texts:
        chars = chinese_chars(text)
        bigrams.extend((chars[i], chars[i + 1]) for i in range(len(chars) - 1))
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(bigrams)


def adjacent_repeat_rate(texts: Sequence[str]) -> float:
    total_pairs = 0
    repeats = 0
    for text in texts:
        chars = chinese_chars(text)
        total_pairs += max(len(chars) - 1, 0)
        repeats += sum(1 for i in range(len(chars) - 1) if chars[i] == chars[i + 1])
    return repeats / total_pairs if total_pairs else 0.0


def unique_ratio(texts: Sequence[str]) -> float:
    normalized = ["".join(chinese_chars(text)) for text in texts]
    normalized = [text for text in normalized if text]
    return len(set(normalized)) / len(normalized) if normalized else 0.0


def train_set_overlap(texts: Sequence[str], train_texts: Iterable[str]) -> float:
    train_set = {"".join(chinese_chars(text)) for text in train_texts}
    train_set.discard("")
    normalized = ["".join(chinese_chars(text)) for text in texts]
    normalized = [text for text in normalized if text]
    if not normalized or not train_set:
        return 0.0
    return sum(1 for text in normalized if text in train_set) / len(normalized)


def build_ngram_set(texts: Iterable[str], n: int = 4) -> set[Tuple[str, ...]]:
    grams = set()
    for text in texts:
        chars = chinese_chars(text)
        grams.update(tuple(chars[i:i + n]) for i in range(max(len(chars) - n + 1, 0)))
    return grams


def train_ngram_overlap(texts: Sequence[str], train_texts: Iterable[str], n: int = 4) -> float:
    train_grams = build_ngram_set(train_texts, n=n)
    if not train_grams:
        return 0.0
    total = 0
    overlap = 0
    for text in texts:
        chars = chinese_chars(text)
        grams = [tuple(chars[i:i + n]) for i in range(max(len(chars) - n + 1, 0))]
        total += len(grams)
        overlap += sum(1 for gram in grams if gram in train_grams)
    return overlap / total if total else 0.0


def _pinyin_final(ch: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return ch
    finals = lazy_pinyin(ch, style=Style.FINALS, strict=False, errors="ignore")
    return finals[0] if finals else ch


def rhyme_consistency(text: str, form_id: Optional[int] = None, form_token: Optional[str] = None) -> float:
    """Approximate rhyme consistency on even lines using pinyin finals if available."""
    lines = split_poem_lines(text)
    spec = form_spec(form_id=form_id, form_token=form_token)
    if spec is None:
        target_indices = [1, 3]
    else:
        target_indices = list(range(1, spec[0], 2))
    endings = [lines[i][-1] for i in target_indices if i < len(lines) and lines[i]]
    if len(endings) < 2:
        return 0.0
    finals = [_pinyin_final(ch) for ch in endings]
    first = finals[0]
    return sum(1 for item in finals[1:] if item == first) / (len(finals) - 1)


def aggregate_generation_metrics(
    samples: Sequence[Dict[str, object]],
    train_texts: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Aggregate automatic metrics for generated sample dictionaries.

    Each sample should contain at least ``poem``. Optional ``form_id`` or
    ``form_token`` values are used for format and rhyme metrics.
    """
    texts = [str(sample.get("poem", "")) for sample in samples]
    if not texts:
        return {
            "format_correctness": 0.0,
            "line_length_accuracy": 0.0,
            "line_count_accuracy": 0.0,
            "rhyme_consistency": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repeat_bigram_ratio": 0.0,
            "adjacent_repeat_rate": 0.0,
            "train_set_overlap": 0.0,
            "train_4gram_overlap": 0.0,
            "unique_ratio": 0.0,
        }

    format_rows = [
        format_scores(
            str(sample.get("poem", "")),
            form_id=sample.get("form_id"),  # type: ignore[arg-type]
            form_token=sample.get("form_token"),  # type: ignore[arg-type]
        )
        for sample in samples
    ]
    rhyme_rows = [
        rhyme_consistency(
            str(sample.get("poem", "")),
            form_id=sample.get("form_id"),  # type: ignore[arg-type]
            form_token=sample.get("form_token"),  # type: ignore[arg-type]
        )
        for sample in samples
    ]

    train_text_list = list(train_texts or [])
    return {
        "format_correctness": _mean(row["format_correctness"] for row in format_rows),
        "line_length_accuracy": _mean(row["line_length_accuracy"] for row in format_rows),
        "line_count_accuracy": _mean(row["line_count_accuracy"] for row in format_rows),
        "rhyme_consistency": _mean(rhyme_rows),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "repeat_bigram_ratio": repeat_bigram_ratio(texts),
        "adjacent_repeat_rate": adjacent_repeat_rate(texts),
        "train_set_overlap": train_set_overlap(texts, train_text_list) if train_text_list else 0.0,
        "train_4gram_overlap": train_ngram_overlap(texts, train_text_list, n=4) if train_text_list else 0.0,
        "unique_ratio": unique_ratio(texts),
    }


@torch.no_grad()
def evaluate_next_token(
    model: torch.nn.Module,
    data_loader,
    pad_id: int = 0,
    device: str | torch.device = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate average NLL, perplexity and top-1 accuracy on non-PAD tokens."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        logits, _hidden = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        non_pad = target_ids != pad_id
        token_count = int(non_pad.sum().item())
        total_nll += float(loss.item()) * token_count
        total_tokens += token_count
        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == target_ids) & non_pad).sum().item())

    avg_nll = total_nll / total_tokens if total_tokens else float("inf")
    return {
        "test_loss": avg_nll,
        "test_ppl": math.exp(avg_nll) if avg_nll < 100 else float("inf"),
        "test_acc": total_correct / total_tokens if total_tokens else 0.0,
    }


def evaluate_model(
    model,
    test_loader,
    generator,
    train_texts: Sequence[str],
    config: Dict[str, object],
) -> Dict[str, float]:
    """Interface-level evaluation entry point used by the project spec."""
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    device = training_cfg.get("device", "cpu") if isinstance(training_cfg, dict) else "cpu"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_metrics = evaluate_next_token(model, test_loader, pad_id=0, device=device)

    prompts = ["春", "山", "月", "风"]
    records = generator.generate_many(
        prompts=prompts,
        strategies=["greedy", "topk", "topp", "constrained"],
        form_ids=[0, 1, 2, 3],
    )
    samples = [
        {
            "strategy": item.strategy,
            "form_token": item.form_token,
            "prompt": item.prompt,
            "poem": item.poem,
        }
        for item in records
    ]
    gen_metrics = aggregate_generation_metrics(samples, train_texts=train_texts)
    return {**lm_metrics, **gen_metrics}


def load_jsonl_texts(path: str, text_key: str = "text") -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line).get(text_key, ""))
    return texts


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

