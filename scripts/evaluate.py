#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run generation and automatic evaluation for Task E."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.train import _MODEL_REGISTRY, _resolve_model_type, build_model
from src.dataset import get_dataloader, load_vocab
from src.generate import FORM_ID_TO_TOKEN, PoetryGenerator
from src.metrics import aggregate_generation_metrics, evaluate_next_token


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw in {"null", "None", "~"}:
        return None
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    config: Dict[str, Any] = {}
    section = None
    list_key = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            text = line.strip()
            if indent == 0 and text.endswith(":"):
                section = text[:-1]
                config[section] = {}
                list_key = None
                continue
            if section is None:
                continue
            if text.startswith("- "):
                if list_key is not None:
                    config[section].setdefault(list_key, []).append(_parse_scalar(text[2:]))
                continue
            if ":" in text:
                key, value = text.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value == "":
                    config[section][key] = []
                    list_key = key
                else:
                    config[section][key] = _parse_scalar(value)
                    list_key = None
    return config


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


def resolve_data_path(path: str) -> str:
    """Resolve data artifacts from the current root, with a synced-copy fallback."""
    candidates = [
        resolve_path(path),
        os.path.join(_PROJECT_ROOT, "pytorch_peot_rnn-develop", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_checkpoint_if_present(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> bool:
    if not checkpoint_path:
        return False
    checkpoint_path = resolve_path(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print(f"[Eval] Checkpoint not found, using current model weights: {checkpoint_path}")
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    print(f"[Eval] Loaded checkpoint: {checkpoint_path}")
    return True


def write_samples(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("=" * 72 + "\n")
            f.write(f"strategy={row['strategy']}  form={row['form_token']}  prompt={row['prompt']}\n")
            f.write(str(row["poem"]) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task E evaluation entry point")
    parser.add_argument("--config", default="configs/improved_model.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--strategies", default="greedy,topk,topp,constrained")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--skip_lm", action="store_true")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    generation_cfg = config.get("generation", {})

    requested_device = args.device or training_cfg.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    word2idx, idx2word = load_vocab(resolve_data_path("data/processed/vocab.json"))
    model_type = _resolve_model_type(model_cfg.get("type", "BaselineLSTM"))
    _model_cls, use_form_token = _MODEL_REGISTRY[model_type]
    model_cfg["vocab_size"] = len(word2idx)
    model = build_model(model_type, len(word2idx), model_cfg).to(device)
    loaded = load_checkpoint_if_present(model, args.checkpoint, device)
    if not loaded:
        print("[Eval] No trained checkpoint was loaded. Generation metrics are for smoke testing only.")

    lm_metrics: Dict[str, Any] = {"test_loss": "", "test_ppl": "", "test_acc": ""}
    if not args.skip_lm:
        test_loader = get_dataloader(
            jsonl_path=resolve_data_path("data/splits/test.jsonl"),
            word2idx=word2idx,
            batch_size=int(training_cfg.get("batch_size", 64)),
            max_len=int(config.get("data", {}).get("max_len", 125)),
            use_form_token=use_form_token,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        lm_metrics = evaluate_next_token(
            model,
            test_loader,
            pad_id=word2idx.get("<PAD>", 0),
            device=device,
            max_batches=args.max_eval_batches,
        )

    train_texts_path = resolve_data_path("data/splits/train_texts.txt")
    with open(train_texts_path, "r", encoding="utf-8") as f:
        train_texts = [line.strip() for line in f if line.strip()]

    generator = PoetryGenerator(
        model=model,
        word2idx=word2idx,
        idx2word=idx2word,
        device=device,
        max_gen_len=int(generation_cfg.get("max_gen_len", 200)),
        use_form_token=use_form_token,
    )

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    prompts = ["春", "山", "月", "风", "江", "花", "夜", "云"][: max(1, args.num_samples)]
    while len(prompts) < args.num_samples:
        prompts.extend(prompts)
    prompts = prompts[:args.num_samples]

    records = generator.generate_many(
        prompts=prompts,
        strategies=strategies,
        form_ids=[0, 1, 2, 3],
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    sample_rows = [
        {
            "strategy": record.strategy,
            "form_token": record.form_token,
            "form_id": {v: k for k, v in FORM_ID_TO_TOKEN.items()}.get(record.form_token, ""),
            "prompt": record.prompt,
            "poem": record.poem,
        }
        for record in records
    ]

    by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_strategy[str(row["strategy"])].append(row)

    metric_rows: List[Dict[str, Any]] = []
    for strategy, rows in by_strategy.items():
        metrics = aggregate_generation_metrics(rows, train_texts=train_texts)
        metric_rows.append({
            "strategy": strategy,
            "n": len(rows),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in lm_metrics.items()},
            **{k: round(v, 6) for k, v in metrics.items()},
        })

    output_dir = resolve_path(args.output_dir)
    write_samples(os.path.join(output_dir, "samples", "generated_poems.txt"), sample_rows)
    write_csv(os.path.join(output_dir, "samples", "generated_poems.csv"), sample_rows)
    write_csv(os.path.join(output_dir, "metrics", "all_metrics.csv"), metric_rows)

    print(f"[Eval] Wrote samples: {os.path.join(output_dir, 'samples', 'generated_poems.txt')}")
    print(f"[Eval] Wrote metrics: {os.path.join(output_dir, 'metrics', 'all_metrics.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
