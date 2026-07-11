#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练入口脚本 — Task D 训练负责人

串联 B（数据）、C（模型）、D（训练器）三模块完成端到端训练。

一次典型运行：
    # 基线模型（E1：修正基线）
    python scripts/train.py --config configs/baseline.yaml

    # 改进模型（E3 / E4）
    python scripts/train.py --config configs/improved_model.yaml

    # 覆盖 GPU 或 epoch 数
    python scripts/train.py --config configs/improved_model.yaml --device cuda:0 --epochs 50

与各模块的对接点
──────────────────────────
  B (dataset.py)
    get_dataloader(jsonl_path, vocab_path, batch_size, max_len, use_form_token, ...)
    load_vocab(vocab_path) → word2idx, idx2word

  C (models/*.py)
    BaselineLSTM(vocab_size, embedding_dim, hidden_dim, ...)
    ConditionalLSTM(vocab_size, embedding_dim, hidden_dim, ...)
    ConditionalAttnLSTM(vocab_size, embedding_dim, hidden_dim, attention_heads, ...)
    统一 forward: output, (h_n, c_n) = model(input)   # input: [B, T]

  D (trainer.py)
    Trainer(model, config, word2idx=..., idx2word=...)
    trainer.train(train_loader, valid_loader) → {"best_epoch", "best_valid_loss", ...}
    trainer.save_checkpoint(path)
    trainer.test(test_loader) → {"test_loss", "test_ppl", "test_acc"}
    trainer.save_training_curves(path) → CSV
"""

import argparse
import inspect
import os
import sys
from typing import Any, Dict, Optional, Tuple, Type

import torch

# ── 把项目根目录加入 sys.path（确保 scripts/ 下运行也能 import src.*）──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.trainer import Trainer, set_seed
from src.dataset import get_dataloader, load_vocab
from src.models import BaselineLSTM, ConditionalLSTM, ConditionalAttnLSTM


# ============================================================================
# 模型注册表
# ============================================================================
# 每个条目记录：(模型类, 是否需要体裁 token)
#   基线模型不需要体裁 token，序列直接以 SOP 开头
#   条件模型需要——Dataset 会在 SOP 前自动插入 <5JUE>/<7JUE>/…

_MODEL_REGISTRY: Dict[str, Tuple[Type[torch.nn.Module], bool]] = {
    "BaselineLSTM":         (BaselineLSTM,         False),
    "ConditionalLSTM":      (ConditionalLSTM,      True),
    "ConditionalAttnLSTM":  (ConditionalAttnLSTM,  True),
    # ── 兼容旧配置中的历史名称 ──
    "PoetryModel2":         (BaselineLSTM,         False),
}


def _resolve_model_type(raw: str) -> str:
    """将配置中的 model.type 规范化为注册表中的键名。"""
    if raw in _MODEL_REGISTRY:
        return raw
    # 尝试大小写不敏感匹配
    lower = raw.lower()
    for key in _MODEL_REGISTRY:
        if key.lower() == lower:
            return key
    raise KeyError(
        f"未知的模型类型: '{raw}'。支持的类型: {sorted(_MODEL_REGISTRY.keys())}"
    )


# ============================================================================
# 配置加载
# ============================================================================

def _load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。优先使用 yaml 库，其次尝试 OmegaConf。"""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    raise ImportError(
        "需要 PyYAML 或 OmegaConf 来读取配置文件。请运行:\n"
        "  pip install pyyaml"
    )


def _resolve_config_path(raw: str) -> str:
    """解析配置文件路径，支持相对路径和绝对路径。"""
    if os.path.isabs(raw):
        return raw
    # 先尝试相对于工作目录
    if os.path.exists(raw):
        return os.path.abspath(raw)
    # 再尝试相对于项目根目录
    candidate = os.path.join(_PROJECT_ROOT, raw)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"找不到配置文件: '{raw}' (工作目录) 或 '{candidate}' (项目根目录)"
    )


# ============================================================================
# 模型构建
# ============================================================================

def build_model(
    model_type: str,
    vocab_size: int,
    model_cfg: Dict[str, Any],
) -> torch.nn.Module:
    """根据配置字典实例化模型。

    Parameters
    ----------
    model_type : str
        注册表中的模型类型名。
    vocab_size : int
        从实际词表获取的 vocab 大小（覆盖配置中的 vocab_size: null）。
    model_cfg : dict
        配置文件中的 ``model`` 段。

    Returns
    -------
    model : nn.Module
    """
    model_cls, _ = _MODEL_REGISTRY[model_type]

    # 收集构造参数：忽略 type / vocab_size（由外部注入）/ form_tokens / vocab_size
    init_params = inspect.signature(model_cls.__init__).parameters
    kwargs: Dict[str, Any] = {}
    for key, value in model_cfg.items():
        if key in ("type", "vocab_size", "form_tokens"):
            continue
        if key in init_params:
            kwargs[key] = value

    kwargs["vocab_size"] = vocab_size

    model = model_cls(**kwargs)
    print(f"[Model] {model_type} 已构建")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {total:,} total / {trainable:,} trainable")
    return model


# ============================================================================
# 数据流水线
# ============================================================================

def build_dataloaders(
    config: Dict[str, Any],
    use_form_token: bool,
) -> Tuple[Any, Any, Any, Dict[str, int], Dict[int, str]]:
    """创建 train / valid / test DataLoader 并返回词表。

    Parameters
    ----------
    config : dict
        完整配置。
    use_form_token : bool
        是否在输入序列中插入体裁控制 token。

    Returns
    -------
    train_loader, valid_loader, test_loader, word2idx, idx2word
    """
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", config)

    batch_size = training_cfg.get("batch_size", 64)
    max_len = data_cfg.get("max_len", 125)

    # ── 路径 ──
    splits_dir = os.path.join(_PROJECT_ROOT, "data", "splits")
    vocab_path = os.path.join(_PROJECT_ROOT, "data", "processed", "vocab.json")

    train_path = os.path.join(splits_dir, "train.jsonl")
    valid_path = os.path.join(splits_dir, "valid.jsonl")
    test_path = os.path.join(splits_dir, "test.jsonl")

    # 检查文件存在
    for path, label in [
        (train_path, "训练集"), (valid_path, "验证集"), (test_path, "测试集"),
        (vocab_path, "词表"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label}文件不存在: {path}\n"
                f"请确认 B（数据负责人）已运行 src/data.py 生成数据划分。"
            )

    # ── 词表 ──
    word2idx, idx2word = load_vocab(vocab_path)
    vocab_size = len(word2idx)
    print(f"[Data] 词表大小: {vocab_size}")

    # ── DataLoader ──
    # Windows 上 num_workers > 0 可能导致死锁，检测后自动设为 0
    num_workers = training_cfg.get("num_workers", 0)
    if os.name == "nt" and num_workers > 0:
        print("[Data] Windows 检测到，num_workers 自动降为 0")
        num_workers = 0

    pin_memory = training_cfg.get("pin_memory", torch.cuda.is_available())
    # valid/test 不应 drop_last，否则会丢失评估样本
    drop_last_train = training_cfg.get("drop_last", True)

    train_loader = get_dataloader(
        jsonl_path=train_path,
        word2idx=word2idx,
        batch_size=batch_size,
        max_len=max_len,
        use_form_token=use_form_token,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last_train,
        pin_memory=pin_memory,
    )

    valid_loader = get_dataloader(
        jsonl_path=valid_path,
        word2idx=word2idx,
        batch_size=batch_size,
        max_len=max_len,
        use_form_token=use_form_token,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    test_loader = get_dataloader(
        jsonl_path=test_path,
        word2idx=word2idx,
        batch_size=batch_size,
        max_len=max_len,
        use_form_token=use_form_token,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    print(
        f"[Data] DataLoader 已就绪: "
        f"train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}, "
        f"test={len(test_loader.dataset)}, "
        f"batch_size={batch_size}, use_form_token={use_form_token}"
    )

    return train_loader, valid_loader, test_loader, word2idx, idx2word


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="诗歌生成模型训练入口 — Task D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/train.py --config configs/baseline.yaml
  python scripts/train.py --config configs/improved_model.yaml --device cuda:0
  python scripts/train.py --config configs/improved_model.yaml --epochs 50 --batch_size 32
        """,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML 配置文件路径（如 configs/baseline.yaml）",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="训练设备（覆盖配置文件中的 device，如 cuda:0 / cpu）",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖配置中的训练轮数",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="覆盖配置中的批次大小",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="覆盖配置中的学习率",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Checkpoint 保存目录（覆盖配置）",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="日志/CSV 保存目录（覆盖配置）",
    )
    parser.add_argument(
        "--no_test", action="store_true",
        help="跳过最终测试集评价（调试用）",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ──
    config_path = _resolve_config_path(args.config)
    config = _load_yaml(config_path)
    print(f"[Config] 已加载: {config_path}")

    # ── 2. 解析模型类型 → 获取 use_form_token ──
    model_cfg = config.get("model", {})
    model_type_raw = model_cfg.get("type", "BaselineLSTM")
    model_type = _resolve_model_type(model_type_raw)
    _, use_form_token = _MODEL_REGISTRY[model_type]
    print(f"[Config] 模型类型: {model_type}, use_form_token={use_form_token}")

    # ── 3. 命令行参数覆盖配置（必须在 DataLoader 创建前处理 batch_size 等）──
    training_cfg = config.get("training", config)
    if args.device is not None:
        training_cfg["device"] = args.device
    if args.epochs is not None:
        training_cfg["num_epoch"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        training_cfg["lr"] = args.lr
    if "training" not in config:
        config["training"] = training_cfg

    # ── 4. 构建 DataLoader（含词表加载）──
    train_loader, valid_loader, test_loader, word2idx, idx2word = \
        build_dataloaders(config, use_form_token)

    vocab_size = len(word2idx)

    # ── 5. 构建模型 ──
    # 把实际 vocab_size 写回 config（checkpoint 可复现时需要）
    model_cfg["vocab_size"] = vocab_size
    model = build_model(model_type, vocab_size, model_cfg)

    # ── 6. 解析输出目录 ──
    paths_cfg = config.get("paths", {})
    checkpoint_dir = args.checkpoint_dir or paths_cfg.get(
        "checkpoint_dir",
        os.path.join(_PROJECT_ROOT, "results", "checkpoints"),
    )
    log_dir = args.log_dir or paths_cfg.get(
        "log_dir",
        os.path.join(_PROJECT_ROOT, "results", "logs"),
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 生成带模型名 + 时间戳的文件名，避免多次运行互相覆盖
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_type.lower()
    checkpoint_path = os.path.join(checkpoint_dir, f"{safe_name}_{timestamp}_best.pt")
    curves_path = os.path.join(log_dir, f"{safe_name}_{timestamp}_curves.csv")

    # ── 7. 固定种子 ──
    seed = training_cfg.get("seed", 2026)
    set_seed(seed)
    print(f"[Seed] {seed}")

    # ── 8. 训练 ──
    trainer = Trainer(
        model=model,
        config=config,
        word2idx=word2idx,
        idx2word=idx2word,
    )

    print("\n" + "=" * 60)
    print(f"  {model_type} — 开始训练")
    print("=" * 60)
    print(f"  设备:     {trainer.device}")
    print(f"  优化器:   {training_cfg.get('optimizer', 'Adam')}")
    print(f"  学习率:   {training_cfg.get('lr', 1e-3)}")
    print(f"  Epochs:   {training_cfg.get('num_epoch', 30)}")
    print(f"  早停:     {training_cfg.get('early_stopping_patience', '关闭')}")
    print(f"  梯度裁剪: {training_cfg.get('gradient_clip', '关闭')}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60 + "\n")

    train_result = trainer.train(train_loader, valid_loader)

    # ── 9. 保存 ──
    trainer.save_checkpoint(
        checkpoint_path,
        epoch=train_result["best_epoch"],
        best_metric=train_result["best_valid_loss"],
    )
    trainer.save_training_curves(curves_path)

    # ── 10. 测试 ──
    if not args.no_test:
        print("\n" + "=" * 60)
        print("  测试集最终评价")
        print("=" * 60)
        # 重新加载最优 checkpoint
        trainer.load_checkpoint(checkpoint_path)
        test_result = trainer.test(test_loader)

        print("\n" + "=" * 60)
        print("  训练完成！")
        print("=" * 60)
        print(f"  最优 Epoch:    {train_result['best_epoch']}")
        print(f"  最优 Valid Loss: {train_result['best_valid_loss']:.4f}")
        print(f"  Test Loss:     {test_result['test_loss']:.4f}")
        print(f"  Test PPL:      {test_result['test_ppl']:.2f}")
        print(f"  Test Acc:      {test_result['test_acc']:.4f}")
        print(f"  Checkpoint:    {checkpoint_path}")
        print(f"  Curves CSV:    {curves_path}")
    else:
        print("\n训练完成（已跳过测试集评价）。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
