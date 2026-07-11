"""
训练模块 — Task D 训练负责人

支持的训练配置：
  - 原始基线训练（无梯度裁剪、无调度器、无早停）
  - 改进训练（AdamW + 梯度裁剪 + ReduceLROnPlateau + Early Stopping）

统一接口约定参考 docs/interface_spec.md

建议用法：
    from src.trainer import Trainer

    trainer = Trainer(model, config)
    result = trainer.train(train_loader, valid_loader)
    trainer.save_checkpoint("results/checkpoints/best.pt")
    trainer.save_training_curves("results/logs/training_curves.csv")
    test_result = trainer.test(test_loader)
"""

import math
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 尽可能让 cuDNN 行为确定化（可能略微影响速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """训练管理器。

    负责：训练循环、验证、测试、梯度裁剪、学习率调度、早停、checkpoint 读写。

    Parameters
    ----------
    model : nn.Module
        模型实例（BaselineLSTM 或 ConditionalAttnLSTM）。
    config : dict
        实验配置字典（与 configs/*.yaml 结构一致）。
    optimizer : torch.optim.Optimizer, optional
        外部传入的优化器；为 None 时根据 config 自动构建。
    scheduler : torch.optim.lr_scheduler, optional
        外部传入的调度器；为 None 时根据 config 自动构建。
    word2idx : dict, optional
        字符 → 索引映射（保存 checkpoint 时使用）。
    idx2word : dict, optional
        索引 → 字符映射（保存 checkpoint 时使用）。
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        word2idx: Optional[Dict[str, int]] = None,
        idx2word: Optional[Dict[int, str]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.training_cfg = config.get("training", config)

        # ---- device ----
        self.device = self._resolve_device()
        self.model.to(self.device)

        # ---- loss ----
        self.pad_id: int = self.training_cfg.get("pad_id", 0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        # ---- optimizer ----
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()

        # ---- scheduler ----
        self.scheduler = scheduler if scheduler is not None else self._build_scheduler()

        # ---- vocabulary ----
        self.word2idx: Dict[str, int] = word2idx or {}
        self.idx2word: Dict[int, str] = idx2word or {}

        # ---- state ----
        self.current_epoch: int = 0
        self.best_metric: float = float("inf")
        self.train_losses: List[float] = []
        self.valid_losses: List[float] = []

    # ------------------------------------------------------------------
    # 内部构建
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        device_str = self.training_cfg.get("device", "auto")
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.training_cfg.get("optimizer", "AdamW")
        lr = self.training_cfg.get("lr", 1e-3)
        wd = self.training_cfg.get("weight_decay", 0.0)

        if name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        if name == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        # fallback: 尝试 torch.optim 下的任意优化器
        cls = getattr(torch.optim, name, None)
        if cls is None:
            raise ValueError(f"未知的优化器: {name}")
        return cls(self.model.parameters(), lr=lr)

    def _build_scheduler(self) -> Optional[Any]:
        name = self.training_cfg.get("scheduler", None)
        if name is None:
            return None
        if name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.training_cfg.get("scheduler_factor", 0.5),
                patience=self.training_cfg.get("scheduler_patience", 2),
            )
        raise ValueError(f"暂不支持的调度器: {name}")

    # ------------------------------------------------------------------
    # 模型前向（统一接口）
    # ------------------------------------------------------------------

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """调用模型前向，返回 [B, T, vocab_size] logits。

        C 的模型接口约定：
            output, (h_n, c_n) = model(input)   # input 是 [B, T] 的 token ids
            output 的 shape 为 [B, T, vocab_size]

        体裁控制 token 已在 Dataset 层嵌入 input_ids，模型不需要额外的
        attention_mask 或 form_ids 参数。
        """
        input_ids = batch["input_ids"].to(self.device)
        output, _hidden = self.model(input_ids)
        return output

    # ------------------------------------------------------------------
    # 训练 / 验证 / 测试
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader) -> float:
        """训练一个 epoch，返回按有效 token 归一化的平均 NLL。"""
        self.model.train()
        total_nll: float = 0.0
        total_tokens: int = 0

        for batch in train_loader:
            logits = self._forward(batch)  # [B, T, V]
            target_ids = batch["target_ids"].to(self.device)  # [B, T]

            self.optimizer.zero_grad()

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),   # [B*T, V]
                target_ids.reshape(-1),                # [B*T]
            )
            loss.backward()

            # 梯度裁剪
            clip_val = self.training_cfg.get("gradient_clip", None)
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

            self.optimizer.step()

            # 统计：排除 PAD 位置的 token 数
            non_pad = (target_ids != self.pad_id).sum().item()
            total_nll += loss.item() * non_pad
            total_tokens += non_pad

        return total_nll / total_tokens if total_tokens > 0 else float("inf")

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        """在给定数据加载器上计算 loss / ppl / top-1 accuracy。

        返回: {"loss": ..., "ppl": ..., "acc": ...}
        """
        self.model.eval()
        total_nll: float = 0.0
        total_tokens: int = 0
        total_correct: int = 0

        for batch in data_loader:
            logits = self._forward(batch)
            target_ids = batch["target_ids"].to(self.device)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            non_pad_mask = target_ids != self.pad_id
            non_pad = non_pad_mask.sum().item()

            total_nll += loss.item() * non_pad
            total_tokens += non_pad

            # Top-1 accuracy（仅非 PAD 位置）
            preds = logits.argmax(dim=-1)  # [B, T]
            correct = (preds == target_ids) & non_pad_mask
            total_correct += correct.sum().item()

        avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
        ppl = math.exp(avg_nll) if avg_nll < 100.0 else float("inf")
        acc = total_correct / total_tokens if total_tokens > 0 else 0.0

        return {"loss": avg_nll, "ppl": ppl, "acc": acc}

    def train(
        self,
        train_loader,
        valid_loader,
    ) -> Dict[str, Any]:
        """完整训练流程，含早停和学习率调度。

        Returns
        -------
        dict
            {"best_epoch": int, "best_valid_loss": float,
             "train_losses": List[float], "valid_losses": List[float]}
        """
        num_epoch = self.training_cfg.get("num_epoch", 30)
        patience = self.training_cfg.get("early_stopping_patience", 0)  # 0 表示不早停

        best_valid_loss = float("inf")
        best_epoch = 0
        no_improve = 0

        # 重置记录
        self.train_losses = []
        self.valid_losses = []

        for epoch in range(1, num_epoch + 1):
            self.current_epoch = epoch

            # ---- 训练一轮 ----
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # ---- 验证 ----
            valid_result = self.evaluate(valid_loader)
            valid_loss = valid_result["loss"]
            self.valid_losses.append(valid_loss)

            # ---- 学习率调度 ----
            if self.scheduler is not None:
                self.scheduler.step(valid_loss)

            # ---- 日志 ----
            print(
                f"Epoch {epoch:3d}/{num_epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"Valid PPL: {valid_result['ppl']:.2f} | "
                f"Valid Acc: {valid_result['acc']:.4f}"
            )

            # ---- 早停判定 ----
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(
                        f"Early stopping: 验证损失连续 {no_improve} 轮未改善，"
                        f"最佳 epoch={best_epoch}, loss={best_valid_loss:.4f}"
                    )
                    break
                if patience > 0:
                    print(
                        f"  No improvement for {no_improve}/{patience} epoch(s). "
                        f"Best: epoch {best_epoch}, loss {best_valid_loss:.4f}"
                    )

        self.best_metric = best_valid_loss
        return {
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
        }

    @torch.no_grad()
    def test(self, test_loader) -> Dict[str, float]:
        """在测试集上评价（仅限最终模型，不得用于模型选择）。

        Returns
        -------
        dict
            {"test_loss": float, "test_ppl": float, "test_acc": float}
        """
        result = self.evaluate(test_loader)
        print(
            f"Test  | Loss: {result['loss']:.4f} | "
            f"PPL: {result['ppl']:.2f} | "
            f"Acc: {result['acc']:.4f}"
        )
        return {
            "test_loss": result["loss"],
            "test_ppl": result["ppl"],
            "test_acc": result["acc"],
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        best_metric: Optional[float] = None,
    ) -> None:
        """保存完整 checkpoint（含模型、优化器、调度器、配置、词表、训练曲线）。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {
            "epoch": epoch if epoch is not None else self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "config": self.config,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "best_metric": best_metric
            if best_metric is not None
            else self.best_metric,
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """从 checkpoint 恢复训练状态。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        sd = checkpoint.get("scheduler_state_dict")
        if sd is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(sd)

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", float("inf"))
        self.word2idx = checkpoint.get("word2idx", {})
        self.idx2word = checkpoint.get("idx2word", {})
        self.train_losses = checkpoint.get("train_losses", [])
        self.valid_losses = checkpoint.get("valid_losses", [])

        print(f"Checkpoint loaded ← {path}  (epoch {self.current_epoch})")
        return checkpoint

    # ------------------------------------------------------------------
    # 训练曲线导出
    # ------------------------------------------------------------------

    def save_training_curves(self, path: str) -> None:
        """将训练 / 验证损失导出为 CSV。

        输出三列：epoch, train_loss, valid_loss
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "valid_loss"])
            for i, (tl, vl) in enumerate(
                zip(self.train_losses, self.valid_losses), start=1
            ):
                writer.writerow([i, tl, vl])
        print(f"Training curves saved → {path}")


# ---------------------------------------------------------------------------
# 快速入口（便于 scripts/train.py 直接调用）
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    config: Dict[str, Any],
    train_loader,
    valid_loader,
    test_loader,
    checkpoint_dir: str = "results/checkpoints",
    log_dir: str = "results/logs",
    word2idx: Optional[Dict[str, int]] = None,
    idx2word: Optional[Dict[int, str]] = None,
) -> Tuple[Trainer, Dict[str, Any], Dict[str, float]]:
    """一键训练→保存最优模型→测试→导出曲线。

    Returns
    -------
    (trainer, train_result, test_result)
    """
    # 固定种子
    seed = config.get("training", config).get("seed", 2026)
    set_seed(seed)

    trainer = Trainer(
        model=model,
        config=config,
        word2idx=word2idx,
        idx2word=idx2word,
    )

    # 训练
    print("\n" + "=" * 55)
    print("开始训练".center(55))
    print("=" * 55)
    train_result = trainer.train(train_loader, valid_loader)

    # 保存最佳模型
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.save_checkpoint(
        os.path.join(checkpoint_dir, "best_model.pt"),
        epoch=train_result["best_epoch"],
        best_metric=train_result["best_valid_loss"],
    )

    # 导出训练曲线
    os.makedirs(log_dir, exist_ok=True)
    trainer.save_training_curves(os.path.join(log_dir, "training_curves.csv"))

    # 最终测试
    print("\n" + "=" * 55)
    print("测试集最终评价".center(55))
    print("=" * 55)
    trainer.load_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"))
    test_result = trainer.test(test_loader)

    return trainer, train_result, test_result
