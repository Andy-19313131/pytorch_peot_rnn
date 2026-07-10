# 基于数据质量分析、体裁条件与因果注意力循环神经网络的古诗生成研究

> 深度学习课程实验项目

## 项目概述

本项目基于字符级两层 LSTM 实现唐诗生成，并在此基础上进行数据质量分析、模型改进和多维评价。

原始仓库: [taishan1994/pytorch_peot_rnn](https://github.com/taishan1994/pytorch_peot_rnn)

## 项目结构

```
poetry-rnn-project/
├─ configs/                     # 实验配置
│  ├─ baseline.yaml             # 原始仓库配置
│  ├─ corrected_baseline.yaml   # 修正基线配置
│  └─ improved_model.yaml       # 改进模型配置
│
├─ data/
│  ├─ raw/                      # 原始数据（不提交到Git）
│  ├─ processed/                # 清洗后数据
│  └─ splits/                   # 训练/验证/测试集划分
│
├─ src/                         # 源代码（阶段二开始填充）
│  ├─ models/
│  │  ├─ baseline_lstm.py
│  │  └─ conditional_attn_lstm.py
│  ├─ trainer.py
│  ├─ generate.py
│  ├─ metrics.py
│  └─ utils.py
│
├─ scripts/                     # 运行脚本
│  ├─ reproduce_original.py     # 原仓库复现脚本
│  ├─ analyze_data.py
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ visualize.py
│
├─ results/
│  ├─ original/                 # 原仓库复现结果
│  ├─ metrics/
│  ├─ samples/
│  ├─ tables/
│  ├─ figures/
│  ├─ logs/
│  └─ checkpoints/
│
├─ tests/                       # 单元测试
├─ docs/                        # 文档
│  ├─ code_audit_report.md      # 代码审计报告
│  ├─ interface_spec.md         # 统一接口约定
│  ├─ sections/
│  └─ report_assets/
│
├─ main.py                      # 原始主程序（保留）
├─ config.py                    # 原始配置（保留）
├─ model.py                     # 原始模型（保留）
├─ process.py                   # 原始数据处理（保留）
├─ utils.py                     # 原始工具函数（保留）
├─ requirements.txt
├─ requirements-lock.txt
└─ .gitignore
```

## 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell

# 安装 PyTorch（根据是否有NVIDIA GPU选择）
# CPU版:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# GPU版（参考 https://pytorch.org/get-started/locally/）:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

## 复现原始仓库

```bash
python -X utf8 scripts/reproduce_original.py
```

结果保存在 `results/original/` 目录下。

## 团队分工

| 成员 | 角色 | 主要职责 | Git 分支 |
|------|------|----------|----------|
| A（组长） | 基线/整合 | 原仓库复现、代码审计、项目整合 | feature/baseline |
| B | 数据 | 数据统计、清洗、去重、体裁标注 | feature/data |
| C | 模型 | 修正基线、体裁条件模型、注意力模型 | feature/model |
| D | 训练 | Trainer、优化器、调度器、早停 | feature/training |
| E | 评价 | 解码、自动指标、人工评价、可视化 | feature/evaluation |

## 版本管理

- `main` — 最终稳定版本
- `develop` — 日常集成版本
- `v0.1-original` — 原始仓库版本（git tag）
- 每位成员从 `develop` 创建 `feature/xxx` 分支

## 参考

> https://github.com/chenyuntc/pytorch-book
