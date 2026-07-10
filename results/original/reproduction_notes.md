# 原始仓库复现说明

## 复现步骤

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装 PyTorch（CPU版，因为没有NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install numpy pandas matplotlib tqdm pyyaml scikit-learn opencc-python-reimplemented pypinyin
```

### 2. 环境信息

| 项目 | 值 |
|------|-----|
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |
| CUDA | 不可用（AMD Radeon RX 9070 GRE，无NVIDIA GPU） |
| 操作系统 | Windows 11 (10.0.26200) |
| Git Commit | 5abd4203d99572b7a18079ad5769ee27ea57c2a4 |
| Git Tag | v0.1-original |

### 3. 数据信息

| 项目 | 值 |
|------|-----|
| 数据总量 | 57,598 条 |
| 词表大小 | 8,840 |
| 最大序列长度 | 125 |
| 数据来源 | data/tang/ 下 58 个 JSON 文件 |
| 预处理 | 繁体→简体（OpenCC）、去注释、去数字 |

### 4. 模型信息

| 项目 | 值 |
|------|-----|
| 模型类型 | PoetryModel2（两层单向 LSTM） |
| Embedding 维度 | 300 |
| 隐藏层维度 | 256 |
| LSTM 层数 | 2 |
| 总参数量 | 6,021,608 |

### 5. 训练结果

| 项目 | 值 |
|------|-----|
| 训练轮数 | 3（复现测试，原配置为20） |
| Batch size | 128 |
| 学习率 | 0.001 |
| 优化器 | Adam（未使用 weight_decay） |
| 训练耗时 | 1,695.5 秒（约28分钟，CPU） |
| 最佳 epoch | 3 |
| 最佳测试损失 | 1,979.30（未归一化，不可直接比较） |

### 6. 生成结果

| 输入 | 生成结果 |
|------|----------|
| 丽日照残春 | 丽日照残春，风风不见春。 |
| 春眠不觉晓 | 春眠不觉晓，春风不见春。 |
| 登高壮观天地间 | 登高壮观天地间，不知一人不可知。 |
| 山色空蒙雨亦奇 | 山色空蒙雨亦奇，一日春风不见春。 |

**观察**: 生成结果存在明显的重复模式（"不见春"、"春风"反复出现），这是贪心解码（topk=1）的典型问题。

### 7. 已发现的问题

详见 `docs/code_audit_report.md`，共发现 7 个主要问题：

1. **严重** - 训练/测试集划分重叠（代码 bug）
2. **严重** - 用测试集选择最佳模型（数据泄漏）
3. **中等** - Padding 掩码不够严谨
4. **中等** - 测试损失未按 token 归一化
5. **轻微** - weight_decay 定义但未使用
6. **中等** - 生成函数依赖全局变量
7. **设计** - 双向 LSTM 存在未来信息泄漏

### 8. 如何复现

```bash
# 确保在项目根目录
cd pytorch_peot_rnn

# 激活虚拟环境
.venv\Scripts\activate  # Windows

# 运行复现脚本
python -X utf8 scripts/reproduce_original.py
```

结果会保存在 `results/original/` 目录下。

### 9. 注意事项

- 原始仓库没有预训练模型文件（checkpoints 目录为空），必须先训练才能推理
- 原始训练使用 CPU 约 28 分钟/epoch（3 epoch），完整 20 epoch 预计约 9.5 小时
- 有 GPU 的情况下训练会快很多
- 原始代码的 `main.py` 中生成函数使用了全局变量 `model`，复现脚本中已改为 `self.model`
