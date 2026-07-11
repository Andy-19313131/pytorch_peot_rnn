# 统一接口约定

> 所有成员在开始编码前必须阅读并遵守本文档。
> 模型负责人（C）和训练负责人（D）不能各自定义一套不同接口。

---

## 1. DataLoader 输出格式

每个 batch 必须返回一个 dict，包含以下字段：

```python
batch = {
    "input_ids":       Tensor, shape [B, T],       # 输入字符索引（含SOP/EOP/PAD）
    "target_ids":      Tensor, shape [B, T],       # 目标字符索引（右移一位）
    "attention_mask":  Tensor, shape [B, T],       # 1=真实字符，0=PAD
    "form_ids":        Tensor, shape [B],           # 体裁编号：0=五绝, 1=七绝, 2=五律, 3=七律
}
```

**说明**:
- `input_ids` 和 `target_ids` 的关系：`target_ids = input_ids 右移一位`（即预测下一个字符）
- `attention_mask` 用于忽略 PAD 位置的损失计算
- `form_ids` 用于体裁条件模型，基线模型可忽略此字段

---

## 2. 模型输入输出

### 基线模型 (Baseline LSTM)

```python
logits = model(
    input_ids=batch["input_ids"],          # [B, T]
    attention_mask=batch["attention_mask"], # [B, T]
)
# 返回: logits, shape [B, T, vocab_size]
```

### 改进模型 (Conditional Attention LSTM)

```python
logits = model(
    input_ids=batch["input_ids"],          # [B, T]
    attention_mask=batch["attention_mask"], # [B, T]
    form_ids=batch["form_ids"],            # [B]
)
# 返回: logits, shape [B, T, vocab_size]
```

**强制约束**:
- 输出 logits 的 shape 必须为 `[B, T, vocab_size]`
- 模型内部不得改变此接口
- 验证方法：`assert logits.shape == (batch_size, seq_len, vocab_size)`

---

## 3. Trainer 接口

```python
class Trainer:
    def __init__(self, model, config, optimizer, scheduler=None):
        ...

    def train(self, train_loader, valid_loader) -> dict:
        """
        训练模型。
        返回: {"best_epoch": int, "best_valid_loss": float, "train_losses": list, "valid_losses": list}
        """
        ...

    def test(self, test_loader) -> dict:
        """
        在测试集上评价。
        返回: {"test_loss": float, "test_ppl": float, "test_acc": float}
        """
        ...

    def load_checkpoint(self, path: str):
        """加载完整 checkpoint（含模型、优化器、scheduler、epoch等）"""
        ...

    def save_checkpoint(self, path: str, epoch: int, best_metric: float):
        """保存完整 checkpoint"""
        ...
```

### Checkpoint 格式

```python
{
    "epoch": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,  # 如果有
    "config": dict,
    "word2idx": dict,
    "idx2word": dict,
    "best_metric": float,
}
```

---

## 4. 评价接口

```python
def evaluate_model(
    model,                    # 已加载的模型
    test_loader,              # 测试集 DataLoader
    generator,                # 生成器对象
    train_texts: list[str],   # 训练集文本列表（用于记忆指标）
    config,                   # 配置
) -> dict:
    """
    返回所有评价指标：
    {
        "test_loss": float,
        "test_ppl": float,
        "test_acc": float,
        "format_correctness": float,       # 格式正确率
        "rhyme_consistency": float,        # 押韵一致率
        "distinct_1": float,               # Distinct-1
        "distinct_2": float,               # Distinct-2
        "repeat_bigram_ratio": float,      # 重复二元组比例
        "adjacent_repeat_rate": float,     # 相邻重复率
        "train_set_overlap": float,        # 与训练集重合比例
        "unique_ratio": float,             # 唯一诗歌比例
    }
    """
    ...
```

---

## 5. 配置文件格式 (YAML)

所有实验配置必须使用 YAML 格式，存放在 `configs/` 目录下。

必须包含的字段：
```yaml
data:
  train_ratio: 0.8
  valid_ratio: 0.1
  test_ratio: 0.1
  max_len: 125

model:
  type: str           # "BaselineLSTM" / "ConditionalAttnLSTM"
  embedding_dim: int
  hidden_dim: int
  num_layers: int

training:
  batch_size: int
  num_epoch: int
  lr: float
  weight_decay: float
  optimizer: str      # "AdamW"
  seed: int
  gradient_clip: float
  early_stopping_patience: int

generation:
  max_gen_len: int
  strategy: str       # "greedy" / "topk" / "topp" / "constrained"
```

---

## 6. 特殊 token 约定

| Token | 索引 | 说明 |
|-------|------|------|
| PAD   | 0    | 填充 |
| UNK   | 1    | 未知字符 |
| SOP   | 2    | 诗歌开始 |
| EOP   | 3    | 诗歌结束 |
| `<5JUE>` | 4 | 五言绝句（改进模型） |
| `<7JUE>` | 5 | 七言绝句（改进模型） |
| `<5LV>`  | 6 | 五言律诗（改进模型） |
| `<7LV>`  | 7 | 七言律诗（改进模型） |

> 注意：基线模型的词表从索引 4 开始为真实字符。改进模型的体裁 token 占用索引 4-7，真实字符从索引 8 开始。

---

## 7. Git 分支和提交规范

### 分支结构
```
main              # 最终稳定版本
develop           # 日常集成版本
archive/original  # 原仓库保留（通过 tag v0.1-original）
feature/baseline  # A组长 - 基线、配置
feature/data      # B - 数据处理
feature/model     # C - 模型
feature/training  # D - 训练
feature/evaluation # E - 评价
```

### 提交信息格式
```
feat(data): add poetry form classification
feat(model): implement conditional LSTM
feat(train): add early stopping
feat(eval): add distinct-n metrics
fix(split): correct train test overlap
fix(loss): ignore padding targets
docs(report): add dataset analysis section
test(model): check output tensor shapes
```

### PR 要求
- 每个 PR 至少由另一名成员 review
- PR 描述需包含：完成内容、测试方法、输出结果、影响文件
