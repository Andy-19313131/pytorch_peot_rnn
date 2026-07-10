# 原始仓库代码审计报告

## 审计对象

仓库: taishan1994/pytorch_peot_rnn (Fork: Andy-19313131/pytorch_peot_rnn)
Commit: 5abd4203d99572b7a18079ad5769ee27ea57c2a4
Git Tag: v0.1-original

---

## 问题 1（严重）：训练集和测试集划分重叠

**文件**: `main.py` 第 21-22 行

**原始代码**:
```python
train_data = data[:train_total]
test_data = data[:train_total]  # BUG: 应为 data[train_total:]
```

**问题描述**: 测试集取的是训练集相同的前 80% 数据，而非后 20%。虽然打印信息显示"测试集xxx条"数量正确，但实际数据完全重叠。

**影响**:
- 测试损失不可信（评估的是训练过的数据）
- 模型选择存在数据泄漏
- 无法真实评价泛化能力

**修复方案**:
```python
test_data = data[train_total:]
```

---

## 问题 2（严重）：用测试集选择最佳模型

**文件**: `main.py` 第 64-69 行

**原始代码**:
```python
if self.config.do_test:
    test_loss = self.test(test_loader)
    if test_loss < best_test_loss:
        torch.save(self.model.state_dict(), self.config.save_path)
        best_test_loss = test_loss
```

**问题描述**: 每轮训练后根据测试损失保存最佳模型，测试集参与了模型选择。

**影响**:
- 测试集信息泄漏到模型选择过程
- 最终报告的测试指标过于乐观

**修复方案**: 划分出独立的验证集用于模型选择，测试集仅在最终评价时使用一次。
```
训练集 80% / 验证集 10% / 测试集 10%
```

---

## 问题 3（中等）：Padding 掩码不够严谨

**文件**: `main.py` 第 49-51 行

**原始代码**:
```python
active = (input > 0).view(-1)
active_output = output[active]
active_target = target.contiguous().view(-1)[active]
```

**问题描述**: 使用 `input > 0` 来判断非 PAD 位置，但 PAD 的索引恰好是 0。这种方式在以下情况可能出错：
- 如果词表中其他字符的索引也恰好为 0（不太可能但不够鲁棒）
- 掩码基于 input 而非 target，当 input 和 target 的 PAD 位置不一致时会产生问题

**修复方案**: 使用 PyTorch 原生的 `ignore_index`：
```python
criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD_ID = 0
loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
```

---

## 问题 4（中等）：测试损失未按 token 归一化

**文件**: `main.py` 第 74-88 行

**原始代码**:
```python
def test(self, test_loader):
    self.model.eval()
    total_loss = 0.
    with torch.no_grad():
        for test_step, test_data in enumerate(test_loader):
            ...
            total_loss = total_loss + loss.item()
    return total_loss
```

**问题描述**: 测试损失是各 batch 损失的直接相加，没有除以 batch 数，也没有按非 PAD token 数统计。

**影响**:
- 数据量不同时不能直接比较
- batch size 改变后数值会改变
- 无法严谨地计算困惑度 (perplexity)

**修复方案**:
```python
total_nll = 0.0
total_tokens = 0
for batch in loader:
    loss = criterion(...)
    total_nll += loss.item() * non_pad_count
    total_tokens += non_pad_count
avg_nll = total_nll / total_tokens
perplexity = math.exp(avg_nll)
```

---

## 问题 5（轻微）：weight_decay 定义但未使用

**文件**: `config.py` 第 16 行 / `main.py` 第 36 行

**配置中定义**:
```python
self.weight_decay = 1e-4
```

**实际优化器**:
```python
optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
```

**问题描述**: `weight_decay` 参数在配置中定义了，但没有传入优化器。

**修复方案**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay
)
```

---

## 问题 6（中等）：生成函数依赖全局变量

**文件**: `main.py` 第 107、112、149、153 行

**原始代码**:
```python
def generate(self, start_words, prefix_words=None):
    ...
    output, hidden = model(input, hidden)  # 使用全局 model
    ...

def gen_acrostic(self, start_words, prefix_words=None):
    ...
    output, hidden = model(input, hidden)  # 使用全局 model
```

**问题描述**: `Trainer.generate()` 和 `gen_acrostic()` 内部调用的是全局变量 `model`，而非 `self.model`。这意味着 Trainer 只能在主程序中恰好存在全局 `model` 变量时正常运行，无法独立使用或测试。

**修复方案**: 全部改为 `self.model(input, hidden)`，并在生成前调用 `self.model.eval()`。

---

## 问题 7（设计问题）：双向 LSTM 不能直接用于自回归生成

**文件**: `model.py` 第 7-34 行

**原始代码**:
```python
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        ...
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim,
                            num_layers=1, batch_first=True,
                            bidirectional=True)  # 双向 LSTM
```

**问题描述**: 仓库定义了双向 LSTM 模型 `PoetryModel`，但主程序实际使用的是单向的 `PoetryModel2`。如果将双向 LSTM 作为改进方案，存在未来信息泄漏问题：
- 训练时：反向分支能看到当前位置之后的输入（包含待预测字符）
- 生成时：没有未来字符可用
- 训练和生成不一致

**结论**: 不应将双向 LSTM 作为主要改进。报告中应指出此问题。

---

## 其他发现

### 8. 数据处理的额外问题

**文件**: `process.py`

- 繁体转简体使用 OpenCC，但转换可能丢失部分字形信息
- 数据只保存为 `peot.txt` 纯文本，丢失了作者、题目等元数据
- `pad_sequences` 使用 post-padding，但掩码检查基于 `input > 0`，对 post-padding 来说 PAD 在序列末尾，与 `input > 0` 的判断一致，但不够直观

### 9. 模型输出形状问题

**文件**: `model.py` 第 60 行

```python
output = self.linear1(output.contiguous().view(seq_len * batch_size, -1))
```

输出形状为 `[seq_len * batch_size, vocab_size]，而非标准的 [batch_size, seq_len, vocab_size]`。这在后续处理中需要额外的 reshape 操作，容易出错。

---

## 总结

| 编号 | 严重程度 | 问题 | 影响 |
|------|----------|------|------|
| 1 | 严重 | 训练/测试集重叠 | 评估完全不可信 |
| 2 | 严重 | 测试集选模型 | 数据泄漏 |
| 3 | 中等 | Padding 掩码不严谨 | 潜在计算错误 |
| 4 | 中等 | 损失未按 token 归一化 | 指标不可比 |
| 5 | 轻微 | weight_decay 未使用 | 缺少正则化 |
| 6 | 中等 | 生成依赖全局变量 | 代码不可移植 |
| 7 | 设计 | 双向 LSTM 信息泄漏 | 训练/生成不一致 |

问题 1-2 属于实验正确性问题，必须修复。问题 3-6 属于代码质量问题，建议修复。问题 7 属于设计问题，需要在报告中分析说明。
