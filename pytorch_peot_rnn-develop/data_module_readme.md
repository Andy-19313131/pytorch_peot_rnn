# 数据模块说明文档（Data Module）

> **负责人**：B（数据负责人）  
> **分支**：`feature/data`  
> **核心文件**：`src/data.py`、`src/dataset.py`  
> **交付日期**：2026-07-10

---

## 一、交付物清单

### 1.1 清洗数据（`data/processed/`）

| 文件名 | 说明 | 样本数 | 用途 |
|--------|------|--------|------|
| `all_clean.jsonl` | 全部清洗后的有效唐诗 | 56,469 首 | 通用生成实验 |
| `regulated.jsonl` | 仅标准体裁（绝句/律诗） | 36,319 首 | 体裁控制实验 |
| `vocab.json` | 字符级词表 | 6,403 | 模型训练共用 |

**词表结构**：
- 索引 0-3：`<PAD>`、`<UNK>`、`<SOP>`、`<EOP>`（所有模型共用）
- 索引 4-7：`<5JUE>`、`<7JUE>`、`<5LV>`、`<7LV>`（改进模型体裁控制 token）
- 索引 8+：真实汉字字符（按频率降序排列）

### 1.2 数据划分（`data/splits/`）

| 文件名 | 说明 | 样本数 | 比例 |
|--------|------|--------|------|
| `train.jsonl` | 训练集 | 45,173 首 | 80% |
| `valid.jsonl` | 验证集 | 5,645 首 | 10% |
| `test.jsonl` | 测试集 | 5,651 首 | 10% |
| `train_texts.txt` | 训练集纯文本（每行一首） | 45,173 首 | 供 E 评价模块使用 |

**划分策略**：按 `form_ids` 分层抽样，确保各集合体裁比例一致。  
**重叠检查**：train ∩ valid = 0，train ∩ test = 0，valid ∩ test = 0。

### 1.3 统计分析结果（`results/`）

**图表（`results/figures/`）**：
- `length_distribution.png` — 诗歌长度分布直方图
- `sentence_distribution.png` — 句子数分布柱状图
- `form_distribution.png` — 体裁分布柱状图
- `form_distribution_pie.png` — 体裁分布饼图
- `top_authors.png` — 作者作品数量 Top 20

**表格与报告（`results/tables/`）**：
- `data_statistics.csv` — 完整统计指标
- `data_examples.csv` — 典型样本与噪声样本
- `split_overlap_check.txt` — 数据划分重叠检查报告
- `split_form_distribution.csv` — 划分后各集合体裁分布
- `data_cleaning_rules.md` — 清洗规则完整说明（含接口约定）
- `data_report_summary.md` — 可直接插入论文的数据分析章节

---

## 二、数据清洗流程

### 2.1 原始数据
- **来源**：58 个 `poet.tang.*.json` 文件
- **总量**：57,612 首
- **字段保留**：`title`、`author`、`paragraphs`、`dynasty`，新增 `text`、`form`、`form_ids`

### 2.2 清洗步骤（按执行顺序）

| 步骤 | 操作 | 影响数量 |
|------|------|----------|
| 1. 繁简转换 | OpenCC `t2s` 繁体转简体 | 全部 |
| 2. 去除注释 | 删除括号内注释，如（一作XXX） | — |
| 3. 去除字母数字 | 删除所有英文字母和阿拉伯数字 | — |
| 4. 去除特殊符号 | 删除※★☆■□▲△▼▽◆◇○◎●等 | — |
| 5. 统一标点 | 半角 `, . ? !` 转全角 `，。？！` | — |
| 6. 去重 | 基于纯汉字文本 MD5 哈希 | 删除 1,129 首 |
| 7. 过滤空文本 | 清洗后长度为 0 | 删除 1 首 |
| 8. 过滤过短 | 汉字字符数 < 5 | 删除 13 首 |
| 9. 标记长篇 | 长度 > 200 字 | 标记 1,625 首为"长篇" |
| 10. 体裁标注 | 按句子数/每句字数分类 | 全部标注 |
| 11. form_ids 映射 | 体裁字符串映射为整数 | 全部标注 |

### 2.3 体裁标注规则（form_ids）

去掉标点后：

| 体裁 | 规则 | form_id |
|------|------|---------|
| 五言绝句 | 4 句，每句 5 字 | 0 |
| 七言绝句 | 4 句，每句 7 字 | 1 |
| 五言律诗 | 8 句，每句 5 字 | 2 |
| 七言律诗 | 8 句，每句 7 字 | 3 |
| 长篇 | 长度 > 200 字 | -1 |
| 其他 | 不符合以上规则 | -1 |

---

## 三、数据质量分析摘要

### 3.1 规模与清洗效果
- 原始数据：57,612 首
- 去重后：56,483 首（去重率 2.0%）
- 最终有效：56,469 首
- 标准体裁（绝句/律诗）：36,319 首（64.3%）
- 长篇：1,625 首（2.9%）
- 其他：18,525 首（32.8%）

### 3.2 作者分布
- 不同作者数：3,635 位
- 前 10 位作者占比：19.8%（未超过 25%，不存在极端集中）
- Top 3：白居易（2,984 首）、杜甫（1,479 首）、李白（1,167 首）

### 3.3 体裁分布
- 五言绝句（form_id=0）：3,863 首（6.8%）
- 七言绝句（form_id=1）：10,417 首（18.4%）
- 五言律诗（form_id=2）：14,238 首（25.2%）
- 七言律诗（form_id=3）：7,801 首（13.8%）
- 长篇/其他（form_id=-1）：20,150 首（35.7%）

**注意**：体裁分布不均衡，五言律诗远多于五言绝句，"其他"体裁包含大量杂言诗、乐府、歌行等。

### 3.4 字符词表
- 唯一字符数：8,288
- 出现 1 次的字符（hapax）：1,173 个
- 低频字（≤5 次）：2,635 个（31.8%）
- 存在明显长尾分布，大量生僻字仅出现少数几次

### 3.5 简繁转换信息损失
- 抽样检查 1,000 首，48.2% 在简体→繁体回转时与原繁体不一致
- 一简对多繁：28 种（如"後/后"→"后"，"復/覆"→"复"）
- OpenCC 过度转换：26 种（如"欲"→"慾"，"峰"→"峯"）
- **结论**：对语言模型训练影响有限，但在严格古典文献研究中需注意

### 3.6 已知局限
1. 数据集仅包含唐诗，不能代表宋词、元曲及全部古典文学
2. 流传并被数字化的作品存在历史收录偏差，盛唐诗人作品可能占比偏高
3. 字符级建模难以显式表达词语和典故
4. 本实验已量化数据分布问题，并通过去重、体裁标注和固定划分降低实验偏差；对于历史收录偏差，只进行分析和限制说明，不声称能够完全修正

---

## 四、接口使用说明

### 4.1 快速开始（供 D 训练负责人使用）

```python
from src.dataset import get_dataloader

# 基线模型（不插入体裁 token）
train_loader = get_dataloader(
    jsonl_path='data/splits/train.jsonl',
    batch_size=64,
    max_len=125,
    use_form_token=False,   # 基线模型
    shuffle=True,
    num_workers=0,          # Windows 建议设为 0
)

# 改进模型（自动插入体裁控制 token）
train_loader = get_dataloader(
    jsonl_path='data/splits/train.jsonl',
    batch_size=64,
    max_len=125,
    use_form_token=True,    # 改进模型：序列开头自动插入 <5JUE> 等
    shuffle=True,
    num_workers=0,
)
```

### 4.2 Batch 输出格式（符合接口约定）

每个 batch 为 `dict`，包含以下字段：

```python
batch = {
    "input_ids":      Tensor[B, T],   # 输入字符索引（含 SOP/EOP/PAD，可能含体裁 token）
    "target_ids":     Tensor[B, T],   # 目标字符索引（input_ids 右移一位）
    "attention_mask": Tensor[B, T],   # 1=真实字符，0=PAD
    "form_ids":       Tensor[B],      # 体裁编号（0=五绝, 1=七绝, 2=五律, 3=七律, -1=其他）
}
```

**训练时示例**：

```python
for batch in train_loader:
    input_ids = batch["input_ids"]           # [B, T]
    target_ids = batch["target_ids"]         # [B, T]
    attention_mask = batch["attention_mask"] # [B, T]
    form_ids = batch["form_ids"]             # [B]

    # 模型前向（基线模型忽略 form_ids）
    logits = model(input_ids, attention_mask, form_ids)  # [B, T, vocab_size]

    # 计算损失（忽略 PAD 位置）
    loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
```

### 4.3 手动加载词表

```python
from src.dataset import load_vocab

word2idx, idx2word = load_vocab('data/processed/vocab.json')
print(f"词表大小: {len(word2idx)}")
print(f"PAD={word2idx['<PAD>']}, UNK={word2idx['<UNK>']}, SOP={word2idx['<SOP>']}, EOP={word2idx['<EOP>']}")
```

### 4.4 手动构建词表（如需重新构建）

```python
from src.dataset import build_vocab

build_vocab(
    jsonl_path='data/processed/regulated.jsonl',
    min_freq=1,
    save_path='data/processed/vocab.json'
)
```

### 4.5 验证 batch 格式

```python
from src.dataset import check_batch_format

batch = next(iter(train_loader))
check_batch_format(batch, vocab_size=6403)  # 断言检查，不通过直接报错
```

### 4.6 供 E 评价模块使用

训练集纯文本已导出至 `data/splits/train_texts.txt`，每行一首清洗后的诗歌文本，用于：
- 训练集重合检查（判断生成诗歌是否直接复制训练集）
- 最大 4-gram 重叠率计算
- 生成诗歌唯一性分析

---

## 五、如何运行

### 5.1 环境要求

```bash
pip install numpy pandas matplotlib tqdm opencc-python-reimplemented
pip install torch  # 用于 dataset.py 验证
```

### 5.2 运行数据清洗与分析

```bash
cd src
python data.py
```

运行后会自动：
1. 读取 `data/tang/` 下的原始 JSON 文件
2. 执行清洗、去重、体裁标注
3. 保存 `all_clean.jsonl`、`regulated.jsonl`
4. 分层划分 train/valid/test
5. 生成所有统计图表和报告
6. 自动调用 `dataset.build_vocab()` 生成词表

### 5.3 验证 Dataset 接口

```bash
cd src
python dataset.py
```

运行后会自动：
1. 加载 `regulated.jsonl` 构建/加载词表
2. 测试基线模型 DataLoader（use_form_token=False）
3. 测试改进模型 DataLoader（use_form_token=True）
4. 验证 batch 格式是否符合接口约定
5. 检查 train/valid/test 划分文件

---

## 六、Git 提交记录

```bash
git checkout -b feature/data
git add src/data.py src/dataset.py
git add data/processed/vocab.json
git add data/splits/*.jsonl data/splits/train_texts.txt
git add results/figures/*.png results/tables/*
git commit -m "feat(data): add data cleaning, analysis, dataset and vocab"
git push -u origin feature/data
```

---

## 七、协作说明

### 7.1 与 C（模型负责人）的接口
- `input_ids` 已包含 `<SOP>`（索引 2）和 `<EOP>`（索引 3），模型无需自行添加
- 基线模型：忽略 `form_ids`，序列格式为 `SOP + 内容 + EOP`
- 改进模型：使用 `form_ids`，Dataset 会自动在序列开头插入对应的体裁 token（如 `<5JUE>`）
- 输出 `logits` 的 shape 必须为 `[B, T, vocab_size]`

### 7.2 与 D（训练负责人）的接口
- 直接调用 `get_dataloader()`，无需手动处理 tokenization
- `attention_mask` 已生成，训练时用于忽略 PAD 位置
- `target_ids` 是 `input_ids` 右移一位，可直接用于 `CrossEntropyLoss`
- 词表统一使用 `vocab.json`，基线和改进模型共用同一套索引

### 7.3 与 E（评价负责人）的接口
- 训练集纯文本：`data/splits/train_texts.txt`
- 测试集：`data/splits/test.jsonl`（含 `form_ids` 字段）
- 体裁分布统计：`results/tables/split_form_distribution.csv`
- 数据清洗规则：`results/tables/data_cleaning_rules.md`

---

## 八、文件目录结构

```
project-root/
├── data/
│   ├── tang/                    # 原始 JSON 数据（不提交 Git）
│   ├── processed/
│   │   ├── all_clean.jsonl      # 全部清洗数据
│   │   ├── regulated.jsonl      # 标准体裁数据
│   │   └── vocab.json           # 词表
│   └── splits/
│       ├── train.jsonl          # 训练集
│       ├── valid.jsonl          # 验证集
│       ├── test.jsonl           # 测试集
│       └── train_texts.txt      # 训练集纯文本
├── src/
│   ├── data.py                  # 数据清洗与分析（B 核心文件）
│   └── dataset.py               # PyTorch Dataset + DataLoader（接口实现）
└── results/
    ├── figures/                 # 统计图表（PNG）
    └── tables/                  # 统计表格与报告（CSV/MD/TXT）
```

---

> **注意**：原始数据目录 `data/tang/` 和大型 JSONL 文件体积较大，建议通过 `.gitignore` 排除，仅在 README 中说明获取方式。词表 `vocab.json` 和划分后的数据文件建议提交，确保实验可复现。
