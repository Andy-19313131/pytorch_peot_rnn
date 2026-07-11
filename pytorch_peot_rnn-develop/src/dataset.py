# src/dataset.py
# 数据负责人：PyTorch Dataset 与 DataLoader 实现（接口兼容版 v1.0）
# 功能：
#   1. 从清洗后的 jsonl 构建/加载词表（word2idx / idx2word）
#   2. PoetryDataset：输出符合接口约定的 batch dict
#   3. get_dataloader：工厂函数，供训练负责人（D）直接调用
#   4. 支持基线模型（use_form_token=False）和改进模型（use_form_token=True）

import os
import json
import collections
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')

# ==================== 接口约定：特殊 token ====================
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<SOP>': 2,
    '<EOP>': 3,
    '<5JUE>': 4,
    '<7JUE>': 5,
    '<5LV>': 6,
    '<7LV>': 7,
}

# 体裁 token 映射（form_id -> token_id）
FORM_ID_TO_TOKEN_ID = {
    0: 4,   # 五言绝句 -> <5JUE>
    1: 5,   # 七言绝句 -> <7JUE>
    2: 6,   # 五言律诗 -> <5LV>
    3: 7,   # 七言律诗 -> <7LV>
}

# 反向映射
TOKEN_ID_TO_FORM_TOKEN = {v: k for k, v in SPECIAL_TOKENS.items() if k.startswith('<') and k[1].isdigit()}


def build_vocab(jsonl_path, min_freq=1, save_path=None):
    """
    从清洗后的 jsonl 文件构建词表。

    词表结构：
        0-3:  PAD, UNK, SOP, EOP（所有模型共用）
        4-7:  <5JUE>, <7JUE>, <5LV>, <7LV>（改进模型体裁控制 token）
        8+:   真实汉字字符，按频率降序排列

    参数:
        jsonl_path: 清洗后的数据文件路径（如 all_clean.jsonl 或 regulated.jsonl）
        min_freq: 字符最小出现频率，低于此频率的字符用 <UNK> 代替
        save_path: 词表保存路径，默认 PROCESSED_DIR/vocab.json

    返回:
        word2idx: dict, 字符 -> 索引
        idx2word: dict, 索引 -> 字符
    """
    print(f"[build_vocab] 从 {jsonl_path} 构建词表...")
    counter = collections.Counter()
    total_lines = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get('text', '')
            counter.update(list(text))
            total_lines += 1

    print(f"  读取 {total_lines} 首诗歌，原始唯一字符数: {len(counter)}")

    # 初始化词表：特殊 token 在前
    vocab = list(SPECIAL_TOKENS.keys())

    # 按频率降序添加真实字符
    for char, freq in counter.most_common():
        if freq >= min_freq and char not in vocab:
            vocab.append(char)

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    print(f"  词表大小: {len(vocab)} (特殊 token: {len(SPECIAL_TOKENS)}, 真实字符: {len(vocab) - len(SPECIAL_TOKENS)})")
    print(f"  低频过滤(<{min_freq}): 丢弃 {len([c for c,f in counter.items() if f < min_freq])} 个字符")

    if save_path is None:
        save_path = os.path.join(PROCESSED_DIR, 'vocab.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'word2idx': word2idx,
            'idx2word': {str(k): v for k, v in idx2word.items()},  # JSON key 必须是字符串
            'special_tokens': SPECIAL_TOKENS,
            'vocab_size': len(vocab),
            'real_char_start': 8,  # 真实字符起始索引
            'source_file': jsonl_path,
            'min_freq': min_freq,
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ 词表已保存: {save_path}")
    return word2idx, idx2word


def load_vocab(vocab_path=None):
    """
    加载已保存的词表。

    返回:
        word2idx: dict
        idx2word: dict（key 为 int）
    """
    if vocab_path is None:
        vocab_path = os.path.join(PROCESSED_DIR, 'vocab.json')

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"词表文件不存在: {vocab_path}\n"
            f"请先运行 build_vocab() 构建词表，或确认 data/processed/vocab.json 已存在。"
        )

    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    word2idx = data['word2idx']
    # JSON 加载后 idx2word 的 key 是字符串，转回 int
    idx2word = {int(k): v for k, v in data['idx2word'].items()}

    print(f"✓ 加载词表: {vocab_path} (大小: {data.get('vocab_size', len(word2idx))})")
    return word2idx, idx2word


class PoetryDataset(Dataset):
    """
    符合接口约定的 PyTorch Dataset。

    输出 batch dict:
        {
            "input_ids":      Tensor[B, T],   # 输入字符索引（含 SOP/EOP/PAD，可能含体裁 token）
            "target_ids":     Tensor[B, T],   # 目标字符索引（input_ids 右移一位）
            "attention_mask": Tensor[B, T],   # 1=真实字符，0=PAD
            "form_ids":       Tensor[B],      # 体裁编号（0=五绝, 1=七绝, 2=五律, 3=七律, -1=其他）
        }

    参数:
        jsonl_path: 数据文件路径
        word2idx: 词表字典
        max_len: 最大序列长度（默认 125，与接口约定一致）
        use_form_token: 是否在序列开头插入体裁控制 token（改进模型用）
        seed: 随机种子（用于调试，实际训练时由 DataLoader shuffle 控制顺序）
    """

    def __init__(self, jsonl_path, word2idx, max_len=125, use_form_token=False, seed=42):
        super().__init__()
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.word2idx = word2idx
        self.max_len = max_len
        self.use_form_token = use_form_token
        self.seed = seed

        # 特殊 token 索引
        self.pad_id = word2idx.get('<PAD>', 0)
        self.unk_id = word2idx.get('<UNK>', 1)
        self.sop_id = word2idx.get('<SOP>', 2)
        self.eop_id = word2idx.get('<EOP>', 3)

        print(f"[PoetryDataset] 加载 {len(self.data)} 首诗歌")
        print(f"  max_len={max_len}, use_form_token={use_form_token}")

    def __len__(self):
        return len(self.data)

    def _text_to_ids(self, text):
        """将文本转为 token id 列表"""
        return [self.word2idx.get(ch, self.unk_id) for ch in text]

    def __getitem__(self, idx):
        poem = self.data[idx]
        text = poem.get('text', '')
        form_id = poem.get('form_ids', -1)

        # 1. 文本转 token ids
        char_ids = self._text_to_ids(text)

        # 2. 构建序列：SOP + [体裁 token] + 内容 + EOP
        # 预留 2 个位置给 SOP 和 EOP（如果 use_form_token，再加 1 个体裁 token）
        reserved = 2 + (1 if self.use_form_token else 0)
        max_content_len = self.max_len - reserved

        # 截断内容
        char_ids = char_ids[:max_content_len]

        token_ids = [self.sop_id]

        # 改进模型：插入体裁控制 token
        if self.use_form_token:
            if form_id in FORM_ID_TO_TOKEN_ID:
                token_ids.append(FORM_ID_TO_TOKEN_ID[form_id])
            else:
                # 对于长篇/其他，不插入体裁 token（或插入 UNK）
                # 这里选择不插入，保持序列长度一致
                pass

        token_ids.extend(char_ids)
        token_ids.append(self.eop_id)

        seq_len = len(token_ids)

        # 3. Padding（post-padding）
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            token_ids = token_ids + [self.pad_id] * pad_len
        else:
            token_ids = token_ids[:self.max_len]
            seq_len = self.max_len

        # 4. 构建 input_ids 和 target_ids
        # target_ids = input_ids 右移一位（预测下一个字符）
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        target_ids = torch.roll(input_ids, shifts=-1, dims=0)
        # 最后一个 target 用 pad_id 填充（会被 CrossEntropyLoss ignore_index 忽略）
        target_ids[-1] = self.pad_id

        # 5. attention_mask：非 PAD 为 1
        attention_mask = (input_ids != self.pad_id).long()

        # 6. form_ids
        form_ids_tensor = torch.tensor(form_id, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "form_ids": form_ids_tensor,
        }

    def get_raw_text(self, idx):
        """获取原始文本（用于调试或生成阶段）"""
        return self.data[idx].get('text', '')

    def get_poem_meta(self, idx):
        """获取诗歌元数据"""
        return {
            'title': self.data[idx].get('title', ''),
            'author': self.data[idx].get('author', '未知'),
            'form': self.data[idx].get('form', '其他'),
            'form_ids': self.data[idx].get('form_ids', -1),
        }


def get_dataloader(
    jsonl_path,
    vocab_path=None,
    word2idx=None,
    batch_size=64,
    max_len=125,
    use_form_token=False,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=True if torch.cuda.is_available() else False,
):
    """
    DataLoader 工厂函数，供训练负责人（D）直接调用。

    参数:
        jsonl_path: 数据文件路径（如 data/splits/train.jsonl）
        vocab_path: 词表路径，默认 data/processed/vocab.json
        word2idx: 直接传入词表（与 vocab_path 二选一）
        batch_size: 批次大小
        max_len: 最大序列长度
        use_form_token: 是否使用体裁控制 token
        shuffle: 是否打乱顺序
        num_workers: DataLoader worker 数（Windows 建议 0）
        drop_last: 是否丢弃最后一个不完整的 batch
        pin_memory: 是否使用 pinned memory（GPU 训练建议 True）

    返回:
        DataLoader 实例
    """
    if word2idx is None:
        word2idx, _ = load_vocab(vocab_path)

    dataset = PoetryDataset(
        jsonl_path=jsonl_path,
        word2idx=word2idx,
        max_len=max_len,
        use_form_token=use_form_token,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    print(f"✓ DataLoader 创建完成: {len(dataset)} 样本, batch_size={batch_size}, "
          f"steps={len(loader)}, shuffle={shuffle}")
    return loader


# ==================== 辅助函数：验证接口格式 ====================

def check_batch_format(batch, vocab_size=None):
    """
    验证 batch 是否符合接口约定格式。
    建议在训练脚本中调用一次进行断言检查。

    参数:
        batch: DataLoader 输出的一个 batch dict
        vocab_size: 词表大小，用于验证 logits 维度（可选）

    返回:
        bool: 是否通过检查
    """
    required_keys = ["input_ids", "target_ids", "attention_mask", "form_ids"]
    for key in required_keys:
        assert key in batch, f"缺少必要字段: {key}"

    B = batch["input_ids"].shape[0]
    T = batch["input_ids"].shape[1]

    assert batch["input_ids"].shape == (B, T),         f"input_ids shape 错误: {batch['input_ids'].shape}, 应为 (B, T)"
    assert batch["target_ids"].shape == (B, T),         f"target_ids shape 错误: {batch['target_ids'].shape}, 应为 (B, T)"
    assert batch["attention_mask"].shape == (B, T),         f"attention_mask shape 错误: {batch['attention_mask'].shape}, 应为 (B, T)"
    assert batch["form_ids"].shape == (B,),         f"form_ids shape 错误: {batch['form_ids'].shape}, 应为 (B,)"

    assert batch["input_ids"].dtype == torch.long, "input_ids 应为 torch.long"
    assert batch["target_ids"].dtype == torch.long, "target_ids 应为 torch.long"
    assert batch["attention_mask"].dtype == torch.long, "attention_mask 应为 torch.long"
    assert batch["form_ids"].dtype == torch.long, "form_ids 应为 torch.long"

    # 验证 attention_mask 与 PAD 位置一致
    pad_id = 0  # <PAD> 索引
    mask_from_ids = (batch["input_ids"] != pad_id).long()
    assert torch.equal(batch["attention_mask"], mask_from_ids),         "attention_mask 与 input_ids 的 PAD 位置不一致"

    # 验证 target_ids 是 input_ids 右移一位
    expected_target = torch.roll(batch["input_ids"], shifts=-1, dims=1)
    expected_target[:, -1] = pad_id
    assert torch.equal(batch["target_ids"], expected_target),         "target_ids 不是 input_ids 的右移版本"

    print(f"✓ Batch 格式检查通过: B={B}, T={T}, vocab_size={vocab_size}")
    return True


# ==================== 主函数：快速测试 ====================

def main():
    """
    快速测试脚本：
    1. 从 regulated.jsonl 构建词表（如果 vocab.json 不存在）
    2. 创建 train/valid/test DataLoader
    3. 验证 batch 格式
    """
    print("=" * 60)
    print("PoetryDataset 接口测试")
    print("=" * 60)

    # 1. 检查数据文件
    regulated_path = os.path.join(PROCESSED_DIR, 'regulated.jsonl')
    all_clean_path = os.path.join(PROCESSED_DIR, 'all_clean.jsonl')

    if os.path.exists(regulated_path):
        data_path = regulated_path
        print(f"使用 regulated 数据: {regulated_path}")
    elif os.path.exists(all_clean_path):
        data_path = all_clean_path
        print(f"使用 all_clean 数据: {all_clean_path}")
    else:
        raise FileNotFoundError(
            f"未找到数据文件。请先运行 data.py 生成清洗数据。\n"
            f"期望路径: {regulated_path} 或 {all_clean_path}"
        )

    # 2. 构建或加载词表
    vocab_path = os.path.join(PROCESSED_DIR, 'vocab.json')
    if not os.path.exists(vocab_path):
        print("\n[1/4] 构建词表...")
        build_vocab(data_path, min_freq=1, save_path=vocab_path)
    else:
        print(f"\n[1/4] 词表已存在，跳过构建: {vocab_path}")

    word2idx, idx2word = load_vocab(vocab_path)
    vocab_size = len(word2idx)
    print(f"  词表大小: {vocab_size}")
    print(f"  特殊 token: {SPECIAL_TOKENS}")
    print(f"  真实字符起始索引: 8")

    # 3. 测试基线模型 DataLoader（use_form_token=False）
    print("\n[2/4] 测试基线模型 DataLoader (use_form_token=False)...")
    train_loader_base = get_dataloader(
        jsonl_path=data_path,
        word2idx=word2idx,
        batch_size=4,
        max_len=125,
        use_form_token=False,
        shuffle=False,
        num_workers=0,
    )

    batch_base = next(iter(train_loader_base))
    check_batch_format(batch_base, vocab_size=vocab_size)
    print(f"  input_ids 示例: {batch_base['input_ids'][0][:20].tolist()}")
    print(f"  form_ids 示例: {batch_base['form_ids'][:8].tolist()}")

    # 4. 测试改进模型 DataLoader（use_form_token=True）
    print("\n[3/4] 测试改进模型 DataLoader (use_form_token=True)...")
    train_loader_cond = get_dataloader(
        jsonl_path=data_path,
        word2idx=word2idx,
        batch_size=4,
        max_len=125,
        use_form_token=True,
        shuffle=False,
        num_workers=0,
    )

    batch_cond = next(iter(train_loader_cond))
    check_batch_format(batch_cond, vocab_size=vocab_size)
    print(f"  input_ids 示例（含体裁 token）: {batch_cond['input_ids'][0][:20].tolist()}")
    # 验证体裁 token 位置
    first_ids = batch_cond['input_ids'][0].tolist()
    assert first_ids[0] == 2, "SOP 应在索引 0"
    if first_ids[1] in [4, 5, 6, 7]:
        print(f"  ✓ 检测到体裁 token: {idx2word[first_ids[1]]} (id={first_ids[1]})")

    # 5. 验证 train/valid/test 划分文件
    print("\n[4/4] 检查划分后的数据文件...")
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(SPLITS_DIR, f'{split}.jsonl')
        if os.path.exists(split_path):
            loader = get_dataloader(
                jsonl_path=split_path,
                word2idx=word2idx,
                batch_size=2,
                max_len=125,
                use_form_token=False,
                shuffle=False,
                num_workers=0,
            )
            batch = next(iter(loader))
            print(f"  {split}: {len(loader.dataset)} 首, batch shape={batch['input_ids'].shape}")
        else:
            print(f"  {split}: 文件不存在 {split_path}")

    print("\n" + "=" * 60)
    print("所有测试通过！接口格式符合约定。")
    print("=" * 60)
    print("\n使用示例:")
    print("  from dataset import get_dataloader, load_vocab")
    print("  loader = get_dataloader('data/splits/train.jsonl', batch_size=64)")
    print("  for batch in loader:")
    print("      logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])")
    print("      loss = criterion(logits.view(-1, vocab_size), batch['target_ids'].view(-1))")


if __name__ == '__main__':
    main()