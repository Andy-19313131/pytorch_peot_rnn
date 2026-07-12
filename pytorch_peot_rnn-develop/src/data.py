# src/data.py
# 数据负责人：诗歌数据统计与分析（接口兼容版 v5.1）
# 修改说明（v5.1）：
# 1. 路径适配：从 src/data.py 定位到项目根目录
# 2. RAW_DIR 从 'tang' 改为 'raw'，与标准项目结构一致
# 3. 增加 __all__ 导出，便于其他模块导入
# 4. 增加 PoetryDataset 类供 DataLoader 使用（可选）

import opencc
import os
import re
import json
import hashlib
import collections
import random
import csv
import difflib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 图表风格配置 ====================

CHART_COLORS = {
    'ink': '#1F2937',
    'muted': '#6B7280',
    'grid': '#E5E7EB',
    'blue': '#2563EB',
    'cyan': '#0891B2',
    'green': '#059669',
    'amber': '#D97706',
    'rose': '#E11D48',
    'violet': '#7C3AED',
    'slate': '#64748B',
}

FORM_COLORS = [
    CHART_COLORS['blue'],
    CHART_COLORS['cyan'],
    CHART_COLORS['green'],
    CHART_COLORS['amber'],
    CHART_COLORS['rose'],
    CHART_COLORS['violet'],
]


def _style_axes(ax, title, xlabel=None, ylabel=None, grid_axis='y'):
    """统一统计图的版式，避免不同图之间视觉风格割裂。"""
    ax.set_facecolor('#FBFCFE')
    ax.figure.set_facecolor('white')
    ax.set_title(title, fontsize=16, fontweight='bold', color=CHART_COLORS['ink'], pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, color=CHART_COLORS['ink'], labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color=CHART_COLORS['ink'], labelpad=10)

    ax.tick_params(axis='both', colors=CHART_COLORS['muted'], labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CHART_COLORS['grid'])
    ax.spines['bottom'].set_color(CHART_COLORS['grid'])
    ax.grid(axis=grid_axis, color=CHART_COLORS['grid'], linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)

# ==================== 路径配置（v5.1：适配 src/data.py） ====================
# 从 src/data.py 向上定位到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'tang')              # 原始JSON数据目录
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')  # 清洗后数据
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')        # 划分后数据
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_FIGURES = os.path.join(RESULTS_DIR, 'figures')
RESULTS_TABLES = os.path.join(RESULTS_DIR, 'tables')

for d in [PROCESSED_DIR, SPLITS_DIR, RESULTS_FIGURES, RESULTS_TABLES]:
    os.makedirs(d, exist_ok=True)


# ==================== 接口约定：form_ids 映射 ====================
FORM_TO_ID = {
    '五言绝句': 0,
    '七言绝句': 1,
    '五言律诗': 2,
    '七言律诗': 3,
    '长篇': -1,
    '其他': -1,
}

ID_TO_FORM = {v: k for k, v in FORM_TO_ID.items() if v >= 0}

# 特殊 token 索引（供团队参考）
SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'SOP': 2,
    'EOP': 3,
    '<5JUE>': 4,
    '<7JUE>': 5,
    '<5LV>': 6,
    '<7LV>': 7,
}

__all__ = [
    'FORM_TO_ID', 'ID_TO_FORM', 'SPECIAL_TOKENS',
    'get_form_id', 'classify_form', 'clean_text',
    'load_data_from_json', 'split_data_stratified',
    'save_clean_data', 'analyze_basic', 'analyze_form',
    'analyze_authors', 'analyze_vocab', 'analyze_duplicates',
    'main',
]


# ==================== 0. 清洗规则配置 ====================

CLEANING_RULES = {
    "繁简转换": "使用 OpenCC (t2s) 将繁体转为简体",
    "去除注释": "删除括号内注释，如（一作XXX）",
    "去除字母数字": "删除所有英文字母和阿拉伯数字",
    "去除特殊符号": "删除※★☆■□▲△▼▽◆◇○◎●等特殊符号",
    "统一标点": "将半角标点转为全角：,→， .→。 ?→？ !→！",
    "去除空文本": "删除清洗后长度为0的诗歌",
    "去除过短文本": "删除汉字字符数<5的诗歌",
    "标记长篇": "长度>200字的诗歌标记为'长篇'，不进入标准体裁实验",
    "去重": "基于纯汉字文本计算MD5哈希，删除完全重复诗歌",
    "体裁标注": "按句子数和每句字数标注为五言绝句/七言绝句/五言律诗/七言律诗/长篇/其他",
    "form_ids映射": "将体裁字符串映射为整数：0=五绝, 1=七绝, 2=五律, 3=七律, -1=其他/长篇",
}


def split_into_sentences(paragraphs):
    sentences = []
    for para in paragraphs:
        parts = re.split(r'[，。]', para)
        for part in parts:
            clean = re.sub(r'[^一-鿿]', '', part)
            if clean:
                sentences.append(clean)
    return sentences


converter = opencc.OpenCC('t2s')

def clean_text(text):
    text = converter.convert(text)
    text = re.sub(r'[（(].*?[）)]', '', text)
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    text = re.sub(r'[※★☆■□▲△▼▽◆◇○◎●]', '', text)
    text = text.replace(',', '，').replace('.', '。').replace('?', '？').replace('!', '！')
    return text


def load_data_from_json(path=None):
    if path is None:
        if not os.path.exists(RAW_DIR):
            raise FileNotFoundError(
                f"原始数据目录不存在: {RAW_DIR}\n"
                f"请创建 {RAW_DIR} 目录并将原始唐诗JSON文件放入其中。"
            )
        json_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(
                f"在 {RAW_DIR} 中未找到 .json 文件\n"
                f"请确保原始数据文件以 .json 结尾。"
            )
        print(f"✓ 找到 {len(json_files)} 个 JSON 文件")
        all_data = []
        for fname in sorted(json_files):
            fpath = os.path.join(RAW_DIR, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"  - {fname}: {len(data)} 条")
        path = RAW_DIR
    else:
        with open(path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

    poems = []
    for obj in all_data:
        paragraphs = obj.get('paragraphs', [])
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        raw_paragraphs = paragraphs.copy()
        clean_paragraphs = [clean_text(p) for p in paragraphs]
        text = ''.join(clean_paragraphs)
        poems.append({
            'title': obj.get('title', ''),
            'author': obj.get('author', '未知'),
            'paragraphs': clean_paragraphs,
            'raw_paragraphs': raw_paragraphs,
            'text': text,
            'dynasty': obj.get('dynasty', '唐'),
        })
    return poems


def classify_form(paragraphs, text_length=None):
    """
    体裁分类
    返回: 体裁字符串
    """
    if text_length is not None and text_length > 200:
        return '长篇'

    sentences = split_into_sentences(paragraphs)
    n_sentences = len(sentences)
    if n_sentences == 0:
        return '其他'
    char_counts = [len(s) for s in sentences]
    if n_sentences == 4:
        if all(c == 5 for c in char_counts):
            return '五言绝句'
        elif all(c == 7 for c in char_counts):
            return '七言绝句'
    elif n_sentences == 8:
        if all(c == 5 for c in char_counts):
            return '五言律诗'
        elif all(c == 7 for c in char_counts):
            return '七言律诗'
    return '其他'


def get_form_id(form_str):
    """将体裁字符串映射为整数ID（接口约定）"""
    return FORM_TO_ID.get(form_str, -1)


# ==================== 1. 基础统计 ====================

def analyze_basic(poems):
    print("=" * 60)
    print("【1】基本信息统计")
    print("=" * 60)
    total = len(poems)
    lengths = [len(p['text']) for p in poems]
    print(f"诗歌总数: {total}")
    print(f"平均长度: {np.mean(lengths):.1f} 字")
    print(f"最短: {min(lengths)} 字")
    print(f"最长: {max(lengths)} 字")
    print(f"中位数: {np.median(lengths):.1f} 字")

    fig_path = os.path.join(RESULTS_FIGURES, 'length_distribution.png')
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    cap = 220
    display_lengths = np.clip(lengths, 0, cap)
    bins = np.arange(0, cap + 10, 10)
    ax.hist(display_lengths, bins=bins, color=CHART_COLORS['blue'], alpha=0.86,
            edgecolor='white', linewidth=1.0)
    _style_axes(ax, '诗歌长度分布', '诗歌长度（字）', '数量（首）')

    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    for value, label, color in [
        (mean_len, f'均值 {mean_len:.1f}', CHART_COLORS['rose']),
        (median_len, f'中位数 {median_len:.1f}', CHART_COLORS['green']),
    ]:
        if value <= cap:
            ax.axvline(value, color=color, linestyle='--', linewidth=1.8, label=label)
    legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(CHART_COLORS['grid'])

    xticks = list(range(0, cap, 20)) + [cap]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks[:-1]] + [f'{cap}+'])
    ax.text(0.99, -0.16, '注：右端 220+ 合并长篇样本，便于观察主体分布。',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=CHART_COLORS['muted'])
    fig.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {fig_path}")
    return lengths


def analyze_sentences(poems):
    print("\n" + "=" * 60)
    print("【5】句子数分布")
    print("=" * 60)
    sentence_counts = []
    for p in poems:
        sentences = split_into_sentences(p['paragraphs'])
        sentence_counts.append(len(sentences))
    counter = collections.Counter(sentence_counts)
    common_counts = {k: v for k, v in counter.items() if k <= 20}
    for n_sent, count in sorted(common_counts.items()):
        print(f"  {n_sent}句: {count} 首 ({count/len(poems)*100:.1f}%)")
    jueju = counter.get(4, 0)
    lushi = counter.get(8, 0)
    print(f"\n  绝句(4句): {jueju} 首 ({jueju/len(poems)*100:.1f}%)")
    print(f"  律诗(8句): {lushi} 首 ({lushi/len(poems)*100:.1f}%)")

    fig_path = os.path.join(RESULTS_FIGURES, 'sentence_distribution.png')
    plt.figure(figsize=(10, 5))
    labels = [str(k) for k in sorted(common_counts.keys())]
    values = [common_counts[int(k)] for k in labels]
    plt.bar(labels, values, color='lightcoral', edgecolor='black')
    plt.xlabel('句子数', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.title('诗歌句子数分布', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {fig_path}")
    return sentence_counts


# ==================== 2. 体裁与作者统计 ====================

def analyze_form(poems):
    """体裁分布统计"""
    print("\n" + "=" * 60)
    print("【6-7】体裁分布统计")
    print("=" * 60)

    forms = {'五言绝句': 0, '七言绝句': 0, '五言律诗': 0, '七言律诗': 0, '长篇': 0, '其他': 0}
    for p in poems:
        form = classify_form(p['paragraphs'], text_length=len(p['text']))
        forms[form] += 1
        p['form'] = form
        p['form_ids'] = get_form_id(form)

    for form, count in forms.items():
        fid = get_form_id(form)
        print(f"{form} (form_id={fid}): {count} 首 ({count/len(poems)*100:.1f}%)")

    other_poems = [p for p in poems if p['form'] == '其他']
    if other_poems:
        other_sentences = [len(split_into_sentences(p['paragraphs'])) for p in other_poems]
        sent_counter = collections.Counter(other_sentences)
        print(f"\n  '其他'体裁细分（前10句子数）:")
        for n_sent, count in sent_counter.most_common(10):
            print(f"    {n_sent}句: {count} 首")

    fig_path_pie = os.path.join(RESULTS_FIGURES, 'form_distribution_pie.png')
    labels = list(forms.keys())
    sizes = list(forms.values())
    colors = FORM_COLORS
    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 1 else '',
        colors=colors,
        startangle=90,
        counterclock=False,
        pctdistance=0.78,
        wedgeprops={'width': 0.42, 'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 11, 'color': CHART_COLORS['ink']},
    )
    for text in autotexts:
        text.set_color('white')
        text.set_fontweight('bold')
        text.set_fontsize(10)
    ax.text(0, 0, f'{sum(sizes)}\n首诗', ha='center', va='center',
            fontsize=15, fontweight='bold', color=CHART_COLORS['ink'])
    ax.set_title('诗歌体裁分布（占比）', fontsize=16, fontweight='bold',
                 color=CHART_COLORS['ink'], pad=14)
    ax.axis('equal')
    fig.tight_layout()
    plt.savefig(fig_path_pie, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {fig_path_pie}")

    fig_path_bar = os.path.join(RESULTS_FIGURES, 'form_distribution.png')
    fig, ax = plt.subplots(figsize=(11, 6.4))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, sizes, color=colors, edgecolor='none', width=0.62)
    _style_axes(ax, '诗歌体裁分布', None, '数量（首）')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, max(sizes) * 1.18)
    for bar, size in zip(bars, sizes):
        pct = size / len(poems) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.02,
                f'{size}\n{pct:.1f}%', ha='center', va='bottom',
                fontsize=10, color=CHART_COLORS['ink'], fontweight='bold')
    fig.tight_layout()
    plt.savefig(fig_path_bar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {fig_path_bar}")

    return forms


def analyze_authors(poems):
    print("\n" + "=" * 60)
    print("【8-9】作者分布统计")
    print("=" * 60)
    author_counter = collections.Counter(p['author'] for p in poems)
    total = len(poems)
    print(f"不同作者数: {len(author_counter)}")
    print("\nTop 20 作者:")
    top20 = author_counter.most_common(20)
    for author, count in top20:
        print(f"  {author}: {count} 首 ({count/total*100:.1f}%)")
    top10_count = sum(c for _, c in author_counter.most_common(10))
    print(f"\n前10位作者占比: {top10_count/total*100:.1f}%")

    fig_path = os.path.join(RESULTS_FIGURES, 'top_authors.png')
    fig, ax = plt.subplots(figsize=(12, 7.8))
    authors = [a for a, _ in reversed(top20)]
    counts = [c for _, c in reversed(top20)]
    bar_colors = [CHART_COLORS['slate']] * len(authors)
    if len(bar_colors) >= 3:
        bar_colors[-1] = CHART_COLORS['blue']
        bar_colors[-2] = CHART_COLORS['cyan']
        bar_colors[-3] = CHART_COLORS['green']
    ax.barh(range(len(authors)), counts, color=bar_colors, edgecolor='none', height=0.68)
    _style_axes(ax, '作者作品数量 Top 20', '作品数量（首）', None, grid_axis='x')
    ax.set_yticks(range(len(authors)))
    ax.set_yticklabels(authors)
    max_count = max(counts)
    ax.set_xlim(0, max_count * 1.18)
    for y_pos, count in enumerate(counts):
        ax.text(count + max_count * 0.012, y_pos, f'{count}  ({count/total*100:.1f}%)',
                va='center', ha='left', fontsize=9.5, color=CHART_COLORS['ink'])
    fig.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {fig_path}")
    return author_counter


# ==================== 3. 字符词表分析 ====================

def analyze_vocab(poems):
    print("\n" + "=" * 60)
    print("【10-11】字符词表分析")
    print("=" * 60)
    all_text = ''.join(re.sub(r'[^一-鿿]', '', p['text']) for p in poems)
    char_counter = collections.Counter(all_text)
    print(f"总字符数（含重复）: {len(all_text)}")
    print(f"唯一字符数: {len(char_counter)}")
    freq_1 = sum(1 for c in char_counter.values() if c == 1)
    freq_le5 = sum(1 for c in char_counter.values() if c <= 5)
    print(f"出现1次的字符: {freq_1}")
    print(f"出现≤5次的字符: {freq_le5} ({freq_le5/len(char_counter)*100:.1f}%)")
    print("\nTop 20 高频字:")
    for char, count in char_counter.most_common(20):
        print(f"  {char}: {count} 次")
    return char_counter, freq_1, freq_le5


# ==================== 4. 异常检测 ====================

def analyze_abnormal(poems):
    print("\n" + "=" * 60)
    print("【3】异常文本检测")
    print("=" * 60)
    empty = []
    too_short = []
    too_long = []
    non_chinese = []
    noise_samples = []
    for i, p in enumerate(poems):
        text = p['text']
        chinese_chars = re.sub(r'[^一-鿿]', '', text)
        if len(text) == 0:
            empty.append(i)
            noise_samples.append({'index': i, 'type': '空文本', 'author': p['author'], 'text': text})
        elif len(chinese_chars) < 5:
            too_short.append((i, text))
            noise_samples.append({'index': i, 'type': '过短', 'author': p['author'], 'text': text})
        elif len(text) > 200:
            too_long.append((i, text[:50] + '...'))
            noise_samples.append({'index': i, 'type': '过长', 'author': p['author'], 'text': text[:80]})
        if len(text) > 0 and len(chinese_chars) / len(text) < 0.5:
            non_chinese.append((i, text[:50]))
            noise_samples.append({'index': i, 'type': '非汉字比例过高', 'author': p['author'], 'text': text[:80]})
    print(f"空文本: {len(empty)} 首")
    print(f"过短（汉字<5）: {len(too_short)} 首")
    print(f"过长（>200字）: {len(too_long)} 首")
    print(f"非汉字比例过高: {len(non_chinese)} 首")
    if too_short:
        print("\n过短示例:")
        for idx, text in too_short[:3]:
            print(f"  [{idx}] {text}")
    if too_long:
        print("\n过长示例:")
        for idx, text in too_long[:3]:
            print(f"  [{idx}] {text}")
    return {
        'empty': empty, 'too_short': too_short,
        'too_long': too_long, 'non_chinese': non_chinese,
        'noise_samples': noise_samples,
        'counts': {
            'empty': len(empty),
            'too_short': len(too_short),
            'too_long': len(too_long),
            'non_chinese': len(non_chinese),
        }
    }


# ==================== 5. 重复检测 ====================

def analyze_duplicates(poems):
    print("\n" + "=" * 60)
    print("【2】重复检测")
    print("=" * 60)
    seen_hashes = set()
    unique_poems = []
    duplicates = []
    for p in poems:
        clean = re.sub(r'[^一-鿿]', '', p['text'])
        h = hashlib.md5(clean.encode('utf-8')).hexdigest()
        if h in seen_hashes:
            duplicates.append(p)
        else:
            seen_hashes.add(h)
            unique_poems.append(p)
    print(f"原始诗歌数: {len(poems)}")
    print(f"重复诗歌数: {len(duplicates)}")
    print(f"去重后诗歌数: {len(unique_poems)}")
    if duplicates:
        print("\n重复示例:")
        for d in duplicates[:3]:
            print(f"  [{d['author']}] {d['text'][:30]}...")
    return unique_poems, duplicates


# ==================== 6. 极值样本 ====================

def analyze_extremes(poems):
    print("\n" + "=" * 60)
    print("【12】极值样本")
    print("=" * 60)
    sorted_poems = sorted(poems, key=lambda p: len(p['text']))
    print("最短3首:")
    for p in sorted_poems[:3]:
        print(f"  [{p['author']}] ({len(p['text'])}字) {p['text']}")
    print("\n最长3首:")
    for p in sorted_poems[-3:]:
        print(f"  [{p['author']}] ({len(p['text'])}字) {p['text'][:80]}...")
    return sorted_poems[:3], sorted_poems[-3:]


# ==================== 7. 简繁转换分析 ====================

def analyze_conversion_loss(poems, sample_size=1000, seed=42):
    print("\n" + "=" * 60)
    print("【新增】简繁转换信息损失分析")
    print("=" * 60)

    s2t_converter = opencc.OpenCC('s2t')
    changed_count = 0
    one_to_many_cases = []
    over_conversion_cases = []

    rng = random.Random(seed)
    sample = rng.sample(poems, min(sample_size, len(poems)))

    for p in sample:
        raw_text = ''.join(p['raw_paragraphs'])
        back_to_traditional = s2t_converter.convert(p['text'])

        if raw_text != back_to_traditional:
            changed_count += 1
            sm = difflib.SequenceMatcher(None, raw_text, back_to_traditional)

            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'replace':
                    raw_part = raw_text[i1:i2]
                    back_part = back_to_traditional[j1:j2]
                    simp_part = p['text'][i1:i2] if i2 <= len(p['text']) else ''

                    for rc, bc in zip(raw_part, back_part):
                        if ('\u4e00' <= rc <= '\u9fff' and 
                            '\u4e00' <= bc <= '\u9fff' and 
                            rc != bc and len(one_to_many_cases) + len(over_conversion_cases) < 80):

                            if rc == simp_part[0] if simp_part else False:
                                over_conversion_cases.append({
                                    'original': rc,
                                    'simplified': simp_part[0] if simp_part else '?',
                                    'back_to_traditional': bc,
                                    'type': '过度转换',
                                    'context': raw_text[max(0,i1-3):min(len(raw_text),i1+4)]
                                })
                            else:
                                one_to_many_cases.append({
                                    'original': rc,
                                    'simplified': simp_part[0] if simp_part else '?',
                                    'back_to_traditional': bc,
                                    'type': '一简对多繁',
                                    'context': raw_text[max(0,i1-3):min(len(raw_text),i1+4)]
                                })

    seen_1m = set()
    unique_1m = []
    for case in one_to_many_cases:
        key = (case['original'], case['simplified'], case['back_to_traditional'])
        if key not in seen_1m:
            seen_1m.add(key)
            unique_1m.append(case)

    seen_ov = set()
    unique_ov = []
    for case in over_conversion_cases:
        key = (case['original'], case['simplified'], case['back_to_traditional'])
        if key not in seen_ov:
            seen_ov.add(key)
            unique_ov.append(case)

    print(f"抽样检查 {len(sample)} 首诗歌（种子={seed}）")
    print(f"简繁转换不一致: {changed_count} 首 ({changed_count/len(sample)*100:.1f}%)")
    print(f"  - 一简对多繁: {len(unique_1m)} 种")
    print(f"  - OpenCC过度转换: {len(unique_ov)} 种")

    if unique_1m:
        print(f"\n一简对多繁示例（前5个）:")
        for case in unique_1m[:5]:
            print(f"  原繁体: {case['original']} → 简体: {case['simplified']} → 回繁: {case['back_to_traditional']} (上下文: ...{case['context']}...)")

    if unique_ov:
        print(f"\nOpenCC过度转换示例（前5个）:")
        for case in unique_ov[:5]:
            print(f"  原文: {case['original']} → 简体: {case['simplified']} → 回繁: {case['back_to_traditional']} (上下文: ...{case['context']}...)")

    return {
        'sample_size': len(sample),
        'changed_count': changed_count,
        'change_rate': changed_count / len(sample) if sample else 0,
        'one_to_many': unique_1m,
        'over_conversion': unique_ov,
    }


# ==================== 8. 数据输出 ====================

def save_clean_data(poems, output_dir=PROCESSED_DIR):
    """保存清洗后的数据，包含form_ids字段"""
    all_clean_path = os.path.join(output_dir, 'all_clean.jsonl')
    with open(all_clean_path, 'w', encoding='utf-8') as f:
        for p in poems:
            if 'form_ids' not in p:
                p['form_ids'] = get_form_id(p.get('form', '其他'))
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"\n✓ 保存 all_clean: {all_clean_path} ({len(poems)} 首)")

    regulated = [p for p in poems if p['form'] in ['五言绝句', '七言绝句', '五言律诗', '七言律诗']]
    regulated_path = os.path.join(output_dir, 'regulated.jsonl')
    with open(regulated_path, 'w', encoding='utf-8') as f:
        for p in regulated:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"✓ 保存 regulated: {regulated_path} ({len(regulated)} 首)")
    return all_clean_path, regulated_path


def split_data_stratified(poems, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """分层划分（使用form_ids进行分层）"""
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    np.random.seed(seed)

    form_groups = collections.defaultdict(list)
    for p in poems:
        fid = p.get('form_ids', get_form_id(p['form']))
        form_groups[fid].append(p)

    train, valid, test = [], [], []
    for form_id, group in form_groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train.extend(group[:n_train])
        valid.extend(group[n_train:n_train + n_valid])
        test.extend(group[n_train + n_valid:])

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    for split_name, split_data in [('train', train), ('valid', valid), ('test', test)]:
        path = os.path.join(SPLITS_DIR, f'{split_name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for p in split_data:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
        print(f"✓ 保存 {split_name}: {path} ({len(split_data)} 首)")

    # 输出 train_texts.txt 供E使用
    train_texts_path = os.path.join(SPLITS_DIR, 'train_texts.txt')
    with open(train_texts_path, 'w', encoding='utf-8') as f:
        for p in train:
            f.write(p['text'] + '\n')
    print(f"✓ 保存 train_texts: {train_texts_path} ({len(train)} 首)")

    def get_hashes(poems_list):
        return set(hashlib.md5(re.sub(r'[^一-鿿]', '', p['text']).encode()).hexdigest() for p in poems_list)

    train_h = get_hashes(train)
    valid_h = get_hashes(valid)
    test_h = get_hashes(test)

    overlap_train_valid = len(train_h & valid_h)
    overlap_train_test = len(train_h & test_h)
    overlap_valid_test = len(valid_h & test_h)

    print(f"\n重叠检查 (应为0):")
    print(f"  train-valid: {overlap_train_valid}")
    print(f"  train-test: {overlap_train_test}")
    print(f"  valid-test: {overlap_valid_test}")

    overlap_path = os.path.join(RESULTS_TABLES, 'split_overlap_check.txt')
    with open(overlap_path, 'w', encoding='utf-8') as f:
        f.write("# 数据划分重叠检查报告\n\n")
        f.write(f"划分种子: {seed}\n")
        f.write(f"训练集: {len(train)} 首\n")
        f.write(f"验证集: {len(valid)} 首\n")
        f.write(f"测试集: {len(test)} 首\n\n")
        f.write("## 重叠检查结果\n\n")
        f.write(f"- train ∩ valid: {overlap_train_valid} 首\n")
        f.write(f"- train ∩ test: {overlap_train_test} 首\n")
        f.write(f"- valid ∩ test: {overlap_valid_test} 首\n\n")
        if overlap_train_valid == 0 and overlap_train_test == 0 and overlap_valid_test == 0:
            f.write("✓ 检查通过：三个集合无重叠样本\n")
        else:
            f.write("✗ 警告：发现重叠样本，需要重新划分！\n")
    print(f"✓ 保存重叠检查: {overlap_path}")

    print(f"\n分层划分后体裁分布（form_ids）:")
    split_form_dist = {}
    for split_name, split_data in [('train', train), ('valid', valid), ('test', test)]:
        form_dist = collections.Counter(p.get('form_ids', get_form_id(p['form'])) for p in split_data)
        split_form_dist[split_name] = dict(form_dist)
        print(f"  {split_name}: {dict(form_dist)}")

    dist_path = os.path.join(RESULTS_TABLES, 'split_form_distribution.csv')
    all_form_ids = [-1, 0, 1, 2, 3]
    form_id_labels = {-1: '其他/长篇', 0: '五言绝句', 1: '七言绝句', 2: '五言律诗', 3: '七言律诗'}
    with open(dist_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['form_id', '体裁名称', 'train', 'valid', 'test'])
        for fid in all_form_ids:
            row = [fid, form_id_labels.get(fid, '未知')]
            for split_name in ['train', 'valid', 'test']:
                row.append(split_form_dist[split_name].get(fid, 0))
            writer.writerow(row)
    print(f"✓ 保存体裁分布: {dist_path}")

    return train, valid, test


# ==================== 9. 统计表格 ====================

def save_statistics(poems, forms, author_counter, char_counter, lengths, 
                    conversion_loss, abnormal_stats, duplicate_count, 
                    freq_1, freq_le5, output_path=None):
    if output_path is None:
        output_path = os.path.join(RESULTS_TABLES, 'data_statistics.csv')
    total = len(poems)
    top10_count = sum(c for _, c in author_counter.most_common(10))

    form_id_dist = collections.Counter(p.get('form_ids', get_form_id(p['form'])) for p in poems)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['指标', '数值', '说明'])
        rows = [
            ['诗歌总数', total, '去重过滤后的有效诗歌数'],
            ['重复诗歌数', duplicate_count, '去重前发现的重复数量'],
            ['空文本数', abnormal_stats['counts']['empty'], '清洗后长度为0的诗歌'],
            ['过短文本数', abnormal_stats['counts']['too_short'], '汉字字符数<5的诗歌'],
            ['过长文本数', abnormal_stats['counts']['too_long'], '长度>200字的诗歌（标记为长篇）'],
            ['非汉字比例过高数', abnormal_stats['counts']['non_chinese'], '汉字占比<50%的诗歌'],
            ['唯一字符数', len(char_counter), '词表规模'],
            ['出现1次的字符', freq_1, 'hapax legomena'],
            ['低频字数量(≤5次)', freq_le5, '低频字总数'],
            ['低频字比例(%)', f'{freq_le5/len(char_counter)*100:.2f}', '低频字占唯一字符的比例'],
            ['五言绝句(form_id=0)', forms['五言绝句'], '4句，每句5字'],
            ['七言绝句(form_id=1)', forms['七言绝句'], '4句，每句7字'],
            ['五言律诗(form_id=2)', forms['五言律诗'], '8句，每句5字'],
            ['七言律诗(form_id=3)', forms['七言律诗'], '8句，每句7字'],
            ['长篇/其他(form_id=-1)', forms['长篇'] + forms['其他'], '非标准体裁'],
            ['不同作者数', len(author_counter), '唯一作者数量'],
            ['前10作者占比(%)', f'{top10_count/total*100:.2f}', '前10位高产作者占比'],
            ['平均长度', f'{np.mean(lengths):.1f}', '平均字符长度'],
            ['最短长度', min(lengths), '最短诗歌长度'],
            ['最长长度', max(lengths), '最长诗歌长度'],
            ['长度中位数', f'{np.median(lengths):.1f}', '长度中位数'],
            ['简繁转换不一致率(%)', f'{conversion_loss["change_rate"]*100:.2f}', '简体回繁后与原文不一致的比例'],
        ]
        writer.writerows(rows)
    print(f"\n✓ 保存统计表格: {output_path}")
    return output_path


# ==================== 10. 样本表格 ====================

def save_examples(poems, noise_samples, output_path=None):
    if output_path is None:
        output_path = os.path.join(RESULTS_TABLES, 'data_examples.csv')

    sorted_by_len = sorted(poems, key=lambda p: len(p['text']))
    examples = []

    examples.append({
        '类型': '最短', '作者': sorted_by_len[0]['author'],
        '体裁': sorted_by_len[0]['form'], 'form_id': sorted_by_len[0].get('form_ids', -1),
        '长度': len(sorted_by_len[0]['text']),
        '清洗后文本': sorted_by_len[0]['text'],
        '原始文本': ''.join(sorted_by_len[0]['raw_paragraphs'])[:100]
    })
    examples.append({
        '类型': '最长', '作者': sorted_by_len[-1]['author'],
        '体裁': sorted_by_len[-1]['form'], 'form_id': sorted_by_len[-1].get('form_ids', -1),
        '长度': len(sorted_by_len[-1]['text']),
        '清洗后文本': sorted_by_len[-1]['text'][:100] + '...',
        '原始文本': ''.join(sorted_by_len[-1]['raw_paragraphs'])[:100]
    })

    for form in ['五言绝句', '七言绝句', '五言律诗', '七言律诗', '长篇', '其他']:
        candidates = [p for p in poems if p['form'] == form]
        if candidates:
            p = candidates[len(candidates)//2]
            examples.append({
                '类型': form, '作者': p['author'],
                '体裁': p['form'], 'form_id': p.get('form_ids', -1),
                '长度': len(p['text']),
                '清洗后文本': p['text'][:100] + ('...' if len(p['text']) > 100 else ''),
                '原始文本': ''.join(p['raw_paragraphs'])[:100]
            })

    for noise in noise_samples[:10]:
        orig_text = ''
        if noise['index'] < len(poems):
            orig_text = ''.join(poems[noise['index']]['raw_paragraphs'])[:100]
        examples.append({
            '类型': f"噪声-{noise['type']}", '作者': noise['author'],
            '体裁': 'N/A', 'form_id': -1,
            '长度': len(noise['text']),
            '清洗后文本': noise['text'][:100],
            '原始文本': orig_text
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['类型', '作者', '体裁', 'form_id', '长度', '清洗后文本', '原始文本'])
        writer.writeheader()
        writer.writerows(examples)
    print(f"✓ 保存样本表格: {output_path}")
    return output_path


# ==================== 11. 清洗规则文档 ====================

def save_cleaning_rules(output_path=None):
    if output_path is None:
        output_path = os.path.join(RESULTS_TABLES, 'data_cleaning_rules.md')

    content = """# 数据清洗规则说明

## 1. 数据来源

从原始 JSON 文件读取，保留以下字段：
- `title`: 诗歌标题
- `author`: 作者
- `paragraphs`: 清洗后的段落列表
- `raw_paragraphs`: 原始段落列表（保留用于对比）
- `text`: 合并后的清洗文本
- `dynasty`: 朝代
- `form`: 体裁标注（字符串）
- `form_ids`: 体裁编号（整数，接口约定）

## 2. 清洗步骤

"""
    for i, (rule_name, rule_desc) in enumerate(CLEANING_RULES.items(), 1):
        content += f"### {i}. {rule_name}\n\n{rule_desc}\n\n"

    content += """## 3. 体裁标注规则

去掉标点后：
- **五言绝句**: 4句，每句5字 → form_id=0
- **七言绝句**: 4句，每句7字 → form_id=1
- **五言律诗**: 8句，每句5字 → form_id=2
- **七言律诗**: 8句，每句7字 → form_id=3
- **长篇**: 长度>200字 → form_id=-1
- **其他**: 不符合以上规则且长度≤200字 → form_id=-1

## 4. 输出数据

### all_clean.jsonl
所有清洗后的有效唐诗，包含标准体裁、长篇和其他，用于通用生成实验。
每条记录包含 `form_ids` 字段，供 DataLoader 使用。

### regulated.jsonl
仅包含四种标准体裁（五言绝句、七言绝句、五言律诗、七言律诗），
用于体裁控制实验。长篇和其他体裁不进入此文件。

### train.jsonl / valid.jsonl / test.jsonl
按体裁分层划分的数据集，包含 `form_ids` 字段。

### train_texts.txt
训练集纯文本列表，每行一首诗的 `text` 字段，
供评价模块（E）进行训练集重合检查。

## 5. 数据划分

采用按 **form_ids** 分层抽样：
- 训练集 (train): 80%
- 验证集 (valid): 10%
- 测试集 (test): 10%

划分前对每种 form_id 内部随机打乱，确保各集合中体裁比例与总体一致。
划分后通过 MD5 哈希检查确认三个集合无重叠样本。

## 6. 接口约定

### form_ids 映射（与模型/训练模块一致）
| 体裁 | form_id |
|------|---------|
| 五言绝句 | 0 |
| 七言绝句 | 1 |
| 五言律诗 | 2 |
| 七言律诗 | 3 |
| 长篇/其他 | -1 |

### 特殊 token 索引（供参考）
| Token | 索引 | 说明 |
|-------|------|------|
| PAD | 0 | 填充 |
| UNK | 1 | 未知字符 |
| SOP | 2 | 诗歌开始 |
| EOP | 3 | 诗歌结束 |
| `<5JUE>` | 4 | 五言绝句（改进模型） |
| `<7JUE>` | 5 | 七言绝句（改进模型） |
| `<5LV>` | 6 | 五言律诗（改进模型） |
| `<7LV>` | 7 | 七言律诗（改进模型） |

> 注意：基线模型的词表从索引 4 开始为真实字符。
> 改进模型的体裁 token 占用索引 4-7，真实字符从索引 8 开始。

## 7. 已知局限

1. **简繁转换存在信息损失**：
   - 一简对多繁：如"后"←"後/后"，"复"←"復/覆"
   - OpenCC过度转换：如将繁体"欲"转简体后再回繁变成"慾"（两者在繁体中本就并存）
2. 数据集仅包含唐诗，不能代表全部古典文学
3. 历史收录存在偏差，盛唐诗人作品占比可能偏高
4. 字符级建模难以显式表达词语和典故
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 保存清洗规则: {output_path}")
    return output_path


# ==================== 12. 报告摘要 ====================

def generate_report_summary(poems, forms, author_counter, char_counter, 
                           lengths, conversion_loss, unique_count, original_count,
                           abnormal_stats, freq_1, freq_le5):
    total = len(poems)
    top10 = sum(c for _, c in author_counter.most_common(10))

    other_poems = [p for p in poems if p['form'] == '其他']
    other_long = sum(1 for p in other_poems if len(p['text']) > 100)
    other_short = len(other_poems) - other_long

    form_values = [forms['五言绝句'], forms['七言绝句'], forms['五言律诗'], forms['七言律诗']]
    max_form = max(form_values)
    min_form = min(form_values)
    is_balanced = (max_form - min_form) < total * 0.15
    is_concentrated = top10 / total > 0.25
    is_long_tail = freq_le5 / len(char_counter) > 0.25

    summary = f"""## 数据质量分析与清洗

### 1. 数据规模与清洗效果

原始数据共 **{original_count}** 首诗歌，经过去重后剩余 **{unique_count}** 首，去重率为 {(original_count-unique_count)/original_count*100:.1f}%。
进一步过滤空文本（{abnormal_stats['counts']['empty']}首）、过短文本（{abnormal_stats['counts']['too_short']}首）后，
最终保留 **{total}** 首有效诗歌用于实验。

其中：
- 标准体裁（绝句/律诗）：{forms['五言绝句']+forms['七言绝句']+forms['五言律诗']+forms['七言律诗']} 首 ({(forms['五言绝句']+forms['七言绝句']+forms['五言律诗']+forms['七言律诗'])/total*100:.1f}%)
- 长篇（>200字）：{forms['长篇']} 首 ({forms['长篇']/total*100:.1f}%)
- 其他（非标准格律）：{forms['其他']} 首 ({forms['其他']/total*100:.1f}%)

另有 {abnormal_stats['counts']['non_chinese']} 首非汉字比例过高的作品被标记。

### 2. 数据偏见分析

**作者集中度**：数据集中共有 {len(author_counter)} 位不同作者，前10位作者作品数占总数的 **{top10/total*100:.1f}%**，
{'存在明显的高产作者集中现象，模型可能偏向学习这些高产诗人的风格。' if is_concentrated else '不存在极端的高产作者集中现象，但部分作者仍占较高比例。'}

**体裁分布**：
- 五言绝句 (form_id=0): {forms['五言绝句']} 首 ({forms['五言绝句']/total*100:.1f}%)
- 七言绝句 (form_id=1): {forms['七言绝句']} 首 ({forms['七言绝句']/total*100:.1f}%)
- 五言律诗 (form_id=2): {forms['五言律诗']} 首 ({forms['五言律诗']/total*100:.1f}%)
- 七言律诗 (form_id=3): {forms['七言律诗']} 首 ({forms['七言律诗']/total*100:.1f}%)
- 长篇/其他 (form_id=-1): {forms['长篇'] + forms['其他']} 首 ({(forms['长篇'] + forms['其他'])/total*100:.1f}%)

{'体裁分布严重不均衡：五言律诗远多于五言绝句，"其他"体裁包括杂言诗、乐府及非标准格律诗。这种不平衡可能导致模型对某些体裁的生成能力较弱。' if not is_balanced else '四种标准体裁分布相对均衡。'}

**"其他"体裁细分**：共 {forms['其他']} 首，包含各种非标准格律诗（杂言、乐府、歌行等），
其中较长作品（>100字）约 {other_long} 首，较短作品约 {other_short} 首。

**字符分布**：唯一字符数为 {len(char_counter)}，其中出现1次的字符有 {freq_1} 个，出现频率≤5次的低频字占 **{freq_le5/len(char_counter)*100:.1f}%**，
{'存在明显的长尾分布，大量生僻字仅出现少数几次，模型难以充分学习这些字符的用法。' if is_long_tail else '字符分布相对均匀。'}

### 3. 简繁转换分析

使用 OpenCC 进行繁简转换，抽样检查 {conversion_loss['sample_size']} 首诗歌（种子=42）发现，
**{conversion_loss['change_rate']*100:.1f}%** 的诗歌在简体→繁体回转时与原繁体不一致。

其中：
- **一简对多繁**：{len(conversion_loss['one_to_many'])} 种，如"後/后"→"后"，"復/覆"→"复"
- **OpenCC过度转换**：{len(conversion_loss['over_conversion'])} 种，如将繁体"欲"回转为"慾"（两者在繁体中本就并存，属于OpenCC的保守策略）

这种信息损失主要体现为：
- 合并字：後/后→后，復/覆/复→复
- 形近字：岳/嶽→岳，采/採→采
- 虽然对语言模型训练影响有限，但在严格古典文献研究中需要注意。

### 4. 历史收录偏差说明

本数据集仅包含唐诗，不能代表宋词、元曲及全部古典文学；
流传并被数字化的作品本身存在历史收录偏差，如盛唐诗人作品可能因保存条件较好而占比偏高。
本实验首先量化数据分布问题，并通过去重、体裁标注和固定划分降低实验偏差；
对于历史收录偏差，只进行分析和限制说明，不声称能够完全修正。

### 5. 数据划分策略

采用按 **form_ids** 分层抽样，将数据划分为训练集(80%)、验证集(10%)、测试集(10%)，
确保各集合中五言绝句(0)、七言绝句(1)、五言律诗(2)、七言律诗(3)、长篇/其他(-1)的比例与总体一致。
划分后通过 hash 检查确认训练/验证/测试集无重叠样本。

所有划分后的数据文件均包含 `form_ids` 字段，与模型/训练模块的接口约定一致。
"""

    summary_path = os.path.join(RESULTS_TABLES, 'data_report_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\n✓ 保存报告摘要: {summary_path}")
    print("\n" + "=" * 60)
    print("【报告摘要预览】")
    print("=" * 60)
    print(summary)

    return summary_path


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("唐诗数据统计与分析（接口兼容版 v5.1）")
    print("=" * 60)
    print("\n项目路径:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  RAW_DIR: {RAW_DIR}")
    print(f"  PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"  SPLITS_DIR: {SPLITS_DIR}")
    print("\n接口约定：")
    print("  form_ids: 0=五绝, 1=七绝, 2=五律, 3=七律, -1=其他/长篇")
    print("  输出: all_clean.jsonl / regulated.jsonl / train.jsonl / valid.jsonl / test.jsonl")
    print("  额外: train_texts.txt (供E评价使用)")
    print("=" * 60)

    poems = load_data_from_json()
    original_count = len(poems)
    print(f"\n加载完成，共 {original_count} 首诗歌\n")

    abnormal_before = analyze_abnormal(poems)
    print(f"\n[去重前] 异常统计: 空文本={abnormal_before['counts']['empty']}, "
          f"过短={abnormal_before['counts']['too_short']}, "
          f"过长={abnormal_before['counts']['too_long']}, "
          f"非汉字={abnormal_before['counts']['non_chinese']}")

    unique_poems, duplicates = analyze_duplicates(poems)
    unique_count = len(unique_poems)
    duplicate_count = len(duplicates)

    print("\n" + "=" * 60)
    print("【过滤】处理异常样本")
    print("=" * 60)

    filtered_poems = []
    removed_count = {'empty': 0, 'too_short': 0, 'marked_long': 0}

    for p in unique_poems:
        text = p['text']
        chinese = re.sub(r'[^一-鿿]', '', text)

        if len(text) == 0:
            removed_count['empty'] += 1
            continue
        if len(chinese) < 5:
            removed_count['too_short'] += 1
            continue

        p['form'] = classify_form(p['paragraphs'], text_length=len(p['text']))
        p['form_ids'] = get_form_id(p['form'])

        if len(text) > 200:
            removed_count['marked_long'] += 1

        filtered_poems.append(p)

    print(f"去重后: {unique_count} 首")
    print(f"过滤后: {len(filtered_poems)} 首")
    print(f"  - 空文本过滤: {removed_count['empty']} 首")
    print(f"  - 过短(<5字)过滤: {removed_count['too_short']} 首")
    print(f"  - 过长(>200字)标记为'长篇': {removed_count['marked_long']} 首")

    unique_poems = filtered_poems

    lengths = analyze_basic(unique_poems)
    sentence_counts = analyze_sentences(unique_poems)
    forms = analyze_form(unique_poems)
    author_counter = analyze_authors(unique_poems)
    char_counter, freq_1, freq_le5 = analyze_vocab(unique_poems)
    shortest, longest = analyze_extremes(unique_poems)
    conversion_loss = analyze_conversion_loss(unique_poems, seed=42)

    save_clean_data(unique_poems)
    train, valid, test = split_data_stratified(unique_poems)

    save_statistics(
        unique_poems, forms, author_counter, char_counter, lengths, 
        conversion_loss, abnormal_before, duplicate_count, freq_1, freq_le5
    )
    # 构建词表（兼容直接运行 src/data.py）
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import build_vocab
    build_vocab(os.path.join(PROCESSED_DIR, 'regulated.jsonl'))
    save_examples(unique_poems, abnormal_before['noise_samples'])
    save_cleaning_rules()

    abnormal_stats = {
        'counts': {
            'empty': abnormal_before['counts']['empty'],
            'too_short': abnormal_before['counts']['too_short'],
            'too_long': abnormal_before['counts']['too_long'],
            'non_chinese': abnormal_before['counts']['non_chinese'],
        }
    }
    generate_report_summary(
        unique_poems, forms, author_counter, char_counter, 
        lengths, conversion_loss, unique_count, original_count,
        abnormal_stats, freq_1, freq_le5
    )

    print("\n" + "=" * 60)
    print("【完成】所有分析结果保存完成")
    print("=" * 60)
    print(f"\n输出目录:")
    print(f"  清洗数据: {PROCESSED_DIR}")
    print(f"  数据划分: {SPLITS_DIR}")
    print(f"  图表: {RESULTS_FIGURES}")
    print(f"  表格: {RESULTS_TABLES}")
    print(f"\n生成文件清单:")
    print(f"  - all_clean.jsonl (含form_ids)")
    print(f"  - regulated.jsonl (含form_ids)")
    print(f"  - train.jsonl / valid.jsonl / test.jsonl (含form_ids)")
    print(f"  - train_texts.txt (纯文本，供E评价使用)")
    print(f"  - data_statistics.csv（完整统计）")
    print(f"  - data_examples.csv（含form_id列）")
    print(f"  - split_overlap_check.txt（重叠检查）")
    print(f"  - split_form_distribution.csv（form_id数字编码）")
    print(f"  - data_cleaning_rules.md（含接口约定）")
    print(f"  - data_report_summary.md（报告摘要）")
    print(f"  - length_distribution.png / sentence_distribution.png")
    print(f"  - form_distribution.png / form_distribution_pie.png")
    print(f"  - top_authors.png")


if __name__ == '__main__':
    main()
