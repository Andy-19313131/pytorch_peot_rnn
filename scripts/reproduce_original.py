"""
阶段一：原仓库复现脚本
记录环境信息、运行原始训练和推理，保存所有结果
"""
import sys
import os
import time
import json
import platform
import torch
import numpy as np

# ===== 环境信息记录 =====
def collect_env_info():
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "numpy_version": np.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "N/A (CPU only)"
        info["cuda_version"] = "N/A"
    return info

if __name__ == '__main__':
    # 收集环境信息
    env_info = collect_env_info()
    print("=" * 60)
    print("环境信息")
    print("=" * 60)
    for k, v in env_info.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # 保存环境信息
    os.makedirs("./results/original", exist_ok=True)
    with open("./results/original/environment.txt", "w", encoding="utf-8") as f:
        for k, v in env_info.items():
            f.write(f"{k}: {v}\n")

    # 获取 git commit
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
        env_info["git_commit"] = commit
        print(f"  git_commit: {commit}")
        with open("./results/original/environment.txt", "a", encoding="utf-8") as f:
            f.write(f"git_commit: {commit}\n")
    except Exception as e:
        print(f"  git_commit: ERROR - {e}")

    # ===== 运行原始训练 =====
    print("\n" + "=" * 60)
    print("开始原始训练流程")
    print("=" * 60)

    # 导入原仓库模块
    sys.path.insert(0, ".")
    from config import Config
    from process import get_data
    from model import PoetryModel2
    from utils import set_seed, set_logger
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import logging

    config = Config()
    set_seed(123)

    # 设置日志输出到文件
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清除已有handler
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    file_handler = logging.FileHandler("./results/original/train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    device = torch.device('cpu')
    config.device = device
    config.do_train = True
    config.do_test = True
    config.do_predict = True
    config.do_load_model = False
    config.num_epoch = 3  # 减少epoch数用于复现测试

    # 加载数据
    print("\n加载数据...")
    data, word2idx, idx2word = get_data(config)
    config.word2idx = word2idx
    config.idx2word = idx2word
    print(f"  数据总量: {len(data)} 条")
    print(f"  词表大小: {len(word2idx)}")

    # 划分训练集和测试集（原始代码有bug）
    import random
    def split_train_test_original(data, train_ratio=0.8, shuffle=True):
        if shuffle:
            random.shuffle(data)
        total = len(data)
        train_total = int(total * train_ratio)
        train_data = data[:train_total]
        test_data = data[:train_total]  # BUG: 应该是 data[train_total:]
        print(f'  总共有数据{total}条')
        print(f'  划分后，训练集{train_total}条')
        print(f'  划分后，测试集{total - train_total}条')
        return train_data, test_data

    train_data, test_data = split_train_test_original(data)

    train_data_tensor = torch.from_numpy(train_data)
    train_loader = DataLoader(train_data_tensor, batch_size=config.batch_size, shuffle=True, num_workers=0)

    test_data_tensor = torch.from_numpy(test_data)
    test_loader = DataLoader(test_data_tensor, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 创建模型
    model = PoetryModel2(len(word2idx), config.embedding_dim, config.hidden_dim)
    model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    with open("./results/original/environment.txt", "a", encoding="utf-8") as f:
        f.write(f"data_count: {len(data)}\n")
        f.write(f"vocab_size: {len(word2idx)}\n")
        f.write(f"total_params: {total_params}\n")
        f.write(f"trainable_params: {trainable_params}\n")

    # 训练
    class OriginalTrainer:
        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.criterion = torch.nn.CrossEntropyLoss()

        def train(self, train_loader, test_loader=None):
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            global_step = 0
            best_test_loss = float("inf")
            best_epoch = None
            total_step = len(train_loader) * self.config.num_epoch
            train_start = time.time()

            for epoch in range(1, self.config.num_epoch + 1):
                total_loss = 0.
                for train_step, train_data_batch in enumerate(train_loader):
                    self.model.train()
                    train_data_batch = train_data_batch.long().to(self.config.device)
                    input_data = train_data_batch[:, :-1]
                    target = train_data_batch[:, 1:]
                    output, _ = self.model(input_data)
                    active = (input_data > 0).view(-1)
                    active_output = output[active]
                    active_target = target.contiguous().view(-1)[active]
                    loss = self.criterion(active_output, active_target)
                    total_loss = total_loss + loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                logger.info('epoch:{} total_loss:{}'.format(epoch, total_loss))

                if test_loader is not None:
                    test_loss = self.test(test_loader)
                    if test_loss < best_test_loss:
                        torch.save(self.model.state_dict(), self.config.save_path)
                        best_test_loss = test_loss
                        best_epoch = epoch
                    logger.info('epoch:{} test_loss:{}'.format(epoch, test_loss))

            train_time = time.time() - train_start
            logger.info(f'训练完成，耗时: {train_time:.1f}秒')
            logger.info(f'在第{best_epoch}个epoch损失最小为：{best_test_loss}')
            return best_epoch, best_test_loss, train_time

        def test(self, test_loader):
            self.model.eval()
            total_loss = 0.
            with torch.no_grad():
                for test_step, test_data_batch in enumerate(test_loader):
                    test_data_batch = test_data_batch.long().to(self.config.device)
                    input_data = test_data_batch[:, :-1]
                    target = test_data_batch[:, 1:]
                    output, _ = self.model(input_data)
                    active = (input_data > 0).view(-1)
                    active_output = output[active]
                    active_target = target.contiguous().view(-1)[active]
                    loss = self.criterion(active_output, active_target)
                    total_loss = total_loss + loss.item()
            return total_loss

        def generate(self, start_words):
            results = list(start_words)
            start_word_len = len(start_words)
            input_t = torch.tensor([self.config.word2idx['SOP']]).view(1, 1).long()
            input_t = input_t.to(self.config.device)
            hidden = None
            for i in range(self.config.max_gen_len):
                output, hidden = self.model(input_t, hidden)
                if i < start_word_len:
                    w = results[i]
                    input_t = input_t.data.new([self.config.word2idx[w]]).view(1, 1)
                else:
                    top_index = output.data[0].topk(1)[1][0].item()
                    w = self.config.idx2word[top_index]
                    results.append(w)
                    input_t = input_t.data.new([top_index]).view(1, 1)
                if w == 'EOP':
                    del results[-1]
                    break
            return results

    trainer = OriginalTrainer(model, config)
    best_epoch, best_test_loss, train_time = trainer.train(train_loader, test_loader)

    # 保存训练结果摘要
    with open("./results/original/train_summary.txt", "w", encoding="utf-8") as f:
        f.write("原始仓库训练复现结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"训练轮数: {config.num_epoch}\n")
        f.write(f"最佳epoch: {best_epoch}\n")
        f.write(f"最佳测试损失(原始): {best_test_loss:.4f}\n")
        f.write(f"训练耗时: {train_time:.1f}秒\n")
        f.write(f"Batch size: {config.batch_size}\n")
        f.write(f"学习率: {config.lr}\n")
        f.write(f"Embedding维度: {config.embedding_dim}\n")
        f.write(f"隐藏层维度: {config.hidden_dim}\n")
        f.write(f"\n注意：测试损失为各batch损失直接相加，未按token归一化\n")
        f.write(f"注意：训练集和测试集存在重叠（代码bug）\n")

    # ===== 推理测试 =====
    print("\n" + "=" * 60)
    print("开始推理测试")
    print("=" * 60)

    # 加载最佳模型
    model.load_state_dict(torch.load(config.save_path, map_location=device, weights_only=True))
    model.eval()

    # 使用 self.model 而不是全局 model
    trainer.model = model

    generated_samples = []

    # 普通诗歌生成
    test_prompts = ['丽日照残春', '春眠不觉晓', '登高壮观天地间', '山色空蒙雨亦奇']
    for prompt in test_prompts:
        result = trainer.generate(prompt)
        poem = "".join(result)
        generated_samples.append({"type": "普通诗歌", "prompt": prompt, "result": poem})
        print(f"  输入: {prompt}")
        print(f"  生成: {poem}")
        print()

    # 保存生成结果
    with open("./results/original/generated_samples.txt", "w", encoding="utf-8") as f:
        f.write("原始仓库生成结果\n")
        f.write("=" * 50 + "\n\n")
        for sample in generated_samples:
            f.write(f"类型: {sample['type']}\n")
            f.write(f"输入: {sample['prompt']}\n")
            f.write(f"生成: {sample['result']}\n")
            f.write("-" * 30 + "\n")

    print("\n所有结果已保存到 results/original/ 目录")
