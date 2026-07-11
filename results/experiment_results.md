# 实验结果总览

| 实验 | 模型 | 配置 | Best Valid Loss | PPL | 最优Epoch |
|------|------|------|:---:|:---:|:---:|
| **E1** | BaselineLSTM | lr=0.001, bs=64, Adam | 4.5141 | 91.3 | 30 |
| **E3** | ConditionalAttnLSTM | lr=0.001, bs=64, AdamW | 4.2534 | 70.4 | 30 |
| **E4a** | ConditionalAttnLSTM | lr=0.0005, bs=64, AdamW | 4.2941 | 73.3 | 30 |
| **E4b** | ConditionalAttnLSTM | **lr=0.001, bs=32**, AdamW | **4.2383** | **69.2** | 30 |
| **E4c** | ConditionalAttnLSTM | lr=0.001, bs=64, **epoch=50** | 4.2393 | 69.4 | 47 |

## 关键结论

- ConditionalAttnLSTM 显著优于 BaselineLSTM（PPL 91.3 → 69.2，降低 24%）
- batch_size=32 效果最好（E4b 的 Valid Loss 最低）
- lr=0.0005 收敛更慢，不如 lr=0.001
- 30 epoch + 早停基本够用（50 epoch 与 30 epoch 无明显差异）
- 早停判定：E4c 在 epoch 47 触发（50 epoch 配置）
