# Reward Model 训练精细调优指南

## 当前训练结果分析

### 参数调整前后对比

**调整前**（1 epoch, lr=2e-5, LoRA r=16）：
- train_loss = 0.8755
- val_loss = 0.5928
- val_acc = 68.45%

**调整后**（2 epochs, lr=1e-5, LoRA r=32）：
- train_loss = 0.6728 ✅ 下降了 23%
- val_loss = 0.6050 ⚠️ 上升了 2%
- val_acc = ?（需要确认）

### 问题分析

1. **训练损失下降**（0.8755 → 0.6728）
   - ✅ 模型在学习，训练过程正常
   - 说明更多的训练轮数和更大的模型容量有帮助

2. **验证损失上升**（0.5928 → 0.6050）
   - ⚠️ 可能的原因：
     - 学习率过小，模型收敛慢，还未充分训练
     - 训练轮数还不够（虽然增加到2，但可能还需要更多）
     - 验证集较小，存在波动
     - 学习率调度器的问题

3. **训练损失 > 验证损失**（0.6728 > 0.6050）
   - 这种情况比较少见，通常表示：
     - 训练过程可能还在初始阶段，模型尚未充分拟合训练数据
     - 验证集可能更容易（或者训练集更难）

---

## 进一步调整建议

### 方案 1：增加训练轮数（推荐）

**问题**：学习率降低后，模型可能需要更多轮数才能充分学习

**建议**：增加到 3-4 个 epoch

```yaml
num_epochs: 3  # 或 4
```

**预期效果**：
- 训练损失继续下降（0.67 → 0.55-0.60）
- 验证损失应该开始下降（0.605 → 0.55-0.58）
- 验证准确率提升（如果之前未查看，应该会看到提升）

**理由**：
- 学习率降低后，每个 epoch 的学习步幅变小
- 需要更多轮数才能达到相同的学习效果

### 方案 2：适度提高学习率

**问题**：学习率可能降得太低（1e-5），导致收敛太慢

**建议**：尝试中等学习率

```yaml
learning_rate: 1.5e-5  # 介于 1e-5 和 2e-5 之间
```

或者回到原来的学习率但增加训练轮数：

```yaml
learning_rate: 2e-5    # 回到原来
num_epochs: 3          # 但增加训练轮数
```

**预期效果**：
- 更快收敛
- 验证损失应该能降到 0.55-0.58

### 方案 3：调整学习率调度策略

**当前**：使用 linear schedule with 10% warmup

**建议**：增加 warmup 比例，或使用 cosine 调度器

修改代码（如果需要）：
```python
# 当前：10% warmup
num_warmup_steps = int(0.1 * num_training_steps)

# 建议：20-30% warmup（如果学习率较高）
num_warmup_steps = int(0.2 * num_training_steps)

# 或使用 cosine 调度器
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(...)
```

### 方案 4：检查数据质量

**可能的问题**：
- 训练集和验证集分布不一致
- 数据标注质量有问题
- 数据量不足

**检查方法**：
```bash
# 检查数据量
wc -l data/train/helpful_pairs.jsonl
wc -l data/val/helpful_pairs.jsonl

# 检查数据质量
head -n 10 data/train/helpful_pairs.jsonl | python3 -m json.tool
```

### 方案 5：降低 LoRA rank（如果过拟合风险）

**当前**：LoRA r=32

**考虑**：如果验证损失上升是由于过拟合，可以尝试：
- 保持 r=32 但增加 dropout（0.1 → 0.2）
- 或者降低 r 回到 24

但这不太可能是当前问题，因为验证损失只上升了 2%。

---

## 推荐的调整策略

### 策略 A：保守策略（推荐先尝试）

**目标**：让验证损失稳定下降

```yaml
num_epochs: 3           # 从 2 增加到 3
learning_rate: 1.5e-5   # 适度提高学习率
lora:
  r: 32
  alpha: 64
  dropout: 0.1
```

**理由**：
- 学习率 1e-5 可能过小，适度提高到 1.5e-5
- 增加 epoch 到 3，给模型更多学习机会

**预期**：
- train_loss: 0.67 → 0.55-0.60
- val_loss: 0.605 → 0.55-0.58
- val_acc: 应该提升到 73-76%

### 策略 B：激进策略

**目标**：快速验证是否学习率问题

```yaml
num_epochs: 2           # 保持 2
learning_rate: 2e-5     # 回到原来的学习率
lora:
  r: 32
  alpha: 64
  dropout: 0.1
```

**理由**：
- 如果原来是 1 epoch 训练不足，现在 2 epochs + 更大的模型可能足够
- 验证是否学习率降低导致的问题

### 策略 C：平衡策略（长期优化）

```yaml
num_epochs: 3
learning_rate: 1.5e-5
batch_size: 6           # 如果显存允许，稍微增加
gradient_accumulation_steps: 2
lora:
  r: 32
  alpha: 64
  dropout: 0.15         # 稍微增加 dropout 防止过拟合
```

---

## 诊断步骤

### 1. 查看完整的训练日志

检查每个 epoch 的损失变化：

```bash
grep "Epoch\|val_loss\|val_acc" logs/training/rm/helpful_*.log
```

**关注**：
- Epoch 1 和 Epoch 2 的损失变化
- 验证准确率是否提升
- 训练损失下降是否平滑

### 2. 检查是否有过拟合迹象

**过拟合的典型表现**：
- 训练损失持续下降，但验证损失开始上升
- 训练损失 << 验证损失（差距很大）

**当前情况**：
- 训练损失 (0.67) > 验证损失 (0.605) ← 这不是过拟合
- 更像是训练不足

### 3. 验证准确率是关键指标

**重要**：请确认验证准确率（val_acc）是多少？

如果验证准确率提升了（例如从 68% 提升到 72%），即使验证损失略有上升，模型实际上是在改进的。

---

## 具体操作建议

### 第一步：确认验证准确率

```bash
# 查看最新的训练日志
tail -n 50 logs/training/rm/helpful_*.log | grep "val_acc"
```

**如果 val_acc 提升了**：
- 即使 val_loss 略有上升，模型也在改进
- 可以继续训练更多 epoch

**如果 val_acc 没有提升或下降**：
- 可能需要调整学习率或其他参数

### 第二步：尝试策略 A（推荐）

修改配置文件：

```yaml
# configs/rm_config.yaml
num_epochs: 3
learning_rate: 1.5e-5  # 适度提高
# 其他参数保持不变
```

重新训练：

```bash
bash scripts/train_rm_distributed.sh helpful 8
```

### 第三步：如果策略 A 不够好

尝试策略 B（回到原来学习率）：

```yaml
num_epochs: 3          # 但保持 3 epochs
learning_rate: 2e-5    # 回到原来的学习率
```

---

## 预期改进路径

### 理想情况下的损失曲线

```
Epoch 1: train_loss=0.87, val_loss=0.59, val_acc=68%
Epoch 2: train_loss=0.67, val_loss=0.61, val_acc=71%  ← 当前
Epoch 3: train_loss=0.58, val_loss=0.57, val_acc=74%  ← 目标
Epoch 4: train_loss=0.52, val_loss=0.55, val_acc=76%  ← 理想
```

### 如果学习率调整后

```
Epoch 1: train_loss=0.87, val_loss=0.59, val_acc=68%
Epoch 2: train_loss=0.65, val_loss=0.58, val_acc=72%
Epoch 3: train_loss=0.56, val_loss=0.54, val_acc=75%
```

---

## 关键观察点

在下次训练时，注意观察：

1. **每个 epoch 的损失变化**
   - 训练损失应该持续下降
   - 验证损失应该在某个点开始下降

2. **验证准确率趋势**
   - 这是更直观的指标
   - 应该随着训练逐渐提升

3. **损失差距**
   - train_loss 和 val_loss 的差距
   - 如果差距过大（>0.2），可能有问题

---

## 总结

**当前状态**：
- ✅ 训练损失明显下降（改进中）
- ⚠️ 验证损失略有上升（需要调整）
- ❓ 验证准确率未知（需要确认）

**立即行动**：

1. **确认验证准确率**（最重要）
   ```bash
   tail -n 50 logs/training/rm/helpful_*.log | grep "val_acc"
   ```

2. **如果 val_acc 提升了**：
   - 继续增加 epoch 到 3-4
   - 或适度提高学习率到 1.5e-5

3. **如果 val_acc 没有提升**：
   - 尝试回到原来学习率（2e-5）但保持 3 epochs
   - 或检查数据质量

**推荐配置**（下一步尝试）：
```yaml
num_epochs: 3
learning_rate: 1.5e-5
lora:
  r: 32
  alpha: 64
```

Good luck! 🎯
