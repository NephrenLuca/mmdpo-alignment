# 显存优化方案总结

## 问题现状

训练 Mistral-7B Reward Model 时，即使在8×64GB GPU上使用：
- `batch_size=1`
- `max_length=384`
- `gradient_accumulation_steps=8`

仍然在优化器初始化时OOM（显存占用约61GB，接近64GB上限）。

## 根本原因

全参数微调（Full Fine-tuning）的显存瓶颈：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| 模型参数（bf16） | ~14GB | 固定，必须加载 |
| **优化器状态（fp32）** | **~28GB** | **AdamW需要2倍参数大小的状态** ⚠️ |
| **梯度（bf16）** | **~14GB** | 与参数大小相同 ⚠️ |
| 激活值 | ~5-8GB | 取决于batch_size和max_length |
| **总计** | **~61GB** | 接近64GB上限 |

**优化器状态是最大的瓶颈**，因为它需要存储每个参数的momentum和variance，且使用fp32精度。

## 解决方案：使用LoRA（⭐推荐）

### LoRA优势

LoRA只训练<1%的参数，可以：

1. **优化器状态显存**：从28GB降到~0.2GB（节省27.8GB）
2. **梯度显存**：从14GB降到~0.2GB（节省13.8GB）
3. **总显存占用**：从~61GB降到~19-23GB（节省约40GB）

### 推荐配置

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.1
batch_size: 4           # 可以使用更大的batch_size
max_length: 512         # 可以使用更长的序列
gradient_accumulation_steps: 2
learning_rate: 2e-5     # 或1e-4
```

### 预期显存占用（LoRA）

- 模型参数（bf16）：~14GB（只加载，不训练）
- 优化器状态（fp32）：~0.2GB ✅
- 梯度（bf16）：~0.2GB ✅
- 激活值（batch_size=4, max_length=512）：~5-8GB
- **总计：~19-23GB** ✅✅✅

### 使用方法

1. 在 `configs/rm_config.yaml` 中启用LoRA：
   ```yaml
   use_lora: true
   lora:
     r: 16
     alpha: 32
     dropout: 0.1
   ```

2. 正常启动训练：
   ```bash
   bash scripts/train_rm_distributed.sh helpful 8
   ```

详细说明请查看 [LORA_OPTIMIZATION.md](LORA_OPTIMIZATION.md)

## 其他优化方案（不推荐）

### 方案1：进一步减小batch_size和max_length

```yaml
batch_size: 1
max_length: 256
gradient_accumulation_steps: 16
```

- **问题**：激活值显存节省有限（从5-8GB降到3-5GB），总计仍需要~57-59GB，容易OOM
- **代价**：训练效率低，序列长度过短可能影响效果

### 方案2：8-bit优化器（bitsandbytes）

使用8-bit AdamW优化器可以：
- 将优化器状态从28GB降到~7GB
- 总显存从~61GB降到~40GB

- **问题**：
  - 需要额外依赖 `bitsandbytes`
  - 可能有数值稳定性问题
  - 仍然不如LoRA节省的显存多

### 方案3：CPU Offload

将优化器状态offload到CPU内存：
- **问题**：训练速度会显著降低（2-3倍慢）

## 关于Attention Warning

```
UserWarning: Torch was not compiled with memory efficient attention
```

**含义**：PyTorch没有编译Flash Attention优化

**影响**：
- 激活值显存可能多占用约2-4GB
- 计算速度较慢

**建议**：**忽略这个警告**

原因：
1. 使用LoRA后，显存问题已解决
2. 显存瓶颈主要在优化器状态，而不是激活值
3. 重新编译PyTorch成本高、收益有限

详细说明请查看 [ATTENTION_WARNING.md](ATTENTION_WARNING.md)

## 总结

| 方案 | 显存占用 | 推荐度 | 说明 |
|------|---------|--------|------|
| **LoRA微调** | **~19-23GB** | ⭐⭐⭐⭐⭐ | **强烈推荐**，显存节省最多 |
| 全参数微调（最小配置） | ~57-59GB | ⭐ | 容易OOM，不推荐 |
| 8-bit优化器 | ~40GB | ⭐⭐⭐ | 不如LoRA，且有稳定性风险 |
| CPU Offload | ~35GB | ⭐⭐ | 训练速度太慢 |

**建议：直接使用LoRA方案，这是最有效且稳定的解决方案。**
