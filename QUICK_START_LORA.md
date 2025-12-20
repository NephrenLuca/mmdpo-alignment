# LoRA快速开始指南

## 为什么使用LoRA？

训练Mistral-7B Reward Model时，全参数微调需要约**61GB显存**，在64GB GPU上容易OOM。

使用LoRA可以将显存占用降低到**19-23GB**，节省约40GB显存！

## 快速配置（3步）

### 1. 修改配置文件

编辑 `configs/rm_config.yaml`：

```yaml
use_lora: true          # 启用LoRA
lora:
  r: 16                 # LoRA rank（推荐16）
  alpha: 32             # LoRA alpha（通常为r的2倍）
  dropout: 0.1          # LoRA dropout

# 使用LoRA后可以使用更大的batch_size和max_length
batch_size: 4
max_length: 512
gradient_accumulation_steps: 2
```

### 2. 启动训练

```bash
bash scripts/train_rm_distributed.sh helpful 8
```

### 3. 验证显存使用

训练开始后，你应该看到：

```
LoRA enabled: 4,194,304 trainable / 7,240,192,000 total parameters (0.06%)
GPU memory allocated: 19.23 GB
GPU memory reserved: 19.48 GB
```

显存占用应该在**20GB左右**，远低于64GB上限 ✅

## 参数说明

### LoRA参数

- **r (rank)**：控制可训练参数数量
  - `r=8`：最少参数，最少显存
  - `r=16`：平衡选择（推荐）⭐
  - `r=32`：更多参数，可能效果更好

- **alpha**：LoRA缩放因子，通常设为 `r * 2`
  - `alpha=16` for `r=8`
  - `alpha=32` for `r=16` ⭐
  - `alpha=64` for `r=32`

- **dropout**：LoRA dropout，推荐 `0.1`

### 训练参数

使用LoRA后，可以：
- **增加batch_size**：从1增加到4或更大
- **增加max_length**：从384增加到512或更大
- **减少gradient_accumulation_steps**：从8减少到2

这样可以在节省显存的同时，保持相同的有效batch_size，甚至提升训练效率。

## 显存对比

| 配置 | 显存占用 | 说明 |
|------|---------|------|
| 全参数微调（最小） | ~61GB | 接近上限，容易OOM |
| LoRA (r=16) | ~19-23GB | **推荐** ⭐⭐⭐⭐⭐ |

## 模型保存

使用LoRA训练的模型会保存为：
- **adapter权重**：只有LoRA的参数（几MB）
- 保存路径：`models/helpful_rm/` 或 `models/harmless_rm/`

加载时需要同时加载：
1. 原始Mistral-7B模型
2. LoRA adapter权重

## 常见问题

### Q: LoRA效果会差很多吗？

A: 通常能达到全参数微调90-95%的效果，对于RM训练来说完全够用。

### Q: 可以继续微调LoRA模型吗？

A: 可以，LoRA模型可以继续使用LoRA微调，也可以合并权重后全参数微调。

### Q: 如果还是OOM怎么办？

A: 
1. 降低`r`值（如从16降到8）
2. 减小`batch_size`（如从4降到2）
3. 减小`max_length`（如从512降到384）

## 更多信息

- 详细说明：[LORA_OPTIMIZATION.md](LORA_OPTIMIZATION.md)
- 优化总结：[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)
- Attention Warning说明：[ATTENTION_WARNING.md](ATTENTION_WARNING.md)
