# 显存问题排查指南

## 当前配置（经过优化的）

```yaml
batch_size: 1
max_length: 384
gradient_accumulation_steps: 8
```

这个配置应该能在 8×64GB GPU 上正常运行。

## 如果仍然OOM

### 1. 检查实际显存使用

在训练开始前，代码会打印：
```
GPU memory allocated: XX.XX GB
GPU memory reserved: XX.XX GB
```

如果已分配的显存接近64GB，说明模型本身占用过大。

### 2. 进一步减小配置

#### 选项A：更小的序列长度
```yaml
batch_size: 1
max_length: 256  # 从384降到256
gradient_accumulation_steps: 16  # 相应增加以保持有效batch_size
```

#### 选项B：更小的batch size（如果已经是1，跳过）
如果 `batch_size=1` 仍然OOM，可能需要考虑：
- 使用CPU offload（会很慢）
- 使用更小的模型
- 使用量化（8-bit或4-bit）

### 3. 检查优化器状态

AdamW优化器的状态（momentum和variance）默认使用fp32，占用约28GB。这是7B模型全参数训练的固有开销，难以避免。

### 4. 使用CPU Offload（最后手段）

如果必须运行但显存不足，可以考虑将优化器状态offload到CPU：

```python
# 这需要修改代码，使用FSDP或自定义优化器
# 注意：这会显著降低训练速度（约2-3倍）
```

### 5. 检查其他进程

确保没有其他进程占用GPU显存：

```bash
nvidia-smi
```

### 6. 重启Python进程

有时显存碎片会导致OOM，即使理论上显存足够。尝试：
1. 完全退出Python进程
2. 重新启动训练

### 7. 验证环境变量

确保 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已设置（启动脚本已包含）。

## 显存占用估算

对于 Mistral-7B Reward Model：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| 模型参数（bf16） | ~14GB | 固定，无法减少 |
| 优化器状态（fp32） | ~28GB | AdamW的momentum+variance，固定 |
| 梯度（bf16） | ~14GB | 与参数相同大小 |
| 激活值 | 可变 | 取决于batch_size和max_length |

**激活值估算公式**（近似）：
- 每个样本约：`max_length * hidden_size * num_layers * 4 bytes / batch_size`
- 对于batch_size=1, max_length=384: 约 3-5GB
- 对于batch_size=2, max_length=512: 约 5-8GB

## 调试技巧

在代码中添加显存监控：

```python
import torch

# 在训练循环中
if step % 10 == 0:
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Step {step}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
```

观察显存使用趋势，找出显存峰值出现的位置。
