# 多GPU分布式训练指南

本项目已支持使用 DistributedDataParallel (DDP) 进行多GPU分布式训练，可以充分利用多块GPU的显存和算力。

## 硬件要求

- **多块GPU**（推荐 4-8 块，每块至少 40GB+ 显存）
- **NCCL backend**（PyTorch 会自动使用，确保已安装对应的 CUDA 版本）

## 快速开始

### 1. 训练 Reward Models (Helpful-RM / Harmless-RM)

使用 8 块 GPU 训练 Helpful-RM：

```bash
bash scripts/train_rm_distributed.sh helpful 8
```

使用 8 块 GPU 训练 Harmless-RM：

```bash
bash scripts/train_rm_distributed.sh harmless 8
```

或者使用 `torchrun` 直接启动：

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task helpful \
    --output_dir models/helpful_rm
```

### 2. 训练 Safe-MM-DPO 策略模型

使用 8 块 GPU 训练对齐后的策略模型：

```bash
bash scripts/train_dpo_distributed.sh 8
```

或者使用 `torchrun` 直接启动：

```bash
torchrun --nproc_per_node=8 --master_port=29501 \
    -m src.training.train_safe_mm_dpo \
    --config configs/dpo_config.yaml \
    --output_dir models/aligned \
    --logging_dir logs/training/safe_mm_dpo
```

## 工作原理

### 分布式训练机制

1. **数据并行**：使用 `DistributedSampler` 将数据集分片到各个GPU，每个GPU处理不同的数据子集
2. **模型并行**：使用 `DistributedDataParallel` 包装可训练模型（policy model），自动同步梯度
3. **评估模型**：参考模型（ref_model）和奖励模型（RMs）在每个GPU上复制，但只在 eval 模式下运行

### 显存优化特性

代码已内置以下显存优化：

- **半精度训练**：自动使用 `bfloat16`（如果GPU支持）或 `float16`
- **Gradient Checkpointing**：通过牺牲部分计算时间换取显存节省
- **`use_cache=False`**：关闭 KV cache，进一步节省显存

### 有效批次大小

在多GPU训练中，**有效批次大小 = batch_size × num_gpus × gradient_accumulation_steps**

例如：
- `batch_size=8` × `8 GPUs` × `gradient_accumulation_steps=1` = **有效批次大小 64**

注意：学习率可能需要根据有效批次大小进行调整（例如使用线性缩放规则）。

## 配置建议

### RM 训练配置 (rm_config.yaml)

对于 8×64GB GPU 的配置建议：

```yaml
batch_size: 4-8          # 每个GPU的batch size
max_length: 1024         # 可根据显存调整到 512-1024
learning_rate: 2e-5      # 可能需要根据有效batch size调整
```

### DPO 训练配置 (dpo_config.yaml)

对于 8×64GB GPU 的配置建议：

```yaml
batch_size: 2-4          # 每个GPU的batch size（DPO需要更多显存）
gradient_accumulation_steps: 4-8  # 通过梯度累积增加有效batch size
learning_rate: 5e-7      # 保持原有学习率，梯度累积会自然增加有效batch
```

## 单GPU训练（向后兼容）

如果只需要使用单GPU，代码会自动检测并回退到单GPU模式：

```bash
# 单GPU训练（无需 torchrun）
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task helpful \
    --output_dir models/helpful_rm
```

## 故障排查

### 1. NCCL 错误

如果遇到 NCCL 初始化错误，尝试：

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果 InfiniBand 不可用
```

### 2. 端口冲突

如果 `master_port` 被占用，修改脚本中的端口号（如 `29502`, `29503` 等）

### 3. 显存不足

即使使用多GPU，如果每个GPU的显存仍然不足，可以：

- 减小 `batch_size`（在配置文件中）
- 减小 `max_length`（在配置文件中）
- 增加 `gradient_accumulation_steps`（仅对 DPO 训练）

## 性能监控

训练过程中，每个GPU都会输出日志。推荐使用 TensorBoard 或 Weights & Biases 来统一监控所有GPU的训练指标。

## 注意事项

1. **模型保存**：只有 rank 0（主GPU）会保存检查点，避免重复保存
2. **日志输出**：只有 rank 0 会打印训练日志，避免输出混乱
3. **数据加载**：使用 `pin_memory=True` 加速数据传输
4. **梯度同步**：DDP 会自动在所有GPU间同步梯度，确保训练一致性
