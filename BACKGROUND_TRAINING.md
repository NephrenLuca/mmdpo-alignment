# 后台训练使用指南

训练脚本现在支持后台运行模式，即使关闭终端，训练也会继续进行。

## 使用方法

### 1. Reward Model 训练

#### 前台运行（默认）
```bash
bash scripts/train_rm_distributed.sh helpful 8
```
- 训练在前台运行，可以看到实时输出
- 按 `Ctrl+C` 可以中断训练
- 日志会自动保存到 `logs/training/rm/helpful_YYYYMMDD_HHMMSS.log`

#### 后台运行（推荐）
```bash
bash scripts/train_rm_distributed.sh helpful 8 --background
# 或简写
bash scripts/train_rm_distributed.sh helpful 8 -b
```

启动后会显示：
```
Training started in background!
Process ID (PID): 12345
Log file: logs/training/rm/helpful_20241220_143022.log
PID file: logs/training/rm/helpful_20241220_143022.pid

To monitor training:
  tail -f logs/training/rm/helpful_20241220_143022.log

To check if training is running:
  ps -p 12345

To stop training:
  kill 12345
```

### 2. Safe-MM-DPO 训练

#### 前台运行（默认）
```bash
bash scripts/train_dpo_distributed.sh 8
```

#### 后台运行（推荐）
```bash
bash scripts/train_dpo_distributed.sh 8 --background
# 或简写
bash scripts/train_dpo_distributed.sh 8 -b
```

## 监控和管理

### 查看训练状态

```bash
bash scripts/check_training.sh status
```

输出示例：
```
=== Checking Training Processes ===

Reward Model Training:
  [RUNNING] helpful (PID: 12345, Started: 20241220_143022)
    Log: logs/training/rm/helpful_20241220_143022.log

Safe-MM-DPO Training:
  [RUNNING] DPO Training (PID: 12346, Started: 20241220_150000)
    Log: logs/training/safe_mm_dpo/train_20241220_150000.log

GPU Usage:
  GPU 0 (NVIDIA A100): 85% util, 45000/80000 MB
  ...
```

### 查看最新日志

```bash
bash scripts/check_training.sh logs
```

这会自动找到最新的日志文件并使用 `tail -f` 实时显示。

### 停止训练

```bash
bash scripts/check_training.sh stop
```

这会停止所有正在运行的训练进程。

## 日志文件

### 日志位置

- **RM训练日志**：`logs/training/rm/<task>_<timestamp>.log`
- **DPO训练日志**：`logs/training/safe_mm_dpo/train_<timestamp>.log`

### 日志内容

日志文件包含：
- 所有标准输出（stdout）
- 所有错误输出（stderr）
- 训练进度信息
- 损失值和指标

### 查看日志

```bash
# 查看完整日志
cat logs/training/rm/helpful_20241220_143022.log

# 实时跟踪日志
tail -f logs/training/rm/helpful_20241220_143022.log

# 查看最后100行
tail -n 100 logs/training/rm/helpful_20241220_143022.log

# 搜索特定内容
grep "lambda" logs/training/safe_mm_dpo/train_20241220_150000.log
```

## 进程管理

### 手动查找进程

```bash
# 查找所有Python训练进程
ps aux | grep train_safe_mm_dpo
ps aux | grep train_rm

# 查找torchrun进程
ps aux | grep torchrun
```

### 手动停止进程

```bash
# 如果知道PID
kill <PID>

# 强制停止（如果普通kill不起作用）
kill -9 <PID>

# 停止所有Python训练进程（谨慎使用）
pkill -f train_safe_mm_dpo
pkill -f train_rm
```

### 使用PID文件

每个后台训练任务都会创建对应的PID文件：

```bash
# 读取PID
cat logs/training/rm/helpful_20241220_143022.pid

# 检查进程是否还在运行
ps -p $(cat logs/training/rm/helpful_20241220_143022.pid)

# 停止特定进程
kill $(cat logs/training/rm/helpful_20241220_143022.pid)
```

## 最佳实践

### 1. 使用后台模式

对于长时间训练，**强烈推荐使用后台模式**：

```bash
bash scripts/train_rm_distributed.sh helpful 8 --background
```

**优势**：
- 即使关闭终端，训练也继续
- SSH断开连接不会中断训练
- 可以随时查看日志

### 2. 使用screen或tmux（可选）

如果需要交互式会话，也可以使用 `screen` 或 `tmux`：

```bash
# 使用screen
screen -S training
bash scripts/train_dpo_distributed.sh 8
# 按 Ctrl+A, D 分离会话

# 重新连接
screen -r training

# 使用tmux
tmux new -s training
bash scripts/train_dpo_distributed.sh 8
# 按 Ctrl+B, D 分离会话

# 重新连接
tmux attach -t training
```

但后台模式更简单，不需要额外的工具。

### 3. 定期检查日志

```bash
# 设置定时检查（每10分钟）
watch -n 600 'bash scripts/check_training.sh status'
```

### 4. 日志轮转（可选）

如果训练时间很长，日志文件可能变得很大。可以手动清理旧的日志：

```bash
# 只保留最近7天的日志
find logs/training -name "*.log" -mtime +7 -delete
find logs/training -name "*.pid" -mtime +7 -delete
```

## 故障排查

### 训练意外停止

1. **检查日志**：查看日志文件的最后几行，寻找错误信息
   ```bash
   tail -n 50 logs/training/safe_mm_dpo/train_*.log
   ```

2. **检查系统资源**：
   ```bash
   # 检查GPU
   nvidia-smi
   
   # 检查内存
   free -h
   
   # 检查磁盘空间
   df -h
   ```

3. **检查进程状态**：
   ```bash
   bash scripts/check_training.sh status
   ```

### SSH断开后训练停止

确保使用后台模式：
```bash
bash scripts/train_dpo_distributed.sh 8 --background
```

而不是：
```bash
bash scripts/train_dpo_distributed.sh 8  # 这样会在SSH断开时停止
```

### 日志文件找不到

检查日志目录是否存在：
```bash
ls -la logs/training/
```

如果目录不存在，训练脚本会自动创建。

## 示例工作流

### 启动训练

```bash
# 1. 启动RM训练（后台）
bash scripts/train_rm_distributed.sh helpful 8 --background

# 2. 检查状态
bash scripts/check_training.sh status

# 3. 查看日志
bash scripts/check_training.sh logs
# 或直接指定日志文件
tail -f logs/training/rm/helpful_*.log
```

### 监控训练

```bash
# 定期检查状态
bash scripts/check_training.sh status

# 实时查看日志
tail -f logs/training/rm/helpful_*.log

# 检查GPU使用
watch -n 5 nvidia-smi
```

### 停止训练

```bash
# 方法1：使用管理脚本
bash scripts/check_training.sh stop

# 方法2：使用PID文件
kill $(cat logs/training/rm/helpful_*.pid)
```
