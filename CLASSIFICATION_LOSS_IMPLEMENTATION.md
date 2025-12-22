# Cost Model Classification Loss 实现说明

根据论文 **2310.12773v1 (Safe RLHF)** 的要求，为 Cost Model（harmless RM）添加了 classification loss。

## 论文要求

根据论文公式 6，Cost Model 的损失函数包含两部分：

```
L_C(θ; D_C) = -E[log σ(C_ψ(y_w, x) - C_ψ(y_l, x))]           (pairwise comparison)
              -E[log σ(s_w · C_ψ(y_w, x)) + log σ(s_l · C_ψ(y_l, x))]  (classification)
```

其中：
- `s(y) = +1` 表示 response 是有害的
- `s(y) = -1` 表示 response 是无害的

## 实现修改

### 1. 数据处理 (`src/data_processing/data_preprocessor.py`)

**修改**：在构建 harmless 偏好对时，提取并保存 safety labels

```python
# 从 PKU-SafeRLHF 数据集中提取 safety labels
# is_response_X_safe = True -> s(y) = -1 (无害)
# is_response_X_safe = False -> s(y) = +1 (有害)
if dimension == "harmless":
    is_safe_chosen = rec.get(f"is_response_{idx}_safe")
    is_safe_rejected = rec.get(f"is_response_{1 - idx}_safe")
    
    if is_safe_chosen is not None and is_safe_rejected is not None:
        safety_labels = {
            "chosen": -1 if is_safe_chosen else +1,
            "rejected": -1 if is_safe_rejected else +1,
        }
        pair_record["safety_labels"] = safety_labels
```

### 2. 数据集 (`src/training/train_rm.py`)

**修改**：`PreferenceDataset` 现在支持读取 safety labels

```python
# 如果存在safety labels，添加到sample中
if "safety_labels" in rec:
    safety_labels = rec["safety_labels"]
    sample["safety_labels"] = {
        "chosen": torch.tensor(safety_labels["chosen"], dtype=torch.float32),
        "rejected": torch.tensor(safety_labels["rejected"], dtype=torch.float32),
    }
```

### 3. Classification Loss 函数

**新增**：`classification_loss` 函数（论文公式 6 的第二项）

```python
def classification_loss(scores: torch.Tensor, safety_labels: torch.Tensor) -> torch.Tensor:
    """
    损失函数：-log(sigmoid(s(y) * C(y, x)))
    - 当 s(y) = +1 且 C > 0 时，损失小（正确预测有害）
    - 当 s(y) = -1 且 C < 0 时，损失小（正确预测无害）
    """
    signed_scores = safety_labels * scores
    return -torch.nn.functional.logsigmoid(signed_scores).mean()
```

### 4. 训练循环 (`src/training/train_rm.py`)

**修改**：`train_one_epoch` 现在支持 classification loss

```python
# Pairwise ranking loss (论文公式 6 的第一项)
loss = preference_loss(scores_c, scores_r)

# Classification loss (论文公式 6 的第二项，仅用于Cost Model)
if classification_loss_weight > 0.0 and "safety_labels" in batch:
    safety_labels_chosen = batch["safety_labels"]["chosen"]
    safety_labels_rejected = batch["safety_labels"]["rejected"]
    
    cls_loss_chosen = classification_loss(scores_c, safety_labels_chosen)
    cls_loss_rejected = classification_loss(scores_r, safety_labels_rejected)
    cls_loss = (cls_loss_chosen + cls_loss_rejected) / 2.0
    
    # 总损失 = ranking loss + classification loss
    loss = loss + classification_loss_weight * cls_loss
```

### 5. 配置 (`configs/rm_config.yaml`)

**新增**：`classification_loss_weight` 参数

```yaml
# Cost Model配置（仅用于harmless任务）
classification_loss_weight: 1.0  # 仅用于harmless RM（Cost Model）
```

## 使用说明

### 1. 数据准备

确保使用 PKU-SafeRLHF 数据集，其中包含 `is_response_0_safe` 和 `is_response_1_safe` 字段：

```bash
python -m src.scripts.prepare_data \
    --input_dir data/raw \
    --output_dir data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### 2. 训练 Helpful RM

Helpful RM 不使用 classification loss（因为数据中没有 safety labels）：

```bash
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task helpful \
    --output_dir models/helpful_rm
```

### 3. 训练 Harmless RM (Cost Model)

Harmless RM 会自动使用 classification loss（如果数据中包含 safety labels）：

```bash
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task harmless \
    --output_dir models/harmless_rm
```

**注意**：
- 如果数据中没有 safety labels，classification loss 会被跳过（不会报错）
- `classification_loss_weight` 控制 classification loss 的权重
- 建议设置为 1.0（与 pairwise loss 同等权重）

## 理论依据

根据论文，Cost Model 需要同时学习：
1. **Pairwise comparison**：学习相对安全性偏好（哪个 response 更安全）
2. **Classification**：学习绝对安全性判断（response 是否有害）

这种设计使得 Cost Model 能够：
- 明确区分安全/不安全簇（如论文 Figure 2a 所示）
- 为动态平衡 helpfulness 和 harmlessness 提供基础（论文 Section 4.2.4）

## 验证

训练 harmless RM 时，应该看到：
- 训练损失包含两部分：pairwise loss 和 classification loss
- Cost Model 能够区分安全/不安全的 responses
- 在 DPO 训练中，J_C 估计更准确（因为 Cost Model 更可靠）

## 参考文献

- Dai, J., Pan, X., Sun, R., et al. (2023). Safe RLHF: Safe Reinforcement Learning from Human Feedback. arXiv:2310.12773v1
- 论文公式 6：Cost Model 的完整损失函数
- 论文 Section 3.2：Preference Model Fitting
- 论文 Section 4.2.4：Design of Cost Preference Model
