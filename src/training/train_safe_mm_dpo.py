"""
Safe-MM-DPO training script.

核心目标：
- 使用两个奖励模型（Helpful-RM / Harmless-RM）分别刻画“有帮助性”和“无害性”；
- 在 DPO 框架上引入 MM-DPO 的动态 β 缩放；
- 使用拉格朗日乘子 λ 动态平衡两个目标（Safe RLHF 思想）。

核心创新点（在实现中有对应注释）：
1. 双评判员：Helpful-RM & Harmless-RM 提供各自的奖励边际 δ_H, δ_S；
2. 动态缩放因子 β(δ)：高置信度样本获得更大权重（MM-DPO 思想）；
3. 拉格朗日乘子 λ：根据有害性成本 J_C 自适应调整，平衡 L_H 与 L_S（Safe RLHF 思想）。

用法（示例）：

    python -m src.training.train_safe_mm_dpo \\
        --config configs/dpo_config.yaml \\
        --output_dir models/aligned \\
        --logging_dir logs/training/safe_mm_dpo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import yaml

from src.models.reward_model import load_reward_model, RewardModelConfig


class PreferenceDataset(Dataset):
    """
    Dataset of preference pairs (prompt, chosen, rejected) stored in jsonl.
    """

    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[Dict] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        rec = self.records[idx]
        prompt = rec["prompt"]
        chosen = rec["chosen_response"]
        rejected = rec["rejected_response"]

        # 这里简单地将 prompt + 响应拼接作为条件序列
        text_w = f"{prompt}\n{chosen}"
        text_l = f"{prompt}\n{rejected}"

        tok_w = self.tokenizer(
            text_w,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok_l = self.tokenizer(
            text_l,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": tok_w["input_ids"].squeeze(0),
            "chosen_attention_mask": tok_w["attention_mask"].squeeze(0),
            "rejected_input_ids": tok_l["input_ids"].squeeze(0),
            "rejected_attention_mask": tok_l["attention_mask"].squeeze(0),
        }


def collate_fn(tokenizer):
    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        def pad(key: str, pad_value: int) -> torch.Tensor:
            return torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch],
                batch_first=True,
                padding_value=pad_value,
            )

        return {
            "chosen_input_ids": pad("chosen_input_ids", tokenizer.pad_token_id),
            "chosen_attention_mask": pad("chosen_attention_mask", 0),
            "rejected_input_ids": pad("rejected_input_ids", tokenizer.pad_token_id),
            "rejected_attention_mask": pad("rejected_attention_mask", 0),
        }

    return _collate


@dataclass
class DPOConfig:
    λ_init: float
    w: float
    k: float
    β_ori: float
    learning_rate: float
    batch_size: int
    kl_coeff: float
    λ_lr: float
    epochs: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    warmup_steps: int

    policy_model_path: str
    ref_model_path: str
    helpful_rm_path: str
    harmless_rm_path: str

    train_helpful_path: Path
    train_harmless_path: Path
    val_helpful_path: Path
    val_harmless_path: Path


def load_dpo_config(path: Path) -> DPOConfig:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return DPOConfig(
        λ_init=float(cfg["λ_init"]),
        w=float(cfg["w"]),
        k=float(cfg["k"]),
        β_ori=float(cfg["β_ori"]),
        learning_rate=float(cfg["learning_rate"]),
        batch_size=int(cfg["batch_size"]),
        kl_coeff=float(cfg["kl_coeff"]),
        λ_lr=float(cfg["λ_lr"]),
        epochs=int(cfg["epochs"]),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        policy_model_path=cfg["policy_model_path"],
        ref_model_path=cfg["ref_model_path"],
        helpful_rm_path=cfg["helpful_rm_path"],
        harmless_rm_path=cfg["harmless_rm_path"],
        train_helpful_path=Path(cfg["train_helpful_path"]),
        train_harmless_path=Path(cfg["train_harmless_path"]),
        val_helpful_path=Path(cfg["val_helpful_path"]),
        val_harmless_path=Path(cfg["val_harmless_path"]),
    )


def compute_log_probs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个样本的平均 token log-prob（近似 log π(y|x)）。

    为简化实现，我们直接对整个序列（包括 prompt）求平均 log-prob。
    理论上 DPO 只需要响应部分，但在大多数实践中这种近似是可接受的。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    # 右移一位，与 labels 对齐
    shift_log_probs = log_probs[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    shift_attention_mask = attention_mask[:, 1:]

    # gather log p(token)
    token_logp = shift_log_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)

    # 按有效 token 求平均
    token_logp = token_logp * shift_attention_mask
    lengths = shift_attention_mask.sum(dim=-1).clamp(min=1)
    seq_logp = token_logp.sum(dim=-1) / lengths
    return seq_logp


def mm_dpo_loss(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    logp_ref_w: torch.Tensor,
    logp_ref_l: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    MM-DPO 损失：
        L = - E[ log σ( β * ( Δθ - Δref ) ) ]
    其中 Δ = log π(y_w|x) - log π(y_l|x)
    """
    delta_theta = logp_w - logp_l
    delta_ref = logp_ref_w - logp_ref_l
    advantage = beta * (delta_theta - delta_ref)
    return -torch.nn.functional.logsigmoid(advantage).mean()


def kl_divergence(
    policy: AutoModelForCausalLM,
    ref: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    近似 KL(π_θ || π_ref)，在一个 batch 上做经验估计。
    """
    with torch.no_grad():
        ref_logits = ref(input_ids=input_ids, attention_mask=attention_mask).logits
    pi_logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits

    pi_logp = torch.log_softmax(pi_logits, dim=-1)
    ref_logp = torch.log_softmax(ref_logits, dim=-1)

    kl = torch.sum(
        torch.exp(pi_logp) * (pi_logp - ref_logp),
        dim=-1,
    )  # (B, T)
    kl = (kl * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
    return kl.mean()


def dynamic_beta(β_ori: float, w: float, k: float, delta: torch.Tensor) -> torch.Tensor:
    """
    MM-DPO 动态缩放因子：
        β(δ) = β_ori * (1 + w * (1 - exp(-k * δ)))
    高置信度（|δ| 大）的样本拥有更大的 β，从而在训练中权重更高。
    """
    return β_ori * (1.0 + w * (1.0 - torch.exp(-k * delta)))


def run_training(
    cfg: DPOConfig,
    output_dir: Path,
    logging_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 加载策略模型与参考模型 ======
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(cfg.policy_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.ref_model_path)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    policy_model.to(device)
    ref_model.to(device)

    # ====== 加载两个奖励模型（双评判员）======
    helpful_rm_cfg = RewardModelConfig(
        base_model_path=cfg.helpful_rm_path,
        tokenizer_name=cfg.helpful_rm_path,
        max_length=tokenizer.model_max_length,
    )
    harmless_rm_cfg = RewardModelConfig(
        base_model_path=cfg.harmless_rm_path,
        tokenizer_name=cfg.harmless_rm_path,
        max_length=tokenizer.model_max_length,
    )
    helpful_rm, _ = load_reward_model(helpful_rm_cfg)
    harmless_rm, _ = load_reward_model(harmless_rm_cfg)
    helpful_rm.to(device).eval()
    harmless_rm.to(device).eval()
    for p in helpful_rm.parameters():
        p.requires_grad = False
    for p in harmless_rm.parameters():
        p.requires_grad = False

    # ====== 构建数据集（helpful / harmless 共用同一结构）======
    train_helpful = PreferenceDataset(cfg.train_helpful_path, tokenizer, tokenizer.model_max_length)
    train_harmless = PreferenceDataset(cfg.train_harmless_path, tokenizer, tokenizer.model_max_length)

    # 合并两个维度的训练数据（简单拼接）
    train_records = train_helpful.records + train_harmless.records
    # 复用 Dataset 逻辑：临时写一个 in-memory 版
    class CombinedDataset(PreferenceDataset):
        def __init__(self, records, tokenizer, max_length):
            self.records = records
            self.tokenizer = tokenizer
            self.max_length = max_length

    train_ds = CombinedDataset(train_records, tokenizer, tokenizer.model_max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer),
    )

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.learning_rate)
    num_training_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 拉格朗日乘子 λ（核心：动态平衡 helpful 与 harmless 损失）
    lambda_param = torch.tensor(cfg.λ_init, device=device)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        policy_model.train()
        for step, batch in enumerate(train_loader, start=1):
            global_step += 1

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # ====== 1. 计算策略与参考模型的 log π(y|x) ======
            logp_w = compute_log_probs(policy_model, chosen_ids, chosen_mask)
            logp_l = compute_log_probs(policy_model, rejected_ids, rejected_mask)
            with torch.no_grad():
                logp_ref_w = compute_log_probs(ref_model, chosen_ids, chosen_mask)
                logp_ref_l = compute_log_probs(ref_model, rejected_ids, rejected_mask)

            # ====== 2. 计算奖励边际 δ_H, δ_S（双评判员）======
            with torch.no_grad():
                # Helpful-RM
                r_helpful_w = helpful_rm(chosen_ids, chosen_mask)
                r_helpful_l = helpful_rm(rejected_ids, rejected_mask)
                delta_H = r_helpful_w - r_helpful_l

                # Harmless-RM
                r_safe_w = harmless_rm(chosen_ids, chosen_mask)
                r_safe_l = harmless_rm(rejected_ids, rejected_mask)
                delta_S = r_safe_w - r_safe_l

            # ====== 3. 动态缩放因子 β_H, β_S（MM-DPO 核心）======
            beta_H = dynamic_beta(cfg.β_ori, cfg.w, cfg.k, delta_H)
            beta_S = dynamic_beta(cfg.β_ori, cfg.w, cfg.k, delta_S)

            # ====== 4. 计算两个维度的 MM-DPO 损失 ======
            loss_H = mm_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta_H)
            loss_S = mm_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta_S)

            # 总损失：Safe RLHF 风格的加权和
            # L_Total = 1/(1+λ) * L_H + λ/(1+λ) * L_S
            weight_helpful = 1.0 / (1.0 + lambda_param)
            weight_safety = lambda_param / (1.0 + lambda_param)
            loss_total = weight_helpful * loss_H + weight_safety * loss_S

            # 可选 KL 约束，防止策略偏离参考模型过远
            if cfg.kl_coeff > 0:
                kl = kl_divergence(policy_model, ref_model, chosen_ids, chosen_mask)
                loss_total = loss_total + cfg.kl_coeff * kl
            else:
                kl = torch.tensor(0.0, device=device)

            loss_total = loss_total / cfg.gradient_accumulation_steps
            loss_total.backward()

            if step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # ====== 5. 更新拉格朗日乘子 λ（控制安全性约束强度）======
                # 简单近似：将有害性成本 J_C 定义为 -E[δ_S]，δ_S 越大（越安全），成本越低。
                with torch.no_grad():
                    J_C = -delta_S.mean()
                    lambda_param = lambda_param + cfg.λ_lr * (J_C - 0.0)
                    lambda_param = lambda_param.clamp(min=0.0)  # λ >= 0

                if global_step % 10 == 0:
                    print(
                        f"Epoch {epoch} step {global_step} "
                        f"loss_H={loss_H.item():.4f} loss_S={loss_S.item():.4f} "
                        f"KL={kl.item():.4f} lambda={lambda_param.item():.4f}"
                    )

        # 一个 epoch 结束后保存 checkpoint
        save_dir = output_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved epoch {epoch} checkpoint to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train policy with Safe-MM-DPO.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dpo_config.yaml.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save aligned policy checkpoints.",
    )
    parser.add_argument(
        "--logging_dir",
        type=Path,
        required=False,
        default=Path("logs/training/safe_mm_dpo"),
        help="Directory for logs (currently used only for structure).",
    )
    args = parser.parse_args()

    cfg = load_dpo_config(args.config)
    run_training(cfg, args.output_dir, args.logging_dir)


if __name__ == "__main__":
    main()


