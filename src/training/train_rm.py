"""
Train reward models (Helpful-RM / Harmless-RM).

Usage (from pretrain_guide.md):

    python -m src.training.train_rm \
        --config configs/rm_config.yaml \
        --task helpful \
        --output_dir models/helpful_rm

    python -m src.training.train_rm \
        --config configs/rm_config.yaml \
        --task harmless \
        --output_dir models/harmless_rm

Expected rm_config.yaml fields (minimal):

    base_model_path: models/base/Mistral-7B-v0.1
    tokenizer_name: null  # optional, defaults to base_model_path
    max_length: 1024

    train_helpful_path: data/train/helpful_pairs.jsonl
    val_helpful_path: data/val/helpful_pairs.jsonl
    train_harmless_path: data/train/harmless_pairs.jsonl
    val_harmless_path: data/val/harmless_pairs.jsonl

    learning_rate: 2e-5
    batch_size: 8
    num_epochs: 1
    weight_decay: 0.01
    max_grad_norm: 1.0

You can extend this config as needed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
import yaml

from src.models.reward_model import RewardModelConfig, load_reward_model


class PreferenceDataset(Dataset):
    """
    Simple dataset for preference pairs stored in jsonl files.

    Each record should contain:
        - prompt
        - chosen_response
        - rejected_response
    """

    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
    ) -> None:
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

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:  # type: ignore[override]
        rec = self.records[idx]
        prompt = rec["prompt"]
        chosen = rec["chosen_response"]
        rejected = rec["rejected_response"]

        prompt_chosen = f"{prompt}\n{chosen}"
        prompt_rejected = f"{prompt}\n{rejected}"

        tok_chosen = self.tokenizer(
            prompt_chosen,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok_rejected = self.tokenizer(
            prompt_rejected,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        sample = {
            "chosen": {
                "input_ids": tok_chosen["input_ids"].squeeze(0),
                "attention_mask": tok_chosen["attention_mask"].squeeze(0),
            },
            "rejected": {
                "input_ids": tok_rejected["input_ids"].squeeze(0),
                "attention_mask": tok_rejected["attention_mask"].squeeze(0),
            },
        }
        return sample


@dataclass
class TrainConfig:
    base_model_path: str
    tokenizer_name: str | None
    max_length: int
    train_path: Path
    val_path: Path
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    max_grad_norm: float


def load_train_config(config_path: Path, task: str) -> TrainConfig:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if task not in {"helpful", "harmless"}:
        raise ValueError(f"Unknown task: {task}")

    train_key = f"train_{task}_path"
    val_key = f"val_{task}_path"
    try:
        train_path = Path(cfg[train_key])
        val_path = Path(cfg[val_key])
    except KeyError as e:
        raise KeyError(f"Missing {e} in rm_config.yaml for task '{task}'") from e

    return TrainConfig(
        base_model_path=cfg["base_model_path"],
        tokenizer_name=cfg.get("tokenizer_name"),
        max_length=int(cfg.get("max_length", 1024)),
        train_path=train_path,
        val_path=val_path,
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        batch_size=int(cfg.get("batch_size", 8)),
        num_epochs=int(cfg.get("num_epochs", 1)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
    )


def preference_loss(
    scores_chosen: torch.Tensor,
    scores_rejected: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise ranking loss:
        -log(sigmoid(score_chosen - score_rejected))
    """
    diff = scores_chosen - scores_rejected
    return -torch.nn.functional.logsigmoid(diff).mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        input_ids_c = chosen["input_ids"].to(device)
        attn_mask_c = chosen["attention_mask"].to(device)
        input_ids_r = rejected["input_ids"].to(device)
        attn_mask_r = rejected["attention_mask"].to(device)

        scores_c = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
        scores_r = model(input_ids=input_ids_r, attention_mask=attn_mask_r)

        loss = preference_loss(scores_c, scores_r)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(1, total_steps)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    total_pairs = 0

    for batch in dataloader:
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        input_ids_c = chosen["input_ids"].to(device)
        attn_mask_c = chosen["attention_mask"].to(device)
        input_ids_r = rejected["input_ids"].to(device)
        attn_mask_r = rejected["attention_mask"].to(device)

        scores_c = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
        scores_r = model(input_ids=input_ids_r, attention_mask=attn_mask_r)

        loss = preference_loss(scores_c, scores_r)

        total_loss += loss.item()
        total_steps += 1

        correct += (scores_c > scores_r).sum().item()
        total_pairs += scores_c.numel()

    return {
        "val_loss": total_loss / max(1, total_steps),
        "val_accuracy": correct / max(1, total_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reward model (helpful/harmless).")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to rm_config.yaml.",
    )
    parser.add_argument(
        "--task",
        choices=["helpful", "harmless"],
        required=True,
        help="Which reward model to train.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the trained reward model.",
    )
    args = parser.parse_args()

    train_cfg = load_train_config(args.config, args.task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rm_cfg = RewardModelConfig(
        base_model_path=train_cfg.base_model_path,
        tokenizer_name=train_cfg.tokenizer_name,
        max_length=train_cfg.max_length,
    )
    model, tokenizer = load_reward_model(rm_cfg)
    model.to(device)

    train_ds = PreferenceDataset(train_cfg.train_path, tokenizer, train_cfg.max_length)
    val_ds = PreferenceDataset(train_cfg.val_path, tokenizer, train_cfg.max_length)

    # We collate manually because our dataset already pads per sample.
    def collate_fn(batch):
        # Just return a dict of lists of tensors; we stack inside training loop.
        # Here we simply stack on dim=0.
        def stack_side(side: str):
            input_ids = [b[side]["input_ids"] for b in batch]
            attention_mask = [b[side]["attention_mask"] for b in batch]
            return {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    attention_mask, batch_first=True, padding_value=0
                ),
            }

        return {
            "chosen": stack_side("chosen"),
            "rejected": stack_side("rejected"),
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    num_training_steps = len(train_loader) * train_cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_val_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            train_cfg.max_grad_norm,
        )
        metrics = evaluate(model, val_loader, device)

        print(
            f"[{args.task}] Epoch {epoch}/{train_cfg.num_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_acc={metrics['val_accuracy']:.4f}"
        )

        # Save best checkpoint
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            save_dir = args.output_dir
            print(f"Saving best model to {save_dir} (val_loss={best_val_loss:.4f})")
            model.model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()


