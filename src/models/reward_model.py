"""
Reward model definition.

According to dev.md, we need a scalar-valued head on top of a base LM
to score (prompt, response) pairs. For simplicity, we reuse
`AutoModelForSequenceClassification` with `num_labels=1`, which works
with most decoder-only LMs (including Mistral-7B) via the
`AutoModelForSequenceClassification` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class RewardModelConfig:
    base_model_path: str
    tokenizer_name: Optional[str] = None
    max_length: int = 1024


class RewardModel(nn.Module):
    """
    Thin wrapper around AutoModelForSequenceClassification (num_labels=1).
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns a scalar score for each sequence in the batch.
        Shape: (batch,)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)  # (batch,)
        return logits


def load_reward_model(cfg: RewardModelConfig) -> tuple[RewardModel, PreTrainedTokenizerBase]:
    """
    Load base model and tokenizer and wrap as a RewardModel.

    - base_model_path: path or HF id for the base LM
    - tokenizer_name: optional, if None we reuse base_model_path
    """
    tokenizer_name = cfg.tokenizer_name or cfg.base_model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        # For decoder-only LMs (e.g. Mistral) we usually don't have a pad token
        # defined by default. Reuse EOS as PAD so that:
        #   - our DataLoader can pad sequences (see train_rm.collate_fn)
        #   - HuggingFace internals (SequenceSummary etc.) know pad_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Make sure the model itself also has pad_token_id set, otherwise some
    # transformer utilities will raise when batch_size > 1.
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_path,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Double‑check on the loaded config as well.
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    model = RewardModel(base_model)
    return model, tokenizer


