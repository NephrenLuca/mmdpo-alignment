"""
Simple preprocessing utilities for BeaverTails-style preference data.

This file is intentionally lightweight. The main entrypoint for users
is the CLI script in src/scripts/prepare_data.py, which uses the helpers
defined here to:
- load raw json/jsonl files;
- construct (prompt, chosen_response, rejected_response) pairs;
- split into train/val/test;
- write to jsonl files consumed by training scripts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple


PreferenceRecord = Dict[str, Any]


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"({self.train_ratio}, {self.val_ratio}, {self.test_ratio})"
            )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a jsonl file into memory."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_raw_files(input_dir: Path, pattern: str = "*.jsonl") -> Iterable[Path]:
    """Yield all raw data files under input_dir matching the given pattern."""
    return sorted(input_dir.glob(pattern))


def build_preference_pairs(
    records: Iterable[Dict[str, Any]],
    dimension: str,
) -> List[PreferenceRecord]:
    """
    Convert raw BeaverTails-style records into preference pairs.

    Expected raw fields (may need to adapt to your actual schema):
    - 'prompt'
    - 'response_chosen'
    - 'response_rejected'
    - 'helpful_label' / 'harmless_label' or similar

    For now, we assume the raw data已经是偏好形式 (chosen/rejected 已给出)，
    因此这里只是做一个字段规范化的封装。
    """
    out: List[PreferenceRecord] = []
    for rec in records:
        prompt = rec.get("prompt")
        chosen = rec.get("response_chosen")
        rejected = rec.get("response_rejected")
        if not prompt or chosen is None or rejected is None:
            # Skip malformed records
            continue

        out.append(
            {
                "prompt": prompt,
                "chosen_response": chosen,
                "rejected_response": rejected,
                "dimension": dimension,
            }
        )
    return out


def split_dataset(
    data: List[PreferenceRecord],
    split_cfg: SplitConfig,
    seed: int = 42,
) -> Tuple[List[PreferenceRecord], List[PreferenceRecord], List[PreferenceRecord]]:
    """Shuffle and split data into train/val/test."""
    split_cfg.validate()
    rng = random.Random(seed)
    rng.shuffle(data)

    n = len(data)
    n_train = int(n * split_cfg.train_ratio)
    n_val = int(n * split_cfg.val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


def write_jsonl(path: Path, records: Iterable[PreferenceRecord]) -> None:
    """Write records to jsonl file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")



