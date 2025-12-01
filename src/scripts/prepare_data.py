"""
Prepare BeaverTails-style preference data for training.

This script implements the CLI described in pretrain_guide.md:

    python -m src.scripts.prepare_data \
        --input_dir data/raw \
        --output_dir data \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1

It expects raw json/jsonl files containing fields:
    - prompt
    - response_chosen
    - response_rejected
and optionally labels such as helpful_label / harmless_label.

For now we assume that the raw files for helpful / harmless dimensions
are already separated by filename pattern:
    - *helpful*.jsonl  -> helpful preference pairs
    - *harmless*.jsonl -> harmless preference pairs

If your actual raw file layout differs, adapt the glob patterns below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.data_processing.data_preprocessor import (
    SplitConfig,
    build_preference_pairs,
    iter_raw_files,
    load_jsonl,
    split_dataset,
    write_jsonl,
)


def _gather_records(input_dir: Path, pattern: str) -> List[dict]:
    records: List[dict] = []
    for path in iter_raw_files(input_dir, pattern=pattern):
        records.extend(load_jsonl(path))
    return records


def run(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> None:
    split_cfg = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Helpful dimension
    helpful_raw = _gather_records(input_dir, pattern="*helpful*.jsonl")
    helpful_pairs = build_preference_pairs(helpful_raw, dimension="helpful")
    h_train, h_val, h_test = split_dataset(helpful_pairs, split_cfg, seed=seed)

    # Harmless / safety dimension
    harmless_raw = _gather_records(input_dir, pattern="*harmless*.jsonl")
    harmless_pairs = build_preference_pairs(harmless_raw, dimension="harmless")
    s_train, s_val, s_test = split_dataset(harmless_pairs, split_cfg, seed=seed)

    # Write out
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    write_jsonl(train_dir / "helpful_pairs.jsonl", h_train)
    write_jsonl(val_dir / "helpful_pairs.jsonl", h_val)
    write_jsonl(test_dir / "helpful_pairs.jsonl", h_test)

    write_jsonl(train_dir / "harmless_pairs.jsonl", s_train)
    write_jsonl(val_dir / "harmless_pairs.jsonl", s_val)
    write_jsonl(test_dir / "harmless_pairs.jsonl", s_test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BeaverTails preference data.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing raw json/jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for processed preference data.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


