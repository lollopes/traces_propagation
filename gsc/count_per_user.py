#!/usr/bin/env python3
# count_per_user.py
#
# Walks a Google‑Speech‑Commands‑style tree and reports,
# for each split (train / test / valid), how many utterances
# each speaker contributes to every class.
# python count_per_user.py /scratch-node/20235438/gsc_v2_data

import argparse
import collections
import os
import re
from pathlib import Path
from typing import Dict, Counter


def extract_user_id(filename: str) -> str:
    """
    Extract the 32‑hex‑digit speaker ID that precedes '_nohash_'
    in every official GSC filename, e.g.
        'c50f55b8_nohash_0.wav'  ->  'c50f55b8'
    Adapt this function if you use a different naming convention.
    """
    return filename.split("_nohash_")[0]


def scan_split(split_path: Path) -> Dict[str, Counter[str]]:
    """
    Return {speaker_id -> Counter({class : count, ...})}
    for all .wav files under `split_path`.
    """
    counts: Dict[str, Counter[str]] = collections.defaultdict(collections.Counter)

    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for wav_file in class_dir.glob("*.wav"):
            user = extract_user_id(wav_file.name)
            counts[user][label] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Count utterances per user per class in train/test/valid splits"
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root folder that contains train/, test/ and/or valid/ sub‑folders",
    )
    args = parser.parse_args()

    # Which split directories actually exist?
    splits = [d for d in ("train", "test", "valid") if (args.data_root / d).is_dir()]
    if not splits:
        raise SystemExit("No train/, test/ or valid/ directories found under data_root")

    print("\n=== SUMMARY ===")
    for split in splits:
        split_path = args.data_root / split
        split_counts = scan_split(split_path)

        # Compute total samples per user
        user_totals = {user: sum(cls_counts.values()) for user, cls_counts in split_counts.items()}
        total_samples = sum(user_totals.values())

        # Find top 3 users by total samples
        top_users = sorted(user_totals.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\n{split.upper()} split:")
        print(f"Total samples: {total_samples}")
        print("Top 3 users by number of recordings:")
        for user, count in top_users:
            print(f"  {user}: {count} samples")

    print("\nDone.")


if __name__ == "__main__":
    main()
