"""Split In-the-Wild (Muller et al.) into a training portion and a genuinely
held-out portion, split at the SPEAKER level (not clip level) so the held-out
eval after retraining still means something — 00_MASTER_PLAN.md §5.4's rule
that real speech splits at speaker level, not just source-clip level.

Without this, folding all of In-the-Wild into training would leave nothing
to check "does this actually generalize to messy real-world audio" with.

Usage:
    python prep_in_the_wild.py --held-out-fraction 0.2
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "in_the_wild" / "release_in_the_wild"
META = SRC_DIR / "meta.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(META, newline="")))
    speakers = sorted({r["speaker"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    n_held = max(1, int(len(speakers) * args.held_out_fraction))
    held_speakers = set(speakers[:n_held])
    print(f"{len(speakers)} speakers total, {len(held_speakers)} held out, "
          f"{len(speakers) - len(held_speakers)} for training")

    dirs = {
        ("train", "bona-fide"): HERE / "real_itw_train",
        ("train", "spoof"): HERE / "fake_itw_train",
        ("held", "bona-fide"): HERE / "real_itw_held",
        ("held", "spoof"): HERE / "fake_itw_held",
    }
    for d in dirs.values():
        d.mkdir(exist_ok=True)

    counts = {k: 0 for k in dirs}
    for row in rows:
        split = "held" if row["speaker"] in held_speakers else "train"
        dest_dir = dirs[(split, row["label"])]
        src = SRC_DIR / row["file"]
        if not src.exists():
            continue
        shutil.copy2(src, dest_dir / row["file"])
        counts[(split, row["label"])] += 1

    for k, v in counts.items():
        print(k, v)


if __name__ == "__main__":
    main()
