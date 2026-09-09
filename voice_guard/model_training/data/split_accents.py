"""Speaker-disjoint train/held split of data/accents/, mirroring
prep_in_the_wild.py's rationale: without holding out whole speakers, folding
accent data into training would leave nothing that actually tests "does the
model generalize on this cell" (00_MASTER_PLAN.md's real-speech splitting
rule, applied here to the new accent cells).

A fake clip inherits its source real speaker's speaker_id (see
gen_accents_fake.py), so splitting by speaker_id keeps a real/clone pair on
the same side — a clone never leaks its own reference speaker's train-time
exposure into the held-out eval.

Needs no network/GPU. Reads data/accent_manifest.csv, so run this after at
least one of prep_accents_real.py / gen_accents_fake.py /
ingest_self_recordings.py has populated it.

Usage:
    python split_accents.py --held-out-fraction 0.2
"""
from __future__ import annotations

import argparse
import random
import shutil

from accent_common import ACCENTS_ROOT, CELLS, LABELS, cell_dir, read_manifest

HERE = ACCENTS_ROOT.parent
SPLIT_ROOT = HERE / "accents_split"


def split_dir(split: str, label: str, cell: str):
    return SPLIT_ROOT / split / label / cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_manifest()
    if not rows:
        print("No manifest rows yet — nothing to split.")
        return

    # Split speakers per (cell, label) independently rather than globally:
    # a global split could, by chance, put every en_native speaker in train
    # and every hi_foreign speaker held out, silently leaving some cells
    # with no held-out eval data at all.
    rng = random.Random(args.seed)
    for label in LABELS:
        for cell in CELLS:
            cell_rows = [r for r in rows if r["label"] == label and r["cell"] == cell]
            if not cell_rows:
                continue
            speakers = sorted({r["speaker_id"] for r in cell_rows})
            rng.shuffle(speakers)
            n_held = max(1, int(len(speakers) * args.held_out_fraction)) if len(speakers) > 1 else 0
            held_speakers = set(speakers[:n_held])

            train_dir = split_dir("train", label, cell)
            held_dir = split_dir("held", label, cell)
            train_dir.mkdir(parents=True, exist_ok=True)
            held_dir.mkdir(parents=True, exist_ok=True)

            n_train = n_held_files = n_missing = 0
            for r in cell_rows:
                src = HERE / r["file"]
                if not src.exists():
                    n_missing += 1
                    continue
                dest_dir = held_dir if r["speaker_id"] in held_speakers else train_dir
                shutil.copy2(src, dest_dir / src.name)
                if dest_dir is held_dir:
                    n_held_files += 1
                else:
                    n_train += 1
            print(f"{label}/{cell}: {len(speakers)} speakers ({len(held_speakers)} held) "
                  f"-> train={n_train} held={n_held_files} missing={n_missing}")


if __name__ == "__main__":
    main()
