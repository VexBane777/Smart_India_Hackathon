"""Per-cell EER breakdown against data/accents_split/held/, instead of one
aggregate number — the whole point of collecting accent-tagged data was to
find out which specific cells (e.g. hi_foreign) a model is still weak on,
which a single aggregate EER can't show (see data/README.md's caveat on
TARGET_MINUTES_PER_CELL being a guess pending exactly this measurement).

Usage (after data/split_accents.py has produced accents_split/held/):
    python eval_accent_cells.py --model runs/voice_guard_v3/model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.accent_common import CELLS
from dataset import build_examples, to_arrays
from model import VoiceGuardMLP
from train import compute_eer

HELD_ROOT = Path(__file__).resolve().parent / "data" / "accents_split" / "held"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    args = ap.parse_args()

    model = VoiceGuardMLP()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    print(f"{'cell':<12} {'windows':>8} {'files':>7}  eer")
    print("-" * 40)
    for cell in CELLS:
        real_dir = HELD_ROOT / "real" / cell
        fake_dir = HELD_ROOT / "fake" / cell
        if not real_dir.exists() or not fake_dir.exists():
            print(f"{cell:<12} {'-':>8} {'-':>7}  no held-out data (run split_accents.py first)")
            continue
        examples = build_examples([real_dir], [fake_dir], channel_recipes=[None])
        if not examples:
            print(f"{cell:<12} {'0':>8} {'0':>7}  no files in held split")
            continue
        X, y = to_arrays(examples)
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(X)), dim=-1)[:, 1].numpy()
        eer = compute_eer(probs, y)
        n_files = len({e.source_file for e in examples})
        print(f"{cell:<12} {len(examples):>8} {n_files:>7}  {eer:.4f}")


if __name__ == "__main__":
    main()
