"""Held-out eval variant that takes plain real/fake WAV dirs (e.g. the
speaker-disjoint held-out split from data/prep_in_the_wild.py) instead of
the meta.csv format eval_held_out.py expects.

Usage:
    python eval_held_out_dirs.py --model runs/voice_guard_v3/model.pt \
        --real data/real_itw_held --fake data/fake_itw_held
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import build_examples, to_arrays
from model import VoiceGuardMLP
from train import compute_eer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, nargs="+")
    ap.add_argument("--fake", type=Path, required=True, nargs="+")
    args = ap.parse_args()

    model = VoiceGuardMLP()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    examples = build_examples(args.real, args.fake, channel_recipes=[None])
    X, y = to_arrays(examples)
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    eer = compute_eer(probs, y)
    print(f"{len(examples)} windows from {len({e.source_file for e in examples})} source files, EER={eer:.4f}")


if __name__ == "__main__":
    main()
