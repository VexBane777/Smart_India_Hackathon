"""Held-out cross-generator eval on the In-the-Wild corpus (Muller et al.) —
messy internet audio from generators the training corpus never saw, matching
00_MASTER_PLAN.md §5.4's "unseen generator" / "harshest external test" protocol.

This is eval-only: it loads a trained model.pt and reports EER, it does not
touch training data or retrain anything.

Usage:
    python eval_held_out.py --model runs/voice_guard_v1/model.pt \
        --wav-dir data/in_the_wild/release_in_the_wild --meta data/in_the_wild/release_in_the_wild/meta.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from dataset import _load_mono_16k  # reuse the exact same loader train-time used
from features import chunk_audio, extract_features
from model import VoiceGuardMLP
from train import compute_eer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    args = ap.parse_args()

    model = VoiceGuardMLP()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    rows = []
    with open(args.meta, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    scores, labels = [], []
    n_missing = 0
    for i, row in enumerate(rows):
        wav_path = args.wav_dir / row["file"]
        if not wav_path.exists():
            n_missing += 1
            continue
        label = 0 if row["label"] == "bona-fide" else 1
        pcm = _load_mono_16k(wav_path)
        for chunk in chunk_audio(pcm):
            feat = extract_features(chunk)
            with torch.no_grad():
                logits = model(torch.from_numpy(feat).unsqueeze(0))
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
            scores.append(prob)
            labels.append(label)
        if i % 500 == 0:
            print(f"{i}/{len(rows)}")

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    eer = compute_eer(scores_arr, labels_arr)
    print(f"In-the-Wild held-out: {len(rows) - n_missing} clips ({n_missing} missing), "
          f"{len(scores)} windows, EER={eer:.4f}")


if __name__ == "__main__":
    main()
