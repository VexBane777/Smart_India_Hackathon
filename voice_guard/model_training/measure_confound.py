"""Reproduces the median-split analysis from
docs/CRITICAL-entity-vs-style-confound.md §2 against a given checkpoint,
to check whether adding physio features shrank the style/entity confound.
Usage: python measure_confound.py --model runs/voice_guard_v10_physio/model.pt --real data/real_itw_held"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import build_examples
from features import N_LFCC
from model import VoiceGuardMLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--real", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    examples = build_examples(args.real, [], channel_recipes=[None])
    X = np.stack([e.features for e in examples])

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    input_dim = state["normalize.mean"].shape[0]
    model = VoiceGuardMLP(input_dim=input_dim)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X)), dim=-1)[:, 1].numpy()

    prosody_start = N_LFCC
    names = ["energyVariance", "pauseRatio", "zcrVariance"]
    # matches extract_prosody's [pauseRatio, energyVariance, zcrVariance] order
    idx = {"pauseRatio": prosody_start + 0, "energyVariance": prosody_start + 1, "zcrVariance": prosody_start + 2}
    for name in names:
        col = X[:, idx[name]]
        median = np.median(col)
        low_mean = probs[col < median].mean()
        high_mean = probs[col >= median].mean()
        print(f"{name}: low-half mean fake-prob={low_mean:.3f}  high-half={high_mean:.3f}  gap={abs(high_mean - low_mean):.3f}")


if __name__ == "__main__":
    main()
