"""measure_confound.py's median-split analysis, adapted for
VoiceGuardSeqCNN's two inputs. The three confound features (pauseRatio,
energyVariance, zcrVariance) live in the scalars vector (extract_scalars =
[pauseRatio, energyVariance, zcrVariance, jitter, shimmer, hnrDb] — see
features.py), not in the LFCC sequence, so the median-split still operates
on X_scalars exactly as measure_confound.py operated on the pooled
feature vector's prosody columns.

Usage:
    python measure_confound_seqcnn.py \
        --model runs/voice_guard_v11_seqcnn_selected/model.pt \
        --norm-stats runs/voice_guard_v11_seqcnn_selected/norm_stats.npz \
        --real data/real_itw_held
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import build_examples, to_arrays
from model import VoiceGuardSeqCNN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--norm-stats", type=Path, required=True)
    ap.add_argument("--real", type=Path, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=1,
                     help="default 1: avoids concurrent torch-importing process pools "
                     "exhausting the paging file when run alongside other eval scripts.")
    args = ap.parse_args()

    examples = build_examples(args.real, [], channel_recipes=[None], workers=args.workers)
    X_seq, X_scalars, _y, _attack_y = to_arrays(examples)

    norm_stats = dict(np.load(args.norm_stats))
    model = VoiceGuardSeqCNN(
        n_frames=int(norm_stats["n_frames"]), n_lfcc=int(norm_stats["n_lfcc"]),
        n_scalars=int(norm_stats["n_scalars"]),
        seq_mean=norm_stats["seq_mean"], seq_std=norm_stats["seq_std"],
        scalar_mean=norm_stats["scalar_mean"], scalar_std=norm_stats["scalar_std"],
    )
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        real_fake_logits, _attack_logits = model(torch.from_numpy(X_seq), torch.from_numpy(X_scalars))
        probs = torch.softmax(real_fake_logits, dim=-1)[:, 1].numpy()

    # extract_scalars order: [pauseRatio, energyVariance, zcrVariance, jitter, shimmer, hnrDb]
    idx = {"pauseRatio": 0, "energyVariance": 1, "zcrVariance": 2}
    for name, col_idx in idx.items():
        col = X_scalars[:, col_idx]
        median = np.median(col)
        low_mean = probs[col < median].mean()
        high_mean = probs[col >= median].mean()
        print(f"{name}: low-half mean fake-prob={low_mean:.3f}  high-half={high_mean:.3f}  gap={abs(high_mean - low_mean):.3f}")


if __name__ == "__main__":
    main()
