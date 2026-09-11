"""eval_held_out_dirs.py's EER + bootstrap-CI check, adapted for
VoiceGuardSeqCNN's two inputs.

Usage:
    python eval_held_out_dirs_seqcnn.py \
        --model runs/voice_guard_v11_seqcnn_selected/model.pt \
        --norm-stats runs/voice_guard_v11_seqcnn_selected/norm_stats.npz \
        --real data/real_itw_held --fake data/fake_itw_held \
        --baseline-eer 0.1538
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import build_examples, to_arrays
from eval_stats import bootstrap_eer_ci
from model import VoiceGuardSeqCNN
from train import compute_eer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--norm-stats", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, nargs="+")
    ap.add_argument("--fake", type=Path, required=True, nargs="+")
    ap.add_argument("--baseline-eer", type=float, default=None,
                     help="exit nonzero if the measured EER is worse (higher) than this.")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1,
                     help="default 1: avoids concurrent torch-importing process pools "
                     "exhausting the paging file when run alongside other eval scripts.")
    args = ap.parse_args()

    norm_stats = dict(np.load(args.norm_stats))
    model = VoiceGuardSeqCNN(
        n_frames=int(norm_stats["n_frames"]), n_lfcc=int(norm_stats["n_lfcc"]),
        n_scalars=int(norm_stats["n_scalars"]),
        seq_mean=norm_stats["seq_mean"], seq_std=norm_stats["seq_std"],
        scalar_mean=norm_stats["scalar_mean"], scalar_std=norm_stats["scalar_std"],
    )
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    examples = build_examples(args.real, args.fake, channel_recipes=[None], workers=args.workers)
    X_seq, X_scalars, y, _attack_y = to_arrays(examples)
    with torch.no_grad():
        real_fake_logits, _attack_logits = model(torch.from_numpy(X_seq), torch.from_numpy(X_scalars))
        probs = torch.softmax(real_fake_logits, dim=-1)[:, 1].numpy()
    eer = compute_eer(probs, y)
    print(f"{len(examples)} windows from {len({e.source_file for e in examples})} source files, EER={eer:.4f}")

    if not args.no_bootstrap:
        source_files = [e.source_file for e in examples]
        ci = bootstrap_eer_ci(probs, y, source_files, n_bootstrap=args.n_bootstrap)
        print(f"95% CI (file-level bootstrap, n={ci['n_sources']} files, "
              f"{args.n_bootstrap} resamples): [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  "
              f"(std={ci['std']:.4f})")

    if args.baseline_eer is not None and eer > args.baseline_eer:
        print(f"REGRESSION: {eer:.4f} is worse than baseline {args.baseline_eer:.4f} - do not deploy.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
