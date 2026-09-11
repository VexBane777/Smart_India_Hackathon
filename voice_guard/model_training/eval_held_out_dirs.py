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
from eval_stats import bootstrap_eer_ci
from model import VoiceGuardMLP
from train import compute_eer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, nargs="+")
    ap.add_argument("--fake", type=Path, required=True, nargs="+")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32],
                     help="must match the checkpoint's train.py --hidden-dims.")
    ap.add_argument("--baseline-eer", type=float, default=None,
                     help="exit nonzero if the measured EER is worse (higher) than this — "
                     "e.g. --baseline-eer 0.1624 to gate against v3. Formalizes the "
                     "manual 'don't deploy a regression' check this project did by hand "
                     "for attempt1/attempt2/ablation/english_only (2026-09-10).")
    ap.add_argument("--no-bootstrap", action="store_true",
                     help="skip the file-level bootstrap CI (faster, but see this session's "
                     "2026-09-10 checkpoint-sweep noise finding for why you usually want it).")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    input_dim = state["normalize.mean"].shape[0]  # derive from checkpoint, not the current INPUT_DIM constant
    model = VoiceGuardMLP(input_dim=input_dim, hidden_dims=tuple(args.hidden_dims))
    model.load_state_dict(state)
    model.eval()

    examples = build_examples(args.real, args.fake, channel_recipes=[None])
    X, y = to_arrays(examples)
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
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
