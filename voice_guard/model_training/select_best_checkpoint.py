"""
Sweep every per-epoch checkpoint from a `train.py --save-every-epoch-
checkpoints` run against the real held-out benchmark, and keep whichever
epoch actually minimizes *held-out* EER — not in-distribution val_eer.

Why this exists (2026-09-10): every training run this session exported
whatever the LAST epoch happened to land on. Checked directly: v6's best
in-distribution epoch was 20/30 (val_eer=0.0772), but the shipped model came
from epoch 30 (0.0835) — a checkpoint already known worse, exported anyway
because nothing was tracking or comparing epochs. Separately, this session's
whole investigation found in-distribution val_eer and cross-generator
held-out EER can diverge or move in *opposite* directions — so even
"keep the best in-distribution epoch" isn't obviously the right criterion.
This sweeps the metric that actually matters and reports every epoch's
number, not just the winner, so the choice is auditable.

Cheap by construction: held-out feature extraction (the expensive part) runs
ONCE; each checkpoint only costs a forward pass over the same cached
features.

Usage:
    python select_best_checkpoint.py --checkpoint-dir runs/voice_guard_v7/checkpoints \
        --real data/real_itw_held --fake data/fake_itw_held \
        --out runs/voice_guard_v7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import build_examples, to_arrays
from model import VoiceGuardMLP
from train import compute_eer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, nargs="+", help="held-out real dir(s)")
    ap.add_argument("--fake", type=Path, required=True, nargs="+", help="held-out fake dir(s)")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--out", type=Path, required=True,
                     help="where to write the winning checkpoint as model.pt + re-exported model.onnx")
    args = ap.parse_args()

    checkpoints = sorted(args.checkpoint_dir.glob("epoch_*.pt")) + sorted(args.checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no epoch_*.pt checkpoints found in {args.checkpoint_dir}")

    print(f"Building held-out features once ({len(checkpoints)} checkpoints to sweep)...")
    examples = build_examples(args.real, args.fake, channel_recipes=[None])
    X, y = to_arrays(examples)
    X_t = torch.from_numpy(X)
    print(f"{len(examples)} windows from {len({e.source_file for e in examples})} source files")

    results = []
    for ckpt_path in checkpoints:
        model = VoiceGuardMLP(hidden_dims=tuple(args.hidden_dims))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_t), dim=-1)[:, 1].numpy()
        eer = compute_eer(probs, y)
        results.append((ckpt_path, eer))
        print(f"  {ckpt_path.name}: held-out EER={eer:.4f}")

    best_path, best_eer = min(results, key=lambda r: r[1])
    print(f"\nBest: {best_path.name}  EER={best_eer:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    best_state = torch.load(best_path, map_location="cpu", weights_only=True)
    torch.save(best_state, args.out / "model.pt")

    model = VoiceGuardMLP(hidden_dims=tuple(args.hidden_dims))
    model.load_state_dict(best_state)
    model.eval()
    dummy = torch.zeros(1, 63)
    torch.onnx.export(
        model, dummy, str(args.out / "model.onnx"),
        input_names=["features"], output_names=["logits"],
        opset_version=13, dynamo=False,
    )

    sweep = {ckpt.name: eer for ckpt, eer in results}
    (args.out / "checkpoint_sweep.json").write_text(json.dumps(
        {"best_checkpoint": best_path.name, "best_held_out_eer": best_eer, "all": sweep}, indent=2))
    print(f"Wrote model.pt / model.onnx / checkpoint_sweep.json -> {args.out}")


if __name__ == "__main__":
    main()
