"""select_best_checkpoint.py's sweep, adapted for VoiceGuardSeqCNN's two
inputs (lfcc_sequence, scalars). Loads norm_stats.npz (written by
train_seq_cnn.py) for the model's shape/normalization instead of deriving
input_dim from the checkpoint state dict (the MLP-only trick used by the
original script does not apply here — the seq-CNN's Conv1d weights don't
encode n_frames/n_scalars).

Usage:
    python select_best_checkpoint_seqcnn.py \
        --checkpoint-dir runs/voice_guard_v11_seqcnn/checkpoints \
        --norm-stats runs/voice_guard_v11_seqcnn/norm_stats.npz \
        --real data/real_noise_aug_split/held/en_native data/real_noise_aug_split/held/hi_native \
        --fake data/fake_itw_held \
        --out runs/voice_guard_v11_seqcnn_selected
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import build_examples, to_arrays
from model import VoiceGuardSeqCNN
from train import compute_eer


def load_model(norm_stats: dict, conv_channels=(32, 16)) -> VoiceGuardSeqCNN:
    return VoiceGuardSeqCNN(
        n_frames=int(norm_stats["n_frames"]), n_lfcc=int(norm_stats["n_lfcc"]),
        n_scalars=int(norm_stats["n_scalars"]),
        seq_mean=norm_stats["seq_mean"], seq_std=norm_stats["seq_std"],
        scalar_mean=norm_stats["scalar_mean"], scalar_std=norm_stats["scalar_std"],
        conv_channels=conv_channels,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--norm-stats", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, nargs="+", help="held-out real dir(s)")
    ap.add_argument("--fake", type=Path, required=True, nargs="+", help="held-out fake dir(s)")
    ap.add_argument("--out", type=Path, required=True,
                     help="where to write the winning checkpoint as model.pt + re-exported model.onnx")
    args = ap.parse_args()

    checkpoints = sorted(args.checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no epoch_*.pt checkpoints found in {args.checkpoint_dir}")

    norm_stats = dict(np.load(args.norm_stats))

    print(f"Building held-out features once ({len(checkpoints)} checkpoints to sweep)...")
    examples = build_examples(args.real, args.fake, channel_recipes=[None])
    X_seq, X_scalars, y, _attack_y = to_arrays(examples)
    X_seq_t = torch.from_numpy(X_seq)
    X_scalars_t = torch.from_numpy(X_scalars)
    print(f"{len(examples)} windows from {len({e.source_file for e in examples})} source files")

    results = []
    for ckpt_path in checkpoints:
        model = load_model(norm_stats)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            real_fake_logits, _attack_logits = model(X_seq_t, X_scalars_t)
            probs = torch.softmax(real_fake_logits, dim=-1)[:, 1].numpy()
        eer = compute_eer(probs, y)
        results.append((ckpt_path, eer))
        print(f"  {ckpt_path.name}: held-out EER={eer:.4f}")

    best_path, best_eer = min(results, key=lambda r: r[1])
    print(f"\nBest: {best_path.name}  EER={best_eer:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    best_state = torch.load(best_path, map_location="cpu", weights_only=True)
    torch.save(best_state, args.out / "model.pt")
    np.savez(args.out / "norm_stats.npz", **norm_stats)

    model = load_model(norm_stats)
    model.load_state_dict(best_state)
    model.eval()
    dummy_seq = torch.zeros(1, int(norm_stats["n_frames"]), int(norm_stats["n_lfcc"]))
    dummy_scalars = torch.zeros(1, int(norm_stats["n_scalars"]))
    torch.onnx.export(
        model, (dummy_seq, dummy_scalars), str(args.out / "model.onnx"),
        input_names=["lfcc_sequence", "scalars"], output_names=["real_fake_logits", "attack_type_logits"],
        opset_version=13, dynamo=False,
    )

    sweep = {ckpt.name: eer for ckpt, eer in results}
    (args.out / "checkpoint_sweep.json").write_text(json.dumps(
        {"best_checkpoint": best_path.name, "best_held_out_eer": best_eer, "all": sweep}, indent=2))
    print(f"Wrote model.pt / model.onnx / norm_stats.npz / checkpoint_sweep.json -> {args.out}")


if __name__ == "__main__":
    main()
