"""Trains VoiceGuardSeqCNN: frame-level LFCC sequence + prosody/physio
scalars -> real/fake (all examples) + attack-type (labeled fakes only,
masked loss). Consolidates remediation tracks 2 (physio), 3 (sequence
CNN), and 4 (attack-type head) into one training pass — see
docs/superpowers/plans/2026-09-11-frame-level-seq-model-and-attack-type-plan.md
Global Constraints for why this replaces three separate retrains.

Usage:
    python train_seq_cnn.py \
        --real data/real data/real2021 data/real_itw_train \
        --fake data/fake data/fake2021 data/fake_itw_train \
        --real-clean data/real_noise_aug_split/train/en_native data/real_noise_aug_split/train/hi_native \
        --channel whatsapp volte none \
        --weight-decay 1e-4 --label-smoothing 0.05 --attack-type-loss-weight 1.0 \
        --save-every-epoch-checkpoints \
        --out runs/voice_guard_v11_seqcnn --epochs 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import build_examples, split_by_source, to_arrays
from train import compute_eer, parse_channel_arg


def compute_masked_attack_type_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropyLoss with ignore_index=-100 (dataset.py's
    IGNORE_ATTACK_TYPE sentinel) — real examples and unlabeled fakes
    contribute zero gradient to the attack-type head. Returns NaN if
    EVERY example in the batch is ignored (documented PyTorch behavior)
    — callers must skip adding this term to the total loss in that case
    (see main()'s training loop below)."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits, targets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, required=True, nargs="+")
    ap.add_argument("--fake", type=Path, required=True, nargs="+")
    ap.add_argument("--real-clean", type=Path, default=[], nargs="*")
    ap.add_argument("--fake-clean", type=Path, default=[], nargs="*")
    ap.add_argument("--channel", nargs="*", default=[None])
    ap.add_argument("--out", type=Path, default=Path("runs/voice_guard_seqcnn"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--attack-type-loss-weight", type=float, default=1.0)
    ap.add_argument("--save-every-epoch-checkpoints", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    from model import VoiceGuardSeqCNN  # local import: keeps torch off the ProcessPoolExecutor workers' import path

    recipes = parse_channel_arg(args.channel)
    print(f"Building examples (channels={recipes})...")
    examples = build_examples(args.real, args.fake, channel_recipes=recipes, workers=args.workers, seed=args.seed)
    if args.real_clean or args.fake_clean:
        examples += build_examples(args.real_clean, args.fake_clean, channel_recipes=[None], workers=args.workers, seed=args.seed)

    train_ex, val_ex = split_by_source(examples)
    X_seq_train, X_scalar_train, y_train, attack_train = to_arrays(train_ex)
    X_seq_val, X_scalar_val, y_val, attack_val = to_arrays(val_ex)
    print(f"{len(train_ex)} train windows / {len(val_ex)} val windows from "
          f"{len({e.source_file for e in examples})} source files")

    seq_mean = X_seq_train.reshape(-1, X_seq_train.shape[-1]).mean(axis=0)
    seq_std = X_seq_train.reshape(-1, X_seq_train.shape[-1]).std(axis=0) + 1e-8
    scalar_mean = X_scalar_train.mean(axis=0)
    scalar_std = X_scalar_train.std(axis=0) + 1e-8

    model = VoiceGuardSeqCNN(
        n_frames=X_seq_train.shape[1], n_lfcc=X_seq_train.shape[2], n_scalars=X_scalar_train.shape[1],
        seq_mean=seq_mean, seq_std=seq_std, scalar_mean=scalar_mean, scalar_std=scalar_std,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    real_fake_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_seq_train), torch.from_numpy(X_scalar_train),
            torch.from_numpy(y_train), torch.from_numpy(attack_train),
        ),
        batch_size=args.batch_size, shuffle=True,
    )
    X_seq_val_t = torch.from_numpy(X_seq_val).to(device)
    X_scalar_val_t = torch.from_numpy(X_scalar_val).to(device)

    checkpoint_dir = args.out / "checkpoints"
    if args.save_every_epoch_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for seq_b, scalar_b, y_b, attack_b in train_loader:
            seq_b, scalar_b, y_b, attack_b = seq_b.to(device), scalar_b.to(device), y_b.to(device), attack_b.to(device)
            opt.zero_grad()
            real_fake_logits, attack_logits = model(seq_b, scalar_b)
            loss = real_fake_loss_fn(real_fake_logits, y_b)
            if (attack_b != -100).any():  # skip the attack-type term entirely if the whole batch is unlabeled (avoids NaN, see compute_masked_attack_type_loss's docstring)
                loss = loss + args.attack_type_loss_weight * compute_masked_attack_type_loss(attack_logits, attack_b)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(seq_b)
        total_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_real_fake_logits, val_attack_logits = model(X_seq_val_t, X_scalar_val_t)
            val_probs = torch.softmax(val_real_fake_logits, dim=-1)[:, 1].cpu().numpy()
        eer = compute_eer(val_probs, y_val)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss:.4f}  val_eer={eer:.4f}")
        if args.save_every_epoch_checkpoints:
            torch.save(model.cpu().state_dict(), checkpoint_dir / f"epoch_{epoch + 1:02d}.pt")
            model = model.to(device)

    model = model.cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    np.savez(
        args.out / "norm_stats.npz",
        seq_mean=seq_mean, seq_std=seq_std, scalar_mean=scalar_mean, scalar_std=scalar_std,
        n_frames=X_seq_train.shape[1], n_lfcc=X_seq_train.shape[2], n_scalars=X_scalar_train.shape[1],
    )  # ONNX export (Task 9) needs these shapes/stats without re-running feature extraction

    dummy_seq = torch.zeros(1, X_seq_train.shape[1], X_seq_train.shape[2])
    dummy_scalars = torch.zeros(1, X_scalar_train.shape[1])
    torch.onnx.export(
        model, (dummy_seq, dummy_scalars), str(args.out / "model.onnx"),
        input_names=["lfcc_sequence", "scalars"], output_names=["real_fake_logits", "attack_type_logits"],
        opset_version=13, dynamo=False,
    )
    print(f"Saved model.pt / model.onnx / norm_stats.npz -> {args.out}")


if __name__ == "__main__":
    main()
