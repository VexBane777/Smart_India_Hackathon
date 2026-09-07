"""
Trains VoiceGuardMLP on a real/ vs fake/ WAV corpus and exports ONNX
(the format-agnostic checkpoint every later step, including the actual
TFLite conversion, builds on).

Usage:
    python train.py --real /path/to/real_wavs --fake /path/to/fake_wavs \
        --out runs/voice_guard_mlp --epochs 30

Where real_wavs/fake_wavs come from (recommended for the actual run, not
this repo's placeholder demo assets — see model_training/README.md):
    ASVspoof 2019 LA bonafide/spoof splits, pulled via:
        kaggle datasets download -d anishsarkar22/asvpoof-2019-dataset-la
    optionally run through vaani's TeleChannel recipes first (--channel).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import build_examples, split_by_source, to_arrays
from model import VoiceGuardMLP


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate: threshold where false-accept rate == false-reject rate."""
    order = np.argsort(scores)
    scores, labels = scores[order], labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    thresholds = np.unique(scores)
    best_gap, best_eer = float("inf"), float("nan")
    for t in thresholds:
        far = (scores[labels == 0] >= t).mean()  # real wrongly flagged
        frr = (scores[labels == 1] < t).mean()  # fake wrongly cleared
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer = gap, (far + frr) / 2
    return float(best_eer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--fake", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/voice_guard_mlp"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--channel",
        nargs="*",
        default=[None],
        help="TeleChannel recipe names to degrade training audio through "
        "(e.g. whatsapp volte cellular_3g); pass 'clean' or omit for none.",
    )
    args = ap.parse_args()

    recipes = [None if r in (None, "clean") else r for r in args.channel]
    print(f"Building examples (channels={recipes})...")
    examples = build_examples(args.real, args.fake, channel_recipes=recipes)
    train_ex, val_ex = split_by_source(examples)
    print(f"{len(train_ex)} train windows / {len(val_ex)} val windows "
          f"from {len({e.source_file for e in examples})} source files")

    X_train, y_train = to_arrays(train_ex)
    X_val, y_val = to_arrays(val_ex)

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = VoiceGuardMLP(norm_mean=mean, norm_std=std)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        total_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val))
            val_probs = torch.softmax(val_logits, dim=-1)[:, 1].numpy()
        eer = compute_eer(val_probs, y_val)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss:.4f}  val_eer={eer:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")

    # Raw (unnormalized) dummy input — normalization is baked into the model
    # itself (FixedNormalize), matching exactly what tflite_io.dart sends.
    dummy = torch.from_numpy(X_val[:1]) if len(X_val) else torch.zeros(1, 63)
    torch.onnx.export(
        model,
        dummy,
        str(args.out / "model.onnx"),
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,  # legacy exporter — avoids the onnxscript dep the new one needs
    )

    metrics = {"final_val_eer": eer, "n_train": len(train_ex), "n_val": len(val_ex)}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model.pt / model.onnx / metrics.json -> {args.out}")
    print("Next: python export_tflite.py --onnx", args.out / "model.onnx", "--out", args.out / "voice_detector.tflite")


if __name__ == "__main__":
    main()
