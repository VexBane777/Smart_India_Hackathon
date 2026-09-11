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

from dataset import build_examples, split_by_source, to_arrays


def parse_channel_arg(items: list[str | None]) -> list[str | None]:
    """Maps the literal string 'none' (any case) to the real no-op sentinel.
    Everything else (including 'clean') passes through unchanged — see
    --channel's help text for why 'clean' is NOT a no-degradation recipe."""
    return [None if r is None or r.lower() == "none" else r for r in items]

# torch/model are imported lazily inside main() — dataset.build_examples()
# spawns a multiprocessing.ProcessPoolExecutor, and on Windows (spawn-only,
# no fork) each worker re-execs this script's module-level code. Importing
# torch (with its CUDA DLLs) up here means every worker pays that cost too,
# which is what blew a 16GB-RAM box's page file when running many workers.


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
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from model import VoiceGuardMLP

    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, required=True, nargs="+", help="one or more real/ WAV dirs (degraded through --channel)")
    ap.add_argument("--fake", type=Path, required=True, nargs="+", help="one or more fake/ WAV dirs (degraded through --channel)")
    ap.add_argument("--real-clean", type=Path, default=[], nargs="*",
                     help="additional real/ WAV dirs used as-is, no --channel degradation applied "
                     "(e.g. already telephony-degraded corpora like ASVspoof2021 LA eval)")
    ap.add_argument("--fake-clean", type=Path, default=[], nargs="*",
                     help="additional fake/ WAV dirs used as-is, no --channel degradation applied")
    ap.add_argument("--out", type=Path, default=Path("runs/voice_guard_mlp"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--channel",
        nargs="*",
        default=[None],
        help="TeleChannel recipe names to degrade training audio through "
        "(e.g. whatsapp volte cellular_3g). Pass the literal string 'none' "
        "for a genuinely undegraded pass — NOTE: TeleChannel's 'clean' "
        "recipe is NOT a no-op; per vaani/telechannel/configs/channels.yaml "
        "it still applies RIR reverb + white noise + mic clipping + packet "
        "loss, only skipping the codec/ffmpeg step (it exists so the "
        "orchestration can be exercised without ffmpeg installed, not to "
        "represent undegraded audio). Using 'clean' where 'none' was meant "
        "was a real bug this project hit (2026-09 sessions) that measurably "
        "hurt cross-generator generalization. Omit --channel entirely for "
        "no processing at all (equivalent to --channel none).",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device for the MLP training loop itself; feature extraction/"
        "channel degradation always run on CPU (ffmpeg subprocess work, "
        "not matrix math, so a GPU wouldn't help there).",
    )
    ap.add_argument("--workers", type=int, default=None, help="process-pool size for feature extraction")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for channel degradation.")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32],
                     help="MLP hidden layer sizes, e.g. --hidden-dims 128 64.")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                     help="L2 regularization on the Adam optimizer (was 0/absent in every run "
                     "this project has done so far) — a direct lever against overfitting to "
                     "spurious per-domain correlations, the exact failure mode this session's "
                     "code-orange investigation was about.")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                     help="CrossEntropyLoss label smoothing, e.g. 0.05-0.1 to reduce overconfident "
                     "fitting to training-set-specific quirks.")
    ap.add_argument("--save-every-epoch-checkpoints", action="store_true",
                     help="save a state_dict per epoch into <out>/checkpoints/epoch_NN.pt. Every "
                     "run so far exported whatever the LAST epoch happened to land on, not the "
                     "best one (found 2026-09-10: v6's best in-distribution epoch was 20/30, but "
                     "the shipped model came from epoch 30, already known worse). Use "
                     "select_best_checkpoint.py afterward to pick by *held-out* EER, not "
                     "in-distribution val_eer — this session's whole finding is that the two can "
                     "diverge.")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    recipes = parse_channel_arg(args.channel)
    if "clean" in recipes:
        print("WARNING: 'clean' is a TeleChannel debug recipe (RIR+noise+clip+loss, "
              "codec skipped only to avoid an ffmpeg dependency) — it is NOT a "
              "no-degradation pass. If you meant 'no processing', pass 'none' instead.")
    print(f"Building examples (channels={recipes})...")
    examples = build_examples(
        args.real, args.fake, channel_recipes=recipes, workers=args.workers, seed=args.seed
    )
    if args.real_clean or args.fake_clean:
        print(f"Building clean (no-degradation) examples from "
              f"{len(args.real_clean)} real-clean + {len(args.fake_clean)} fake-clean dirs...")
        examples += build_examples(
            args.real_clean or [], args.fake_clean or [], channel_recipes=[None],
            workers=args.workers, seed=args.seed,
        )
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

    model = VoiceGuardMLP(
        input_dim=X_train.shape[1], norm_mean=mean, norm_std=std, hidden_dims=tuple(args.hidden_dims)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    X_val_t = torch.from_numpy(X_val).to(device)

    checkpoint_dir = args.out / "checkpoints"
    if args.save_every_epoch_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        total_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
        eer = compute_eer(val_probs, y_val)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss:.4f}  val_eer={eer:.4f}")
        if args.save_every_epoch_checkpoints:
            torch.save(model.cpu().state_dict(), checkpoint_dir / f"epoch_{epoch + 1:02d}.pt")
            model = model.to(device)

    model = model.cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")

    # Raw (unnormalized) dummy input — normalization is baked into the model
    # itself (FixedNormalize), matching exactly what tflite_io.dart sends.
    dummy = torch.from_numpy(X_val[:1]) if len(X_val) else torch.zeros(1, X_train.shape[1])
    torch.onnx.export(
        model,
        dummy,
        str(args.out / "model.onnx"),
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,  # legacy exporter — avoids the onnxscript dep the new one needs
    )

    metrics = {"final_val_eer": eer, "n_train": len(train_ex), "n_val": len(val_ex),
               "hidden_dims": list(args.hidden_dims),
               "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model.pt / model.onnx / metrics.json -> {args.out}")
    print("Next: python export_tflite.py --onnx", args.out / "model.onnx", "--out", args.out / "voice_detector.tflite")


if __name__ == "__main__":
    main()
