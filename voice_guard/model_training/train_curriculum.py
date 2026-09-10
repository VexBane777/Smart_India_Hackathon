"""
Two-stage curriculum training, per the "1+1<2" negative-transfer finding
surfaced in this session's code-orange literature review
(voice_guard/docs/superpowers/specs/2026-09-10-model-regression-design.md,
idea I4): jointly training on a mix of broad, many-generator sources and a
few single/dual-generator ("harmful", strong-fingerprint) sources can
actively degrade cross-domain generalization, because the model can latch
onto each harmful source's distinctive generator fingerprint instead of
learning transferable spoof-detection features. The proposed mitigation is
a curriculum: establish system-invariant representations on the broad,
weak-fingerprint sources first, then fold the narrow, strong-fingerprint
sources in gradually rather than all at once.

This project's own confound-fixing session (same day) got v6 to the best
in-distribution fit and second-best cross-generator held-out EER of any
post-v3 attempt, but still didn't beat v3's 0.1624 baseline — and, notably,
`ablation` (old, purely broad-generator corpus, no accent cells at all)
still beat v6 (broad corpus + confound-free accent cells). That's the
"1+1<2" pattern exactly: removing shortcuts stopped the model cheating: it
didn't give it a way to reconcile domains it was never taught to expect.

Phase 1 (`--phase1-*`): broad, many-generator sources — base ASVspoof2019/
2021, In-the-Wild train, `en_native` (already a ~20-generator mix per
`accent_manifest.csv`). Trains for `--phase1-epochs` on this alone.

Phase 2 (`--phase2-*`): narrow, single/dual-generator sources — `en_foreign`
(YourTTS), `hi_native`/`hi_foreign` (MMS-TTS + a little XTTS). Folded in for
`--phase2-epochs` more epochs, training on phase1+phase2 combined (not
phase2 alone — the point is reconciling domains, not forgetting phase 1).

A single train/val split (by source, as always) is taken across the full
combined corpus up front, so val_eer is tracked on the same fixed set
through both phases and comparable to every other run this session.

Usage:
    python train_curriculum.py \
        --phase1-real data/real --phase1-fake data/fake \
        --phase1-real-clean data/real2021 data/real_itw_train data/accents_split/train/real/en_native \
        --phase1-fake-clean data/fake2021 data/fake_itw_train data/accents_split/train/fake/en_native \
        --phase2-real-clean data/accents_split/train/real/en_foreign data/accents_split/train/real/hi_native_capped data/accents_split/train/real/hi_foreign \
        --phase2-fake-clean data/accents_split/train/fake/en_foreign data/accents_split/train/fake/hi_native data/accents_split/train/fake/hi_foreign \
        --channel whatsapp volte none \
        --phase1-epochs 20 --phase2-epochs 10 \
        --out runs/voice_guard_curriculum
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import build_examples, split_by_source, to_arrays
from train import compute_eer, parse_channel_arg


def _resolved(paths: list[Path]) -> set[str]:
    return {str(Path(p).resolve()) for p in paths}


def main() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from model import VoiceGuardMLP

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-real", type=Path, required=True, nargs="+")
    ap.add_argument("--phase1-fake", type=Path, required=True, nargs="+")
    ap.add_argument("--phase1-real-clean", type=Path, default=[], nargs="*")
    ap.add_argument("--phase1-fake-clean", type=Path, default=[], nargs="*")
    ap.add_argument("--phase2-real-clean", type=Path, required=True, nargs="+",
                     help="narrow/single-generator ('harmful') real dirs, folded in during phase 2")
    ap.add_argument("--phase2-fake-clean", type=Path, required=True, nargs="+",
                     help="narrow/single-generator ('harmful') fake dirs, folded in during phase 2")
    ap.add_argument("--channel", nargs="*", default=[None],
                     help="applied to --phase1-real/--phase1-fake only, same semantics as train.py.")
    ap.add_argument("--phase1-epochs", type=int, default=20)
    ap.add_argument("--phase2-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("runs/voice_guard_curriculum"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                     help="L2 regularization on the Adam optimizer — see train.py's --weight-decay help.")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--save-every-epoch-checkpoints", action="store_true",
                     help="save a state_dict per epoch (both phases) into <out>/checkpoints/"
                     "epoch_NN.pt — use select_best_checkpoint.py afterward to pick by held-out "
                     "EER, not in-distribution val_eer. See train.py's flag of the same name.")
    ap.add_argument("--checkpoint-every-n-steps", type=int, default=None,
                     help="ALSO save a state_dict every N optimizer steps (mini-batches), into "
                     "<out>/checkpoints/step_NNNNN.pt — finer-grained than per-epoch. Found "
                     "2026-09-10: whole-epoch checkpoints showed epoch 1 (~908 steps for a "
                     "58K-window phase1) as the best-generalizing checkpoint of the whole run, "
                     "with every later epoch worse — this exists to check whether the true peak "
                     "is earlier (or later) within that first epoch, since whole-epoch "
                     "granularity can't see that.")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    recipes = parse_channel_arg(args.channel)
    if "clean" in recipes:
        print("WARNING: 'clean' is a TeleChannel debug recipe, not a no-degradation pass "
              "— see train.py --channel's help text.")

    phase1_dirs = _resolved(args.phase1_real + args.phase1_fake + args.phase1_real_clean + args.phase1_fake_clean)
    phase2_dirs = _resolved(args.phase2_real_clean + args.phase2_fake_clean)
    assert not (phase1_dirs & phase2_dirs), "a directory can't be in both phase1 and phase2"

    print(f"Building phase1 (broad/weak-fingerprint) examples, channels={recipes}...")
    phase1_examples = build_examples(
        args.phase1_real, args.phase1_fake, channel_recipes=recipes,
        workers=args.workers, seed=args.seed,
    )
    if args.phase1_real_clean or args.phase1_fake_clean:
        phase1_examples += build_examples(
            args.phase1_real_clean, args.phase1_fake_clean, channel_recipes=[None],
            workers=args.workers, seed=args.seed,
        )

    print("Building phase2 (narrow/strong-fingerprint, 'harmful') examples...")
    phase2_examples = build_examples(
        args.phase2_real_clean, args.phase2_fake_clean, channel_recipes=[None],
        workers=args.workers, seed=args.seed,
    )

    all_examples = phase1_examples + phase2_examples
    train_ex, val_ex = split_by_source(all_examples)
    phase1_train_ex = [e for e in train_ex if e.source_dir in phase1_dirs]
    phase2_train_ex = [e for e in train_ex if e.source_dir in phase2_dirs]
    assert len(phase1_train_ex) + len(phase2_train_ex) == len(train_ex), \
        "every train example's source_dir must resolve into exactly one phase"

    print(f"{len(train_ex)} train windows ({len(phase1_train_ex)} phase1 + {len(phase2_train_ex)} phase2) "
          f"/ {len(val_ex)} val windows from {len({e.source_file for e in all_examples})} source files")

    X_train_all, y_train_all = to_arrays(train_ex)
    X_train_p1, y_train_p1 = to_arrays(phase1_train_ex)
    X_val, y_val = to_arrays(val_ex)

    # Normalization is fit on the FULL combined training set (what the model
    # sees by the end of phase 2), not phase1 alone — matching what the
    # exported FixedNormalize will actually be used against at inference.
    mean, std = X_train_all.mean(axis=0), X_train_all.std(axis=0) + 1e-8

    model = VoiceGuardMLP(norm_mean=mean, norm_std=std, hidden_dims=tuple(args.hidden_dims)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    X_val_t = torch.from_numpy(X_val).to(device)

    checkpoint_dir = args.out / "checkpoints"
    if args.save_every_epoch_checkpoints or args.checkpoint_every_n_steps:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    def run_epochs(X: np.ndarray, y: np.ndarray, n_epochs: int, phase_name: str, epoch_offset: int) -> float:
        nonlocal global_step
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=args.batch_size, shuffle=True,
        )
        eer = float("nan")
        for i in range(n_epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(xb)
                global_step += 1
                if args.checkpoint_every_n_steps and global_step % args.checkpoint_every_n_steps == 0:
                    model.eval()
                    torch.save(model.cpu().state_dict(), checkpoint_dir / f"step_{global_step:06d}.pt")
                    model.to(device)
                    model.train()
            total_loss /= len(loader.dataset)

            model.eval()
            with torch.no_grad():
                val_probs = torch.softmax(model(X_val_t), dim=-1)[:, 1].cpu().numpy()
            eer = compute_eer(val_probs, y_val)
            epoch_num = epoch_offset + i + 1
            print(f"[{phase_name}] epoch {epoch_num}/{args.phase1_epochs + args.phase2_epochs}  "
                  f"train_loss={total_loss:.4f}  val_eer={eer:.4f}")
            if args.save_every_epoch_checkpoints:
                torch.save(model.cpu().state_dict(), checkpoint_dir / f"epoch_{epoch_num:02d}.pt")
                model.to(device)
        return eer

    run_epochs(X_train_p1, y_train_p1, args.phase1_epochs, "phase1", 0)
    final_eer = run_epochs(X_train_all, y_train_all, args.phase2_epochs, "phase2", args.phase1_epochs)

    model = model.cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")

    dummy = torch.from_numpy(X_val[:1]) if len(X_val) else torch.zeros(1, 63)
    torch.onnx.export(
        model, dummy, str(args.out / "model.onnx"),
        input_names=["features"], output_names=["logits"],
        opset_version=13, dynamo=False,
    )

    metrics = {
        "final_val_eer": final_eer,
        "n_train_phase1": len(phase1_train_ex),
        "n_train_phase2": len(phase2_train_ex),
        "n_val": len(val_ex),
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "hidden_dims": list(args.hidden_dims),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model.pt / model.onnx / metrics.json -> {args.out}")


if __name__ == "__main__":
    main()
