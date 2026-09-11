"""Trains a VoiceGuard sequence model (default: VoiceGuardSeqTCN,
"seqtcn_v2") from the on-disk feature cache.

v12 changes vs v11 (docs/superpowers/plans/2026-09-11-v12-eval-hardening-handoff.md):
- cache-backed memmap data, so no OOM and phone channels are back
  (v11 had to train --channel none);
- the channel policy is enforced: no clean-only training unless --application bank;
- AdamW (wd 0.01), 1-epoch linear warmup then cosine decay, 30 epochs,
  batch 256, EMA weights (0.999) saved per epoch: v11 was still improving
  at epoch 25 on a constant LR, and epoch-to-epoch selection was noisy;
- class-weighted real/fake CE, label smoothing 0.05, attack-type weight 0.5;
- leave-attack-out: A11 (TTS) and A18 (VC) are masked from the attack-type
  loss (still trained as fakes) so evaluate.py can score the attack head on
  systems it never saw labels for;
- per-epoch log: val EER (all channels, and phone-only), attack-type
  balanced accuracy.

Checkpoint selection is NOT done here: select_best_checkpoint_seqcnn.py
picks an epoch on the `select` half of the held-out sets only.

Usage (the v12 run):
    python train_seq_cnn.py --out runs/voice_guard_v12 --cache-root <cache>
Custom dirs (e.g. the smoke test):
    python train_seq_cnn.py --real DIR... --fake DIR... --out runs/x --epochs 1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dataset import IGNORE_ATTACK_TYPE, stable_unit
from eval_protocol import APPLICATIONS, TRAIN_CHANNELS, channel_name, phone_subset, resolve_channels
from eval_stats import balanced_accuracy, compute_eer

# torch is imported lazily (inside functions) on purpose: the cache build
# spawns a Windows process pool, and spawn re-imports this script in every
# worker. A module-level `import torch` makes each worker load the CUDA
# DLLs, which exhausted the page file on this 16 GB machine before
# (README "Windows gotcha").


def compute_masked_attack_type_loss(logits, targets):
    """CrossEntropyLoss with ignore_index=-100 (IGNORE_ATTACK_TYPE): reals,
    unlabeled fakes and leave-out attacks contribute zero gradient. Returns
    NaN if every target is ignored (documented PyTorch behavior); callers
    skip the term then."""
    from torch import nn

    return nn.CrossEntropyLoss(ignore_index=IGNORE_ATTACK_TYPE)(logits, targets)


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-5) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


class EMA:
    """Exponential moving average of parameters (buffers such as BN running
    stats are copied). Decay ramps up over the first steps."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps = 0

    def update(self, model) -> None:
        import torch

        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        with torch.no_grad():
            for pe, p in zip(self.model.parameters(), model.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)
            for be, b in zip(self.model.buffers(), model.buffers()):
                be.copy_(b)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")[-60:]


def build_training_collection(args, channels):
    """Builds/loads the cache units; returns (CacheCollection, unit dirs)."""
    from concurrent.futures import ProcessPoolExecutor

    from build_attack_type_maps import build_attack_type_maps
    from corpus import TRAIN_SETS_V12, SetDef, training_specs, training_units
    from feature_cache import CacheCollection, build_unit

    maps = build_attack_type_maps()
    if args.real or args.fake:
        sets = tuple(SetDef(_safe_name(str(d)), str(Path(d).resolve()), label)
                     for label, dirs in ((0, args.real), (1, args.fake)) for d in dirs)
    else:
        sets = TRAIN_SETS_V12
    specs_by_set = training_specs(sets, attack_type_maps=maps, balance_pad=not args.no_pad_balance,
                                  duration_cache=args.cache_root / "train_durations.json", workers=args.workers)
    units = training_units(specs_by_set, args.seed, tuple(channels))
    dirs = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for name, specs, ch in units:
            dirs.append(build_unit(name, specs, ch, args.cache_root, seed=args.seed, workers=args.workers,
                                   rebuild=args.rebuild_stale, pool=pool if args.workers > 1 else None))
    return CacheCollection(dirs), dirs


def compute_norm_stats(coll, train_idx: np.ndarray, arch: str, seed: int, n_sample: int = 20000) -> dict:
    """Per-LFCC-coefficient mean/std over frames of a random train sample
    (accumulated in batches: 20k windows at float64 would be ~1.8 GB), and
    scalar mean/std over all train windows."""
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(train_idx, size=min(n_sample, len(train_idx)), replace=False))
    s1 = s2 = 0.0
    count = 0
    for i in range(0, len(sample), 1000):
        x = coll.get_seq(sample[i:i + 1000]).astype(np.float64)
        x = x.reshape(-1, x.shape[-1])
        s1 = s1 + x.sum(axis=0)
        s2 = s2 + (x**2).sum(axis=0)
        count += len(x)
    seq_mean = s1 / count
    seq_std = np.sqrt(np.maximum(s2 / count - seq_mean**2, 0.0))
    scal = coll.scalars[train_idx].astype(np.float64)
    return {
        "arch": np.array(arch),
        "seq_mean": seq_mean.astype(np.float32), "seq_std": (seq_std + 1e-6).astype(np.float32),
        "scalar_mean": scal.mean(axis=0).astype(np.float32), "scalar_std": (scal.std(axis=0) + 1e-6).astype(np.float32),
        "n_frames": np.array(184), "n_lfcc": np.array(60), "n_scalars": np.array(coll.scalars.shape[1]),
    }


def score(model, coll, idx: np.ndarray, device: str, batch: int = 2048):
    """(p_fake (n,), attack_probs (n, 2) or None) for windows idx of a CacheCollection."""
    import torch

    model.eval()
    p_fake, att = [], []
    has_attack = True
    with torch.no_grad():
        for _b, seq, scal in coll.iter_batches(idx, batch):
            rf, at = model(torch.from_numpy(seq).to(device), torch.from_numpy(scal).to(device))
            p_fake.append(torch.softmax(rf, -1)[:, 1].float().cpu().numpy())
            if at is None:
                has_attack = False
            else:
                att.append(torch.softmax(at, -1).float().cpu().numpy())
    if not p_fake:
        return np.zeros(0), (np.zeros((0, 2)) if has_attack else None)
    return np.concatenate(p_fake), (np.concatenate(att) if has_attack else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, nargs="*", default=[], help="custom real dirs (default: corpus TRAIN_SETS_V12)")
    ap.add_argument("--fake", type=Path, nargs="*", default=[], help="custom fake dirs")
    ap.add_argument("--channel", "--channels", dest="channels", nargs="*", default=None,
                    help="training channels (default: none whatsapp volte cellular_3g)")
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
    ap.add_argument("--arch", default="seqtcn_v2", choices=["seqtcn_v2", "seqcnn_v1"])
    ap.add_argument("--out", type=Path, default=Path("runs/voice_guard_v12"))
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--attack-type-loss-weight", type=float, default=0.5)
    ap.add_argument("--leave-out-attacks", nargs="*", default=None, help="default: corpus.LEAVE_OUT_ATTACKS")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--no-pad-balance", action="store_true")
    ap.add_argument("--rebuild-stale", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch import nn

    from attack_labels import attack_id_to_int
    from build_caches import default_cache_root
    from corpus import LEAVE_OUT_ATTACKS
    from model import build_model_from_norm_stats, export_onnx

    args.cache_root = args.cache_root or default_cache_root()
    channels = resolve_channels(args.channels if args.channels is not None else TRAIN_CHANNELS,
                                args.application, purpose="train")
    leave_out = tuple(args.leave_out_attacks if args.leave_out_attacks is not None else LEAVE_OUT_ATTACKS)
    leave_out_codes = [attack_id_to_int(a) for a in leave_out]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    print(f"device={device} channels={[channel_name(c) for c in channels]} leave_out={leave_out}", flush=True)

    coll, unit_dirs = build_training_collection(args, channels)
    is_val = np.array([stable_unit(f, args.seed, "val") < args.val_fraction for f in coll.file_id])
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    if len(val_idx) == 0 or len(np.unique(coll.label[val_idx])) < 2:  # tiny custom corpora
        val_idx = train_idx
    attack_target = coll.attack_type.copy()
    attack_target[np.isin(coll.attack_id, leave_out_codes)] = IGNORE_ATTACK_TYPE
    n_real, n_fake = int((coll.label[train_idx] == 0).sum()), int((coll.label[train_idx] == 1).sum())
    print(f"{coll.n} windows ({len(train_idx)} train / {len(val_idx)} val) from {len(set(coll.file_id))} files; "
          f"train real={n_real} fake={n_fake}; attack-labeled train={int((attack_target[train_idx] >= 0).sum())}",
          flush=True)

    norm_stats = compute_norm_stats(coll, train_idx, args.arch, args.seed)
    model = build_model_from_norm_stats(norm_stats).to(device)
    ema = EMA(model, args.ema_decay)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = torch.tensor([len(train_idx) / (2 * max(n_real, 1)), len(train_idx) / (2 * max(n_fake, 1))],
                     dtype=torch.float32, device=device)
    rf_loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)

    steps_per_epoch = max(1, math.ceil(len(train_idx) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "norm_stats.npz", **norm_stats)
    phone_val = val_idx[np.isin(coll.channel[val_idx], phone_subset(channels))]
    def json_safe(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [json_safe(x) for x in v]
        return v

    config = {k: json_safe(v) for k, v in vars(args).items()}
    config.update({"channels": [channel_name(c) for c in channels], "leave_out_attacks": list(leave_out),
                   "unit_dirs": [str(d) for d in unit_dirs], "n_params": n_params,
                   "n_windows": coll.n, "n_train": len(train_idx), "n_val": len(val_idx),
                   "feature_version": coll.manifests[0]["feature_version"] if coll.manifests else None})
    (args.out / "train_config.json").write_text(json.dumps(config, indent=1))
    print(f"arch={args.arch} params={n_params} steps/epoch={steps_per_epoch}", flush=True)

    history = []
    step = 0
    prefetch = ThreadPoolExecutor(max_workers=2)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx)
        batches = [np.sort(perm[i:i + args.batch_size]) for i in range(0, len(perm), args.batch_size)]
        fut = prefetch.submit(coll.get_seq, batches[0])
        total_loss = 0.0
        for bi, b in enumerate(batches):
            seq = fut.result()
            if bi + 1 < len(batches):
                fut = prefetch.submit(coll.get_seq, batches[bi + 1])
            seq_t = torch.from_numpy(seq).to(device, non_blocking=True)
            scal_t = torch.from_numpy(coll.scalars[b]).to(device)
            y_t = torch.from_numpy(coll.label[b]).to(device)
            a_t = torch.from_numpy(attack_target[b]).to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, args.lr)
            rf_logits, at_logits = model(seq_t, scal_t)
            loss = rf_loss_fn(rf_logits, y_t)
            if (a_t != IGNORE_ATTACK_TYPE).any():
                loss = loss + args.attack_type_loss_weight * compute_masked_attack_type_loss(at_logits, a_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            step += 1
            total_loss += loss.item() * len(b)
        total_loss /= len(train_idx)

        p_fake, att = score(ema.model, coll, val_idx, device)
        y_val = coll.label[val_idx]
        val_eer = compute_eer(p_fake, y_val)
        pm = np.isin(val_idx, phone_val)
        val_eer_phone = compute_eer(p_fake[pm], y_val[pm]) if pm.any() else float("nan")
        am = attack_target[val_idx] >= 0
        att_bacc = balanced_accuracy(attack_target[val_idx][am], att[am].argmax(1)) if am.any() else float("nan")
        rec = {"epoch": epoch + 1, "train_loss": total_loss, "val_eer": val_eer, "val_eer_phone": val_eer_phone,
               "val_attack_bacc": att_bacc, "lr_end": lr_at(step - 1, total_steps, warmup_steps, args.lr),
               "seconds": time.time() - t0}
        history.append(rec)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss:.4f} val_eer={val_eer:.4f} "
              f"val_eer_phone={val_eer_phone:.4f} attack_bacc={att_bacc:.3f} ({rec['seconds']:.0f}s)", flush=True)
        torch.save({k: v.cpu() for k, v in ema.model.state_dict().items()}, ckpt_dir / f"epoch_{epoch + 1:02d}.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=1))
    prefetch.shutdown()

    final = ema.model.cpu().eval()
    torch.save(final.state_dict(), args.out / "model.pt")
    export_onnx(final, args.out / "model.onnx")
    print(f"Saved model.pt (last-epoch EMA) / model.onnx / norm_stats.npz / history.json -> {args.out}. "
          "Next: select_best_checkpoint_seqcnn.py (select split only), then evaluate.py --split test.", flush=True)


if __name__ == "__main__":
    main()
