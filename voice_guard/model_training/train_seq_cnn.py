"""Trains a VoiceGuard sequence model (default: VoiceGuardSeqTCN,
"seqtcn_v2") from the on-disk feature cache.

v12 changes vs v11 (docs/superpowers/plans/2026-09-11-v12-eval-hardening-handoff.md):
- cache-backed memmap data, so no OOM and phone channels are back
  (v11 had to train --channel none);
- the channel policy is enforced: no clean-only training unless --application bank;
- AdamW (wd 0.01), 1-epoch linear warmup then cosine decay to min_lr (1e-5),
  30 epochs, batch 256, EMA weights (0.999) saved per epoch. v11 was still
  improving at epoch 25 on a constant LR, and epoch-to-epoch selection was noisy;
- cosine-then-hold via --min-lr-epochs: hold the LR at min_lr for that many
  epochs after the cosine completes, so extending --epochs adds real low-LR
  polish instead of re-stretching the cosine (default 0 = current behavior);
- class-weighted real/fake CE, label smoothing 0.05, attack-type weight 0.5;
- leave-attack-out: A11 (TTS) and A18 (VC) are masked from the attack-type
  loss (still trained as fakes) so evaluate.py can score the attack head on
  systems it never saw labels for;
- per-epoch log: val EER (all channels, and phone-only), attack-type
  balanced accuracy.

Post-v13 rework levers (docs/2026-09-training-improvement-plan.md, 2026-09-16).
Every one is OFF by default and reduces to the exact v12/v13 behavior when off:
- capacity, persisted in norm_stats.npz so every loader rebuilds the same
  architecture: --model-channels, --model-dilations, --model-hidden,
  --model-dropout;
- paired regularization, also persisted: --model-stochastic-depth (train-only
  per-sample residual skips), --model-cmvn (per-utterance normalization inside
  the graph -- deterministic and ONNX-safe), --model-se (squeeze-excitation);
- --focal-gamma (0 = plain weighted CE), --attack-type-warmup-epochs (ramp the
  auxiliary head 0 -> --attack-type-loss-weight), --mixup-alpha,
  --specaugment-freq-masks / --specaugment-time-masks (+ widths);
- --min-lr-epochs (cosine-then-hold) and --grad-clip.
The ONNX I/O contract is unchanged by any of them (inputs 1x184x60 + 1x6 ->
real_fake_logits 1x2 + attack_type_logits 1x2).

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
from eval_protocol import APPLICATIONS, TRAIN_CHANNELS, acoustic_subset, channel_name, phone_subset, resolve_channels
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


# --------------------------------------------------------------------------
# Post-v13 rework levers (docs/2026-09-training-improvement-plan.md axes F/D3).
# Every one of these is OFF by default and reduces to the exact pre-rework
# behavior when off, so v12/v13 runs stay reproducible.
# --------------------------------------------------------------------------
def attack_type_weight(epoch: int, warmup_epochs: float, base_weight: float) -> float:
    """Linear ramp 0 -> base_weight over `warmup_epochs` (0-based `epoch`).

    Axis F "warm the attack-type head": at default warmup 0 this returns
    base_weight for every epoch, i.e. the constant weight v12/v13 used."""
    if warmup_epochs <= 0:
        return base_weight
    return base_weight * min(1.0, (epoch + 1) / warmup_epochs)


def focal_cross_entropy(logits, targets, weight=None, gamma: float = 0.0, label_smoothing: float = 0.0):
    """Weighted cross-entropy with optional focal modulation (1 - p_t)^gamma.

    gamma=0 calls nn.CrossEntropyLoss(weight, label_smoothing) directly, so the
    default is bit-identical to the v12/v13 real-fake loss. gamma>0 multiplies
    each sample's CE by (1 - p_t)^gamma and normalizes by the summed weights,
    matching PyTorch's `reduction="mean"` convention for weighted losses."""
    from torch import nn

    if gamma <= 0:
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)(logits, targets)
    import torch

    n_classes = logits.shape[1]
    logp = torch.log_softmax(logits, dim=-1)
    target = torch.full_like(logp, label_smoothing / n_classes)
    target.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + label_smoothing / n_classes)
    per_sample = -(target * logp).sum(dim=-1)
    with torch.no_grad():
        p_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
    per_sample = per_sample * (1.0 - p_t).clamp(min=0.0).pow(gamma)
    if weight is not None:
        w = weight[targets]
        return (per_sample * w).sum() / w.sum().clamp(min=1e-12)
    return per_sample.mean()


def soft_target_cross_entropy(logits, soft_target, weight=None, label_smoothing: float = 0.0):
    """-sum(target * log_softmax) for a (n, K) soft target (Mixup), normalized
    like nn.CrossEntropyLoss(weight=...): weighted mean over samples."""
    import torch

    if label_smoothing > 0:
        k = soft_target.shape[1]
        soft_target = (1.0 - label_smoothing) * soft_target + label_smoothing / k
    logp = torch.log_softmax(logits, dim=-1)
    per_sample = -(soft_target * logp).sum(dim=-1)
    if weight is None:
        return per_sample.mean()
    w = (soft_target * weight).sum(dim=-1)
    return (per_sample * w).sum() / w.sum().clamp(min=1e-12)


def sample_mixup_lambda(alpha: float, rng) -> float:
    """Beta(alpha, alpha) draw from a numpy Generator, so augmentation is
    reproducible from --seed and does not perturb torch's batch-order RNG."""
    return float(rng.beta(alpha, alpha))


def mixup_batch(seq, scal, label, attack, lam: float, perm=None):
    """Mixup (Zhang et al. 2018) over a batch with explicit mixing weight `lam`.

    Returns (seq_mix, scal_mix, soft_rf, soft_attack, attack_mask):
    - soft_rf (n, 2): one-hot mixture for the real/fake head -> feed to
      soft_target_cross_entropy;
    - soft_attack (n, 2): one-hot mixture of the attack labels, only meaningful
      where attack_mask is True;
    - attack_mask (n,): partners that BOTH carry a real attack label (mixing
      IGNORE_ATTACK_TYPE is undefined), mirroring the leave-attack-out masking."""
    import torch

    n = seq.shape[0]
    if perm is None:
        perm = torch.randperm(n, device=seq.device)
    perm = perm.to(seq.device)
    seq_mix = lam * seq + (1.0 - lam) * seq[perm]
    scal_mix = lam * scal + (1.0 - lam) * scal[perm]

    def onehot(t):
        return torch.nn.functional.one_hot(t.long(), 2).to(seq.dtype)

    soft_rf = lam * onehot(label) + (1.0 - lam) * onehot(label[perm])
    mask = (attack >= 0) & (attack[perm] >= 0)
    soft_attack = torch.zeros_like(soft_rf)
    if bool(mask.any()):
        soft_attack[mask] = lam * onehot(attack[mask]) + (1.0 - lam) * onehot(attack[perm][mask])
    return seq_mix, scal_mix, soft_rf, soft_attack, mask


def specaugment(seq, n_freq_masks: int = 0, freq_width: int = 8, n_time_masks: int = 0,
                time_width: int = 8, rng=None):
    """Park et al. 2019 SpecAugment on a (B, T, C) LFCC batch, in a clone.

    Random-width frequency bands / time spans are zeroed per sample (zero is
    the LFCC mean-ish value and is what the model's FixedNormalizeSeq expects
    to see; the raw sequence is augmented BEFORE normalization). No-op when
    every count is 0, which is the default."""
    import torch

    if n_freq_masks <= 0 and n_time_masks <= 0:
        return seq
    out = seq.clone()
    n, n_frames, n_lfcc = out.shape
    rng = rng or np.random.default_rng(0)
    for i in range(n):
        for _ in range(max(0, n_freq_masks)):
            if freq_width <= 0:
                break
            w = int(rng.integers(1, freq_width + 1))
            f0 = int(rng.integers(0, max(1, n_lfcc - w + 1)))
            out[i, :, f0:f0 + w] = 0.0
        for _ in range(max(0, n_time_masks)):
            if time_width <= 0:
                break
            w = int(rng.integers(1, time_width + 1))
            t0 = int(rng.integers(0, max(1, n_frames - w + 1)))
            out[i, t0:t0 + w, :] = 0.0
    return out


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
    units = training_units(specs_by_set, args.seed, tuple(channels),
                           playback_fraction=args.playback_fraction)
    dirs = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for name, specs, ch in units:
            dirs.append(build_unit(name, specs, ch, args.cache_root, seed=args.seed, workers=args.workers,
                                   rebuild=args.rebuild_stale, pool=pool if args.workers > 1 else None))
    return CacheCollection(dirs), dirs


def compute_norm_stats(coll, train_idx: np.ndarray, arch: str, seed: int,
                       capacity: dict | None = None, n_sample: int = 20000) -> dict:
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
    stats = {
        "arch": np.array(arch),
        "seq_mean": seq_mean.astype(np.float32), "seq_std": (seq_std + 1e-6).astype(np.float32),
        "scalar_mean": scal.mean(axis=0).astype(np.float32), "scalar_std": (scal.std(axis=0) + 1e-6).astype(np.float32),
        "n_frames": np.array(184), "n_lfcc": np.array(60), "n_scalars": np.array(coll.scalars.shape[1]),
    }
    # Persist capacity so build_model_from_norm_stats rebuilds the SAME arch when loading
    # model.pt (select/checkpoint + evaluate + validate_fp16 all rebuild from norm_stats).
    if capacity is not None and arch == "seqtcn_v2":
        stats["channels"] = np.array(int(capacity["channels"]))
        stats["dilations"] = np.array(capacity["dilations"])
        stats["dropout"] = np.array(float(capacity["dropout"]))
        stats["hidden"] = np.array(int(capacity["hidden"]))
        stats["stochastic_depth"] = np.array(float(capacity.get("stochastic_depth", 0.0)))
        stats["cmvn"] = np.array(bool(capacity.get("cmvn", False)))
        stats["se"] = np.array(bool(capacity.get("se", False)))
    return stats


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
    ap.add_argument("--playback-fraction", type=float, default=0.5,
                    help="hash-selected share of training files that ALSO get a third "
                         "playback rendition (deterministic via stable_unit; default "
                         "0.5 = the pre-registered v13 recipe)")
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
    ap.add_argument("--arch", default="seqtcn_v2", choices=["seqtcn_v2", "seqcnn_v1"])
    ap.add_argument("--out", type=Path, default=Path("runs/voice_guard_v12"))
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-lr-epochs", type=float, default=0.0,
                    help="hold LR at min_lr for this many trailing epochs AFTER the cosine "
                         "completes (so --epochs N --min-lr-epochs M = (N-M) cosine + M hold). "
                         "Growing --epochs without this just re-stretches the cosine; 0 keeps the "
                         "original behavior.")
        # --- model capacity (seqtcn_v2); persisted in norm_stats.npz so loaders rebuild the same arch ---
    ap.add_argument("--model-channels", type=int, default=64, help="seqtcn_v2 Conv1d width (default 64).")
    ap.add_argument("--model-dilations", type=str, default="1,2,4,8,16",
                    help="seqtcn_v2 residual dilations, comma-separated (default '1,2,4,8,16').")
    ap.add_argument("--model-hidden", type=int, default=64, help="seqtcn_v2 trunk/head width (default 64).")
    ap.add_argument("--model-dropout", type=float, default=0.1, help="seqtcn_v2 dropout (default 0.1).")
    ap.add_argument("--model-stochastic-depth", type=float, default=0.0,
                    help="per-sample residual-block skip probability, linearly ramped by block depth "
                         "(0.0 = off). Train-time only: eval/ONNX takes the identity path, so the "
                         "exported graph is unaffected.")
    ap.add_argument("--model-cmvn", action="store_true",
                    help="per-utterance mean/var normalization of the LFCC sequence inside the graph "
                         "(deterministic, ONNX-safe; attacks the channel/level confound). Persisted "
                         "in norm_stats.npz.")
    ap.add_argument("--model-se", action="store_true",
                    help="squeeze-excitation channel attention in each residual block (pairs with "
                         "--model-channels). Persisted in norm_stats.npz.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="clip global grad norm to this max (None = no clip). 1.0 recommended for "
                         "larger models / higher LR.")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="focal-loss focusing exponent for the real/fake term (0.0 = plain weighted "
                         "CE, the pre-rework behavior). Standard for spoofing: down-weights easy "
                         "examples instead of adding epochs.")
    ap.add_argument("--attack-type-warmup-epochs", type=float, default=0.0,
                    help="ramp the attack-type loss weight linearly 0 -> --attack-type-loss-weight "
                         "over this many epochs (0.0 = constant weight, the pre-rework behavior), so "
                         "the auxiliary head cannot perturb the real/fake head early.")
    ap.add_argument("--mixup-alpha", type=float, default=0.0,
                    help="Beta(alpha, alpha) Mixup on (lfcc_sequence, scalars) with soft real/fake "
                         "targets; the attack-type term mixes one-hot labels only where BOTH partners "
                         "are attack-labeled (0.0 = off). Train-time only.")
    ap.add_argument("--specaugment-freq-masks", type=int, default=0,
                    help="number of frequency masks per sample (0 = off). Applied to the raw LFCC "
                         "sequence before normalization, train-time only.")
    ap.add_argument("--specaugment-freq-width", type=int, default=8, help="max width of a frequency mask.")
    ap.add_argument("--specaugment-time-masks", type=int, default=0,
                    help="number of time masks per sample (0 = off).")
    ap.add_argument("--specaugment-time-width", type=int, default=8, help="max width of a time mask.")
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

    norm_stats = compute_norm_stats(coll, train_idx, args.arch, args.seed,
                                    capacity={"channels": args.model_channels,
                                              "dilations": tuple(int(x) for x in args.model_dilations.split(",")),
                                              "dropout": args.model_dropout, "hidden": args.model_hidden,
                                              "stochastic_depth": args.model_stochastic_depth,
                                              "cmvn": args.model_cmvn, "se": args.model_se})
    model = build_model_from_norm_stats(norm_stats).to(device)
    ema = EMA(model, args.ema_decay)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = torch.tensor([len(train_idx) / (2 * max(n_real, 1)), len(train_idx) / (2 * max(n_fake, 1))],
                     dtype=torch.float32, device=device)
    seg_rng = np.random.default_rng(args.seed + 1)  # SpecAugment/Mixup stream, separate from batch order

    steps_per_epoch = max(1, math.ceil(len(train_idx) / args.batch_size))
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))
    # Cosine decays over `total_steps`; the trailing `min_lr_steps` epochs hold flat at
    # min_lr (lr_at clamps progress to 1.0 beyond total_steps). This decouples "more
    # epochs" from "higher LR longer": --min-lr-epochs adds real low-LR polish instead of
    # re-stretching the cosine. Clamp total_steps >= warmup+1 to keep the cosine phase valid.
    min_lr_steps = max(0, int(round(args.min_lr_epochs * steps_per_epoch)))
    total_steps = max(warmup_steps + 1, steps_per_epoch * args.epochs - min_lr_steps)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "norm_stats.npz", **norm_stats)
    phone_val = val_idx[np.isin(coll.channel[val_idx], phone_subset(channels))]
    acous_val = val_idx[np.isin(coll.channel[val_idx], acoustic_subset(channels))]
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
                   "feature_version": coll.manifests[0]["feature_version"] if coll.manifests else None,
                   "playback_fraction": args.playback_fraction})
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
        at_w = attack_type_weight(epoch, args.attack_type_warmup_epochs, args.attack_type_loss_weight)
        for bi, b in enumerate(batches):
            seq = fut.result()
            if bi + 1 < len(batches):
                fut = prefetch.submit(coll.get_seq, batches[bi + 1])
            seq_t = torch.from_numpy(seq).to(device, non_blocking=True)
            scal_t = torch.from_numpy(coll.scalars[b]).to(device)
            y_t = torch.from_numpy(coll.label[b]).to(device)
            a_t = torch.from_numpy(attack_target[b]).to(device)
            if args.specaugment_freq_masks > 0 or args.specaugment_time_masks > 0:
                seq_t = specaugment(seq_t, args.specaugment_freq_masks, args.specaugment_freq_width,
                                    args.specaugment_time_masks, args.specaugment_time_width, rng=seg_rng)
            if args.mixup_alpha > 0:
                # pair each window with a random partner (same stream as SpecAugment)
                lam = sample_mixup_lambda(args.mixup_alpha, seg_rng)
                partner = torch.from_numpy(seg_rng.permutation(len(b))).to(device)
                seq_t, scal_t, soft_rf, soft_attack, att_mask = mixup_batch(seq_t, scal_t, y_t, a_t, lam,
                                                                           perm=partner)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, args.lr)
            rf_logits, at_logits = model(seq_t, scal_t)
            if args.mixup_alpha > 0:
                loss = soft_target_cross_entropy(rf_logits, soft_rf, weight=w,
                                                label_smoothing=args.label_smoothing)
                if at_w > 0 and bool(att_mask.any()):
                    loss = loss + at_w * soft_target_cross_entropy(at_logits[att_mask], soft_attack[att_mask])
            else:
                loss = focal_cross_entropy(rf_logits, y_t, weight=w, gamma=args.focal_gamma,
                                           label_smoothing=args.label_smoothing)
                if at_w > 0 and (a_t != IGNORE_ATTACK_TYPE).any():
                    loss = loss + at_w * compute_masked_attack_type_loss(at_logits, a_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        amask = np.isin(val_idx, acous_val)
        val_eer_acoustic = compute_eer(p_fake[amask], y_val[amask]) if amask.any() else float("nan")
        am = attack_target[val_idx] >= 0
        att_bacc = balanced_accuracy(attack_target[val_idx][am], att[am].argmax(1)) if am.any() else float("nan")
        rec = {"epoch": epoch + 1, "train_loss": total_loss, "val_eer": val_eer, "val_eer_phone": val_eer_phone,
               "val_eer_acoustic": val_eer_acoustic,
               "val_attack_bacc": att_bacc, "lr_end": lr_at(step - 1, total_steps, warmup_steps, args.lr),
               "attack_weight": at_w, "seconds": time.time() - t0}
        history.append(rec)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss:.4f} val_eer={val_eer:.4f} "
              f"val_eer_phone={val_eer_phone:.4f} val_eer_acoustic={val_eer_acoustic:.4f} "
              f"attack_bacc={att_bacc:.3f} ({rec['seconds']:.0f}s)", flush=True)
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
