"""Picks a training run's best epoch checkpoint using ONLY the `select`
half of the held-out sets (eval_splits/held_out_split_v1.json), scored
under the channel protocol (metric: phone-pooled EER on the core sets).

Why only `select`: v11 picked its epoch on fake_itw_held, which was also
the fake half of its headline EER test, so that EER was optimistic. This
script never touches `test`; test_eval_protocol.py fails the build if it
ever references the test split. The final number comes from
`evaluate.py --split test`, run once per candidate.

Usage:
    python select_best_checkpoint_seqcnn.py --run runs/voice_guard_v12 --out runs/voice_guard_v12_selected
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from eval_protocol import APPLICATIONS, channel_name, resolve_channels, acoustic_subset
from eval_stats import compute_eer

SPLIT = "select"  # the only split this script may read


def _epoch_of(name: str) -> int:
    """'epoch_06.pt' -> 6 (epoch number, for stability-ordered selection)."""
    m = re.search(r"epoch_(\d+)", name)
    return int(m.group(1)) if m else -1


def _objective(row: dict, metric: str, channels: list[str]) -> float:
    """Selection objective for one checkpoint row.

    "pooled" = the pre-registered v13 objective (EER pooled over phone+acoustic
    windows). "channel_balanced" = the mean of the per-channel EERs, so the two
    worst codecs (gsm_2g, tandem_xnet) stop dominating the argmin -- rework axis
    D1: the pooled argmin was chasing noisy channels, not generalization."""
    if metric == "channel_balanced":
        vals = [float(row[f"eer_{c}"]) for c in channels if f"eer_{c}" in row]
        vals = [v for v in vals if not math.isnan(v)]
        return float(sum(vals) / len(vals)) if vals else float("nan")
    return float(row["pooled_eer"])


def _clears_floor(row: dict, floor: float | None) -> bool:
    """Multi-objective floor (axis H): rows with no attack-head measurement are
    not eligible, unless the caller already dropped the floor (no history.json)."""
    if floor is None:
        return True
    val = row.get("val_attack_bacc")
    return val is not None and not math.isnan(float(val)) and float(val) >= floor


def attack_bacc_from_history(history: list[dict]) -> dict[str, float]:
    """{'epoch_06.pt': val_attack_bacc} from a training run's history.json rows.

    The selector reads the attack-head metric from the run's own per-epoch val
    log rather than re-scoring the attack head on `select`, because the eval
    cache does not carry attack labels (evaluate.py scores that head on test)."""
    out = {}
    for rec in history:
        ep = int(rec.get("epoch", -1))
        if ep > 0:
            out[f"epoch_{ep:02d}.pt"] = float(rec.get("val_attack_bacc", float("nan")))
    return out


def select_best(sweep: dict, tolerance: float = 0.0, metric: str = "pooled",
                channels: list[str] | tuple[str, ...] = (), min_attack_bacc: float | None = None) -> str:
    """Best checkpoint by `metric`, with two optional robustness rules:

    - `tolerance` > 0: return the EARLIEST epoch within `tolerance` of the best
      objective (stability selection: less training, same val EER, less
      chance-dip bias) instead of the raw argmin;
    - `min_attack_bacc`: multi-objective floor -- only epochs whose attack-type
      balanced accuracy clears it are eligible, so an EER-min checkpoint with a
      weak attack head cannot win. If nothing clears the floor, it is dropped
      (callers warn) rather than failing the selection."""
    chans = list(channels)

    def key_of(k: str) -> tuple[float, int, str]:
        val = _objective(sweep[k], metric, chans)
        return (float("inf") if math.isnan(val) else val, _epoch_of(k), k)

    eligible = [k for k in sweep if _clears_floor(sweep[k], min_attack_bacc)]
    if not eligible:
        eligible = list(sweep)
    best_key = min(eligible, key=key_of)
    if tolerance <= 0:
        return best_key
    threshold = key_of(best_key)[0] + tolerance
    for key in sorted(eligible, key=_epoch_of):
        if key_of(key)[0] <= threshold:
            return key
    return best_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training run dir (checkpoints/, norm_stats.npz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--channels", nargs="*", default=None, help="default: none + all phone channels")
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
    ap.add_argument("--stability-tolerance", type=float, default=0.0,
                    help="if >0, pick the EARLIEST epoch within this absolute EER of the argmin "
                         "instead of the raw argmin (stability selection: less training, same val "
                         "EER, less overfit); 0 = raw argmin (current behavior).")
    ap.add_argument("--metric", default="pooled", choices=["pooled", "channel_balanced"],
                    help="selection objective: 'pooled' = the pre-registered phone+acoustic pooled "
                         "EER (v13 default, unchanged); 'channel_balanced' = mean of the per-channel "
                         "EERs, so gsm_2g/tandem_xnet cannot dominate the argmin (rework axis D1).")
    ap.add_argument("--min-attack-bacc", type=float, default=None,
                    help="multi-objective floor (rework axis H): only epochs whose attack-type "
                         "balanced accuracy clears this are eligible, so an EER-min checkpoint with "
                         "a weak attack head cannot win. Read per epoch from the run's history.json; "
                         "dropped with a warning if that is unavailable.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    import torch

    from build_caches import default_cache_root
    from corpus import CORE_EVAL_SETS
    from evaluate import load_eval_collection
    from model import build_model_from_norm_stats, export_onnx
    from train_seq_cnn import score

    channels = resolve_channels(args.channels, args.application, "eval")
    phone = [c for c in channels if c is not None]
    acous = acoustic_subset(channels)
    # v13 pre-registered selection objective: EER pooled over phone + acoustic
    # channels (the acoustic loop is a deploy gate now, so selection must care
    # about it, not just phone). Phone-only EER is kept as a diagnostic row.
    pool = phone + acous
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoints = sorted((args.run / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no epoch_*.pt in {args.run / 'checkpoints'}")
    norm_stats = dict(np.load(args.run / "norm_stats.npz"))

    coll = load_eval_collection(SPLIT, channels, args.cache_root or default_cache_root(),
                                set_names=CORE_EVAL_SETS, workers=args.workers)
    idx = np.arange(coll.n)
    pool_mask = np.isin(coll.channel, pool)
    phone_mask = np.isin(coll.channel, phone)
    print(f"{coll.n} {SPLIT}-split windows ({int(pool_mask.sum())} phone+acoustic, "
          f"{int(phone_mask.sum())} phone) from {len(set(coll.file_id))} files")

    sweep = {}
    for ckpt in checkpoints:
        model = build_model_from_norm_stats(norm_stats)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        p, _att = score(model.to(device), coll, idx, device)
        row = {"pooled_eer": compute_eer(p[pool_mask], coll.label[pool_mask]),
               "phone_eer": compute_eer(p[phone_mask], coll.label[phone_mask])}
        for c in channels:
            mm = coll.channel == channel_name(c)
            row[f"eer_{channel_name(c)}"] = compute_eer(p[mm], coll.label[mm])
        sweep[ckpt.name] = row
        print(f"  {ckpt.name}: pooled EER={row['pooled_eer']:.4f}  phone={row['phone_eer']:.4f}  "
              f"none={row['eer_none']:.4f}")

    # per-epoch attack-head balanced accuracy from the run's own history (axis H floor)
    history_path = args.run / "history.json"
    if history_path.exists():
        for name, bacc in attack_bacc_from_history(json.loads(history_path.read_text())).items():
            if name in sweep:
                sweep[name]["val_attack_bacc"] = bacc
    floor = args.min_attack_bacc
    if floor is not None and not any("val_attack_bacc" in row for row in sweep.values()):
        print("WARNING: no usable history.json/val_attack_bacc -> --min-attack-bacc dropped")
        floor = None

    metric_channels = [channel_name(c) for c in pool]
    best = select_best(sweep, args.stability_tolerance, args.metric, metric_channels, floor)
    sel_mode = "argmin" if args.stability_tolerance <= 0 else f"earliest-within-{args.stability_tolerance:g}"
    if floor is not None:
        sel_mode += f", attack_bacc>={floor:g}"
    print(f"\nBest on {SPLIT} ({args.metric}, {sel_mode}): {best}  "
          f"pooled EER (phone+acoustic)={sweep[best]['pooled_eer']:.4f}  "
          f"phone={sweep[best]['phone_eer']:.4f}  "
          f"balanced={_objective(sweep[best], 'channel_balanced', metric_channels):.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.run / "checkpoints" / best, args.out / "model.pt")
    np.savez(args.out / "norm_stats.npz", **norm_stats)
    model = build_model_from_norm_stats(norm_stats)
    model.load_state_dict(torch.load(args.out / "model.pt", map_location="cpu", weights_only=True))
    export_onnx(model, args.out / "model.onnx")
    (args.out / "checkpoint_sweep.json").write_text(json.dumps({
        "split": SPLIT, "metric": "pooled_eer over phone+acoustic core eval sets (v13, pre-registered)",
        "selection_metric": args.metric, "min_attack_bacc": floor,
        "channels": [channel_name(c) for c in channels], "best_checkpoint": best,
        "selection_mode": sel_mode, "stability_tolerance": args.stability_tolerance,
        "best": sweep[best], "all": sweep, "source_run": str(args.run),
    }, indent=1))
    print(f"Wrote model.pt / model.onnx / norm_stats.npz / checkpoint_sweep.json -> {args.out}")


if __name__ == "__main__":
    main()
