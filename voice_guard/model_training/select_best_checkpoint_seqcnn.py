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
import shutil
from pathlib import Path

import numpy as np

from eval_protocol import APPLICATIONS, channel_name, resolve_channels
from eval_stats import compute_eer

SPLIT = "select"  # the only split this script may read


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training run dir (checkpoints/, norm_stats.npz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--channels", nargs="*", default=None, help="default: none + all phone channels")
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
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
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoints = sorted((args.run / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no epoch_*.pt in {args.run / 'checkpoints'}")
    norm_stats = dict(np.load(args.run / "norm_stats.npz"))

    coll = load_eval_collection(SPLIT, channels, args.cache_root or default_cache_root(),
                                set_names=CORE_EVAL_SETS, workers=args.workers)
    idx = np.arange(coll.n)
    phone_mask = np.isin(coll.channel, phone)
    print(f"{coll.n} {SPLIT}-split windows ({int(phone_mask.sum())} phone) from {len(set(coll.file_id))} files")

    sweep = {}
    for ckpt in checkpoints:
        model = build_model_from_norm_stats(norm_stats)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        p, _att = score(model.to(device), coll, idx, device)
        row = {"phone_pooled_eer": compute_eer(p[phone_mask], coll.label[phone_mask])}
        for c in channels:
            m = coll.channel == channel_name(c)
            row[f"eer_{channel_name(c)}"] = compute_eer(p[m], coll.label[m])
        sweep[ckpt.name] = row
        print(f"  {ckpt.name}: phone-pooled EER={row['phone_pooled_eer']:.4f}  none={row['eer_none']:.4f}")

    best = min(sweep, key=lambda k: sweep[k]["phone_pooled_eer"])
    print(f"\nBest on {SPLIT}: {best}  phone-pooled EER={sweep[best]['phone_pooled_eer']:.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.run / "checkpoints" / best, args.out / "model.pt")
    np.savez(args.out / "norm_stats.npz", **norm_stats)
    model = build_model_from_norm_stats(norm_stats)
    model.load_state_dict(torch.load(args.out / "model.pt", map_location="cpu", weights_only=True))
    export_onnx(model, args.out / "model.onnx")
    (args.out / "checkpoint_sweep.json").write_text(json.dumps({
        "split": SPLIT, "metric": "phone_pooled_eer over core eval sets",
        "channels": [channel_name(c) for c in channels], "best_checkpoint": best,
        "best": sweep[best], "all": sweep, "source_run": str(args.run),
    }, indent=1))
    print(f"Wrote model.pt / model.onnx / norm_stats.npz / checkpoint_sweep.json -> {args.out}")


if __name__ == "__main__":
    main()
