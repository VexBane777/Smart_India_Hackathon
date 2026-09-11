"""Measures whether storing LFCC sequences as float16 in the feature cache
changes model outputs. Scores the same windows from fp32 features and from
fp16-rounded features, and reports max/mean |delta p_fake| and decision flips.

Windows come from real held-out files (select split, ITW real + fake),
rendered through `none` and two phone channels.

Usage:
    python validate_fp16.py --model runs/voice_guard_v11_seqcnn_selected/model.pt \
        --model runs/voice_guard_v9_noisefix_final/model.pt --out runs/fp16_validation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_protocol import APPLICATIONS, resolve_channels

MAX_ABS_DELTA_GATE = 0.01


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, action="append", required=True)
    ap.add_argument("--n-files", type=int, default=150, help="per set")
    ap.add_argument("--channels", nargs="*", default=["none", "whatsapp", "gsm_2g"])
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    from build_attack_type_maps import build_attack_type_maps
    from corpus import eval_specs
    from dataset import process_file, stable_unit
    from model import load_scoring_model

    channels = resolve_channels(args.channels, args.application, "eval")
    specs_by_set = eval_specs("select", ("itw_real", "itw_fake"), attack_type_maps=build_attack_type_maps())
    specs = []
    for s in specs_by_set.values():
        specs += sorted(s, key=lambda x: stable_unit(x.file_id, "fp16"))[: args.n_files]
    seqs, scal = [], []
    for spec in specs:
        for ch in channels:
            ws, _st, _d = process_file(spec, ch, 0)
            seqs += [w.lfcc_seq for w in ws]
            scal += [w.scalars for w in ws]
    seq32 = np.stack(seqs).astype(np.float32)
    seq16 = seq32.astype(np.float16).astype(np.float32)
    scal = np.stack(scal).astype(np.float32)
    print(f"{len(seq32)} windows; max |seq32 - seq16| = {np.abs(seq32 - seq16).max():.4f} "
          f"(max |value| {np.abs(seq32).max():.1f})")

    report = {"n_windows": int(len(seq32)), "channels": args.channels,
              "max_abs_feature_delta": float(np.abs(seq32 - seq16).max()), "models": {}}
    for mp in args.model:
        model, arch = load_scoring_model(mp)
        with torch.no_grad():
            p = []
            for x in (seq32, seq16):
                out = []
                for i in range(0, len(x), 2048):
                    rf, _ = model(torch.from_numpy(x[i:i + 2048]), torch.from_numpy(scal[i:i + 2048]))
                    out.append(torch.softmax(rf, -1)[:, 1].numpy())
                p.append(np.concatenate(out))
        d = np.abs(p[0] - p[1])
        r = {"arch": arch, "max_abs_delta_p": float(d.max()), "mean_abs_delta_p": float(d.mean()),
             "flips_at_0.5": int(((p[0] >= 0.5) != (p[1] >= 0.5)).sum()),
             "flips_at_0.6": int(((p[0] >= 0.6) != (p[1] >= 0.6)).sum()),
             "pass": bool(d.max() <= MAX_ABS_DELTA_GATE)}
        report["models"][str(mp)] = r
        print(f"{mp} ({arch}): max|dp|={r['max_abs_delta_p']:.2e} mean={r['mean_abs_delta_p']:.2e} "
              f"flips@0.5={r['flips_at_0.5']} flips@0.6={r['flips_at_0.6']} -> {'PASS' if r['pass'] else 'FAIL'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if not all(r["pass"] for r in report["models"].values()):
        raise SystemExit(f"fp16 storage changes outputs by more than {MAX_ABS_DELTA_GATE}; switch SEQ_DTYPE to float32")


if __name__ == "__main__":
    main()
