"""
Near-duplicate dedup across train/select/test splits (rework axis D2).

Catches files that are near-identical across the train/select/test boundary
so a model can't "memorize" a held-out file via an almost-identical training
rendition. Two mechanisms:

1. Exact audio hash (MD5 of the decoded float32 PCM) — catches bit-identical
   files that got copied into another split, or the same source rendered
   through two recipes that round-trip to identical PCM.

2. Perceptual near-dup (spectral band-energy fingerprint, quantized) — catches
   "the same clip rendered twice through slightly different pipelines" without
   paying for a full chroma/audio-fingerprint library.

Both are cheap relative to a training run and are meant to be run as a
*preflight gate* (like check_corpus.py), not during training.

Usage:
    python dedup.py --splits eval_splits/held_out_split_v1.json --data-root <DATA_ROOT>
    # prints duplicates across splits; exits nonzero if any found
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset import DATA_ROOT

# A file_id can appear in multiple split groups in the manifest. When two
# different file_ids (or the same file_id referenced from two groups) decode to
# the same audio, that's either a true duplicate (exact MD5) or a near-dup
# (perceptual fingerprint). Cross-split duplicates let a model "memorize" a
# held-out file via an almost-identical training rendition, so we flag them as
# a preflight gate (like check_corpus.py), not during training.

# Allowed group names in the manifest for this check.
_SPLIT_GROUPS = ("select", "test", "train")


def _md5_of_path(path: Path) -> str | None:
    """MD5 of decoded float32 mono PCM; None on any read/decode error."""
    try:
        data, _ = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None
    if data is None:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return hashlib.md5(data.tobytes()).hexdigest()


def _perceptual_fp(path: Path) -> str | None:
    """Quantized spectral-band fingerprint: 32 log-mag bands, 4 bits each -> 16 hex chars.

    Cheap enough for a preflight scan; not a replacement for a full fingerprint
    library, but catches "same clip rendered through two slightly different pipelines".
    Returns None on read error or a too-short file.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None
    if data is None or len(data) < 256:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    # Resample to 16 kHz if needed (linear interp; we only want band shape).
    if sr != 16000:
        old_t = np.arange(len(data)) / sr
        new_t = np.arange(0, old_t[-1], 1 / 16000)
        data = np.interp(new_t, old_t, data).astype(np.float32)
    window = data * np.hanning(len(data))
    spec = np.abs(np.fft.rfft(window))
    log_spec = np.log1p(spec)
    freqs = np.fft.rfftfreq(len(data), 1 / 16000)
    edges = np.linspace(freqs[0], freqs[-1], _NFP + 1)
    bands = np.zeros(_NFP, dtype=np.float32)
    for b in range(_NFP):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if mask.any():
            bands[b] = log_spec[mask].mean()
    mx = bands.max()
    if mx <= 0:
        return None
    q = np.clip((bands / mx) * 15, 0, 15).astype(np.uint8)
    return bytes(q).hex()
# Allowed group names in the manifest for this check.
_SPLIT_GROUPS = ("select", "test", "train")


def find_cross_split_duplicates(
    manifest_path,
    *,
    data_root=None,
    perceptual=False,
):
    """Find near-duplicate audio crossing train/select/test split boundaries.

    Returns list of dicts:
        kind: "exact" | "perceptual"
        detail: md5 hex or fingerprint hex
        groups: list of (split, set_name, file_id, path)
    """
    data_root = data_root or DATA_ROOT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact_index = defaultdict(list)
    perc_index = defaultdict(list)
    for split_name, split_groups in manifest.items():
        if split_name not in _SPLIT_GROUPS:
            continue
        for set_name, file_ids in split_groups.items():
            for fid in file_ids:
                p = data_root / fid
                if not p.is_file():
                    continue
                m = _md5_of_path(p)
                if m:
                    exact_index[m].append((split_name, set_name, fid, p))
                if perceptual:
                    f = _perceptual_fp(p)
                    if f:
                        perc_index[f].append((split_name, set_name, fid, p))
    out = []
    for md5, recs in exact_index.items():
        ss = {(r[0], r[1]) for r in recs}
        if len(ss) > 1:
            out.append({"kind": "exact", "detail": md5, "groups": recs})
    if perceptual:
        for fp, recs in perc_index.items():
            ss = {(r[0], r[1]) for r in recs}
            if len(ss) > 1:
                out.append({"kind": "perceptual", "detail": fp, "groups": recs})
    out.sort(key=lambda d: (0 if d["kind"] == "exact" else 1, -len(d["groups"])))
    return out


def _format_dup(d):
    kind = d["kind"].upper()
    lines = [f"[{kind}] {d['detail']}"]
    for split_name, set_name, fid, path in d["groups"]:
        try:
            rel = path.relative_to(DATA_ROOT)
        except ValueError:
            rel = path.name
        lines.append(f"  {split_name:6s} {set_name:22s} {fid:38s} {rel}")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Near-duplicate dedup across train/select/test splits (axis D2).")
    ap.add_argument("--splits", type=Path, default=SPLIT_MANIFEST,
                    help="split manifest JSON (default: %(default)s)")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT,
                    help="audio root (default: %(default)s)")
    ap.add_argument("--perceptual", action="store_true",
                    help="also flag perceptual near-dups")
    args = ap.parse_args(argv)
    if not args.splits.is_file():
        raise FileNotFoundError(f"split manifest not found: {args.splits}")
    dups = find_cross_split_duplicates(args.splits, data_root=args.data_root,
                                       perceptual=args.perceptual)
    if not dups:
        print("No cross-split duplicates found.")
        return 0
    print(f"Found {len(dups)} cross-split duplicate group(s):\n")
    for d in dups:
        print(_format_dup(d))
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
