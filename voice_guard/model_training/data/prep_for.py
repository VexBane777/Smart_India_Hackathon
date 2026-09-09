"""Ingest the Fake-or-Real (FoR) dataset into data/accents/fake/en_native —
English, 33 synthetic voices across major cloud/commercial TTS providers
(Google Cloud TTS, Amazon AWS Polly, Microsoft Azure TTS, Baidu Cloud TTS,
WaveNet, DeepVoice 3) — another distinct generator family alongside
ASVspoof/XTTS/CodecFake/MLAAD already in this pipeline.

Source: Kaggle `mohammedabdeldayem/the-fake-or-real-dataset` (17.2GB, all 4
variants: for-original, for-norm, for-2sec, for-rerec). Uses **for-2sec**
only (already normalized, mono, 16kHz, silence-trimmed, 2s clips) — cleanest
variant, no extra preprocessing needed, and using just one variant avoids
quadruple-counting near-duplicate content across variants.

No accent/speaker metadata ships with FoR — routed entirely to en_native,
same reasoning as prep_codecfake.py/prep_mlaad.py (no per-clip nativity tag
available). License: check the Kaggle page before commercial use.

Usage (after the Kaggle download lands in data/for_dataset/):
    python prep_for.py --limit 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

HERE = Path(__file__).resolve().parent
SOURCE_SAMPLE_RATE = 16000
CELL = "en_native"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE / "for_dataset")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # for-2sec's exact nesting varies by mirror snapshot (seen: for-2sec/for-2seconds/<split>/<label>/*.wav) —
    # search broadly rather than hardcoding one path depth.
    fake_files = sorted(args.root.glob("**/for-2sec*/**/fake/*.wav"))
    if not fake_files:
        print(f"No for-2sec fake/*.wav files found under {args.root} — "
              "check the Kaggle download landed (see module docstring).", file=sys.stderr)
        raise SystemExit(1)
    if args.limit is not None:
        fake_files = fake_files[: args.limit]

    ensure_dirs()
    rows = []
    for i, src in enumerate(fake_files):
        try:
            array, sr = sf.read(str(src), dtype="float32", always_2d=False)
        except Exception as e:
            print(f"skip {src.name}: {e}", file=sys.stderr)
            continue
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != SOURCE_SAMPLE_RATE:
            import librosa  # local import: heavy dep, only needed for resampling

            array = librosa.resample(array, orig_sr=sr, target_sr=SOURCE_SAMPLE_RATE)
            sr = SOURCE_SAMPLE_RATE

        out_name = f"for_{i:06d}.wav"
        out_path = cell_dir("fake", CELL) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "fake",
            "cell": CELL,
            "source": "fake_or_real",
            "speaker_id": "unknown",
            "duration_s": f"{duration_s:.3f}",
        })
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(fake_files)} converted", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} manifest rows appended to fake/{CELL}")


if __name__ == "__main__":
    main()
