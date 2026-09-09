"""Pull real Hindi speech from IIT Madras/AI4Bharat's IndicTTS Hindi corpus
into data/accents/real/hi_native/ — studio-quality, high-volume Hindi
speech to close the hi_native gap (Common Voice's Hindi coverage is small;
see prep_accents_real.py's docstring).

Source: `SPRINGLab/IndicTTS-Hindi` on Hugging Face — **not gated**, no HF
login needed (confirmed via unauthenticated streaming pull). Audio decoded
via soundfile from raw bytes, same torchcodec-avoidance as
prep_accents_real.py/prep_vctk.py.

All rows go to hi_native (not hi_foreign) — this is a monolingual Hindi
studio TTS-training corpus by a Chennai-based lab reading standard Hindi,
not accent-tagged, so there's no foreign/native split to make here; use
Common Voice / manual supplement for hi_foreign.

Caveat: the HF dataset exposes no per-speaker id (only `gender`), so every
row here gets speaker_id="unknown". split_accents.py's speaker-disjoint
logic sends a (cell, label) with only one distinct speaker_id entirely to
`train` (never `held`) — meaning this corpus contributes to training volume
but not to the held-out eval for hi_native. Acceptable tradeoff for a real
volume boost; Common Voice's per-clip client_id still anchors hi_native's
held-out eval.

License: check https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi
before any commercial use — IIT Madras's IndicTTS has historically been
research/non-commercial-oriented; treat like XTTS-v2's CPML flag (fine for
this prototype, revisit before release).

Usage:
    python prep_indictts_hindi.py --limit 3000
"""
from __future__ import annotations

import argparse
import io
import sys

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

DATASET = "SPRINGLab/IndicTTS-Hindi"
SOURCE_SAMPLE_RATE = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="cap on examples pulled; omit for the full split")
    args = ap.parse_args()

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Missing dependency: pip install datasets", file=sys.stderr)
        raise

    ensure_dirs()
    cell = "hi_native"
    ds = load_dataset(DATASET, split="train", streaming=True, token=False)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    n_pulled = 0
    for i, example in enumerate(ds):
        if args.limit is not None and n_pulled >= args.limit:
            break
        audio_bytes = example["audio"]["bytes"]
        if not audio_bytes:
            continue
        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != SOURCE_SAMPLE_RATE:
            import librosa  # local import: heavy dep, only needed for resampling

            array = librosa.resample(array, orig_sr=sr, target_sr=SOURCE_SAMPLE_RATE)
            sr = SOURCE_SAMPLE_RATE

        out_name = f"indictts_hi_{i:06d}.wav"
        out_path = cell_dir("real", cell) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "real",
            "cell": cell,
            "source": "indictts_hindi",
            "speaker_id": "unknown",
            "duration_s": f"{duration_s:.3f}",
        })
        n_pulled += 1
        if n_pulled % 200 == 0:
            print(f"{n_pulled} pulled", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} manifest rows appended to hi_native")


if __name__ == "__main__":
    main()
