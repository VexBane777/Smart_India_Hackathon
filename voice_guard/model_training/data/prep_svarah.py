"""Ingest AI4Bharat's Svarah corpus (genuinely Indian-accented English, 117
speakers across 65 districts/19 states) into data/accents/real/en_foreign/ —
the highest-value single source for that cell found this session: unlike
Common Voice's coarse self-reported accent tags, every Svarah speaker is
verifiably Indian (native_place_state/district + primary_language columns).

Source: downloaded via Kaggle (`shivanshpotdar/svarah-indian-english-
complete-dataset`), a single `combined.parquet` — sidesteps the official
`ai4bharat/Svarah` HF repo, which IS gated (confirmed via unauthenticated
test — unlike Common Voice/VCTK/IndicTTS-Hindi/IndicVoices-R, this one
genuinely needs an HF login on the official path). Run manually first:

    kaggle datasets download -d shivanshpotdar/svarah-indian-english-complete-dataset \
        -p data/svarah --unzip

Speaker IDs: the parquet has no stable per-speaker column (the audio
filename's numeric prefix is per-recording-session, not per-speaker — 2938
distinct prefixes across only 117 known speakers). Used (gender, age-group,
native_place_district) as a pseudo speaker_id instead (110 distinct groups,
close to the true 117) — coarser than a real id, documented rather than
silently assumed exact.

License: check https://huggingface.co/datasets/ai4bharat/Svarah — AI4Bharat
datasets have historically been research/non-commercial-leaning; treat like
every other non-ASVspoof source in this pipeline (fine for this prototype,
flag before commercial use).

Usage (after data/svarah/combined.parquet exists):
    python prep_svarah.py --limit 3000
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

PARQUET_PATH = Path(__file__).resolve().parent / "svarah" / "combined.parquet"
SOURCE_SAMPLE_RATE = 16000
CELL = "en_foreign"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not PARQUET_PATH.exists():
        print(
            f"Missing {PARQUET_PATH} — pull it first:\n"
            "  kaggle datasets download -d shivanshpotdar/svarah-indian-english-complete-dataset "
            "-p data/svarah --unzip",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        import pandas as pd
    except ImportError:
        print("Missing dependency: pip install pandas pyarrow", file=sys.stderr)
        raise

    ensure_dirs()
    df = pd.read_parquet(PARQUET_PATH)
    if args.limit is not None:
        df = df.iloc[: args.limit]

    rows = []
    for i, row in df.iterrows():
        audio = row["audio_filepath"]
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not audio_bytes:
            continue
        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != SOURCE_SAMPLE_RATE:
            import librosa  # local import: heavy dep, only needed for resampling

            array = librosa.resample(array, orig_sr=sr, target_sr=SOURCE_SAMPLE_RATE)
            sr = SOURCE_SAMPLE_RATE

        pseudo_speaker = f"{row.get('gender')}_{row.get('age-group')}_{row.get('native_place_district')}"
        out_name = f"svarah_{i:06d}.wav"
        out_path = cell_dir("real", CELL) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "real",
            "cell": CELL,
            "source": "svarah",
            "speaker_id": pseudo_speaker,
            "duration_s": f"{duration_s:.3f}",
        })
        if len(rows) % 500 == 0:
            print(f"{len(rows)} pulled", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} manifest rows appended to real/{CELL}")


if __name__ == "__main__":
    main()
