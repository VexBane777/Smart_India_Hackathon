"""Ingest OpenSLR SLR103 (Microsoft Hindi speech corpus) into
data/accents/real/hi_native/ — 95hrs/59 speakers (train) + 5.5hrs/19 speakers
(test, speaker-disjoint from train), by far the largest single real-Hindi
source in this pipeline. Also genuinely telephony-relevant: recorded at
**8kHz** natively (not downsampled from wideband), unlike every other Hindi
source here — closer to what a real phone call actually sounds like.

Source: https://www.openslr.org/103/ (Microsoft Speech Corpus, permissive
MS Open Data-style license, no login needed). Downloaded via
runs/download_openslr_musan.ps1 (Hindi_train.tar.gz + Hindi_test.tar.gz,
extracted to data/openslr_hindi/{train,test}/).

Structure (undocumented on the OpenSLR page itself, determined by
inspection): `{train,test}/audio/<speaker_id>_<utt_id>.wav`, 8kHz mono PCM16,
plus a `transcription.txt` (space-separated `<speaker_id>_<utt_id> <Hindi
text>` per line) — not used here since this pipeline only needs audio +
speaker id, not transcripts. speaker_id is the 4-digit prefix before the
underscore.

Usage (after data/openslr_hindi/{train,test}/audio/ exist):
    python prep_openslr_hindi.py --limit 5000
    python prep_openslr_hindi.py --split test --limit 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

HERE = Path(__file__).resolve().parent
OPENSLR_ROOT = HERE / "openslr_hindi"
SOURCE_SAMPLE_RATE = 16000
CELL = "hi_native"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    audio_dir = OPENSLR_ROOT / args.split / "audio"
    if not audio_dir.exists():
        print(
            f"Missing {audio_dir} — run runs/download_openslr_musan.ps1 first "
            "(or extract Hindi_{train,test}.tar.gz from https://www.openslr.org/103/ manually).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ensure_dirs()
    wav_files = sorted(audio_dir.glob("*.wav"))
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    rows = []
    for i, wav_path in enumerate(wav_files):
        speaker_id = wav_path.stem.split("_")[0]
        array, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != SOURCE_SAMPLE_RATE:
            import librosa  # local import: heavy dep, only needed for resampling

            array = librosa.resample(array, orig_sr=sr, target_sr=SOURCE_SAMPLE_RATE)
            sr = SOURCE_SAMPLE_RATE

        out_name = f"openslr103_{args.split}_{wav_path.stem}.wav"
        out_path = cell_dir("real", CELL) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "real",
            "cell": CELL,
            "source": f"openslr103_{args.split}",
            "speaker_id": f"openslr103_{speaker_id}",
            "duration_s": f"{duration_s:.3f}",
        })
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(wav_files)} converted", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} manifest rows appended to real/{CELL} (split={args.split})")


if __name__ == "__main__":
    main()
