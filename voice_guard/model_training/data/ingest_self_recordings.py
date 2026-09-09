"""Fold your own recordings into data/accents/real/{en,hi}_native/.

Unlike prep_accents_real.py / gen_accents_fake.py, this one needs no
network access or GPU — safe to run on any machine with ffmpeg + soundfile.

Usage:
    1. Record short (~10-30s) clips any way that's convenient (voice memo,
       WhatsApp note, etc.) — deliberately include some quiet takes and some
       with real background noise/chatter, per voice_guard/state.md's finding
       that noisy live-mic speech was never actually eval'd against this
       model before.
    2. Drop the raw files into data/incoming_self/ (created on first run).
    3. python ingest_self_recordings.py --lang en
       python ingest_self_recordings.py --lang hi

Files are converted to 16kHz mono WAV (matching the rest of the training
corpus) via ffmpeg, moved into real/<lang>_native/, and a manifest row is
appended per file. Source files in incoming_self/ are deleted after a
successful convert+move (so re-running doesn't reprocess them) — originals
aren't otherwise needed once the WAV copy exists in accents/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

HERE = Path(__file__).resolve().parent
INCOMING_DIR = HERE / "incoming_self"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    args = ap.parse_args()

    ensure_dirs()
    INCOMING_DIR.mkdir(exist_ok=True)

    files = sorted(p for p in INCOMING_DIR.iterdir() if p.is_file())
    if not files:
        print(f"No files in {INCOMING_DIR} — drop recordings there first.", file=sys.stderr)
        return

    cell = f"{args.lang}_native"
    out_dir = cell_dir("real", cell)
    rows = []
    for src in files:
        out_name = f"self_{args.lang}_{src.stem}.wav"
        out_path = out_dir / out_name
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(out_path)],
            capture_output=True,
        )
        if result.returncode != 0 or not out_path.exists():
            print(f"skip {src.name}: {result.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
            continue
        info = sf.info(out_path)
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "real",
            "cell": cell,
            "source": "self_recorded",
            "speaker_id": "self",
            "duration_s": f"{info.frames / info.samplerate:.3f}",
        })
        src.unlink()

    manifest_writer_append(rows)
    print(f"Ingested {len(rows)} recording(s) into real/{cell}/, manifest rows appended")


if __name__ == "__main__":
    main()
