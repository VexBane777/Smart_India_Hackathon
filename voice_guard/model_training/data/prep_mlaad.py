"""Ingest MLAAD (Multi-Language Audio Anti-Spoofing Dataset) into
data/accents/fake/<cell> — by far the broadest fake-generator diversity in
this pipeline: the English sample alone spans **117 different TTS
systems/architectures** (ElevenLabs v2/v3/Turbo, ChatTTS, F5-TTS,
FireRedTTS-2.0, Cartesia Sonic-3, Edge-TTS, Chatterbox, DeepGram, Meta's
MMS-TTS, and many more) — modern, current-generation generators, not just
the classic ASVspoof/XTTS families already in this corpus.

Source: official dataset (`mueller91/MLAAD` on HF) is gated; used instead an
unofficial Kaggle re-upload by `smraj0198` — `mlaad-english-500` (5 samples
per TTS model, ~91MB, fast) for English. **No confirmed Hindi slice found**
this session (MLAAD's README only documents 8 original + ~54 "computer-
generated" languages without listing them, and paginating the 45GB full
Kaggle mirror's file listing didn't reach past "ar" alphabetically in
reasonable time) — if a Hindi slice turns up later, extend LANG_CELL below
and re-run with --lang hi.

License: CC BY-NC 4.0 (non-commercial) — treat like every other non-
ASVspoof source in this pipeline (fine for this prototype, flag before
commercial use).

Structure: `fake/<lang>/<tts_model_name>/*.flac` + a `meta.csv` per model
dir (not used here — this pipeline only needs audio + a coarse speaker
proxy, not MLAAD's own accent/architecture metadata columns).

Usage (after pulling a Kaggle MLAAD slice into data/mlaad_en500/, or
whatever --root you point at):
    python prep_mlaad.py --lang en --root mlaad_en500 --limit 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

SOURCE_SAMPLE_RATE = 16000
LANG_CELL = {"en": "en_native", "hi": "hi_native"}  # extend if a hi slice is found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=list(LANG_CELL), required=True)
    ap.add_argument("--root", type=Path, required=True,
                     help="dir containing fake/<lang>/<model>/*.flac (e.g. data/mlaad_en500)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    lang_dir = args.root / "fake" / args.lang
    if not lang_dir.exists():
        print(f"Missing {lang_dir} — pull an MLAAD slice first (see module docstring).",
              file=sys.stderr)
        raise SystemExit(1)

    ensure_dirs()
    cell = LANG_CELL[args.lang]
    flac_files = sorted(lang_dir.glob("*/*.flac")) + sorted(lang_dir.glob("*/*.wav"))
    if args.limit is not None:
        flac_files = flac_files[: args.limit]

    rows = []
    for i, src in enumerate(flac_files):
        model_name = src.parent.name
        array, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != SOURCE_SAMPLE_RATE:
            import librosa  # local import: heavy dep, only needed for resampling

            array = librosa.resample(array, orig_sr=sr, target_sr=SOURCE_SAMPLE_RATE)
            sr = SOURCE_SAMPLE_RATE

        safe_model = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_name = f"mlaad_{args.lang}_{safe_model}_{i:06d}.wav"
        out_path = cell_dir("fake", cell) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "fake",
            "cell": cell,
            "source": f"mlaad_{model_name}",
            "speaker_id": "unknown",
            "duration_s": f"{duration_s:.3f}",
        })
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(flac_files)} converted", file=sys.stderr)

    manifest_writer_append(rows)
    n_models = len({r["source"] for r in rows})
    print(f"Done. {len(rows)} manifest rows appended to fake/{cell} "
          f"({n_models} distinct TTS models/architectures)")


if __name__ == "__main__":
    main()
