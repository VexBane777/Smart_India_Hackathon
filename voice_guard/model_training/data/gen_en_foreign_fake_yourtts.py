"""Generate additional en_foreign fake (voice-cloned) speech via Coqui
YourTTS instead of XTTS-v2.

Why (2026-09-10): en_foreign's entire fake side was 100% XTTS-v2 cloning,
which outputs natively at 22050Hz while every en_foreign real clip is
natively 16000Hz — a perfect, trivially learnable sample-rate/generator
shortcut (see dataset_audit.py / test_dataset_integrity.py, and
gen_hindi_mms_fake.py's docstring for the same issue in hi_native/hi_foreign,
fixed there via facebook/mms-tts-hin). Unlike that fix, en_foreign needs a
voice-*cloning* model (not a fixed single voice) to preserve the "foreign-
accented English" property genuinely — a plain TTS voice would swap the
sample-rate confound for an accent confound instead of fixing anything.
YourTTS supports zero-shot cloning via speaker_wav (same interface as
gen_accents_fake.py's XTTS usage) and outputs natively at 16000Hz — matching
real audio's rate, so this genuinely breaks the confound rather than trading
it for a different one.

Same reference-cloning approach as gen_accents_fake.py: for each en_foreign
real clip in the manifest, clone that speaker's voice reading a stock
English sentence, writing into fake/en_foreign.

Usage:
    python gen_en_foreign_fake_yourtts.py --limit 400
"""
from __future__ import annotations

import argparse
import sys

import soundfile as sf

from accent_common import ACCENTS_ROOT, cell_dir, ensure_dirs, manifest_writer_append, read_manifest
from gen_accents_fake import STOCK_SENTENCES, _patch_torchaudio_load_with_soundfile

CELL = "en_foreign"
SOURCE_TAG = "your_tts_clone"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    _patch_torchaudio_load_with_soundfile()

    from TTS.api import TTS

    ensure_dirs()
    rows = [r for r in read_manifest() if r["label"] == "real" and r["cell"] == CELL]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print(f"No real/{CELL} rows in the manifest.", file=sys.stderr)
        return

    tts = TTS("tts_models/multilingual/multi-dataset/your_tts")
    sentences = STOCK_SENTENCES["en"]

    existing_manifest_files = {r["file"] for r in read_manifest() if r["source"] == SOURCE_TAG}

    n_written = n_skipped = 0
    for i, r in enumerate(rows):
        speaker_wav = ACCENTS_ROOT.parent / r["file"]
        out_name = f"your_tts_clone_en_foreign_{i:06d}.wav"
        out_path = cell_dir("fake", CELL) / out_name
        out_rel = str(out_path.relative_to(ACCENTS_ROOT.parent))
        if out_rel in existing_manifest_files:
            n_skipped += 1
            continue
        text = sentences[i % len(sentences)]
        try:
            tts.tts_to_file(text=text, speaker_wav=str(speaker_wav), language="en",
                             file_path=str(out_path))
        except Exception as e:
            print(f"skip {r['file']}: {e}", file=sys.stderr)
            continue
        manifest_writer_append([{
            "file": out_rel,
            "label": "fake",
            "cell": CELL,
            "source": SOURCE_TAG,
            "speaker_id": r["speaker_id"],
            "duration_s": f"{sf.info(str(out_path)).duration:.3f}",
        }])
        n_written += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} cloned", file=sys.stderr)

    print(f"Done. {n_written} new YourTTS clones written ({n_skipped} already-done skipped).")


if __name__ == "__main__":
    main()
