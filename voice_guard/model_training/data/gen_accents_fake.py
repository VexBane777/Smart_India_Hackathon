"""Generate "fake" (voice-cloned/TTS) accent-diverse speech to match the
real clips prep_accents_real.py pulled, using Coqui XTTS-v2 (open-weight,
free, runs locally, supports voice cloning from a short reference clip).

NOT RUN YET — written on a non-GPU machine; XTTS-v2 needs real GPU inference
time and isn't installed here. Run on the GPU machine, after
data/prep_accents_real.py has populated data/accents/real/. See data/README.md.

Model: https://huggingface.co/coqui/XTTS-v2
License note: XTTS-v2 ships under Coqui's CPML (non-commercial research/
personal use). Fine for a hackathon prototype and for training a detector
you don't sell, but flag it before any commercial use of this project.

For each real clip in the manifest, this clones that speaker's voice
(speaker_wav=<the real clip>) reading a short stock sentence in the same
language, and writes the result into fake/<same cell> — i.e. a same-speaker,
same-accent-bucket clone, which is the "does the model catch a clone of a
voice/accent it's also seen genuine speech from" case, matching what a real
attacker cloning a specific target sounds like.

Usage (on the GPU machine, after `pip install TTS`):
    python gen_accents_fake.py --lang en --limit 200
    python gen_accents_fake.py --lang hi --limit 200
"""
from __future__ import annotations

import argparse
import sys

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append, read_manifest

# Short, phonetically varied stock sentences per language. Content doesn't
# need to match the reference clip's content — XTTS-v2 clones timbre/prosody
# style from speaker_wav independent of what text it's asked to read.
STOCK_SENTENCES = {
    "en": [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Can you confirm the delivery address before tomorrow afternoon?",
        "I was trying to reach the support desk about my recent order.",
    ],
    "hi": [
        "कल शाम आपकी मीटिंग किस समय रखी गई थी?",
        "कृपया अपना पता और फोन नंबर पुन: कंफ़र्म करें।",
        "मुझे सहायता चाहिए, क्या आप अभी बात कर सकते हैं?",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    args = ap.parse_args()

    try:
        from TTS.api import TTS
    except ImportError:
        print(
            "Missing dependency: pip install TTS (Coqui TTS)\n"
            "See https://huggingface.co/coqui/XTTS-v2 for model details/license.",
            file=sys.stderr,
        )
        raise

    ensure_dirs()
    rows = [r for r in read_manifest() if r["label"] == "real" and r["cell"].startswith(args.lang)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print(f"No real/{args.lang}_* rows in the manifest yet — run prep_accents_real.py first.",
              file=sys.stderr)
        return

    tts = TTS(args.model_name)  # downloads weights on first run (several GB)
    sentences = STOCK_SENTENCES[args.lang]

    out_rows = []
    for i, r in enumerate(rows):
        # manifest "file" is stored relative to ACCENTS_ROOT's parent (data/) —
        # see prep_accents_real.py's relative_to() call.
        speaker_wav = ACCENTS_ROOT.parent / r["file"]
        cell = r["cell"]
        out_name = f"clone_{args.lang}_{i:06d}.wav"
        out_path = cell_dir("fake", cell) / out_name
        text = sentences[i % len(sentences)]
        try:
            tts.tts_to_file(text=text, speaker_wav=str(speaker_wav), language=args.lang,
                             file_path=str(out_path))
        except Exception as e:
            print(f"skip {r['file']}: {e}", file=sys.stderr)
            continue
        out_rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "fake",
            "cell": cell,
            "source": "xtts_v2_clone",
            "speaker_id": r["speaker_id"],
            "duration_s": "",  # filled in by report_accent_coverage.py via soundfile if needed
        })
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} cloned", file=sys.stderr)

    manifest_writer_append(out_rows)
    print(f"Done. lang={args.lang} -> {len(out_rows)} clones written, manifest rows appended")


if __name__ == "__main__":
    main()
