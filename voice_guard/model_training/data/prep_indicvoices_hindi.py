"""Pull real, spontaneous/conversational Hindi speech from IndicVoices-R
into data/accents/real/hi_native/ — a different register than Common Voice
(read sentences) and IndicTTS-Hindi (studio TTS-training reads): natural
conversational speech across many districts/states, closer to what a real
phone call sounds like.

Source: `SPRINGLab/IndicVoices-R_Hindi` on Hugging Face — **not gated**
(the original `ai4bharat/IndicVoices` IS gated; this SPRINGLab re-hosting
isn't — confirmed via unauthenticated streaming pull). Has a real per-clip
`speaker_id` field (unlike IndicTTS-Hindi/Svarah, no pseudo-id needed here).
Also carries a real `snr` field per clip — not used for cell bucketing here,
but worth knowing about if noise-level-aware training becomes a separate
axis later (see augment_with_noise.py for the noise-robustness angle this
session took instead).

Audio decoded via soundfile from raw bytes (torchcodec's DLL is broken on
Windows, same workaround as every other prep_*.py script here).

License: check https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Hindi
/ ai4bharat/IndicVoices before commercial use.

Usage:
    python prep_indicvoices_hindi.py --limit 3000
"""
from __future__ import annotations

import argparse
import io
import sys

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

DATASET = "SPRINGLab/IndicVoices-R_Hindi"
SOURCE_SAMPLE_RATE = 16000
CELL = "hi_native"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Missing dependency: pip install datasets", file=sys.stderr)
        raise

    ensure_dirs()
    ds = load_dataset(DATASET, split="train", streaming=True, token=False)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    n_pulled = 0
    for example in ds:
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

        speaker_id = example.get("speaker_id", "unknown")
        out_name = f"indicvoices_hi_{n_pulled:06d}.wav"
        out_path = cell_dir("real", CELL) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "real",
            "cell": CELL,
            "source": "indicvoices_r",
            "speaker_id": speaker_id,
            "duration_s": f"{duration_s:.3f}",
        })
        n_pulled += 1
        if n_pulled % 200 == 0:
            print(f"{n_pulled} pulled", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} manifest rows appended to real/{CELL}")


if __name__ == "__main__":
    main()
