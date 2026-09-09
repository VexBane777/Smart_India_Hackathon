"""Pull English fake/spoofed speech from CodecFake (Interspeech 2024) into
data/accents/fake/en_native/ — a fake-generation family genuinely different
from ASVspoof (2019/2021, mostly classic TTS/VC) and XTTS-v2 cloning used
elsewhere in this pipeline: **neural-codec resynthesis** artifacts (speech
round-tripped through 31 different open-source neural audio codecs, e.g.
EnCodec/FunCodec-style models). Broadens what "fake" looks like to the
detector rather than just adding volume.

Source: `rogertseng/CodecFake` on Hugging Face — **not gated**, no HF login
needed (confirmed via unauthenticated streaming pull). Audio decoded via
soundfile from raw bytes (torchcodec's DLL is broken on Windows, same
workaround as every other prep_*.py script here).

Speaker IDs here follow VCTK's p### convention (this corpus's "real" side is
built from VCTK/LibriSpeech-style English speech) — but no per-example
accent/nativity tag ships with it, so unlike prep_vctk.py this can't be
split native/foreign. Everything lands in en_native/fake as a pragmatic
default (VCTK, its likely source, is majority native-English-accent per
prep_vctk.py's own speaker-info.txt bucketing) — a coarser assumption than
the rest of this pipeline makes elsewhere, flagged here rather than hidden.

Only rows labeled "spoofing" are pulled (the genuine/bonafide side would
duplicate real speech we already have from Common Voice/VCTK without any
codec-specific fake signal to learn from).

License: check https://huggingface.co/datasets/rogertseng/CodecFake before
commercial use — research dataset, treat like XTTS-v2/IndicTTS's non-
commercial-leaning flags elsewhere in this pipeline.

Usage:
    python prep_codecfake.py --limit 3000
"""
from __future__ import annotations

import argparse
import io
import sys

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

DATASET = "rogertseng/CodecFake"
SOURCE_SAMPLE_RATE = 16000
CELL = "en_native"  # see module docstring: no per-example accent tag available


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="cap on spoof examples pulled; omit for the full split")
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
    n_skipped_nonspoof = 0
    for example in ds:
        if args.limit is not None and n_pulled >= args.limit:
            break
        if example.get("label") != "spoofing":
            n_skipped_nonspoof += 1
            continue
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
        codec_name = example.get("codec_name", "unknown")
        out_name = f"codecfake_{speaker_id}_{n_pulled:06d}.wav"
        out_path = cell_dir("fake", CELL) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "fake",
            "cell": CELL,
            "source": f"codecfake_{codec_name}",
            "speaker_id": speaker_id,
            "duration_s": f"{duration_s:.3f}",
        })
        n_pulled += 1
        if n_pulled % 200 == 0:
            print(f"{n_pulled} pulled", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} spoof manifest rows appended to fake/{CELL} "
          f"(skipped {n_skipped_nonspoof} non-spoof rows)")


if __name__ == "__main__":
    main()
