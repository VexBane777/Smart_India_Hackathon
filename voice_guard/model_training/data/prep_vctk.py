"""Pull real English speech from the VCTK corpus (110 speakers, explicitly
multi-accent by design — unlike Common Voice, every VCTK speaker's accent is
professionally annotated, not self-reported/sparse) into data/accents/real/
en_<native|foreign>/, adding volume and accent diversity beyond Common Voice.

Audio: `jspaulsen/vctk` on Hugging Face (unofficial mirror, no auth needed —
the official `CSTR-Edinburgh/vctk` repo uses a deprecated loading-script
format the current `datasets` library no longer supports). Decoded via
soundfile from raw bytes, same torchcodec-avoidance as prep_accents_real.py.
Only `mic1` recordings are used (mic2 is a near-duplicate second recording
for most speakers, mic1 matches every prior VCTK release — see
coqui-tts's own formatters.py convention). Resampled 48kHz -> 16kHz.
License: CC-BY-4.0 (VCTK Corpus v0.92).

Accent labels: `speaker-info.txt` (ID/AGE/GENDER/ACCENTS/REGION per speaker)
is NOT bundled in any working HF mirror — only inside the 10.94GB official
zip. Pulled instead via Kaggle's `kynthesis/vctk-corpus` mirror, which lists
it as a standalone 3.6KB file (`kaggle datasets download -d
kynthesis/vctk-corpus -f VCTK-Corpus/speaker-info.txt`), avoiding the full
corpus download just for metadata. Run that once first; this script reads
the result from data/speaker-info.txt (same dir as accent_manifest.csv).

ACCENTS column values found (108 speakers): English, American, Scottish,
Irish, NorthernIrish, Canadian, SouthAfrican, Indian, Australian, Welsh,
NewZealand. Bucketed the same way prep_accents_real.py buckets Common Voice:
British-Isles/anglophone-settler accents -> native, SouthAfrican/Indian ->
foreign. This is the same coarse proxy accent_common.py's docstring already
flags, not a stronger claim just because the source label is closer to
ground truth than Common Voice's self-reported tags.

Usage (after data/speaker-info.txt exists):
    python prep_vctk.py --limit 3000
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

VCTK_DATASET = "jspaulsen/vctk"
SOURCE_SAMPLE_RATE = 16000
SPEAKER_INFO_PATH = Path(__file__).resolve().parent / "speaker-info.txt"

# VCTK ACCENTS column -> native-anglophone-region bucket, same reasoning as
# prep_accents_real.py's EN_NATIVE_ACCENTS. Anything not listed here
# (SouthAfrican, Indian) buckets "foreign".
VCTK_NATIVE_ACCENTS = {
    "english", "american", "scottish", "irish", "northernirish",
    "canadian", "australian", "welsh", "newzealand",
}


def _load_speaker_accents() -> dict[str, str]:
    if not SPEAKER_INFO_PATH.exists():
        print(
            f"Missing {SPEAKER_INFO_PATH} — pull it first:\n"
            "  kaggle datasets download -d kynthesis/vctk-corpus "
            "-f VCTK-Corpus/speaker-info.txt -p data",
            file=sys.stderr,
        )
        raise SystemExit(1)
    mapping = {}
    with open(SPEAKER_INFO_PATH, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            speaker_id, accent = parts[0], parts[3]
            mapping[f"p{speaker_id}"] = accent.strip().lower()
    return mapping


def _bucket(accent: str) -> str:
    return "native" if accent in VCTK_NATIVE_ACCENTS else "foreign"


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
    speaker_accents = _load_speaker_accents()

    ds = load_dataset(VCTK_DATASET, split="train", streaming=True, token=False)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    counts = {"native": 0, "foreign": 0, "skipped_mic2": 0, "unknown_speaker": 0}
    n_pulled = 0
    for example in ds:
        if args.limit is not None and n_pulled >= args.limit:
            break
        if example.get("mic_id") != "mic1":
            counts["skipped_mic2"] += 1
            continue
        speaker_id = example["speaker_id"]
        accent = speaker_accents.get(speaker_id)
        if accent is None:
            counts["unknown_speaker"] += 1
            continue
        cell_nat = _bucket(accent)
        cell = f"en_{cell_nat}"

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

        out_name = f"vctk_{example['id']}.wav"
        out_path = cell_dir("real", cell) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "real",
            "cell": cell,
            "source": "vctk_0.92",
            "speaker_id": speaker_id,
            "duration_s": f"{duration_s:.3f}",
        })
        counts[cell_nat] += 1
        n_pulled += 1
        if n_pulled % 200 == 0:
            print(f"{n_pulled} pulled (native={counts['native']} foreign={counts['foreign']})",
                  file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. native={counts['native']} foreign={counts['foreign']} "
          f"skipped_mic2={counts['skipped_mic2']} unknown_speaker={counts['unknown_speaker']} "
          f"-> {len(rows)} manifest rows appended")


if __name__ == "__main__":
    main()
