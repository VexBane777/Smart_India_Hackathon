"""Pull "real" (bona-fide) accent-diverse speech from Common Voice 17.0
into the data/accents/real/<cell>/ layout (see accent_common.py).

**2026-09-09 update:** the official `mozilla-foundation/common_voice_17_0`
HF repo is deprecated (Mozilla moved distribution to Mozilla Data Collective,
https://datacollective.mozillafoundation.org, effective October 2025) and the
old repo now serves a notice page instead of data. Switched to
`fixie-ai/common_voice_17_0`, an unofficial but verified-working re-upload of
the same 17.0 corpus — **no gated terms/HF login needed** (confirmed via
unauthenticated streaming pull, both `en` and `hi` configs). Its `audio`
column decodes via `torchcodec` by default, which fails to load its native
DLL on Windows (`libtorchcodec_core5.dll` / FFmpeg version mismatch) — worked
around by pulling raw bytes (`Audio(decode=False)`) and decoding with
`soundfile` instead, same as every other script in this pipeline. Audio here
is 32kHz, not 16kHz — resampled via librosa to match the rest of the corpus.

Source: https://huggingface.co/datasets/fixie-ai/common_voice_17_0

English accent tags (config "en") are Common Voice's own free-text-ish
self-reported categories, e.g. "united states english", "india and south
asia (india, pakistan, sri lanka)". Bucketed into native-anglophone-region
vs. everything else. This is a coarse proxy, not a verified L1 claim — see
accent_common.py's docstring.

Hindi (config "hi") accent tagging in Common Voice is sparse to nonexistent
in practice — expect most/all pulled Hindi clips to land in hi_native by
default here, with hi_foreign left thin. That's a known, expected gap for
the manual-supplement pass (data/report_accent_coverage.py flags it) —
don't treat an empty hi_foreign after running this script as a bug.

Usage (just `pip install datasets soundfile librosa` — no HF login needed):
    python prep_accents_real.py --lang en --limit 200
    python prep_accents_real.py --lang hi --limit 200
    # omit --limit for a fuller pull; Common Voice releases are large,
    # start small and raise it once the mechanics are confirmed working.
"""
from __future__ import annotations

import argparse
import io
import sys

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

COMMON_VOICE_DATASET = "fixie-ai/common_voice_17_0"
SOURCE_SAMPLE_RATE = 16000  # resample target, matches the rest of the corpus

# Common Voice "en" accent string -> native-anglophone-region bucket.
# Anything not in this set (e.g. "india and south asia...", "german english",
# "filipino", empty/unspecified) is bucketed "foreign" by default — see
# module docstring on why this is a coarse proxy, not a linguistic claim.
# fixie-ai's accent field can carry multiple comma-separated tags (e.g.
# "United States English,Midwestern,Low,Demure") — only the first tag is the
# region, the rest are voice-quality descriptors, so bucket on that alone.
EN_NATIVE_ACCENTS = {
    "united states english",
    "england english",
    "canadian english",
    "australian english",
    "scottish english",
    "irish english",
    "new zealand english",
}


def _bucket_en(accent: str) -> str:
    first_tag = (accent or "").split(",")[0].strip().lower()
    return "native" if first_tag in EN_NATIVE_ACCENTS else "foreign"


def _bucket_hi(accent: str) -> str:
    # See module docstring: Hindi accent tags are sparse. Anything explicitly
    # tagged non-Hindi-belt goes to "foreign"; everything else (including
    # untagged, which is most of the corpus) defaults to "native" — meaning
    # hi_foreign relies almost entirely on the manual-supplement pass.
    accent = (accent or "").strip().lower()
    if accent and "hindi" not in accent and "india" not in accent:
        return "foreign"
    return "native"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap on examples pulled; omit for the full split")
    ap.add_argument("--split", default="validated",
                     help="Common Voice split to pull from (default: validated)")
    args = ap.parse_args()

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Missing dependency: pip install datasets", file=sys.stderr)
        raise

    ensure_dirs()
    bucket_fn = _bucket_en if args.lang == "en" else _bucket_hi

    ds = load_dataset(COMMON_VOICE_DATASET, args.lang, split=args.split, streaming=True, token=False)
    # decode=False: pull raw bytes and decode with soundfile ourselves — this
    # mirror's default decode path goes through torchcodec, which fails to
    # load its native DLL on Windows (see module docstring).
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    counts = {"native": 0, "foreign": 0}
    for i, example in enumerate(ds):
        if args.limit is not None and i >= args.limit:
            break
        cell_nat = bucket_fn(example.get("accent", ""))
        cell = f"{args.lang}_{cell_nat}"
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
        speaker_id = example.get("client_id", "unknown")
        out_name = f"cv_{args.lang}_{i:06d}.wav"
        out_path = cell_dir("real", cell) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            # relative to data/ (ACCENTS_ROOT.parent), matching every other
            # script's convention (ingest_self_recordings.py,
            # gen_accents_fake.py) — NOT relative to ACCENTS_ROOT itself,
            # which would drop the "accents/" prefix and break every
            # downstream lookup of this file.
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "real",
            "cell": cell,
            "source": "common_voice_17_0",
            "speaker_id": speaker_id,
            "duration_s": f"{duration_s:.3f}",
        })
        counts[cell_nat] += 1
        if (i + 1) % 100 == 0:
            print(f"{i + 1} pulled (native={counts['native']} foreign={counts['foreign']})",
                  file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. lang={args.lang} native={counts['native']} foreign={counts['foreign']} "
          f"-> {len(rows)} manifest rows appended")


if __name__ == "__main__":
    main()
