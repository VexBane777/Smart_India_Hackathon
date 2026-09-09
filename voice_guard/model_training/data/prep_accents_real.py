"""Pull "real" (bona-fide) accent-diverse speech from Mozilla Common Voice
into the data/accents/real/<cell>/ layout (see accent_common.py).

NOT RUN YET as of this writing — written on a non-GPU machine with neither
`datasets` nor a Hugging Face token available, per instructions to preserve
the download method rather than execute it here. Run this on the machine
that will also do training/generation. See data/README.md for setup.

Source: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
Requires a free Hugging Face account + token (`huggingface-cli login`) and
accepting that dataset's terms on its page once, in a browser, first.

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

Usage (on the GPU/training machine, after `pip install datasets` and
`huggingface-cli login`):
    python prep_accents_real.py --lang en --limit 200
    python prep_accents_real.py --lang hi --limit 200
    # omit --limit for a fuller pull; Common Voice releases are large,
    # start small and raise it once the mechanics are confirmed working.
"""
from __future__ import annotations

import argparse
import sys

import soundfile as sf

from accent_common import ensure_dirs, cell_dir, manifest_writer_append

COMMON_VOICE_DATASET = "mozilla-foundation/common_voice_17_0"

# Common Voice "en" accent string -> native-anglophone-region bucket.
# Anything not in this set (e.g. "india and south asia...", "german english",
# "filipino", empty/unspecified) is bucketed "foreign" by default — see
# module docstring on why this is a coarse proxy, not a linguistic claim.
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
    accent = (accent or "").strip().lower()
    return "native" if accent in EN_NATIVE_ACCENTS else "foreign"


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
        print(
            "Missing dependency: pip install datasets\n"
            "Also run `huggingface-cli login` with a free HF token, and accept "
            f"the dataset terms once at https://huggingface.co/datasets/{COMMON_VOICE_DATASET}",
            file=sys.stderr,
        )
        raise

    ensure_dirs()
    bucket_fn = _bucket_en if args.lang == "en" else _bucket_hi

    ds = load_dataset(COMMON_VOICE_DATASET, args.lang, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    rows = []
    counts = {"native": 0, "foreign": 0}
    for i, example in enumerate(ds):
        if args.limit is not None and i >= args.limit:
            break
        cell_nat = bucket_fn(example.get("accent", ""))
        cell = f"{args.lang}_{cell_nat}"
        audio = example["audio"]
        speaker_id = example.get("client_id", "unknown")
        out_name = f"cv_{args.lang}_{i:06d}.wav"
        out_path = cell_dir("real", cell) / out_name
        sf.write(out_path, audio["array"], audio["sampling_rate"])
        duration_s = len(audio["array"]) / audio["sampling_rate"]
        rows.append({
            "file": str(out_path.relative_to(cell_dir("real", cell).parent.parent)),
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
