"""Generate Hindi fake/synthetic speech via facebook/mms-tts-hin (Meta's
Massively Multilingual Speech VITS checkpoint) into accents/fake/hi_native
(and a smaller slice into hi_foreign).

Why this exists (2026-09-10): hi_native's and hi_foreign's entire fake side
was, until now, 100% XTTS-v2 voice cloning — which outputs natively at
22050Hz while every real clip in those cells is natively 16000Hz. That's a
perfect, trivially learnable sample-rate/generator shortcut, caught by
dataset_audit.py / test_dataset_integrity.py. MMS-TTS's Hindi checkpoint
outputs natively at 16000Hz (confirmed: `VitsModel.config.sampling_rate`) —
matching the real audio's rate exactly — so adding these breaks the
confound directly, rather than papering over it with post-hoc resampling.

Real, matching-language Hindi text comes from this project's own already-
downloaded OpenSLR103 transcription file (data/openslr_hindi/train/
transcription.txt, 99,925 lines) — no internet text source needed.

MMS-TTS-hin is a single fixed voice (no multi-speaker conditioning), so this
adds generator/sample-rate diversity, not speaker diversity — that's an
intentional, acknowledged limitation, not something this script tries to
paper over. It's a supplement to XTTS's fake side, not a replacement.

Each output clip concatenates transcript lines one at a time until it hits a
per-clip target duration sampled from the *real* hi_native/hi_foreign clip
duration distribution (measured 2026-09-10: hi_native p10/p50/p90 =
2.12/3.48/6.52s; hi_foreign 3.97/5.69/7.17s) — not a fixed line count. An
earlier version of this script used a fixed 3-line concatenation, which
reliably cleared features.chunk_audio's 3s cutoff but produced clips running
~2-3x longer than real ones (median ~8s fake vs ~3.5s real) — itself a
measurable real/fake shortcut on `duration_s`
(dataset_audit.py::find_acoustic_shortcuts), found via the same 2026-09-10
code-orange review that found the leading/trailing-silence confound this
script's fixed-count design was never checked against. Matching the target
*distribution*, not just clearing the minimum, is the actual fix — a
uniformly-longer "fake" clip population is exactly the kind of shortcut this
whole exercise exists to avoid re-introducing.

Usage:
    python gen_hindi_mms_fake.py --n-native 500 --n-foreign 60
"""
from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

from accent_common import ACCENTS_ROOT, cell_dir, ensure_dirs, manifest_writer_append

MODEL_ID = "facebook/mms-tts-hin"
TRANSCRIPTION_PATH = "openslr_hindi/train/transcription.txt"
MIN_SECONDS = 3.2  # a little over chunk_audio's 3s cutoff, so the whole clip survives
MAX_LINES_PER_CLIP = 6  # safety cap so a target duration can't runaway-concatenate

# Per-cell target duration ranges, matching the real corpus's measured
# distribution (see module docstring) rather than a fixed line count.
TARGET_DURATION_RANGE = {
    "hi_native": (3.1, 5.0),  # real median 3.48s; narrowed from (3.2, 6.5) after
                              # a first pass still measured AUC=0.15 vs real
    "hi_foreign": (3.9, 7.2),
}


def load_transcript_lines(path: str) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                lines.append(parts[1])
    return lines


def synth_clip(model, tokenizer, lines: list[str], rng: random.Random, device: str,
                target_duration_s: float) -> np.ndarray:
    """Concatenates lines until target_duration_s is reached, then hard-trims
    to it. A single synthesized line is often already close to (or past) a
    short target, so early-stopping alone systematically overshoots (a first
    attempt at this script measured median 6.6s vs. real hi_native's 3.48s —
    better than a fixed line count's ~8s, but still a measurable gap) — the
    trim is what actually bounds the output to match the real distribution,
    not just "stop concatenating soon enough"."""
    parts = []
    total_samples = 0
    sr = model.config.sampling_rate
    target_samples = int(target_duration_s * sr)
    for _ in range(MAX_LINES_PER_CLIP):
        text = rng.choice(lines)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            waveform = model(**inputs).waveform
        parts.append(waveform.squeeze(0).cpu().numpy())
        total_samples += parts[-1].shape[0]
        if total_samples >= target_samples:
            break
        parts.append(np.zeros(int(0.2 * sr), dtype=np.float32))  # brief pause between lines
        total_samples += parts[-1].shape[0]
    audio = np.concatenate(parts).astype(np.float32)
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    # A hard cut can land mid-voiced-frame with no natural trailing quiet —
    # dataset.trim_edge_silence then has no room to add its own randomized
    # pad, measuring trail_silence_s=0 (found this the hard way: an earlier
    # version of this trim produced a *new*, self-inflicted silence-based
    # shortcut — real hi_foreign clips have a small natural trailing pad,
    # fake had a hard AUC=0.89 zero-vs-nonzero split). A short low-amplitude
    # tail gives trim_edge_silence real quiet content to detect and pad
    # around, same as any real recording's room-tone trail.
    tail = np.random.default_rng().normal(0, 2e-4, int(0.15 * sr)).astype(np.float32)
    return np.concatenate([audio, tail])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-native", type=int, default=500, help="clips to add to fake/hi_native")
    ap.add_argument("--n-foreign", type=int, default=60, help="clips to add to fake/hi_foreign")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs()
    rng = random.Random(args.seed)

    lines = load_transcript_lines(TRANSCRIPTION_PATH)
    print(f"Loaded {len(lines)} Hindi transcript lines from {TRANSCRIPTION_PATH}", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    sr = model.config.sampling_rate
    print(f"Loaded {MODEL_ID} on {device}, native sampling_rate={sr}", file=sys.stderr)

    rows = []
    for cell, n_clips in (("hi_native", args.n_native), ("hi_foreign", args.n_foreign)):
        out_dir = cell_dir("fake", cell)
        lo, hi = TARGET_DURATION_RANGE[cell]
        n_written = 0
        attempt = 0
        while n_written < n_clips and attempt < n_clips * 3:
            attempt += 1
            target = rng.uniform(lo, hi)
            audio = synth_clip(model, tokenizer, lines, rng, device, target)
            duration_s = len(audio) / sr
            if duration_s < MIN_SECONDS:
                continue  # would be silently dropped by chunk_audio anyway
            out_name = f"mms_tts_hin_{cell}_{n_written:05d}.wav"
            out_path = out_dir / out_name
            sf.write(out_path, audio, sr)
            rows.append({
                "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
                "label": "fake",
                "cell": cell,
                "source": "mms_tts_hin",
                "speaker_id": "mms_tts_hin_fixed_voice",
                "duration_s": f"{duration_s:.3f}",
            })
            n_written += 1
            if n_written % 100 == 0:
                print(f"  {cell}: {n_written}/{n_clips}", file=sys.stderr)
        print(f"{cell}: wrote {n_written} clips ({attempt} attempts)", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} manifest rows appended.")


if __name__ == "__main__":
    main()
