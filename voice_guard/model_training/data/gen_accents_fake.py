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

**2026-09-09 update:** on a congested link, letting `TTS.api.TTS()` download
its own weights is a trap — its downloader is plain `requests`, no resume,
so any connection drop restarts the ~1.9GB download from zero (observed
here: died at 1.18GB, twice). Fixed by pre-downloading via
`huggingface_hub.snapshot_download` (properly resumable) and passing
`--local-model-dir`:
    python -c "from huggingface_hub import snapshot_download; \
        snapshot_download(repo_id='coqui/XTTS-v2', local_dir='data/xtts_v2_local')"
    python gen_accents_fake.py --lang en --local-model-dir data/xtts_v2_local
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append, read_manifest

# Short, phonetically varied stock sentences per language. Content doesn't
# need to match the reference clip's content — XTTS-v2 clones timbre/prosody
# style from speaker_wav independent of what text it's asked to read.


def _patch_torchaudio_load_with_soundfile() -> None:
    """XTTS's own code (TTS/tts/models/xtts.py's load_audio) calls
    torchaudio.load(audiopath) directly to read the speaker reference clip —
    no way to inject a preloaded tensor through the public tts_to_file API.
    torchaudio's default backend on this torch version hard-requires
    torchcodec, which fails to load its native DLL on Windows (FFmpeg
    shared-library mismatch — same root cause worked around everywhere else
    in this pipeline via soundfile). Monkeypatch torchaudio.load itself with
    a soundfile-based implementation returning the same (Tensor, int)
    contract, so XTTS's internal call sites need no changes."""
    import torch
    import torchaudio
    import soundfile as sf

    def _load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # torchaudio.load returns (channels, samples); soundfile gives (samples, channels).
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _load


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
    ap.add_argument(
        "--local-model-dir", type=Path, default=None,
        help="Load weights from a local huggingface_hub snapshot instead of letting "
        "TTS.api.TTS() download them itself. Coqui's own downloader (plain requests, "
        "no resume) dies and restarts from zero on any connection drop — on a "
        "congested link that never completes. `huggingface_hub.snapshot_download` "
        "resumes properly; see this script's module docstring for the exact command.",
    )
    args = ap.parse_args()

    _patch_torchaudio_load_with_soundfile()

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

    if args.local_model_dir is not None:
        tts = TTS(
            model_path=str(args.local_model_dir),
            config_path=str(args.local_model_dir / "config.json"),
        )
    else:
        tts = TTS(args.model_name)  # downloads weights on first run (several GB)
    sentences = STOCK_SENTENCES[args.lang]

    # Existing clone files already on disk from a prior (killed/interrupted) run.
    # Manifest rows are now written incrementally (below), one per clip, not
    # batched to the end — but a run killed *before* this fix (or killed at
    # just the wrong instant) can still leave a .wav with no manifest row yet.
    # Backfill those now (real duration via soundfile) rather than either
    # re-cloning them (wasted GPU time — the exact failure mode this whole
    # change exists to avoid) or leaving them permanently invisible to
    # report_accent_coverage.py / train.py.
    existing_manifest_files = {
        r["file"] for r in read_manifest() if r["source"] == "xtts_v2_clone"
    }
    on_disk = {p.name: p for p in ACCENTS_ROOT.glob(f"fake/*/clone_{args.lang}_*.wav")}
    orphaned = []
    for name, path in on_disk.items():
        rel = str(path.relative_to(ACCENTS_ROOT.parent))
        if rel not in existing_manifest_files:
            orphaned.append((name, path, rel))
    if orphaned:
        # speaker_id for a backfilled row can't be recovered exactly (which
        # source real row it was cloned from isn't stored in the filename),
        # but that's a cosmetic gap, not a correctness one — split_accents.py
        # only needs *some* id to key its train/held split on, and these are
        # a small, already-mixed set of speakers by construction.
        backfill_rows = [{
            "file": rel,
            "label": "fake",
            "cell": path.parent.name,
            "source": "xtts_v2_clone",
            "speaker_id": "unknown_backfilled",
            "duration_s": f"{sf.info(str(path)).duration:.3f}",
        } for name, path, rel in orphaned]
        manifest_writer_append(backfill_rows)
        print(f"Backfilled {len(orphaned)} manifest rows for clone files from a prior "
              f"interrupted run (found on disk, no manifest row yet)", file=sys.stderr)
        existing_manifest_files |= {rel for _, _, rel in orphaned}

    n_written = n_skipped_existing = 0
    for i, r in enumerate(rows):
        # manifest "file" is stored relative to ACCENTS_ROOT's parent (data/) —
        # see prep_accents_real.py's relative_to() call.
        speaker_wav = ACCENTS_ROOT.parent / r["file"]
        cell = r["cell"]
        out_name = f"clone_{args.lang}_{i:06d}.wav"
        out_path = cell_dir("fake", cell) / out_name
        out_rel = str(out_path.relative_to(ACCENTS_ROOT.parent))
        if out_rel in existing_manifest_files:
            n_skipped_existing += 1
            continue
        text = sentences[i % len(sentences)]
        try:
            tts.tts_to_file(text=text, speaker_wav=str(speaker_wav), language=args.lang,
                             file_path=str(out_path))
        except Exception as e:
            print(f"skip {r['file']}: {e}", file=sys.stderr)
            continue
        # Written immediately per-clip, not batched to the end — a kill/crash
        # mid-run now loses at most the one clip in flight, not everything
        # already done.
        manifest_writer_append([{
            "file": out_rel,
            "label": "fake",
            "cell": cell,
            "source": "xtts_v2_clone",
            "speaker_id": r["speaker_id"],
            "duration_s": f"{sf.info(str(out_path)).duration:.3f}",
        }])
        n_written += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} cloned", file=sys.stderr)

    print(f"Done. lang={args.lang} -> {n_written} new clones written "
          f"({n_skipped_existing} already-done clones skipped), manifest rows appended incrementally")


if __name__ == "__main__":
    main()
