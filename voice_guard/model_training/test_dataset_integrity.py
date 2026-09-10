"""
Dataset/pipeline integrity tests — the gap `test_pipeline_smoke.py` leaves
open on purpose (it only proves the code *runs*, on synthetic 4s tones that
never exercise the corners a real, messy, multi-source corpus hits).

Written 2026-09-10 after four retraining attempts in a row (attempt1,
attempt2, ablation, english_only) regressed cross-generator held-out EER
despite ever-larger training sets. Root causes found and fixed in that
session, each with a regression test here so it can't silently recur:

  1. `features.chunk_audio` silently drops any clip <3s — fine on its own
     (matches the real 3s on-device inference window), but nobody was
     measuring how unevenly that drop hits different sources (69% of
     ASVspoof2021-fake, ~0% of some accent cells) until this session.
  2. `dataset.split_by_source` used to key on bare filename, not full path —
     a latent cross-directory collision risk (no observed collisions in the
     current corpus, but the bug was real regardless).
  3. TeleChannel's `'clean'` recipe is NOT a no-degradation pass (it applies
     RIR reverb + noise + mic clipping + packet loss, only skipping the
     ffmpeg/codec step) — using it where "no processing" was intended
     measurably hurt generalization.
  4. Several accent-expansion cells (`en_foreign`, `hi_native`, `hi_foreign`)
     have every fake example natively at one TTS engine's output sample
     rate (22050Hz) and every real example natively at 16000Hz — a perfect,
     trivially learnable shortcut unrelated to genuine spoof detection.
     Fixed 2026-09-10 (same session) by sourcing real alternative-generator
     fake audio (facebook/mms-tts-hin, Coqui YourTTS) natively at 16000Hz.
  5. **Found via code-orange literature review, same day, after fix #4 alone
     didn't recover generalization**: leading/trailing silence duration also
     correlates with real/fake label — a well-documented confound in
     ASVspoof-lineage corpora (Kwak et al. 2021, arXiv:2106.12914) — and,
     worse, **the direction is inconsistent across this project's own
     cells** (ASVspoof-derived data + the ITW held-out benchmark: real has
     somewhat *more* silence than fake; `en_foreign`/`hi_native`'s
     TTS-generated fakes: the *opposite*, ~3x more trailing silence than
     their real counterparts). Not yet fixed — `find_acoustic_shortcuts`
     below is the detector, added so this can't silently recur either.

Tests here that touch the real corpus directories SKIP (not fail) if those
directories aren't present — this suite must still run in a fresh checkout
without the multi-GB dataset downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dataset import Example, build_examples, split_by_source
from dataset_audit import audit_directory_pair, find_acoustic_shortcuts, find_technical_shortcuts
from features import SAMPLE_RATE, chunk_audio
from train import parse_channel_arg

MODEL_TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_TRAINING_DIR / "data"

VAANI_ROOT = MODEL_TRAINING_DIR.parents[1] / "vaani"
if str(VAANI_ROOT) not in sys.path:
    sys.path.append(str(VAANI_ROOT))


def _skip_if_missing(*dirs: Path):
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        pytest.skip(f"corpus dir(s) not present in this checkout: {missing}")


# --- 1. chunk_audio's <3s cutoff: explicit, documented contract ------------

def test_chunk_audio_drops_short_clips():
    short = np.zeros(int(2.9 * SAMPLE_RATE), dtype=np.float32)
    assert chunk_audio(short) == []


def test_chunk_audio_keeps_clips_at_or_above_3s():
    exact = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
    assert len(chunk_audio(exact)) == 1
    long = np.zeros(7 * SAMPLE_RATE, dtype=np.float32)
    assert len(chunk_audio(long)) == 2  # floor(7/3), remainder dropped, not padded


# --- 2. split_by_source: full path, not basename ---------------------------

def test_split_by_source_keys_on_full_path_not_basename(tmp_path: Path):
    """Two different directories each containing a file named 'clip.wav' —
    a bare-basename key would silently merge these into one 'source' for
    train/val splitting, which is exactly the bug fixed 2026-09-10."""
    dir_a = tmp_path / "corpus_a" / "real"
    dir_b = tmp_path / "corpus_b" / "real"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    t = np.linspace(0, 4.0, int(SAMPLE_RATE * 4.0), endpoint=False)
    sf.write(dir_a / "clip.wav", (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), SAMPLE_RATE)
    sf.write(dir_b / "clip.wav", (0.1 * np.sin(2 * np.pi * 400 * t)).astype(np.float32), SAMPLE_RATE)

    fake_dir = tmp_path / "fake_empty"
    fake_dir.mkdir()

    examples = build_examples([dir_a, dir_b], fake_dir, channel_recipes=[None])
    source_files = {e.source_file for e in examples}
    assert len(source_files) == 2, (
        f"expected 2 distinct sources (one per directory), got {source_files} — "
        "split_by_source's basename-keying bug has regressed."
    )


def test_split_by_source_never_shares_a_source_across_train_and_val():
    examples = [
        Example(features=np.zeros(63, dtype=np.float32), label=0,
                source_file=f"/a/{i}.wav", source_dir="/a")
        for i in range(20)
    ]
    train_ex, val_ex = split_by_source(examples, val_fraction=0.3)
    assert {e.source_file for e in train_ex} & {e.source_file for e in val_ex} == set()


# --- 3. TeleChannel recipe semantics ----------------------------------------

def test_none_channel_recipe_is_a_true_noop():
    from telechannel.pipeline import process_clip

    pcm = (0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, SAMPLE_RATE))).astype(np.float32)
    # dataset._maybe_channel short-circuits on recipe is None without ever
    # calling process_clip — this test documents that contract directly.
    from dataset import _maybe_channel
    out = _maybe_channel(pcm, None, np.random.default_rng(0))
    assert np.array_equal(out, pcm)


def test_clean_recipe_is_not_a_noop():
    """Guards against re-assuming 'clean' means undegraded — it's a
    debug/orchestration recipe (RIR+noise+clip+loss, no codec), per
    vaani/telechannel/configs/channels.yaml. If this test ever starts
    failing because 'clean' became a true no-op, that's a real behavior
    change worth knowing about, not a bug in the test."""
    from telechannel.pipeline import process_clip

    pcm = (0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, SAMPLE_RATE))).astype(np.float32)
    out = process_clip(pcm, "clean", SAMPLE_RATE, rng=np.random.default_rng(0))
    assert not np.allclose(out, pcm, atol=1e-3), (
        "'clean' produced (near-)identical output to the input — if TeleChannel's "
        "'clean' recipe has genuinely become a no-op, update train.py's --channel "
        "help text and the WARNING it prints, and this test's assumption."
    )


def test_parse_channel_arg_maps_none_string_to_real_none():
    assert parse_channel_arg(["whatsapp", "None", "NONE", "none"]) == ["whatsapp", None, None, None]
    assert parse_channel_arg(["clean"]) == ["clean"]  # passes through unchanged, deliberately


# --- 4. Real-corpus checks (skip if the dataset isn't downloaded) ---------

def test_no_overlap_between_itw_train_and_held_out():
    real_train, fake_train = DATA_DIR / "real_itw_train", DATA_DIR / "fake_itw_train"
    real_held, fake_held = DATA_DIR / "real_itw_held", DATA_DIR / "fake_itw_held"
    _skip_if_missing(real_train, fake_train, real_held, fake_held)

    for train_dir, held_dir, label in (
        (real_train, real_held, "real"), (fake_train, fake_held, "fake")
    ):
        overlap = {f.name for f in train_dir.glob("*.wav")} & {f.name for f in held_dir.glob("*.wav")}
        assert not overlap, f"{label}: train/held-out overlap (leakage): {list(overlap)[:5]}..."


ACCENT_CELLS = ["en_native", "en_foreign", "hi_native", "hi_foreign"]


@pytest.mark.parametrize("cell", ACCENT_CELLS)
def test_no_shortcut_in_accent_cell(cell: str):
    """THE gate that would have caught the 2026-09-10 XTTS-sample-rate
    confound before four wasted training runs — and, since audit_directory_
    pair now also runs find_acoustic_shortcuts, the silence-duration confound
    found via the code-orange literature review the same day. A failure here
    means: do not train on this cell's real/fake pair until the underlying
    source data is fixed — it is not a bug in this test to leave failing
    while that's still true. As of this writing, `en_foreign`/`hi_native`
    are EXPECTED to fail here on the silence check (sample-rate is fixed;
    silence is not) — see voice_guard/docs/superpowers/specs/
    2026-09-10-model-regression-design.md."""
    real_dir = DATA_DIR / "accents_split" / "train" / "real" / cell
    fake_dir = DATA_DIR / "accents_split" / "train" / "fake" / cell
    if cell == "hi_native" and not real_dir.is_dir():
        real_dir = DATA_DIR / "accents_split" / "train" / "real" / "hi_native_capped"
    _skip_if_missing(real_dir, fake_dir)

    issues = audit_directory_pair(real_dir, fake_dir)
    assert not issues, "\n".join(issues)


def test_find_acoustic_shortcuts_detects_a_synthetic_silence_confound():
    rng = np.random.default_rng(0)
    real_meta = [{"lead_silence_s": v, "trail_silence_s": 0.2, "duration_s": 4.0}
                 for v in rng.normal(0.5, 0.05, 50)]
    fake_meta = [{"lead_silence_s": v, "trail_silence_s": 0.2, "duration_s": 4.0}
                 for v in rng.normal(0.05, 0.02, 50)]
    issues = find_acoustic_shortcuts(real_meta, fake_meta)
    assert issues and "lead_silence_s" in issues[0]


def test_find_acoustic_shortcuts_clean_on_overlapping_synthetic_data():
    rng = np.random.default_rng(0)
    real_meta = [{"lead_silence_s": v, "trail_silence_s": v, "duration_s": 4.0}
                 for v in rng.normal(0.3, 0.15, 50)]
    fake_meta = [{"lead_silence_s": v, "trail_silence_s": v, "duration_s": 4.0}
                 for v in rng.normal(0.3, 0.15, 50)]
    assert find_acoustic_shortcuts(real_meta, fake_meta) == []


BASE_CORPUS_PAIRS = [
    ("real", "fake"),
    ("real2021", "fake2021"),
    ("real_itw_train", "fake_itw_train"),
]


@pytest.mark.parametrize("real_name,fake_name", BASE_CORPUS_PAIRS)
def test_no_technical_shortcut_in_base_corpus(real_name: str, fake_name: str):
    real_dir, fake_dir = DATA_DIR / real_name, DATA_DIR / fake_name
    _skip_if_missing(real_dir, fake_dir)

    issues = audit_directory_pair(real_dir, fake_dir)
    assert not issues, "\n".join(issues)


def test_find_technical_shortcuts_detects_a_synthetic_confound():
    """Unit test for the detector itself, independent of real data: a
    perfectly rate-confounded synthetic pair must be flagged."""
    real_meta = [{"samplerate": 16000, "channels": 1, "subtype": "PCM_16"} for _ in range(50)]
    fake_meta = [{"samplerate": 22050, "channels": 1, "subtype": "PCM_16"} for _ in range(50)]
    issues = find_technical_shortcuts(real_meta, fake_meta)
    assert issues and "samplerate" in issues[0]


def test_find_technical_shortcuts_clean_on_balanced_synthetic_data():
    rng = np.random.default_rng(0)
    real_meta = [{"samplerate": int(r), "channels": 1, "subtype": "PCM_16"}
                 for r in rng.choice([16000, 22050], size=50)]
    fake_meta = [{"samplerate": int(r), "channels": 1, "subtype": "PCM_16"}
                 for r in rng.choice([16000, 22050], size=50)]
    assert find_technical_shortcuts(real_meta, fake_meta) == []
