"""
Builds a (features, label) dataset for the voice_guard TFLite model from
two directories of WAV files: real speech and synthetic/cloned speech.

Label convention matches TFLiteService.infer's softmax indexing:
  0 = real (VERIFIED_HUMAN), 1 = synthetic/cloned (AI_DETECTED)

Splits at the SOURCE FILE level (not chunk level) — chunks from the same
clip never span train/val, matching the leakage rule already documented
in vaani/00_MASTER_PLAN.md §5.4 for the main TeleChannel corpus.

Optionally degrades clips through vaani's TeleChannel pipeline before
feature extraction, so the trained model sees phone-channel-shaped audio
matching what a real call sounds like on-device — the same reasoning
that makes TeleChannel the project's differentiator, not just an aside.
"""
from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from features import SAMPLE_RATE, chunk_audio, extract_features

VAANI_ROOT = Path(__file__).resolve().parents[2] / "vaani"
if str(VAANI_ROOT) not in sys.path:
    # append, not insert(0): vaani/ has its own train.py, and inserting at
    # the front shadowed this package's train.py for anything importing
    # both (e.g. eval_held_out.py's `from train import compute_eer`).
    sys.path.append(str(VAANI_ROOT))


@dataclass
class Example:
    features: np.ndarray
    label: int
    source_file: str


def _load_mono_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa  # local import: heavy dep, only needed for resampling

        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data.astype(np.float32)


def _maybe_channel(
    pcm: np.ndarray, recipe: str | None, rng: np.random.Generator
) -> np.ndarray:
    if recipe is None:
        return pcm
    from telechannel.pipeline import process_clip  # vaani package

    return process_clip(pcm, recipe, SAMPLE_RATE, rng=rng)


def _process_file(
    args: tuple[Path, int, list[str | None], np.random.SeedSequence],
) -> list[Example]:
    wav_path, label, channel_recipes, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    pcm = _load_mono_16k(wav_path)
    out: list[Example] = []
    for recipe in channel_recipes:
        degraded = _maybe_channel(pcm, recipe, rng)
        for chunk in chunk_audio(degraded):
            out.append(
                Example(
                    features=extract_features(chunk),
                    label=label,
                    source_file=wav_path.name,
                )
            )
    return out


def _as_dir_list(dirs: Path | list[Path]) -> list[Path]:
    return [dirs] if isinstance(dirs, (str, Path)) else list(dirs)


def build_examples(
    real_dir: Path | list[Path],
    fake_dir: Path | list[Path],
    channel_recipes: list[str | None] = (None,),
    workers: int | None = None,
    seed: int = 0,
) -> list[Example]:
    """Extracts (features, label) examples from one or more real/fake WAV
    dirs, optionally degraded through each channel recipe.

    Parallelized across files with a process pool: each file's channel
    degradation (ffmpeg subprocess roundtrips) and feature extraction are
    CPU-bound and independent, so this is a straightforward multi-core win
    on a large corpus. Each file gets its own child SeedSequence (spawned
    from the single top-level seed) so degradation stays reproducible and
    independent per file even though workers run as separate processes —
    a single shared np.random.Generator can't be meaningfully advanced
    across a process pool the way it can in-process."""
    tasks: list[tuple[Path, int, list[str | None]]] = []
    for label, dirs in ((0, real_dir), (1, fake_dir)):
        for directory in _as_dir_list(dirs):
            for wav_path in sorted(Path(directory).glob("*.wav")):
                tasks.append((wav_path, label, list(channel_recipes)))

    seed_seqs = np.random.SeedSequence(seed).spawn(len(tasks))
    tasks = [(*t, ss) for t, ss in zip(tasks, seed_seqs)]

    workers = workers or os.cpu_count() or 1
    examples: list[Example] = []
    if workers <= 1 or len(tasks) < 2:
        for task in tasks:
            examples.extend(_process_file(task))
        return examples

    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = 0
        for result in pool.map(_process_file, tasks, chunksize=4):
            examples.extend(result)
            done += 1
            if done % 500 == 0:
                print(f"  ...{done}/{len(tasks)} source files processed", file=sys.stderr)
    return examples


def split_by_source(
    examples: list[Example], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    sources = sorted({e.source_file for e in examples})
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_val = max(1, int(len(sources) * val_fraction))
    val_sources = set(sources[:n_val])
    train = [e for e in examples if e.source_file not in val_sources]
    val = [e for e in examples if e.source_file in val_sources]
    return train, val


def to_arrays(examples: list[Example]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([e.features for e in examples]).astype(np.float32)
    y = np.array([e.label for e in examples], dtype=np.int64)
    return X, y
