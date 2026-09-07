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

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from features import SAMPLE_RATE, chunk_audio, extract_features

VAANI_ROOT = Path(__file__).resolve().parents[2] / "vaani"
if str(VAANI_ROOT) not in sys.path:
    sys.path.insert(0, str(VAANI_ROOT))


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


def _maybe_channel(pcm: np.ndarray, recipe: str | None) -> np.ndarray:
    if recipe is None:
        return pcm
    from telechannel.pipeline import process_clip  # vaani package

    return process_clip(pcm, recipe, SAMPLE_RATE)


def build_examples(
    real_dir: Path,
    fake_dir: Path,
    channel_recipes: list[str | None] = (None,),
) -> list[Example]:
    examples: list[Example] = []
    for label, directory in ((0, real_dir), (1, fake_dir)):
        for wav_path in sorted(Path(directory).glob("*.wav")):
            pcm = _load_mono_16k(wav_path)
            for recipe in channel_recipes:
                degraded = _maybe_channel(pcm, recipe)
                for chunk in chunk_audio(degraded):
                    examples.append(
                        Example(
                            features=extract_features(chunk),
                            label=label,
                            source_file=wav_path.name,
                        )
                    )
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
