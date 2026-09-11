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

from attack_labels import attack_type_for_file
from features import SAMPLE_RATE, chunk_audio, extract_lfcc_sequence, extract_scalars

VAANI_ROOT = Path(__file__).resolve().parents[2] / "vaani"
if str(VAANI_ROOT) not in sys.path:
    # append, not insert(0): vaani/ has its own train.py, and inserting at
    # the front shadowed this package's train.py for anything importing
    # both (e.g. eval_held_out.py's `from train import compute_eer`).
    sys.path.append(str(VAANI_ROOT))


ATTACK_TYPE_TO_INT = {"tts": 0, "vc": 1}  # "unknown" and real both map to IGNORE_ATTACK_TYPE
IGNORE_ATTACK_TYPE = -100  # PyTorch CrossEntropyLoss's default ignore_index


@dataclass
class Example:
    lfcc_seq: np.ndarray  # (n_frames, 60), unpooled
    scalars: np.ndarray  # (6,) = 3 prosody + 3 physio
    label: int
    attack_type: int  # ATTACK_TYPE_TO_INT value, or IGNORE_ATTACK_TYPE
    source_file: str
    source_dir: str  # which --real/--fake/--real-clean/--fake-clean dir this came from


SILENCE_TRIM_THRESHOLD = 0.01  # matches dataset_audit.py's leading/trailing-silence measure
SILENCE_TRIM_PAD_RANGE = (0.05, 0.15)  # seconds; randomized per file, not a fixed constant


def trim_edge_silence(
    pcm: np.ndarray, sr: int = SAMPLE_RATE, thresh: float = SILENCE_TRIM_THRESHOLD,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Trims leading/trailing near-silence, leaving a small randomized pad
    (not zero, and not one fixed constant — either would just be a new,
    cleaner boundary artifact instead of the one being removed).

    Why this exists (2026-09-10, code-orange investigation,
    voice_guard/docs/superpowers/specs/2026-09-10-model-regression-design.md):
    leading/trailing silence duration is a documented confound in
    ASVspoof-lineage corpora (Kwak et al. 2021, arXiv:2106.12914 — bonafide
    clips run systematically longer silence than spoofed ones there), and
    this project's own TTS-generated accent cells (en_foreign, hi_native)
    were measured to have the *opposite* bias (fake clips ~3x more trailing
    silence than real). A corpus mixing both isn't "no shortcut" — it's an
    inconsistent one a model can still exploit locally per sub-corpus. This
    trims the clip-boundary artifact (specific to how each source was
    recorded/synthesized) while leaving internal content — including natural
    mid-speech pauses `extract_prosody`'s pauseRatio genuinely measures —
    untouched. Applied unconditionally to every clip, real and fake, every
    source, at load time: the fix is "make the corpus consistent," not
    "correct just the cells found contaminated so far."

    Only affects the first/last chunk_audio window of a clip in practice —
    a clip trimmed below 3s is then correctly dropped by chunk_audio's
    existing >=3s cutoff, same as any other short clip (see its docstring)."""
    rng = rng or np.random.default_rng()
    above = np.abs(pcm) > thresh
    if not above.any():
        return pcm  # fully silent/near-silent input; chunk_audio will drop it
    first = int(np.argmax(above))
    last = len(pcm) - 1 - int(np.argmax(above[::-1]))
    pad = int(rng.uniform(*SILENCE_TRIM_PAD_RANGE) * sr)
    start = max(0, first - pad)
    end = min(len(pcm), last + 1 + pad)
    return pcm[start:end]


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
    args: tuple[Path, int, str, list[str | None], np.random.SeedSequence, int],
) -> list[Example]:
    # attack_type is resolved in build_examples (parent process), not here:
    # ProcessPoolExecutor uses spawn on Windows, so a worker subprocess never
    # sees attack_labels.DIRECTORY_DEFAULT_ATTACK_TYPE mutations made in the
    # parent after the pool starts — same class of issue this module's RNG
    # handling already works around. Resolving up front sidesteps it rather
    # than relying on cross-process global state.
    wav_path, label, source_dir, channel_recipes, seed_seq, attack_type = args
    rng = np.random.default_rng(seed_seq)
    pcm = _load_mono_16k(wav_path)
    pcm = trim_edge_silence(pcm, rng=rng)
    out: list[Example] = []
    for recipe in channel_recipes:
        degraded = _maybe_channel(pcm, recipe, rng)
        for chunk in chunk_audio(degraded):
            out.append(
                Example(
                    # float32, not extract_lfcc_sequence's native float64: a
                    # sequence Example is ~167x larger than track 2's flat
                    # 66-float vector (184x60 vs 66), so accumulating tens of
                    # thousands of them in float64 during extraction — before
                    # to_arrays' own eventual .astype(np.float32) even runs —
                    # was measured to double peak RAM on a real full-corpus
                    # run and made the process unviable on a 16GB machine.
                    # Downcast immediately instead of deferring to to_arrays.
                    lfcc_seq=extract_lfcc_sequence(chunk).astype(np.float32),
                    scalars=extract_scalars(chunk).astype(np.float32),
                    label=label,
                    attack_type=attack_type,
                    # Full resolved path, not wav_path.name: several corpora sourced
                    # into this pipeline use sequentially-numbered filenames
                    # (1.wav, 10005.wav, ...) that collide across directories.
                    # split_by_source groups by this field, so a bare basename
                    # would silently merge unrelated clips from different
                    # corpora/labels into one "source" for train/val splitting.
                    source_file=str(wav_path.resolve()),
                    source_dir=source_dir,
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
    attack_type_maps: dict[str, dict[str, str] | None] | None = None,
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
    across a process pool the way it can in-process.

    attack_type_maps: optional {resolved_source_dir: per_file_map_or_None}
    — see attack_labels.attack_type_for_file. Directories absent from this
    dict fall back to the directory-level default / "unknown", same as
    passing an explicit None entry."""
    attack_type_maps = attack_type_maps or {}
    tasks: list[tuple[Path, int, str, list[str | None], int]] = []
    for label, dirs in ((0, real_dir), (1, fake_dir)):
        for directory in _as_dir_list(dirs):
            source_dir = str(Path(directory).resolve())
            per_file_map = attack_type_maps.get(source_dir)
            for wav_path in sorted(Path(directory).glob("*.wav")):
                if label == 1:
                    attack_type_str = attack_type_for_file(wav_path, source_dir, per_file_map)
                    attack_type = ATTACK_TYPE_TO_INT.get(attack_type_str, IGNORE_ATTACK_TYPE)
                else:
                    attack_type = IGNORE_ATTACK_TYPE  # real examples are never attack-typed
                tasks.append((wav_path, label, source_dir, list(channel_recipes), attack_type))

    seed_seqs = np.random.SeedSequence(seed).spawn(len(tasks))
    tasks = [(t[0], t[1], t[2], t[3], ss, t[4]) for t, ss in zip(tasks, seed_seqs)]

    workers = workers or os.cpu_count() or 1
    examples: list[Example] = []
    if workers <= 1 or len(tasks) < 2:
        for task in tasks:
            examples.extend(_process_file(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = 0
            for result in pool.map(_process_file, tasks, chunksize=4):
                examples.extend(result)
                done += 1
                if done % 500 == 0:
                    print(f"  ...{done}/{len(tasks)} source files processed", file=sys.stderr)

    report_yield(tasks, examples)
    return examples


def report_yield(
    tasks: list[tuple], examples: list[Example]
) -> dict[str, dict]:
    """Per-source-directory yield report: how many input files actually
    survived chunk_audio's 3s-minimum cutoff to become >=1 training example,
    vs. how many were silently dropped entirely. Directory file counts alone
    are not the training corpus composition — this is (found the hard way,
    2026-09-10: some sources lose ~70% of files this way, others ~0%, which
    silently and unevenly reweights every corpus mix)."""
    files_per_dir: dict[str, set[str]] = {}
    for task in tasks:
        wav_path, _label, source_dir, _recipes = task[0], task[1], task[2], task[3]
        files_per_dir.setdefault(source_dir, set()).add(str(wav_path.resolve()))

    surviving_per_dir: dict[str, set[str]] = {}
    windows_per_dir: dict[str, int] = {}
    for ex in examples:
        surviving_per_dir.setdefault(ex.source_dir, set()).add(ex.source_file)
        windows_per_dir[ex.source_dir] = windows_per_dir.get(ex.source_dir, 0) + 1

    report = {}
    print("Per-source-directory yield (files in -> files surviving >=3s chunking -> windows out):",
          file=sys.stderr)  # noqa: keep ASCII-only, printed to Windows consoles
    for source_dir, all_files in sorted(files_per_dir.items()):
        n_in = len(all_files)
        n_survived = len(surviving_per_dir.get(source_dir, set()))
        n_windows = windows_per_dir.get(source_dir, 0)
        frac = n_survived / n_in if n_in else float("nan")
        report[source_dir] = {
            "files_in": n_in, "files_survived": n_survived,
            "windows_out": n_windows, "survival_frac": frac,
        }
        flag = "  <-- >50% of files produced ZERO training examples" if frac < 0.5 else ""
        print(f"  {source_dir}: {n_in} in -> {n_survived} survived ({frac:.0%}) "
              f"-> {n_windows} windows{flag}", file=sys.stderr)
    return report


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


def to_arrays(examples: list[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_seq = np.stack([e.lfcc_seq for e in examples]).astype(np.float32)
    X_scalars = np.stack([e.scalars for e in examples]).astype(np.float32)
    y = np.array([e.label for e in examples], dtype=np.int64)
    attack_y = np.array([e.attack_type for e in examples], dtype=np.int64)
    return X_seq, X_scalars, y, attack_y
