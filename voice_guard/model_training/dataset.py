"""
Audio -> windows -> (LFCC sequence, scalars) for VoiceGuard, the one place
windowing and per-file randomness are defined. Every train and eval path
goes through `process_file`, directly or via feature_cache.py.

Label convention matches TFLiteService.infer's softmax indexing:
  0 = real (VERIFIED_HUMAN), 1 = synthetic/cloned (AI_DETECTED)

## Windowing contract (v12, 2026-09-11)

The app scores a 3 s window every second and never pads: a window always
holds 3 s of whatever the call carried, pauses and line noise included.
Training/eval windows are built to look like that:

1. load mono 16 kHz, trim edge silence (`trim_edge_silence`);
2. clip < 1.0 s after trimming: dropped (reported, never silent);
3. 1.0-3.0 s: padded to 3 s + 0.1 s margin with Gaussian noise at the
   clip's own noise floor (quietest-10% 20 ms frame RMS, floor 1e-4),
   split randomly between front and back, **before** the channel, so the
   pad carries channel noise like the silent part of a real call window;
4. >= 3.0 s: the channel runs on the whole clip, then non-overlapping 3 s
   windows, plus one end-aligned tail window if >= 1 s is left over.

Before v12 every clip under 3 s was dropped. That silently removed ~74%
of training files and 44-75% of held-out files, unevenly per source.

Each window records `pad_fraction` (padded share of its 3 s). Padding is a
new potential shortcut: in this corpus the share of short clips differs by
source and label (e.g. fake2021 ~68% short vs real2021 ~45%). So:
- **training** equalizes it per source set with `compute_pad_policy`: sets
  with too few padded windows turn some long-clip windows into random
  1-3 s crops (padded the same way), and sets with too many keep only a
  fraction of their short clips, so every set lands near
  PAD_TARGET_FRACTION padded windows;
- **eval** never rebalances; the report measures pad_fraction as a
  confound feature (evaluate.py).

## Determinism

Every random draw (trim pad, noise pad, crop, channel) is seeded from a
stable hash of (file_id, purpose, channel, base seed), never from a task
index. File ids are paths relative to `data/` (junction-safe, identical
across worktrees). So a file gets the same windows no matter which other
files are in the run, and caches stay reusable. Before v12 seeds came from
the task index: the same fake_itw_held dir yielded 762 files in one script
and 757 in another, because trim padding differed and ~3.0 s files flipped
across the cutoff.
"""
from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from attack_labels import (
    ATTACK_ID_UNKNOWN,
    attack_id_for_file,
    attack_id_to_int,
    attack_type_for_file,
)
from eval_protocol import channel_name
from features import SAMPLE_RATE, extract_lfcc_sequence, extract_scalars

VAANI_ROOT = Path(__file__).resolve().parents[2] / "vaani"
if str(VAANI_ROOT) not in sys.path:
    sys.path.append(str(VAANI_ROOT))

MODEL_TRAINING_DIR = Path(os.path.abspath(__file__)).parent
DATA_ROOT = MODEL_TRAINING_DIR / "data"
AUDIO_EXTENSIONS = (".wav", ".flac")

ATTACK_TYPE_TO_INT = {"tts": 0, "vc": 1}  # "unknown" and real both map to IGNORE_ATTACK_TYPE
IGNORE_ATTACK_TYPE = -100  # PyTorch CrossEntropyLoss's default ignore_index

WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)  # 48000, = features.CHUNK_SAMPLES
MIN_CLIP_SECONDS = 1.0
TAIL_MIN_SECONDS = 1.0
PAD_MARGIN_SECONDS = 0.1
NOISE_FLOOR_FRAME = 320  # 20 ms
NOISE_FLOOR_QUANTILE = 0.10
NOISE_FLOOR_MIN = 1e-4
CROP_SECONDS_RANGE = (1.0, 3.0)
PAD_TARGET_FRACTION = 0.5
# Bounds on pad balancing: a set made almost entirely of short clips keeps
# at least 30% of them (it must never vanish), and at most 60% of a
# long-clip set's windows become crops (long context must survive).
KEEP_SHORT_PROB_MIN = 0.3
CONVERT_PROB_MAX = 0.6

SILENCE_TRIM_THRESHOLD = 0.01  # matches dataset_audit.py's leading/trailing-silence measure
SILENCE_TRIM_PAD_RANGE = (0.05, 0.15)  # seconds; randomized per file, not a fixed constant

STATUS_OK = "ok"
STATUS_TOO_SHORT = "too_short"
STATUS_SILENT = "silent"
STATUS_LOAD_ERROR = "load_error"
STATUS_BALANCED_OUT = "pad_balanced_out"


# ---------------------------------------------------------------- identity

def file_id(path: Path | str) -> str:
    """Stable id: posix path relative to data/ when under it (computed with
    abspath, not resolve, so NTFS junctions keep the worktree-local
    spelling), else the absolute posix path."""
    p = os.path.abspath(str(path))
    root = str(DATA_ROOT)
    try:
        rel = os.path.relpath(p, root)
    except ValueError:  # different drive on Windows
        rel = None
    if rel is not None and not rel.startswith(".."):
        return Path(rel).as_posix()
    return Path(p).as_posix()


def stable_seed(*parts) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def stable_unit(*parts) -> float:
    """Deterministic pseudo-uniform in [0, 1) from the parts."""
    return stable_seed(*parts) / 2**64


def list_audio_files(directory: Path | str, recursive: bool = False) -> list[Path]:
    d = Path(directory)
    it = d.rglob("*") if recursive else d.glob("*")
    files = [p for p in it if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()]
    return sorted(files, key=lambda p: file_id(p))


# ---------------------------------------------------------------- audio ops

def trim_edge_silence(
    pcm: np.ndarray, sr: int = SAMPLE_RATE, thresh: float = SILENCE_TRIM_THRESHOLD,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Trims leading/trailing near-silence, leaving a small randomized pad
    (not zero, and not one fixed constant: either would just be a new,
    cleaner boundary artifact instead of the one being removed).

    Why (2026-09-10, voice_guard/docs/superpowers/specs/
    2026-09-10-model-regression-design.md): edge-silence duration is a
    documented confound in ASVspoof-lineage corpora (Kwak et al. 2021,
    arXiv:2106.12914), and this project's own TTS accent cells showed the
    opposite bias. Trimming every clip makes the corpus consistent while
    leaving internal pauses (which pauseRatio measures) untouched.

    Returns `pcm` unchanged if nothing exceeds `thresh` (process_file drops
    such clips as silent)."""
    rng = rng or np.random.default_rng()
    above = np.abs(pcm) > thresh
    if not above.any():
        return pcm
    first = int(np.argmax(above))
    last = len(pcm) - 1 - int(np.argmax(above[::-1]))
    pad = int(rng.uniform(*SILENCE_TRIM_PAD_RANGE) * sr)
    start = max(0, first - pad)
    end = min(len(pcm), last + 1 + pad)
    return pcm[start:end]


def noise_floor(pcm: np.ndarray) -> float:
    """RMS of the quietest 10% of 20 ms frames, floored at 1e-4."""
    n = len(pcm) // NOISE_FLOOR_FRAME
    if n == 0:
        return NOISE_FLOOR_MIN
    frames = pcm[: n * NOISE_FLOOR_FRAME].reshape(n, NOISE_FLOOR_FRAME).astype(np.float64)
    rms = np.sqrt((frames**2).mean(axis=1))
    return float(max(np.quantile(rms, NOISE_FLOOR_QUANTILE), NOISE_FLOOR_MIN))


def pad_to_window(pcm: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Pads a <3 s clip to 3 s + margin with noise-floor Gaussian noise,
    split randomly front/back. Returns (padded, pad_samples_inside_first_3s)."""
    total = WINDOW_SAMPLES + int(PAD_MARGIN_SECONDS * SAMPLE_RATE)
    n_pad = max(0, total - len(pcm))
    front = int(rng.integers(0, n_pad + 1)) if n_pad > 0 else 0
    back = n_pad - front
    sigma = noise_floor(pcm)
    out = np.concatenate([
        rng.normal(0.0, sigma, front),
        np.asarray(pcm, dtype=np.float64),
        rng.normal(0.0, sigma, back),
    ]).astype(np.float32)
    margin = total - WINDOW_SAMPLES
    pad_in_window = min(front, WINDOW_SAMPLES) + max(0, back - margin)
    return out, int(min(pad_in_window, WINDOW_SAMPLES))


def _fit_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x[:WINDOW_SAMPLES], dtype=np.float32)
    if len(x) < WINDOW_SAMPLES:  # codec length jitter; a few ms at most
        x = np.concatenate([x, np.zeros(WINDOW_SAMPLES - len(x), dtype=np.float32)])
    return x


def load_audio(path: Path | str) -> np.ndarray:
    """Mono float32 at 16 kHz (WAV or FLAC)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa  # local import: heavy dep, only needed for resampling

        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.asarray(data, dtype=np.float32)


_CHANNEL_CONFIG = None


def apply_channel(pcm: np.ndarray, recipe: str | None, rng: np.random.Generator) -> np.ndarray:
    """TeleChannel degradation; recipe None is a true no-op."""
    if recipe is None:
        return pcm
    global _CHANNEL_CONFIG
    from telechannel.pipeline import load_config, process_clip  # vaani package

    if _CHANNEL_CONFIG is None:
        _CHANNEL_CONFIG = load_config()
    return process_clip(pcm, recipe, SAMPLE_RATE, config=_CHANNEL_CONFIG, rng=rng).astype(np.float32)


# ---------------------------------------------------------------- specs / windows

@dataclass(frozen=True)
class FileSpec:
    """One source file and how to window it."""
    path: str
    file_id: str
    label: int
    source_set: str
    attack_type: int = IGNORE_ATTACK_TYPE
    attack_id: int = ATTACK_ID_UNKNOWN
    convert_prob: float = 0.0  # training pad balancing: long-clip window -> padded crop
    keep_short_prob: float = 1.0  # training pad balancing: keep a short (padded) clip


@dataclass
class Window:
    lfcc_seq: np.ndarray  # (184, 60) float32, unpooled
    scalars: np.ndarray  # (6,) float32 = 3 prosody + 3 physio
    label: int
    attack_type: int
    attack_id: int
    file_id: str
    source_set: str
    channel: str
    window_index: int
    pad_fraction: float
    duration: float  # trimmed clip duration, seconds


def estimate_windows(duration: float) -> tuple[int, int]:
    """(padded_windows, full_windows) a clip of this trimmed duration yields."""
    if duration < MIN_CLIP_SECONDS:
        return 0, 0
    if duration < WINDOW_SECONDS:
        return 1, 0
    n = int(duration // WINDOW_SECONDS)
    rem = duration - n * WINDOW_SECONDS
    return 0, n + (1 if rem >= TAIL_MIN_SECONDS else 0)


def estimate_duration_seconds(path: Path) -> float:
    """Clip duration from the file header only — no audio decode. Used by
    the training pad-policy pre-pass (compute_pad_policy needs raw durations
    for the whole corpus; decoding ~100k files just to time them took the
    v12 train-cache build over the edge). Trim-edge effects are tiny
    relative to a 1–3 s policy decision."""
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


def compute_pad_policy(durations: list[float], target: float = PAD_TARGET_FRACTION) -> tuple[float, float, float]:
    """(convert_prob, keep_short_prob, natural_padded_fraction) moving one
    source set's padded-window share to `target`, from clip durations."""
    padded = full = 0
    for d in durations:
        p, f = estimate_windows(d)
        padded += p
        full += f
    total = padded + full
    if total == 0:
        return 0.0, 1.0, float("nan")
    frac = padded / total
    if frac < target:
        return float(min(CONVERT_PROB_MAX, (target - frac) / (1 - frac))), 1.0, frac
    if frac > target:
        return 0.0, float(max(KEEP_SHORT_PROB_MIN, target * (1 - frac) / (frac * (1 - target)))), frac
    return 0.0, 1.0, frac


def process_file(spec: FileSpec, recipe: str | None, base_seed: int = 0) -> tuple[list[Window], str, float]:
    """All windows of one file under one channel recipe.
    Returns (windows, status, trimmed_duration_seconds)."""
    chan = channel_name(recipe)
    try:
        pcm = load_audio(spec.path)
    except Exception:  # corrupt/undecodable file: reported by callers, never fatal
        return [], STATUS_LOAD_ERROR, 0.0
    if not (np.abs(pcm) > SILENCE_TRIM_THRESHOLD).any():
        return [], STATUS_SILENT, len(pcm) / SAMPLE_RATE

    file_rng = np.random.default_rng(stable_seed(spec.file_id, base_seed, "trim"))
    pcm = trim_edge_silence(pcm, rng=file_rng)
    duration = len(pcm) / SAMPLE_RATE
    if duration < MIN_CLIP_SECONDS:
        return [], STATUS_TOO_SHORT, duration

    chan_rng = np.random.default_rng(stable_seed(spec.file_id, base_seed, "channel", chan))
    segments: list[tuple[np.ndarray, float]] = []
    if duration < WINDOW_SECONDS:
        if spec.keep_short_prob < 1.0 and stable_unit(spec.file_id, base_seed, "keep_short") >= spec.keep_short_prob:
            return [], STATUS_BALANCED_OUT, duration
        padded, n_pad = pad_to_window(pcm, file_rng)
        segments.append((_fit_window(apply_channel(padded, recipe, chan_rng)), n_pad / WINDOW_SAMPLES))
    else:
        degraded = apply_channel(pcm, recipe, chan_rng)
        # max(1, ...): a codec can shave a few ms off a ~3.0 s clip; that is
        # still one window (zero-filled by _fit_window), never zero windows.
        n_full = max(1, len(degraded) // WINDOW_SAMPLES)
        starts = [i * WINDOW_SAMPLES for i in range(n_full)]
        if len(degraded) - n_full * WINDOW_SAMPLES >= TAIL_MIN_SECONDS * SAMPLE_RATE:
            starts.append(len(degraded) - WINDOW_SAMPLES)
        for wi, s in enumerate(starts):
            if spec.convert_prob > 0 and stable_unit(spec.file_id, base_seed, "convert", wi) < spec.convert_prob:
                crop_rng = np.random.default_rng(stable_seed(spec.file_id, base_seed, "crop", wi))
                length = int(crop_rng.uniform(*CROP_SECONDS_RANGE) * SAMPLE_RATE)
                lo = min(s, max(0, len(pcm) - length))
                c0 = int(crop_rng.integers(lo, max(lo, min(s + WINDOW_SAMPLES - length, len(pcm) - length)) + 1))
                padded, n_pad = pad_to_window(pcm[c0:c0 + length], crop_rng)
                segments.append((_fit_window(apply_channel(padded, recipe, chan_rng)), n_pad / WINDOW_SAMPLES))
            else:
                segments.append((_fit_window(degraded[s:s + WINDOW_SAMPLES]), 0.0))

    windows = [
        Window(
            # float32 immediately: float64 sequences doubled peak RAM on a
            # full-corpus run (2026-09-11, v11).
            lfcc_seq=extract_lfcc_sequence(seg).astype(np.float32),
            scalars=extract_scalars(seg).astype(np.float32),
            label=spec.label, attack_type=spec.attack_type, attack_id=spec.attack_id,
            file_id=spec.file_id, source_set=spec.source_set, channel=chan,
            window_index=wi, pad_fraction=float(pad_frac), duration=float(duration),
        )
        for wi, (seg, pad_frac) in enumerate(segments)
    ]
    return windows, STATUS_OK, duration


def make_file_specs(
    directory: Path | str,
    label: int,
    source_set: str | None = None,
    attack_type_maps: dict[str, dict[str, str] | None] | None = None,
    default_attack_type: str | None = None,
    cap: int | None = None,
    recursive: bool = False,
    files: list[Path] | None = None,
) -> list[FileSpec]:
    """FileSpecs for one directory. `cap` keeps a deterministic hash
    subsample (same files every run, independent of directory order).
    Fakes get attack type/id from the per-file map (keyed by resolved dir),
    else the registered directory default, else `default_attack_type`."""
    directory = Path(directory)
    source_set = source_set or file_id(directory)
    files = list_audio_files(directory, recursive) if files is None else sorted(files, key=file_id)
    if cap is not None and len(files) > cap:
        files = sorted(files, key=lambda p: stable_unit(file_id(p), "cap"))[:cap]
        files.sort(key=file_id)
    resolved_dir = str(directory.resolve())
    per_file_map = (attack_type_maps or {}).get(resolved_dir)
    specs = []
    for f in files:
        at, aid = IGNORE_ATTACK_TYPE, ATTACK_ID_UNKNOWN
        if label == 1:
            t = attack_type_for_file(f, resolved_dir, per_file_map)
            if t == "unknown" and default_attack_type:
                t = default_attack_type
            at = ATTACK_TYPE_TO_INT.get(t, IGNORE_ATTACK_TYPE)
            aid = attack_id_to_int(attack_id_for_file(f, per_file_map))
        specs.append(FileSpec(path=str(f), file_id=file_id(f), label=label, source_set=source_set,
                              attack_type=at, attack_id=aid))
    return specs


# ---------------------------------------------------------------- in-memory API

def _process_task(task: tuple[FileSpec, str | None, int]):
    spec, recipe, seed = task
    windows, status, duration = process_file(spec, recipe, seed)
    return spec, recipe, windows, status


def build_examples(
    real_dir: Path | list[Path],
    fake_dir: Path | list[Path],
    channel_recipes: list[str | None],
    workers: int | None = None,
    seed: int = 0,
    attack_type_maps: dict[str, dict[str, str] | None] | None = None,
) -> list[Window]:
    """In-memory windows for small jobs (tests, quick probes). Real
    training/eval runs use feature_cache.py, which writes the same windows
    to disk in shards so RAM stays bounded. `channel_recipes` is required:
    there is deliberately no clean-only default (see eval_protocol.py)."""
    specs: list[FileSpec] = []
    for label, dirs in ((0, real_dir), (1, fake_dir)):
        for d in ([dirs] if isinstance(dirs, (str, Path)) else list(dirs)):
            specs += make_file_specs(d, label, attack_type_maps=attack_type_maps)
    tasks = [(s, r, seed) for s in specs for r in channel_recipes]
    workers = workers or min(os.cpu_count() or 1, 8)
    results = []
    if workers <= 1 or len(tasks) < 2:
        results = [_process_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_process_task, tasks, chunksize=4))
    windows = [w for _s, _r, ws, _st in results for w in ws]
    report_yield([(s, r, len(ws), st) for s, r, ws, st in results])
    return windows


def report_yield(results: list[tuple[FileSpec, str | None, int, str]], stream=None) -> dict[str, dict]:
    """Per-source-set yield: files in, per-status counts, windows out. Every
    dropped file is counted by reason; nothing is silently lost."""
    stream = stream or sys.stderr
    report: dict[str, dict] = {}
    for spec, recipe, n_windows, status in results:
        r = report.setdefault(spec.source_set, {"renditions": 0, "windows": 0})
        r["renditions"] += 1
        r["windows"] += n_windows
        r[status] = r.get(status, 0) + 1
    print("Per-source-set yield (file x channel renditions -> status counts -> windows):", file=stream)
    for name, r in sorted(report.items()):
        ok = r.get(STATUS_OK, 0)
        frac = ok / r["renditions"] if r["renditions"] else float("nan")
        r["ok_frac"] = frac
        dropped = {k: v for k, v in r.items() if k not in ("renditions", "windows", "ok_frac", STATUS_OK)}
        flag = "  <-- >10% of renditions produced no window" if frac < 0.9 else ""
        print(f"  {name}: {r['renditions']} in -> {ok} ok ({frac:.0%}), dropped={dropped} "
              f"-> {r['windows']} windows{flag}", file=stream)
    return report


def split_by_source(
    windows: list[Window], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Window], list[Window]]:
    """Source-file-level split by stable hash: every window and channel
    rendition of a file lands on the same side, and a file's side doesn't
    depend on which other files are present."""
    val = [w for w in windows if stable_unit(w.file_id, seed, "val") < val_fraction]
    train = [w for w in windows if stable_unit(w.file_id, seed, "val") >= val_fraction]
    return train, val


def to_arrays(windows: list[Window]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_seq = np.stack([w.lfcc_seq for w in windows]).astype(np.float32)
    X_scalars = np.stack([w.scalars for w in windows]).astype(np.float32)
    y = np.array([w.label for w in windows], dtype=np.int64)
    attack_y = np.array([w.attack_type for w in windows], dtype=np.int64)
    return X_seq, X_scalars, y, attack_y
