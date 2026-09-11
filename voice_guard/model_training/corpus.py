"""
Committed corpus definitions for training and evaluation, so a run is
reproducible from a name rather than from whatever --real/--fake a shell
history held.

TRAIN_SETS_V12: the v11 corpus (the v9_noisefix corpus plus its noise-aug
reals) with fake2021 capped at 40,000 files by stable hash. Uncapped, fakes
outnumber reals ~4.5:1 by file. Each file is rendered twice: once with
channel `none` and once with ONE phone channel drawn deterministically from
TRAIN_PHONE_CHANNELS by file hash. That's 2x the files, a compromise
between v9's 3x and memory/time. Short-clip padding is balanced per set
(dataset.compute_pad_policy).

Eval sets come from the committed split manifest
(`eval_splits/held_out_split_v1.json`, written by make_eval_splits.py), so
no script can accidentally evaluate on files that aren't in it, or pick
up files added to a directory later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import soundfile as sf

from dataset import DATA_ROOT, FileSpec, compute_pad_policy, make_file_specs, stable_seed
from eval_protocol import TRAIN_PHONE_CHANNELS

MODEL_TRAINING_DIR = Path(__file__).resolve().parent
SPLIT_MANIFEST = MODEL_TRAINING_DIR / "eval_splits" / "held_out_split_v1.json"

# Attack-type head leave-attack-out (v12): masked from the attack-type loss
# only (still trained as fake for real/fake), then scored as unseen systems.
LEAVE_OUT_ATTACKS: tuple[str, ...] = ("A11", "A18")  # one TTS, one VC


@dataclass(frozen=True)
class SetDef:
    name: str
    rel_dir: str
    label: int
    cap: int | None = None
    recursive: bool = False
    default_attack_type: str | None = None
    test_only: bool = False
    group: str = "core"  # core | unseen_generator | accent

    @property
    def path(self) -> Path:
        return DATA_ROOT / self.rel_dir


TRAIN_SETS_V12: tuple[SetDef, ...] = (
    SetDef("real", "real", 0),
    SetDef("real2021", "real2021", 0),
    SetDef("real_itw_train", "real_itw_train", 0),
    SetDef("noiseaug_train_en", "real_noise_aug_split/train/en_native", 0),
    SetDef("noiseaug_train_hi", "real_noise_aug_split/train/hi_native", 0),
    SetDef("fake", "fake", 1),
    SetDef("fake2021", "fake2021", 1, cap=40_000),
    SetDef("fake_itw_train", "fake_itw_train", 1),
)

EVAL_SETS: tuple[SetDef, ...] = (
    # core: split 50/50 into select/test; the headline numbers
    SetDef("itw_real", "real_itw_held", 0),
    SetDef("itw_fake", "fake_itw_held", 1),
    SetDef("noiseaug_real_en", "real_noise_aug_split/held/en_native", 0),
    SetDef("noiseaug_real_hi", "real_noise_aug_split/held/hi_native", 0),
    # unseen modern generators (~20 TTS systems, FLAC, nested), test only
    SetDef("mlaad_fake", "mlaad_en500/fake", 1, recursive=True, default_attack_type="tts",
           test_only=True, group="unseen_generator"),
    # accent cells (formerly eval_accent_cells.py), test only, capped
    SetDef("accent_real_en_native", "accents_split/held/real/en_native", 0, cap=300, test_only=True, group="accent"),
    SetDef("accent_real_en_foreign", "accents_split/held/real/en_foreign", 0, cap=300, test_only=True, group="accent"),
    SetDef("accent_real_hi_native", "accents_split/held/real/hi_native", 0, cap=300, test_only=True, group="accent"),
    SetDef("accent_fake_en_native", "accents_split/held/fake/en_native", 1, cap=300, test_only=True,
           default_attack_type="tts", group="accent"),
    SetDef("accent_fake_hi_native", "accents_split/held/fake/hi_native", 1, cap=300, test_only=True,
           default_attack_type="tts", group="accent"),
)
EVAL_SET_BY_NAME = {s.name: s for s in EVAL_SETS}
CORE_EVAL_SETS = tuple(s.name for s in EVAL_SETS if s.group == "core")


def train_phone_channel(file_id: str, seed: int = 0) -> str:
    """The one phone channel a training file is rendered through (besides none)."""
    return train_phone_channel_from(file_id, TRAIN_PHONE_CHANNELS, seed)


# Backwards-compat alias: estimate_duration_seconds used to live here as the
# private _raw_duration. dataset's copy is now the canonical one.
def _raw_duration(path: str) -> float:
    from dataset import estimate_duration_seconds

    return estimate_duration_seconds(Path(path))


def _durations_for(specs: list, durations: dict, workers: int = 10) -> dict:
    """Header-only duration reads for specs missing from `durations`, in a
    process pool (this corpus's ~120k files take minutes single-threaded;
    only the missing ones are read). Returns the updated dict."""
    from concurrent.futures import ProcessPoolExecutor

    from dataset import estimate_duration_seconds

    missing = [s for s in specs if s.file_id not in durations]
    if not missing:
        return durations
    if workers <= 1 or len(missing) < 2:
        for s in missing:
            durations[s.file_id] = estimate_duration_seconds(Path(s.path))
        return durations
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for s, d in zip(missing, pool.map(estimate_duration_seconds, (Path(s.path) for s in missing), chunksize=256)):
            durations[s.file_id] = d
    return durations


def training_specs(
    sets: tuple[SetDef, ...] = TRAIN_SETS_V12, attack_type_maps=None, balance_pad: bool = True,
    duration_cache: Path | None = None, workers: int = 10,
) -> dict[str, list[FileSpec]]:
    """{set name: specs}, with per-set pad-balancing probabilities filled in.
    Durations (header reads) are cached in `duration_cache` (json) if given."""
    durations: dict[str, float] = {}
    if duration_cache and Path(duration_cache).exists():
        durations = json.loads(Path(duration_cache).read_text())
    out: dict[str, list[FileSpec]] = {}
    for sd in sets:
        specs = make_file_specs(sd.path, sd.label, sd.name, attack_type_maps,
                                default_attack_type=sd.default_attack_type, cap=sd.cap, recursive=sd.recursive)
        if balance_pad:
            durations = _durations_for(specs, durations, workers)
            conv, keep, frac = compute_pad_policy([durations[s.file_id] for s in specs])
            print(f"[corpus] {sd.name}: {len(specs)} files, natural padded-window share ~{frac:.2f} -> "
                  f"convert_prob={conv:.3f} keep_short_prob={keep:.3f}")
            specs = [replace(s, convert_prob=conv, keep_short_prob=keep) for s in specs]
        out[sd.name] = specs
    if duration_cache:
        Path(duration_cache).parent.mkdir(parents=True, exist_ok=True)
        Path(duration_cache).write_text(json.dumps(durations))
    return out


def train_phone_channel_from(file_id: str, phone_channels: tuple[str, ...], seed: int = 0) -> str:
    return phone_channels[stable_seed(file_id, seed, "train_channel") % len(phone_channels)]


def training_units(
    specs_by_set: dict[str, list[FileSpec]], seed: int = 0,
    channels: tuple[str | None, ...] | None = None) -> list[tuple[str, list[FileSpec], str | None]]:
    """(unit name, specs, recipe) for every training rendition: each set
    once with `none` (if in `channels`), and each file once more through ONE
    phone channel from `channels` chosen by file hash. Default channels:
    TRAIN_CHANNELS."""
    channels = tuple(channels) if channels is not None else (None,) + TRAIN_PHONE_CHANNELS
    phone = tuple(c for c in channels if c is not None)
    units = []
    for name, specs in specs_by_set.items():
        if None in channels:
            units.append((f"train_{name}", specs, None))
        for ch in phone:
            sub = [s for s in specs if train_phone_channel_from(s.file_id, phone, seed) == ch]
            if sub:
                units.append((f"train_{name}", sub, ch))
    return units


def load_split_manifest(path: Path = SPLIT_MANIFEST) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing. Run make_eval_splits.py (it is committed; never regenerate "
                                "it casually: a new split invalidates every comparison made on the old one).")
    return json.loads(Path(path).read_text())


def eval_specs(split: str, set_names: tuple[str, ...] | None = None, manifest_path: Path = SPLIT_MANIFEST,
               attack_type_maps=None) -> dict[str, list[FileSpec]]:
    """{set name: specs} for one split ("select" or "test") of the committed
    manifest. Every listed file must exist; a missing file is an error,
    not a silent shrink of the eval set."""
    if split not in ("select", "test"):
        raise ValueError(f"split must be 'select' or 'test', got {split!r}")
    manifest = load_split_manifest(manifest_path)
    out = {}
    for name, entry in manifest["sets"].items():
        if set_names and name not in set_names:
            continue
        ids = entry[split]
        if not ids:
            continue
        sd = EVAL_SET_BY_NAME[name]
        files = [DATA_ROOT / fid for fid in ids]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"{name}/{split}: {len(missing)} manifest files missing, e.g. {missing[:3]}")
        out[name] = make_file_specs(sd.path, sd.label, name, attack_type_maps,
                                    default_attack_type=sd.default_attack_type, files=files)
    return out
