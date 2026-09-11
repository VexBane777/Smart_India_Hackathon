"""Builds feature-cache units ahead of training/evaluation (long, resumable:
re-run the same command after a crash and finished shards are kept).

Usage:
    # held-out eval caches (both splits x DEFAULT_EVAL_CHANNELS):
    python build_caches.py --eval
    # training corpus (TRAIN_SETS_V12, none + one phone channel per file):
    python build_caches.py --train

--cache-root defaults to $VOICEGUARD_CACHE_ROOT, else model_training/cache
(gitignored). Keep it outside any temporary worktree so caches outlive it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from build_attack_type_maps import build_attack_type_maps
from corpus import EVAL_SETS, eval_specs, training_specs, training_units
from eval_protocol import APPLICATIONS, channel_name, resolve_channels
from feature_cache import DEFAULT_CACHE_ROOT, build_unit, unit_dir


def default_cache_root() -> Path:
    return Path(os.environ.get("VOICEGUARD_CACHE_ROOT", DEFAULT_CACHE_ROOT))


def eval_units(split: str, channels, set_names=None, attack_type_maps=None):
    """[(unit name, specs, recipe)] for one split of the committed manifest."""
    specs_by_set = eval_specs(split, tuple(set_names) if set_names else None, attack_type_maps=attack_type_maps)
    return [(f"eval_{split}_{name}", specs, ch) for name, specs in specs_by_set.items() for ch in channels]


def train_units_for(cache_root: Path, seed: int = 0, attack_type_maps=None, workers: int = 10):
    specs_by_set = training_specs(attack_type_maps=attack_type_maps,
                                  duration_cache=cache_root / "train_durations.json", workers=workers)
    return training_units(specs_by_set, seed)


def build_all(units, cache_root: Path, workers: int, seed: int, rebuild: bool) -> list[Path]:
    dirs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (name, specs, ch) in enumerate(units):
            dirs.append(build_unit(name, specs, ch, cache_root, seed=seed, workers=workers, rebuild=rebuild, pool=pool))
            print(f"[build] unit {i + 1}/{len(units)} {name}/{channel_name(ch)} done, "
                  f"{(time.time() - t0) / 60:.1f} min elapsed", file=sys.stderr, flush=True)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="build held-out eval caches")
    ap.add_argument("--splits", nargs="+", default=["select", "test"], choices=["select", "test"])
    ap.add_argument("--eval-sets", nargs="*", default=None, help=f"subset of {[s.name for s in EVAL_SETS]}")
    ap.add_argument("--channels", nargs="*", default=None, help="eval channels (default: none + all phone channels)")
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
    ap.add_argument("--train", action="store_true", help="build training caches")
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-stale", action="store_true")
    args = ap.parse_args()
    if not (args.eval or args.train):
        ap.error("pass --eval and/or --train")

    cache_root = args.cache_root or default_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    maps = build_attack_type_maps()
    units = []
    if args.eval:
        channels = resolve_channels(args.channels, args.application, "eval")
        for split in args.splits:
            units += eval_units(split, channels, args.eval_sets, maps)
    if args.train:
        units += train_units_for(cache_root, args.seed, maps, args.workers)
    n_files = sum(len(s) for _n, s, _c in units)
    print(f"[build] {len(units)} units, {n_files} file-renditions -> {cache_root}", file=sys.stderr, flush=True)
    todo = [u for u in units if not (unit_dir(cache_root, u[0], u[1], u[2], args.seed) / "manifest.json").exists()]
    print(f"[build] {len(units) - len(todo)} already built (version-checked on use)", file=sys.stderr, flush=True)
    build_all(units, cache_root, args.workers, args.seed, args.rebuild_stale)
    print("[build] all units complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
