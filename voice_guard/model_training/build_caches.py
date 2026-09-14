"""Builds feature-cache units ahead of training/evaluation (long, resumable:
re-run the same command after a crash and finished shards are kept).

Usage:
    # held-out eval caches (both splits x DEFAULT_EVAL_CHANNELS):
    python build_caches.py --eval
    # training corpus (TRAIN_SETS_V12, v13: none + one phone channel per file
    # plus a hash-selected ~50% playback rendition):
    python build_caches.py --train

--cache-root defaults to $VOICEGUARD_CACHE_ROOT, else model_training/cache
(gitignored). Keep it outside any temporary worktree so caches outlive it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from build_attack_type_maps import build_attack_type_maps
from corpus import EVAL_SETS, eval_specs, training_specs, training_units
from eval_protocol import APPLICATIONS, channel_name, resolve_channels
from feature_cache import DEFAULT_CACHE_ROOT, build_unit, revalidate, unit_dir


def default_cache_root() -> Path:
    return Path(os.environ.get("VOICEGUARD_CACHE_ROOT", DEFAULT_CACHE_ROOT))


def _ensure_unit(unit, cache_root: Path, workers: int, seed: int, revalidate_stale: bool) -> Path:
    """Build a unit, but if a stale manifest already exists and
    --revalidate-stale, MEASURE whether the output is truly unchanged under
    the current code (revalidate) and re-stamp instead of rebuilding; rebuild
    only if it genuinely changed (v13 post-v12 plan step 3b)."""
    from feature_cache import feature_version

    name, specs, ch = unit
    d = unit_dir(cache_root, name, specs, ch, seed)
    mp = d / "manifest.json"
    if not mp.exists():
        return build_unit(name, specs, ch, cache_root, seed=seed, workers=workers)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if manifest.get("feature_version") == feature_version(ch, all_recipes=False)["hash"]:
        return d  # fresh: nothing to do
    if not revalidate_stale:
        # loud staleness: let build_unit raise StaleCacheError
        return build_unit(name, specs, ch, cache_root, seed=seed, workers=workers)
    res = revalidate(d, seed=seed, max_sample_files=12)
    if res["equivalent"]:
        print(f"[revalidate] {name}/{channel_name(ch)}: unchanged, re-stamped "
              f"(max_abs_diff={res['max_abs_diff']:.2e})", file=sys.stderr, flush=True)
        return d
    print(f"[revalidate] {name}/{channel_name(ch)}: changed, rebuilding", file=sys.stderr, flush=True)
    return build_unit(name, specs, ch, cache_root, seed=seed, workers=workers, rebuild=True)


def eval_units(split: str, channels, set_names=None, attack_type_maps=None):
    """[(unit name, specs, recipe)] for one split of the committed manifest."""
    specs_by_set = eval_specs(split, tuple(set_names) if set_names else None, attack_type_maps=attack_type_maps)
    return [(f"eval_{split}_{name}", specs, ch) for name, specs in specs_by_set.items() for ch in channels]


def train_units_for(cache_root: Path, seed: int = 0, attack_type_maps=None, workers: int = 10):
    specs_by_set = training_specs(attack_type_maps=attack_type_maps,
                                  duration_cache=cache_root / "train_durations.json", workers=workers)
    return training_units(specs_by_set, seed)


def build_all(units, cache_root: Path, workers: int, seed: int, rebuild: bool,
              revalidate_stale: bool) -> list[Path]:
    dirs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, unit in enumerate(units):
            name, specs, ch = unit
            if rebuild:
                dirs.append(build_unit(name, specs, ch, cache_root, seed=seed, workers=workers,
                                       rebuild=True, pool=pool))
            else:
                dirs.append(_ensure_unit(unit, cache_root, workers, seed, revalidate_stale))
            print(f"[build] unit {i + 1}/{len(units)} {name}/{channel_name(ch)} done, "
                  f"{(time.time() - t0) / 60:.1f} min elapsed", file=sys.stderr, flush=True)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="build held-out eval caches")
    ap.add_argument("--splits", nargs="+", default=["select", "test"], choices=["select", "test"])
    ap.add_argument("--eval-sets", nargs="*", default=None, help=f"subset of {[s.name for s in EVAL_SETS]}")
    ap.add_argument("--channels", nargs="*", default=None, help="eval channels (default: none + every phone and acoustic channel)")
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
    ap.add_argument("--train", action="store_true", help="build training caches (incl. the v13 playback rendition)")
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-stale", action="store_true")
    ap.add_argument("--revalidate-stale", action="store_true",
                    help="reuse stale units whose output is unchanged (measure, don't assume) "
                         "and re-stamp them; rebuild only units that truly changed (v13 step 3b)")
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
    build_all(units, cache_root, args.workers, args.seed, args.rebuild_stale, args.revalidate_stale)
    print("[build] all units complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
