"""
Preflight gate: run this BEFORE launching a long train.py job, not after.

Checks, cheaply (header reads / small samples, no full feature extraction):
  1. Per-directory yield estimate: what fraction of files is long enough
     (>= dataset.MIN_CLIP_SECONDS = 1 s) to yield a window under the v12
     windowing contract (dataset.py docstring). Before v12 the cutoff was 3 s
     and ~74% of training files yielded nothing. The real per-file outcome
     is recorded in every feature-cache manifest.
  2. Technical-shortcut scan on every real/fake directory pair you pass in
     (see dataset_audit.py) — catches e.g. one class being a single TTS
     engine's native sample rate in disguise.
  3. Acoustic-shortcut scan (added after the 2026-09-10 code-orange review) —
     catches e.g. leading/trailing silence duration correlating with label
     (a documented ASVspoof-lineage confound, Kwak et al. 2021), which fixing
     #2 alone does not fix. Slower (decodes ~150 files/side), still cheap
     relative to a training run.

Exits nonzero if anything looks like it would produce a corpus not worth
training on. This does not replace held-out eval after training — it's a
cheap filter to avoid discovering these issues 40 minutes into a training
run, the way this project did on 2026-09-10.

Usage:
    python check_corpus.py --pair data/real data/fake \
        --pair data/accents_split/train/real/en_foreign data/accents_split/train/fake/en_foreign \
        --pair data/accents_split/train/real/hi_native_capped data/accents_split/train/fake/hi_native
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import soundfile as sf

from dataset_audit import audit_directory_pair


def estimate_yield(directory: Path, sample_n: int = 300, seed: int = 0) -> tuple[int, float]:
    """Returns (n_files, fraction at or above MIN_CLIP_SECONDS), estimated
    from a header-only sample (fast: no audio decode; edge trimming can
    shorten a clip slightly, so this is an upper bound)."""
    from dataset import MIN_CLIP_SECONDS

    files = sorted(Path(directory).glob("*.wav"))
    if not files:
        return 0, float("nan")
    sample = files if len(files) <= sample_n else random.Random(seed).sample(files, sample_n)
    survived = 0
    for f in sample:
        try:
            if sf.info(str(f)).duration >= MIN_CLIP_SECONDS:
                survived += 1
        except Exception:
            pass
    return len(files), survived / len(sample)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, metavar=("REAL_DIR", "FAKE_DIR"), action="append", default=[],
                     help="a real/fake directory pair to audit; repeatable.")
    ap.add_argument("--min-yield", type=float, default=0.15,
                     help="warn if a directory's estimated >=1s-survival fraction is below this.")
    args = ap.parse_args()

    if not args.pair:
        ap.error("pass at least one --pair REAL_DIR FAKE_DIR")

    had_issue = False
    for real_dir, fake_dir in args.pair:
        print(f"\n=== {real_dir}  vs  {fake_dir} ===")
        for label, d in (("real", real_dir), ("fake", fake_dir)):
            n, frac = estimate_yield(d)
            flag = "  <-- LOW YIELD" if frac < args.min_yield else ""
            print(f"  [{label}] {d}: {n} files, ~{frac:.0%} long enough to window (>=1 s){flag}")
            if frac < args.min_yield:
                had_issue = True

        issues = audit_directory_pair(Path(real_dir), Path(fake_dir))
        if issues:
            had_issue = True
            print("  SHORTCUT RISK:")
            for w in issues:
                print(f"    - {w}")
        else:
            print("  no obvious technical or acoustic shortcut found")

        from build_attack_type_maps import build_attack_type_maps
        from attack_labels import attack_type_for_file

        attack_maps = build_attack_type_maps()
        fake_resolved = str(Path(fake_dir).resolve())
        per_file_map = attack_maps.get(fake_resolved)
        fake_files = sorted(Path(fake_dir).glob("*.wav"))
        if fake_files:
            labeled = sum(
                1 for f in fake_files
                if attack_type_for_file(f, fake_resolved, per_file_map) != "unknown"
            )
            print(f"  attack-type coverage: {labeled}/{len(fake_files)} fake files labeled "
                  f"({labeled / len(fake_files):.0%})")

    if had_issue:
        print("\ncheck_corpus: at least one directory pair looks risky to train on as-is. "
              "Review before running train.py.")
        raise SystemExit(1)
    print("\ncheck_corpus: all pairs look OK.")


if __name__ == "__main__":
    main()
