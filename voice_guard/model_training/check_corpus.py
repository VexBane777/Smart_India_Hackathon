"""
Preflight gate: run this BEFORE launching a long train.py job, not after.

Checks, cheaply (header reads / small samples, no full feature extraction):
  1. Per-directory chunk-yield estimate — what fraction of files in each dir
     will actually survive chunk_audio's 3s-minimum cutoff (see dataset.py's
     report_yield / features.py's chunk_audio docstring).
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
    """Returns (n_files, fraction surviving chunk_audio's >=3s cutoff),
    estimated from a header-only sample (fast: no audio decode)."""
    files = sorted(Path(directory).glob("*.wav"))
    if not files:
        return 0, float("nan")
    sample = files if len(files) <= sample_n else random.Random(seed).sample(files, sample_n)
    survived = 0
    for f in sample:
        try:
            if sf.info(str(f)).duration >= 3.0:
                survived += 1
        except Exception:
            pass
    return len(files), survived / len(sample)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, metavar=("REAL_DIR", "FAKE_DIR"), action="append", default=[],
                     help="a real/fake directory pair to audit; repeatable.")
    ap.add_argument("--min-yield", type=float, default=0.15,
                     help="warn if a directory's estimated >=3s-survival fraction is below this.")
    args = ap.parse_args()

    if not args.pair:
        ap.error("pass at least one --pair REAL_DIR FAKE_DIR")

    had_issue = False
    for real_dir, fake_dir in args.pair:
        print(f"\n=== {real_dir}  vs  {fake_dir} ===")
        for label, d in (("real", real_dir), ("fake", fake_dir)):
            n, frac = estimate_yield(d)
            flag = "  <-- LOW YIELD" if frac < args.min_yield else ""
            print(f"  [{label}] {d}: {n} files, ~{frac:.0%} survive >=3s chunking{flag}")
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

    if had_issue:
        print("\ncheck_corpus: at least one directory pair looks risky to train on as-is. "
              "Review before running train.py.")
        raise SystemExit(1)
    print("\ncheck_corpus: all pairs look OK.")


if __name__ == "__main__":
    main()
