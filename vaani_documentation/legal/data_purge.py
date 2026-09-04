"""
data_purge.py — DPDP Act 2023 "Right to Erasure" (Sections 11-13) enforcer.

Scans every .parquet manifest under data/manifests/ and removes all rows
belonging to a given (withdrawn) speaker, then re-writes the manifest in
place. This is step 4 of the revocation workflow described in
TDD_MOD_A_01_Consent.md, section 2.2.

Usage:
    python data_purge.py --id spk_001
    python data_purge.py --id spk_001 --manifests-dir /path/to/data/manifests
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"
SPEAKER_COLUMN = "speaker_anon"


def purge_speaker(speaker_id: str, manifests_dir: Path = DEFAULT_MANIFESTS_DIR) -> dict:
    """
    Remove all rows where `speaker_anon == speaker_id` from every .parquet
    file found in `manifests_dir`, rewriting each modified file in place.

    Returns a dict mapping manifest filename -> number of rows removed,
    for every manifest that actually contained the speaker.
    """
    manifests_dir = Path(manifests_dir)
    if not manifests_dir.exists():
        raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")

    removed_summary = {}

    for manifest_path in sorted(manifests_dir.glob("*.parquet")):
        df = pd.read_parquet(manifest_path)

        if SPEAKER_COLUMN not in df.columns:
            continue

        mask = df[SPEAKER_COLUMN] == speaker_id
        num_removed = int(mask.sum())
        if num_removed == 0:
            continue

        df = df[~mask]
        _atomic_write_parquet(df, manifest_path)
        removed_summary[manifest_path.name] = num_removed

    return removed_summary


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Write `df` to `path` as parquet atomically: write to a temp file in the
    same directory, then os.replace() over the target so a crash mid-write
    can never leave `path` truncated or corrupted.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)  # pandas/pyarrow will reopen the path by name
    try:
        df.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge a speaker's rows from all manifest parquet files.")
    parser.add_argument("--id", required=True, dest="speaker_id", help="Anonymous speaker_id (e.g. spk_001)")
    parser.add_argument(
        "--manifests-dir",
        default=str(DEFAULT_MANIFESTS_DIR),
        help="Directory containing .parquet manifests (default: data/manifests)",
    )
    args = parser.parse_args()

    try:
        summary = purge_speaker(args.speaker_id, Path(args.manifests_dir))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if summary:
        print(f"Purged '{args.speaker_id}' from manifests: {summary}")
    else:
        print(f"No rows for '{args.speaker_id}' found in any manifest under {args.manifests_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
