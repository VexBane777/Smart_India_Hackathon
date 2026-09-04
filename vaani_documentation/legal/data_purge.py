"""
data_purge.py — DPDP Act 2023 "Right to Erasure" (Sections 11-13) enforcer.

Scans the manifest.parquet file (same file and default location as
telechannel/manifest.py's DEFAULT_MANIFEST_PATH / run_corpus.yaml's
`manifest.path: data/manifest.parquet`) and removes all rows belonging to
a given (withdrawn) speaker, then re-writes the manifest in place. This is
step 4 of the revocation workflow described in TDD_MOD_A_01_Consent.md,
section 2.2.

Usage:
    python data_purge.py --id spk_001
    python data_purge.py --id spk_001 --manifest-path /path/to/data/manifest.parquet
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.parquet"
SPEAKER_COLUMN = "speaker_anon"


def purge_speaker(speaker_id: str, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict:
    """
    Remove all rows where `speaker_anon == speaker_id` from the manifest
    Parquet file at `manifest_path`, rewriting it in place.

    Returns a dict mapping manifest filename -> number of rows removed,
    (empty if the manifest didn't contain the speaker or lacks the
    speaker_anon column), for symmetry with the previous multi-manifest
    return shape and to keep the CLI summary output unchanged.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    removed_summary = {}

    df = pd.read_parquet(manifest_path)

    if SPEAKER_COLUMN not in df.columns:
        return removed_summary

    mask = df[SPEAKER_COLUMN] == speaker_id
    num_removed = int(mask.sum())
    if num_removed == 0:
        return removed_summary

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
    parser = argparse.ArgumentParser(description="Purge a speaker's rows from the manifest parquet file.")
    parser.add_argument("--id", required=True, dest="speaker_id", help="Anonymous speaker_id (e.g. spk_001)")
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the manifest .parquet file (default: data/manifest.parquet)",
    )
    args = parser.parse_args()

    try:
        summary = purge_speaker(args.speaker_id, Path(args.manifest_path))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if summary:
        print(f"Purged '{args.speaker_id}' from manifest: {summary}")
    else:
        print(f"No rows for '{args.speaker_id}' found in manifest at {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
