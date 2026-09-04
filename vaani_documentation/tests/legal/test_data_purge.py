"""
Unit tests for legal/data_purge.py.

Most tests operate on throwaway parquet fixtures under pytest's tmp_path
fixture — those never touch vaani_documentation/data/manifest.parquet.
One test (test_purge_finds_rows_at_real_default_manifest_path) deliberately
exercises the ACTUAL default location (DEFAULT_MANIFEST_PATH), since that
real-default-path case is exactly what earlier tmp_path-only tests failed
to cover and is where a manifest-location mismatch would silently produce
a false "no rows found" success on this legal right-to-erasure path.
"""

import pandas as pd
import pytest

from data_purge import DEFAULT_MANIFEST_PATH, purge_speaker


def _write_manifest(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_purge_removes_matching_rows(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.parquet",
        {
            "clip_id": ["c1", "c2", "c3", "c4"],
            "speaker_anon": ["spk_001", "spk_001", "spk_002", "spk_003"],
            "duration_s": [3.2, 4.1, 2.9, 5.0],
        },
    )

    summary = purge_speaker("spk_001", manifest)

    assert summary == {"manifest.parquet": 2}
    remaining = pd.read_parquet(manifest)
    assert sorted(remaining["speaker_anon"].tolist()) == ["spk_002", "spk_003"]
    assert "spk_001" not in remaining["speaker_anon"].values


def test_purge_with_no_matching_rows_is_a_noop(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.parquet",
        {
            "clip_id": ["c3", "c4"],
            "speaker_anon": ["spk_002", "spk_003"],
            "duration_s": [2.9, 5.0],
        },
    )
    original_bytes = manifest.read_bytes()

    summary = purge_speaker("spk_001", manifest)

    assert summary == {}
    # File must be left byte-for-byte untouched when there is nothing to purge.
    assert manifest.read_bytes() == original_bytes


def test_purge_skips_manifest_missing_speaker_anon_column(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest_no_speaker_col.parquet",
        {
            "clip_id": ["c1", "c2"],
            "duration_s": [3.2, 4.1],
        },
    )
    original_bytes = manifest.read_bytes()

    summary = purge_speaker("spk_001", manifest)

    assert summary == {}
    assert manifest.read_bytes() == original_bytes


def test_purge_missing_manifest_file_raises_filenotfounderror(tmp_path):
    missing_manifest = tmp_path / "does_not_exist.parquet"

    with pytest.raises(FileNotFoundError):
        purge_speaker("spk_001", missing_manifest)


def test_purge_write_is_atomic_no_temp_file_left_behind(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.parquet",
        {"clip_id": ["c1"], "speaker_anon": ["spk_001"], "duration_s": [1.0]},
    )

    purge_speaker("spk_001", manifest)

    leftover_tmp_files = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover_tmp_files == []


def test_purge_finds_rows_at_real_default_manifest_path():
    """
    Exercise the ACTUAL default location (DEFAULT_MANIFEST_PATH), with no
    tmp_path override, and prove purge_speaker actually finds and removes
    matching rows there.

    This is the case neither side's previous tests covered: data_purge.py
    used to default-scan a `data/manifests/` directory while
    telechannel/manifest.py (and run_corpus.yaml) write to the single file
    `data/manifest.parquet`, so a real corpus run's manifest was silently
    invisible to a default-path purge. DEFAULT_MANIFEST_PATH must now be
    the single source of truth shared with telechannel/manifest.py.
    """
    backup_bytes = DEFAULT_MANIFEST_PATH.read_bytes() if DEFAULT_MANIFEST_PATH.exists() else None
    DEFAULT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_manifest(
            DEFAULT_MANIFEST_PATH,
            {
                "clip_id": ["c1", "c2", "c3"],
                "speaker_anon": ["spk_001", "spk_001", "spk_002"],
                "duration_s": [3.2, 4.1, 2.9],
            },
        )

        summary = purge_speaker("spk_001")  # no manifest_path override

        assert summary == {DEFAULT_MANIFEST_PATH.name: 2}
        remaining = pd.read_parquet(DEFAULT_MANIFEST_PATH)
        assert remaining["speaker_anon"].tolist() == ["spk_002"]
        assert "spk_001" not in remaining["speaker_anon"].values
    finally:
        if backup_bytes is None:
            DEFAULT_MANIFEST_PATH.unlink(missing_ok=True)
        else:
            DEFAULT_MANIFEST_PATH.write_bytes(backup_bytes)
