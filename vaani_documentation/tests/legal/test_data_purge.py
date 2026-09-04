"""
Unit tests for legal/data_purge.py.

Every test operates on throwaway parquet fixtures under pytest's tmp_path
fixture — no fixture here touches vaani_documentation/data/manifests/.
"""

import pandas as pd
import pytest

from data_purge import purge_speaker


def _write_manifest(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_purge_removes_matching_rows(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest_00.parquet",
        {
            "clip_id": ["c1", "c2", "c3", "c4"],
            "speaker_anon": ["spk_001", "spk_001", "spk_002", "spk_003"],
            "duration_s": [3.2, 4.1, 2.9, 5.0],
        },
    )

    summary = purge_speaker("spk_001", tmp_path)

    assert summary == {"manifest_00.parquet": 2}
    remaining = pd.read_parquet(manifest)
    assert sorted(remaining["speaker_anon"].tolist()) == ["spk_002", "spk_003"]
    assert "spk_001" not in remaining["speaker_anon"].values


def test_purge_with_no_matching_rows_is_a_noop(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest_00.parquet",
        {
            "clip_id": ["c3", "c4"],
            "speaker_anon": ["spk_002", "spk_003"],
            "duration_s": [2.9, 5.0],
        },
    )
    original_bytes = manifest.read_bytes()

    summary = purge_speaker("spk_001", tmp_path)

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

    summary = purge_speaker("spk_001", tmp_path)

    assert summary == {}
    assert manifest.read_bytes() == original_bytes


def test_purge_across_multiple_manifests(tmp_path):
    _write_manifest(
        tmp_path / "manifest_a.parquet",
        {"clip_id": ["c1"], "speaker_anon": ["spk_001"], "duration_s": [1.0]},
    )
    _write_manifest(
        tmp_path / "manifest_b.parquet",
        {"clip_id": ["c2", "c3"], "speaker_anon": ["spk_001", "spk_002"], "duration_s": [1.0, 2.0]},
    )

    summary = purge_speaker("spk_001", tmp_path)

    assert summary == {"manifest_a.parquet": 1, "manifest_b.parquet": 1}
    remaining_a = pd.read_parquet(tmp_path / "manifest_a.parquet")
    remaining_b = pd.read_parquet(tmp_path / "manifest_b.parquet")
    assert remaining_a.empty
    assert remaining_b["speaker_anon"].tolist() == ["spk_002"]


def test_purge_missing_manifests_dir_raises_filenotfounderror(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        purge_speaker("spk_001", missing_dir)


def test_purge_write_is_atomic_no_temp_file_left_behind(tmp_path):
    _write_manifest(
        tmp_path / "manifest_00.parquet",
        {"clip_id": ["c1"], "speaker_anon": ["spk_001"], "duration_s": [1.0]},
    )

    purge_speaker("spk_001", tmp_path)

    leftover_tmp_files = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover_tmp_files == []
