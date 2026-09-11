"""feature_cache: build/consolidate/read round trip, resume, loud staleness,
code-only version hashing, and dropped-file accounting."""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

import feature_cache
from dataset import make_file_specs, process_file
from feature_cache import CacheCollection, StaleCacheError, _code_fingerprint, build_unit
from features import SAMPLE_RATE


def _corpus(tmp_path):
    d = tmp_path / "wavs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i, secs in enumerate([0.5, 1.8, 3.4, 7.3, 2.2]):
        t = np.arange(int(secs * SAMPLE_RATE)) / SAMPLE_RATE
        sf.write(str(d / f"c{i}.wav"), (0.3 * np.sin(2 * np.pi * (150 + 20 * i) * t)
                                         + rng.normal(0, 0.01, len(t))).astype(np.float32), SAMPLE_RATE)
    return make_file_specs(d, 0, "unit_test_set")


def test_round_trip_matches_process_file(tmp_path):
    specs = _corpus(tmp_path)
    d = build_unit("t", specs, None, tmp_path / "cache", workers=1, shard_files=2)
    coll = CacheCollection([d])
    expected = [w for s in specs for w in process_file(s, None)[0]]
    assert coll.n == len(expected) == 6  # 0 + 1 + 1 + 3 + 1
    idx = np.array([4, 0, 2])  # unsorted on purpose
    got = coll.get_seq(idx)
    for k, i in enumerate(idx):
        assert np.array_equal(got[k], expected[i].lfcc_seq.astype(np.float16).astype(np.float32))
        assert np.allclose(coll.scalars[i], expected[i].scalars)
        assert coll.file_id[i] == expected[i].file_id
        assert coll.pad_fraction[i] == pytest.approx(expected[i].pad_fraction)
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["n_windows"] == 6
    assert list(manifest["dropped_files"].values()) == ["too_short"]
    assert not (d / "shards").exists()


def test_rebuild_is_a_no_op_and_stale_is_loud(tmp_path, monkeypatch):
    specs = _corpus(tmp_path)
    d = build_unit("t", specs, None, tmp_path / "cache", workers=1)
    mtime = (d / "seq.npy").stat().st_mtime
    assert build_unit("t", specs, None, tmp_path / "cache", workers=1) == d
    assert (d / "seq.npy").stat().st_mtime == mtime

    monkeypatch.setattr(feature_cache, "_FEATURE_VERSION", {"hash": "different", "components": {}})
    with pytest.raises(StaleCacheError):
        CacheCollection([d])
    with pytest.raises(StaleCacheError):
        build_unit("t", specs, None, tmp_path / "cache", workers=1)
    d2 = build_unit("t", specs, None, tmp_path / "cache", workers=1, rebuild=True)
    assert CacheCollection([d2]).n == 6


def test_unit_dir_changes_with_file_list_channel_and_seed(tmp_path):
    specs = _corpus(tmp_path)
    root = tmp_path / "cache"
    dirs = {feature_cache.unit_dir(root, "t", specs, None, 0), feature_cache.unit_dir(root, "t", specs[:-1], None, 0),
            feature_cache.unit_dir(root, "t", specs, "pstn", 0), feature_cache.unit_dir(root, "t", specs, None, 1)}
    assert len(dirs) == 4


def test_finished_shards_are_reused_on_resume(tmp_path):
    specs = _corpus(tmp_path)
    shard = tmp_path / "shard_00000.npz"
    first = feature_cache._build_shard((str(shard), specs, None, 0))
    mtime = shard.stat().st_mtime
    second = feature_cache._build_shard((str(shard), specs, None, 0))
    assert first == second and shard.stat().st_mtime == mtime


def test_code_fingerprint_ignores_docs_and_comments_but_not_code(tmp_path):
    a = tmp_path / "a.py"
    a.write_text('"""doc"""\ndef f(x):\n    """inner doc"""\n    return x + 1  # comment\n')
    b = tmp_path / "b.py"
    b.write_text('"""other doc"""\n\ndef f(x):\n    # different comment\n    return x + 1\n')
    c = tmp_path / "c.py"
    c.write_text('def f(x):\n    return x + 2\n')
    assert _code_fingerprint(a) == _code_fingerprint(b) != _code_fingerprint(c)
    y1, y2 = tmp_path / "x.yaml", tmp_path / "y.yaml"
    y1.write_text("a: 1  # c\nb: [1, 2]\n")
    y2.write_text("b: [1, 2]\na: 1\n")
    assert _code_fingerprint(y1) == _code_fingerprint(y2)


def test_feature_version_covers_features_dataset_and_telechannel():
    comps = feature_cache.feature_version()["components"]
    keys = " ".join(comps)
    for needle in ("features.py", "dataset.py", "channels.yaml", "pipeline.py", "codec.py", "ffmpeg"):
        assert needle in keys
