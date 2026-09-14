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

    # Make the recipe-scoped version hash different (this unit is undegraded ->
    # key "none"), so both reads and rebuilds must refuse it until rebuilt.
    feature_cache._VERSION_CACHE["none"] = {"hash": "different", "components": {}}
    with pytest.raises(StaleCacheError):
        CacheCollection([d])
    with pytest.raises(StaleCacheError):
        build_unit("t", specs, None, tmp_path / "cache", workers=1)
    d2 = build_unit("t", specs, None, tmp_path / "cache", workers=1, rebuild=True)
    assert CacheCollection([d2]).n == 6
    # don't let the poisoned key leak: later tests need the real "none" hash
    feature_cache._VERSION_CACHE.pop("none", None)


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
def test_recipe_scoped_version_ignores_unrelated_recipe_changes(monkeypatch):
    """post-v12 plan step 3a: a change to one recipe must not bump the version
    of every unit; only that recipe's own scope."""
    base_whatsapp = feature_cache.feature_version("whatsapp", all_recipes=False)["hash"]
    base_playback = feature_cache.feature_version("playback", all_recipes=False)["hash"]
    assert base_whatsapp != base_playback

    # simulate a change that touches ONLY the playback recipe (snr_db 12 -> 13)
    cfg = feature_cache._channels_config()
    cfg["recipes"]["playback"]["noise"]["snr_db"] = 13
    cfg["recipes"]["playback"]["bandlimit"]["high_hz"] = 4000
    monkeypatch.setattr(feature_cache, "_channels_config", lambda: cfg)
    feature_cache._VERSION_CACHE.clear()

    assert feature_cache.feature_version("whatsapp", all_recipes=False)["hash"] == base_whatsapp
    assert feature_cache.feature_version("playback", all_recipes=False)["hash"] != base_playback


def test_scoped_components_carry_the_recipe_but_not_other_recipes():
    comps_wh = feature_cache.feature_version("whatsapp", all_recipes=False)["components"]
    assert "channels.yaml[whatsapp]" in comps_wh
    assert "channels.yaml[playback]" not in comps_wh and "channels.yaml[gsm_2g]" not in comps_wh
    # undegraded carries no channel yaml scope at all
    comps_none = feature_cache.feature_version(None, all_recipes=False)["components"]
    assert all("channels.yaml[" not in k for k in comps_none)


def test_revalidate_accepts_unchanged_unit(tmp_path):
    specs = _corpus(tmp_path)
    d = build_unit("t", specs, None, tmp_path / "cache", workers=1)
    res = feature_cache.revalidate(d, max_sample_files=3)
    assert res["equivalent"] is True
    assert "restamped_to" in res
    m = json.loads((d / "manifest.json").read_text())
    assert m["feature_version"] == feature_cache.feature_version(None, all_recipes=False)["hash"]
    assert "max_abs_diff" in m["revalidated_from"]


def test_revalidate_detects_planted_change(tmp_path):
    """A behavioral change to feature extraction must be detected by comparison,
    not assumed equivalent."""
    import dataset as ds

    specs = _corpus(tmp_path)
    d = build_unit("t", specs, None, tmp_path / "cache", workers=1)
    assert feature_cache.revalidate(d, max_sample_files=3)["equivalent"] is True

    orig = ds.extract_lfcc_sequence

    def shifted(pcm):
        return orig(pcm) + np.float32(5.0)

    ds.extract_lfcc_sequence = shifted
    try:
        res = feature_cache.revalidate(d, max_sample_files=3)
    finally:
        ds.extract_lfcc_sequence = orig
    assert res["equivalent"] is False
    assert any(not v.get("equivalent") for v in res["per_file"].values())
    # untouched by a failed revalidate: still flagged stale when the version no
    # longer matches (here it still matches, but the file was NOT re-stamped)
    assert "restamped_to" not in res
