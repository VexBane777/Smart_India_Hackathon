"""The v12 windowing contract (dataset.process_file): short-clip padding,
the 1 s floor, tail windows, pad_fraction, pad balancing, and per-file
determinism that doesn't depend on which other files are in the run."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from dataset import (
    DATA_ROOT,
    KEEP_SHORT_PROB_MIN,
    STATUS_BALANCED_OUT,
    STATUS_OK,
    STATUS_SILENT,
    STATUS_TOO_SHORT,
    WINDOW_SAMPLES,
    build_examples,
    compute_pad_policy,
    estimate_windows,
    file_id,
    make_file_specs,
    noise_floor,
    pad_to_window,
    process_file,
    split_by_source,
)
from features import SAMPLE_RATE

SR = SAMPLE_RATE


def _tone(path, seconds, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    sig = amp * np.sin(2 * np.pi * 180 * t) + rng.normal(0, 0.01, len(t))  # above the trim threshold throughout
    sf.write(str(path), sig.astype(np.float32), SR)
    return path


def _spec(path, label=0):
    return make_file_specs(path.parent, label, files=[path])[0]


@pytest.mark.parametrize("seconds,expected", [(0.8, 0), (1.5, 1), (2.9, 1), (3.0, 1), (6.5, 2), (7.2, 3), (9.0, 3)])
def test_window_counts(tmp_path, seconds, expected):
    ws, status, _d = process_file(_spec(_tone(tmp_path / "a.wav", seconds)), None)
    assert len(ws) == expected
    assert status == (STATUS_TOO_SHORT if expected == 0 else STATUS_OK)
    for w in ws:
        assert w.lfcc_seq.shape == (184, 60) and w.lfcc_seq.dtype == np.float32
        assert w.scalars.shape == (6,)


def test_estimate_windows_matches_process_file_for_untrimmed_audio(tmp_path):
    for s in (1.2, 2.5, 3.4, 4.1, 6.0, 7.5):
        ws, _st, d = process_file(_spec(_tone(tmp_path / f"{s}.wav", s)), None)
        padded, full = estimate_windows(d)
        assert padded + full == len(ws)


def test_short_clip_is_padded_with_correct_pad_fraction(tmp_path):
    ws, _st, d = process_file(_spec(_tone(tmp_path / "s.wav", 1.5)), None)
    assert len(ws) == 1
    # 3.1 s total, 1.5 s speech; the first 3 s keep all speech -> pad share ~ (3.0 - 1.5) / 3.0
    assert ws[0].pad_fraction == pytest.approx(0.5, abs=0.04)
    long_ws, _st, _d = process_file(_spec(_tone(tmp_path / "l.wav", 7.2)), None)
    assert all(w.pad_fraction == 0.0 for w in long_ws)


def test_pad_to_window_uses_noise_floor_not_digital_silence():
    rng = np.random.default_rng(0)
    clip = np.concatenate([np.zeros(4000), 0.3 * np.ones(16000)]).astype(np.float32) + rng.normal(0, 0.002, 20000).astype(np.float32)
    out, n_pad = pad_to_window(clip, np.random.default_rng(1))
    assert len(out) == WINDOW_SAMPLES + int(0.1 * SR)
    assert 0 < n_pad <= WINDOW_SAMPLES
    assert noise_floor(clip) == pytest.approx(0.002, rel=0.5)


def test_silent_file_is_reported_not_windowed(tmp_path):
    sf.write(str(tmp_path / "z.wav"), np.zeros(5 * SR, dtype=np.float32), SR)
    ws, status, _d = process_file(_spec(tmp_path / "z.wav"), None)
    assert ws == [] and status == STATUS_SILENT


def test_deterministic_and_independent_of_run_composition(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    for i in range(3):
        _tone(a_dir / f"x{i}.wav", 2.0 + 1.3 * i, seed=i)
        _tone(b_dir / f"y{i}.wav", 3.3, seed=10 + i)
    alone = build_examples(a_dir, [], channel_recipes=[None], workers=1)
    mixed = build_examples([b_dir, a_dir], [], channel_recipes=[None], workers=1)
    mixed_a = [w for w in mixed if w.file_id.startswith(file_id(a_dir))]
    assert len(alone) == len(mixed_a)
    for x, y in zip(sorted(alone, key=lambda w: (w.file_id, w.window_index)),
                    sorted(mixed_a, key=lambda w: (w.file_id, w.window_index))):
        assert np.array_equal(x.lfcc_seq, y.lfcc_seq) and np.array_equal(x.scalars, y.scalars)


def test_same_file_same_trim_across_channels(tmp_path):
    spec = _spec(_tone(tmp_path / "c.wav", 5.0))
    a, _s, d1 = process_file(spec, None)
    b, _s, d2 = process_file(spec, "whatsapp")
    assert d1 == d2 and len(a) == len(b)
    assert not np.allclose(a[0].lfcc_seq, b[0].lfcc_seq)  # the channel really degraded it


def test_convert_prob_turns_long_windows_into_padded_crops(tmp_path):
    spec = replace(_spec(_tone(tmp_path / "v.wav", 9.0)), convert_prob=1.0)
    ws, _st, _d = process_file(spec, None)
    assert len(ws) == 3 and all(0 < w.pad_fraction < 0.7 for w in ws)


def test_keep_short_prob_zero_drops_short_clips_with_a_reason(tmp_path):
    spec = replace(_spec(_tone(tmp_path / "k.wav", 2.0)), keep_short_prob=0.0)
    ws, status, _d = process_file(spec, None)
    assert ws == [] and status == STATUS_BALANCED_OUT


def test_compute_pad_policy():
    conv, keep, frac = compute_pad_policy([6.0] * 10)  # all long: 20 unpadded windows
    assert keep == 1.0 and frac == 0.0 and conv == pytest.approx(0.5)
    conv, keep, frac = compute_pad_policy([2.0] * 3 + [3.5] * 1)  # 3 padded, 1 full
    assert conv == 0.0 and frac == 0.75 and keep == pytest.approx(max(KEEP_SHORT_PROB_MIN, 0.5 * 0.25 / (0.75 * 0.5)))
    conv, keep, _frac = compute_pad_policy([2.0] * 10)  # all short: floor keeps the set alive
    assert keep == KEEP_SHORT_PROB_MIN
    assert compute_pad_policy([0.5])[1] == 1.0  # nothing usable: no policy


def test_file_id_is_data_relative_for_corpus_paths_and_absolute_elsewhere(tmp_path):
    assert file_id(DATA_ROOT / "fake_itw_held" / "1.wav") == "fake_itw_held/1.wav"
    assert file_id(tmp_path / "q.wav").endswith("/q.wav") and ":" in file_id(tmp_path / "q.wav")[:3]


def test_split_by_source_is_stable_under_composition(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    for i in range(12):
        _tone(d / f"f{i}.wav", 3.2, seed=i)
    ws = build_examples(d, [], channel_recipes=[None], workers=1)
    _tr, val = split_by_source(ws, val_fraction=0.5)
    _tr2, val2 = split_by_source(ws[: len(ws) // 2], val_fraction=0.5)
    assert {w.file_id for w in val2} <= {w.file_id for w in val}
