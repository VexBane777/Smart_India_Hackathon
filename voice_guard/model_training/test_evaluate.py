"""evaluate.py's confound gates and per-model metrics on synthetic scores."""
from __future__ import annotations

import numpy as np

from evaluate import RATE_ABS_FLOOR, confound_table, evaluate_model


def _synthetic(n=4000, leak=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    scal = rng.normal(0, 1, (n, 6)).astype(np.float32)
    pad = np.where(rng.random(n) < 0.4, rng.uniform(0, 0.6, n), 0.0).astype(np.float32)
    logit = 3 * (y - 0.5) + leak * scal[:, 0] + rng.normal(0, 1, n)
    return y, scal, pad, 1 / (1 + np.exp(-logit))


def test_confound_passes_when_score_is_independent_of_features():
    y, scal, pad, p = _synthetic(leak=0.0)
    out = confound_table(p, y, scal, pad, np.ones(len(y), bool), 0.5)
    assert out["passed"], out["failed"]
    assert {r["feature"] for r in out["rows"]} >= {"pauseRatio", "hnr_db", "pad_fraction"}
    assert {r["class"] for r in out["rows"]} == {"real", "fake"}


def test_confound_fails_when_score_tracks_pause_ratio():
    y, scal, pad, p = _synthetic(leak=2.0)
    out = confound_table(p, y, scal, pad, np.ones(len(y), bool), 0.5)
    assert not out["passed"]
    assert "real:pauseRatio" in out["failed"] and "fake:pauseRatio" in out["failed"]


def test_rate_gate_noise_floor():
    # both halves have near-zero FPR: ratio could be huge, absolute difference is tiny -> passes
    rng = np.random.default_rng(1)
    n = 2000
    y = np.zeros(n, int)
    scal = rng.normal(0, 1, (n, 6))
    p = np.full(n, 0.1)
    p[:3] = 0.9  # 3 false positives total
    out = confound_table(p, y, scal, np.zeros(n), np.ones(n, bool), 0.5)
    for r in out["rows"]:
        if r["feature"] != "pad_fraction":
            assert abs(r["rate_high"] - r["rate_low"]) <= RATE_ABS_FLOOR and r["gate_rate"]


class _Coll:
    def __init__(self, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        sets = np.array(["itw_real", "itw_fake", "noiseaug_real_en", "mlaad_fake"])
        n_files = n // 3
        file_set = rng.integers(0, 4, n_files)
        f = np.repeat(np.arange(n_files), 3)
        self.source_set = sets[file_set[f]]
        self.label = np.isin(self.source_set, ["itw_fake", "mlaad_fake"]).astype(np.int64)
        self.file_id = np.array([f"file{i}" for i in f])
        self.channel = rng.choice(["none", "whatsapp", "pstn"], len(f))
        self.scalars = rng.normal(0, 1, (len(f), 6)).astype(np.float32)
        self.pad_fraction = np.where(rng.random(len(f)) < 0.3, 0.4, 0.0).astype(np.float32)
        self.n = len(f)


def test_evaluate_model_structure_and_headline_is_phone_pooled_core():
    c = _Coll()
    rng = np.random.default_rng(2)
    p = 1 / (1 + np.exp(-(3 * (c.label - 0.5) + rng.normal(0, 1, c.n))))
    att = np.stack([np.full(c.n, 0.8), np.full(c.n, 0.2)], 1)
    res = evaluate_model(p, att, c, 0.5, [None, "whatsapp", "pstn"], n_boot=20)
    h = res["headline"]
    core_phone = np.isin(c.source_set, ["itw_real", "itw_fake", "noiseaug_real_en"]) & (c.channel != "none")
    assert h["n_windows"] == int(core_phone.sum())
    assert h["ci_lo"] <= h["eer"] <= h["ci_hi"]
    assert set(res["per_channel"]) == {"none", "whatsapp", "pstn"}
    assert res["channel_groups"]["unseen_phone"]["channels"] == ["pstn"]
    assert res["attack_mlaad"]["tts_share"] == 1.0
    assert "core_plus_mlaad" in res and res["per_set"]["itw_real"]["label"] == 0
