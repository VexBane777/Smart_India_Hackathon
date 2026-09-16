"""train_seq_cnn.py: masked attack loss, LR schedule, EMA, the channel
policy at the CLI, and an end-to-end cache-backed smoke run with ONNX
export checked through onnxruntime."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from train_seq_cnn import (EMA, attack_type_weight, compute_masked_attack_type_loss,
                           focal_cross_entropy, lr_at, mixup_batch, sample_mixup_lambda,
                           soft_target_cross_entropy, specaugment)

HERE = Path(__file__).parent


def test_masked_loss_ignores_ignore_index_examples():
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    targets = torch.tensor([0, 1, -100, -100])
    assert compute_masked_attack_type_loss(logits, targets).item() < 0.01


def test_masked_loss_all_ignored_is_nan_and_caller_must_guard():
    loss = compute_masked_attack_type_loss(torch.zeros(3, 2), torch.tensor([-100, -100, -100]))
    assert torch.isnan(loss)


def test_lr_schedule_warmup_then_cosine():
    total, warm, base = 1000, 100, 1e-3
    assert lr_at(0, total, warm, base) == base / 100
    assert lr_at(99, total, warm, base) == base
    assert lr_at(999, total, warm, base) < 2e-5
    lrs = [lr_at(s, total, warm, base) for s in range(100, 1000)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_ema_tracks_parameters():
    m = torch.nn.Linear(2, 1)
    ema = EMA(m, decay=0.9)
    with torch.no_grad():
        m.weight.fill_(1.0)
    for _ in range(200):
        ema.update(m)
    assert torch.allclose(ema.model.weight, m.weight, atol=1e-3)


def _corpus(tmp_path):
    real_dir, fake_dir = tmp_path / "real", tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    rng = np.random.default_rng(0)
    sr = 16000
    for i in range(6):
        for d, f0 in ((real_dir, 150), (fake_dir, 220)):
            secs = 2.0 + 0.5 * i
            t = np.arange(int(secs * sr)) / sr
            sf.write(str(d / f"x{i}.wav"), (0.3 * np.sin(2 * np.pi * f0 * t) + rng.normal(0, 0.02, len(t))).astype(np.float32), sr)
    return real_dir, fake_dir


def test_clean_only_training_is_refused(tmp_path):
    real_dir, fake_dir = _corpus(tmp_path)
    r = subprocess.run([sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
                        "--channels", "none", "--out", str(tmp_path / "run"), "--cache-root", str(tmp_path / "c")],
                       cwd=HERE, capture_output=True, text=True)
    assert r.returncode != 0 and "CleanOnlyEvalError" in r.stderr
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").iterdir())


def _make_file_specs(tmp_path, label, source_set, n=6):
    """Minimal synthetic FileSpec list for testing corpus helpers that don't
    touch the real DATA_ROOT / cache. Creates WAV files under tmp_path."""
    import soundfile as sf
    from dataset import FileSpec

    d = tmp_path / ("real" if label == 0 else "fake")
    d.mkdir(parents=True, exist_ok=True)
    sr = 16000
    specs = []
    for i in range(n):
        secs = 2.0 + 0.5 * i
        t = np.arange(int(secs * sr)) / sr
        p = d / f"x{i}.wav"
        sf.write(str(p), (0.3 * np.sin(2 * np.pi * 220 * t) + np.random.default_rng(i).normal(0, 0.02, len(t))).astype(np.float32), sr)
        specs.append(FileSpec(path=str(p), file_id=f"x{i}", label=label, source_set=source_set))
    return specs


def test_playback_fraction_full_fold(tmp_path):
    """playback_fraction=1.0 puts every file into the playback unit; 0.0
    skips it entirely. v13 default 0.5 is ~half by stable hash."""
    from corpus import training_units

    specs_by_set = {"train": _make_file_specs(tmp_path, 0, "train", n=8)}
    # 1.0 = every file gets a playback rendition
    full = training_units(specs_by_set, seed=123, playback_fraction=1.0)
    pb_units = [u for u in full if u[2] == "playback"]
    assert len(pb_units) == 1
    assert len(pb_units[0][1]) == 8  # all 8 files
    # 0.0 = no playback unit at all
    none = training_units(specs_by_set, seed=123, playback_fraction=0.0)
    assert not any(u[2] == "playback" for u in none)
    # 0.5 default picks a deterministic subset
    half = training_units(specs_by_set, seed=123, playback_fraction=0.5)
    half_pb = [u for u in half if u[2] == "playback"]
    assert 0 < len(half_pb[0][1]) < 8
    # deterministic: same seed -> same subset
    half2 = training_units(specs_by_set, seed=123, playback_fraction=0.5)
    assert half2 == half


def test_room_renditions_training_only(tmp_path):
    """Room renditions are added when room_fraction > 0 and are rejected as
    eval channels; playback_fraction and phone channels stay independent."""
    from corpus import training_units, TRAIN_ROOM_CHANNELS
    from eval_protocol import (
        resolve_channels,
        room_subset,
        TRAIN_ROOM_CHANNELS as RC,
        CleanOnlyEvalError,
    )

    specs_by_set = {"train": _make_file_specs(tmp_path, 0, "train", n=12)}
    phone = tuple(c for c in ("whatsapp", "volte") if c is not None)

    # With room_fraction=1.0, every file gets a room rendition through one of
    # the room channels, split by stable hash across the three recipes.
    units = training_units(specs_by_set, seed=7, channels=phone, room_fraction=1.0)
    room_units = [u for u in units if u[2] in RC]
    recipes_seen = {u[2] for u in room_units}
    # Each room recipe should get at least some files (deterministic hash).
    assert recipes_seen == set(RC)
    # The room units partition the files (each file in exactly one room recipe).
    room_file_ids = [f.file_id for u in room_units for f in u[1]]
    assert len(room_file_ids) == len(set(room_file_ids)) == 12

    # Room channels are NOT valid eval channels.
    for ch in RC:
        with pytest.raises(ValueError, match="training-only room/loudspeaker"):
            resolve_channels([ch], "phone", "eval")
    # A plain eval with phone channels does not include room channels.
    resolved = resolve_channels([None, "whatsapp", "playback"], "phone", "eval")
    assert not any(c in RC for c in resolved)
    # room_subset extracts only room renditions from a list.
    assert room_subset([None, "whatsapp_room", "playback", "volte_room"]) == ["whatsapp_room", "volte_room"]
    assert room_subset([None, "whatsapp", "playback"]) == []

    # Room rendering is independent of phone rendering (tagged stable hash).
    from corpus import train_phone_channel_from, room_channel_from
    fid = "x42"
    for seed in (0, 1, 99):
        phone_ch = train_phone_channel_from(fid, phone, seed)
        room_ch = room_channel_from(fid, RC, seed)
        assert phone_ch in phone and room_ch in RC
        # Same seed can route the same file to different phone vs room recipes --
        # they're independent dimensions, not correlated.
    # Different seeds for phone vs room on the same file: independence guaranteed
    # by the different tags ("train_channel" vs "room_channel").
    p0 = train_phone_channel_from(fid, phone, 0)
    r1 = room_channel_from(fid, RC, 1)
    assert p0 in phone and r1 in RC



def test_train_seq_cnn_end_to_end_smoke(tmp_path):
    import onnxruntime as ort

    real_dir, fake_dir = _corpus(tmp_path)
    out_dir = tmp_path / "run"
    r = subprocess.run(
        [sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
         "--channels", "none", "whatsapp", "--out", str(out_dir), "--cache-root", str(tmp_path / "cache"),
         "--epochs", "2", "--workers", "1", "--batch-size", "8"],
        cwd=HERE, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    for f in ("model.pt", "model.onnx", "norm_stats.npz", "history.json", "train_config.json",
              "checkpoints/epoch_01.pt", "checkpoints/epoch_02.pt"):
        assert (out_dir / f).exists(), f
    assert str(np.load(out_dir / "norm_stats.npz")["arch"]) == "seqtcn_v2"
    sess = ort.InferenceSession(str(out_dir / "model.onnx"))
    outs = sess.run(None, {"lfcc_sequence": np.zeros((1, 184, 60), np.float32), "scalars": np.zeros((1, 6), np.float32)})
    assert [o.shape for o in outs] == [(1, 2), (1, 2)]


def test_lr_holds_at_min_past_cosine_endpoint():
    # lr_at clamps progress to 1.0 at/after the cosine endpoint => flat hold at min_lr.
    # This is what --min-lr-epochs relies on: the call site ends the cosine early (a
    # smaller total_steps) and trailing epochs step past it, where lr_at returns min_lr.
    total, warm, base = 1000, 100, 1e-3
    on_end = lr_at(total, total, warm, base)
    past_end = lr_at(total + 500, total, warm, base)
    assert math.isclose(on_end, 1e-5, abs_tol=1e-9)
    assert math.isclose(past_end, 1e-5, abs_tol=1e-9)
    assert math.isclose(on_end, past_end, abs_tol=1e-12)


def test_train_smoke_with_min_lr_hold_and_grad_clip(tmp_path):
    import onnxruntime as ort

    real_dir, fake_dir = _corpus(tmp_path)
    out_dir = tmp_path / "run_hold"
    r = subprocess.run(
        [sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
         "--channels", "none", "whatsapp", "--out", str(out_dir), "--cache-root",
         str(tmp_path / "cache"), "--epochs", "3", "--min-lr-epochs", "1", "--grad-clip", "1.0",
         "--workers", "1", "--batch-size", "8"],
        cwd=HERE, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    hist = json.loads((out_dir / "history.json").read_text())
    assert len(hist) == 3
    cfg = json.loads((out_dir / "train_config.json").read_text())
    assert cfg["min_lr_epochs"] == 1.0
    assert cfg["grad_clip"] == 1.0
    sess = ort.InferenceSession(str(out_dir / "model.onnx"))
    outs = sess.run(None, {"lfcc_sequence": np.zeros((1, 184, 60), np.float32),
                           "scalars": np.zeros((1, 6), np.float32)})
    assert [o.shape for o in outs] == [(1, 2), (1, 2)]


# ---------------------------------------------------------------------------
# Post-v13 rework levers (docs/2026-09-training-improvement-plan.md axes F/D3).
# Every lever must be exact-default OFF: gamma=0 / warmup=0 / mixup=0 /
# specaugment=0 must reproduce the v12/v13 loss and pipeline bit-for-bit.
# ---------------------------------------------------------------------------
def test_attack_type_weight_default_is_constant_and_ramps_when_asked():
    base = 0.5
    for ep in range(5):
        assert attack_type_weight(ep, 0.0, base) == base      # legacy behavior
    assert attack_type_weight(0, 2.0, base) == base * 0.5     # epoch 1 of 2
    assert attack_type_weight(1, 2.0, base) == base           # epoch 2 of 2
    assert attack_type_weight(9, 2.0, base) == base           # stays clamped


def test_focal_gamma_zero_matches_cross_entropy_loss():
    torch.manual_seed(0)
    logits = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))
    w = torch.tensor([1.5, 0.75])
    want = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)(logits, targets)
    got = focal_cross_entropy(logits, targets, weight=w, gamma=0.0, label_smoothing=0.05)
    assert torch.allclose(got, want, atol=1e-6), (got, want)
    assert torch.allclose(focal_cross_entropy(logits, targets),
                          torch.nn.CrossEntropyLoss()(logits, targets), atol=1e-6)


def test_focal_gamma_downweights_confident_examples_more_than_hard_ones():
    targets = torch.tensor([0, 1])
    easy = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])            # p_t ~ 1
    hard = torch.tensor([[0.01, -0.01], [-0.01, 0.01]])        # p_t ~ 0.5

    def suppression(x):
        """focal/CE = (1 - p_t)^gamma, the per-sample down-weighting factor."""
        return float(focal_cross_entropy(x, targets, gamma=2.0) / focal_cross_entropy(x, targets, gamma=0.0))

    assert suppression(easy) < 1e-8            # (1 - p_t)^2 with p_t ~ 1
    assert 0.2 < suppression(hard) < 0.3       # (1 - 0.5)^2 = 0.25
    assert suppression(easy) < suppression(hard)  # easy examples lose almost all their weight


def test_soft_target_cross_entropy_matches_ce_for_one_hot_targets():
    torch.manual_seed(1)
    logits = torch.randn(16, 2)
    targets = torch.randint(0, 2, (16,))
    onehot = torch.nn.functional.one_hot(targets, 2).float()
    assert torch.allclose(soft_target_cross_entropy(logits, onehot),
                          torch.nn.CrossEntropyLoss()(logits, targets), atol=1e-6)
    assert torch.allclose(soft_target_cross_entropy(logits, onehot, label_smoothing=0.05),
                          torch.nn.CrossEntropyLoss(label_smoothing=0.05)(logits, targets), atol=1e-6)


def test_mixup_batch_mixes_inputs_and_masks_ignored_attack_labels():
    seq = torch.zeros(4, 3, 2)
    seq[3] = 1.0
    scal = torch.arange(8.0).reshape(4, 2)
    label = torch.tensor([0, 1, 0, 1])
    attack = torch.tensor([0, 1, -100, 1])  # window 2 carries no attack label
    perm = torch.tensor([3, 2, 1, 0])

    seq_mix, scal_mix, soft_rf, soft_attack, mask = mixup_batch(seq, scal, label, attack, 0.25, perm=perm)

    assert torch.allclose(seq_mix[0], 0.25 * seq[0] + 0.75 * seq[3])
    assert torch.allclose(scal_mix[0], 0.25 * scal[0] + 0.75 * scal[3])
    assert torch.allclose(soft_rf.sum(dim=1), torch.ones(4))
    # label 0 mixed with label[perm][0] = label[3] = 1
    assert torch.allclose(soft_rf[0], torch.tensor([0.25, 0.75]))
    # only windows whose partner ALSO has a label are usable for the attack head
    assert mask.tolist() == [True, False, False, True]
    assert torch.allclose(soft_attack[0], torch.tensor([0.25, 0.75]))
    assert torch.allclose(soft_attack[3], torch.tensor([0.75, 0.25]))


def test_mixup_lambda_is_deterministic_per_seed_and_in_range():
    a = sample_mixup_lambda(0.4, np.random.default_rng(7))
    b = sample_mixup_lambda(0.4, np.random.default_rng(7))
    assert a == b and 0.0 < a < 1.0


def test_specaugment_zeroes_masks_and_is_a_noop_when_off():
    seq = torch.ones(3, 184, 60)
    assert specaugment(seq, 0, 8, 0, 8) is seq             # default: no-op, same object
    out = specaugment(seq, 2, 6, 1, 5, rng=np.random.default_rng(3))
    assert out.shape == seq.shape
    assert torch.equal(seq, torch.ones(3, 184, 60))        # input untouched
    assert (out == 0).sum() > 0                            # something was masked
    assert (out.flatten(1).sum(dim=1) < 184 * 60).all()    # per-sample masks landed

def test_train_smoke_with_rework_levers(tmp_path):
    """End-to-end: wider + regularized model, focal loss, attack-head warmup,
    Mixup, SpecAugment, CMVN/SE/stochastic depth, LR hold, grad clip -- one run,
    asserting the ONNX contract still holds and provenance is written."""
    import onnxruntime as ort

    real_dir, fake_dir = _corpus(tmp_path)
    out_dir = tmp_path / "run_rework"
    r = subprocess.run(
        [sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
         "--channels", "none", "whatsapp", "--out", str(out_dir), "--cache-root", str(tmp_path / "cache"),
         "--epochs", "2", "--min-lr-epochs", "1", "--grad-clip", "1.0", "--workers", "1", "--batch-size", "8",
         "--model-channels", "32", "--model-dilations", "1,2", "--model-hidden", "32", "--model-dropout", "0.2",
         "--model-stochastic-depth", "0.1", "--model-cmvn", "--model-se",
         "--focal-gamma", "2.0", "--attack-type-warmup-epochs", "2.0",
         "--mixup-alpha", "0.4", "--specaugment-freq-masks", "2", "--specaugment-time-masks", "1"],
        cwd=HERE, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    cfg = json.loads((out_dir / "train_config.json").read_text())
    for key, want in (("focal_gamma", 2.0), ("attack_type_warmup_epochs", 2.0), ("mixup_alpha", 0.4),
                      ("specaugment_freq_masks", 2), ("specaugment_time_masks", 1),
                      ("model_channels", 32), ("model_hidden", 32), ("model_stochastic_depth", 0.1)):
        assert cfg[key] == want, key
    assert cfg["model_cmvn"] is True and cfg["model_se"] is True
    ns = dict(np.load(out_dir / "norm_stats.npz"))
    assert int(ns["channels"]) == 32 and int(ns["hidden"]) == 32
    assert tuple(int(d) for d in ns["dilations"]) == (1, 2)
    assert float(ns["stochastic_depth"]) == 0.1
    assert bool(ns["cmvn"]) and bool(ns["se"])
    hist = json.loads((out_dir / "history.json").read_text())
    assert len(hist) == 2
    # the 2-epoch warmup ramp is visible in the per-epoch log: half weight, then full
    assert hist[0]["attack_weight"] == cfg["attack_type_loss_weight"] / 2
    assert hist[1]["attack_weight"] == cfg["attack_type_loss_weight"]
    # the rebuilt model must match the checkpoint (capacity round-trip through norm_stats)
    from model import build_model_from_norm_stats

    model = build_model_from_norm_stats(ns)
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True))
    assert sum(p.numel() for p in model.parameters()) == cfg["n_params"]
    sess = ort.InferenceSession(str(out_dir / "model.onnx"))
    outs = sess.run(None, {"lfcc_sequence": np.zeros((1, 184, 60), np.float32),
                           "scalars": np.zeros((1, 6), np.float32)})
    assert [o.shape for o in outs] == [(1, 2), (1, 2)]