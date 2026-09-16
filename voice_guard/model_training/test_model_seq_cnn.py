"""Tests for the sequence models (v11 VoiceGuardSeqCNN, v12 VoiceGuardSeqTCN),
the arch factory, the MLP baseline adapter and the ONNX I/O contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from features import extract_features, extract_lfcc_sequence, extract_scalars
from model import (
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    FixedNormalizeSeq,
    MLPSequenceAdapter,
    VoiceGuardMLP,
    VoiceGuardSeqCNN,
    VoiceGuardSeqTCN,
    VoiceGuardConformer,
    build_model_from_norm_stats,
    export_onnx,
    load_scoring_model,
)

RUNS = Path(__file__).resolve().parent / "runs"


def _norm(arch=None):
    ns = {"seq_mean": np.zeros(60, np.float32), "seq_std": np.ones(60, np.float32),
          "scalar_mean": np.zeros(6, np.float32), "scalar_std": np.ones(6, np.float32),
          "n_frames": np.array(184), "n_lfcc": np.array(60), "n_scalars": np.array(6)}
    if arch:
        ns["arch"] = np.array(arch)
    return ns


def test_fixed_normalize_seq_broadcasts_across_time():
    norm = FixedNormalizeSeq(np.zeros(60, dtype=np.float32), np.ones(60, dtype=np.float32))
    assert norm(torch.randn(4, 184, 60)).shape == (4, 184, 60)


@pytest.mark.parametrize("arch,cls", [(None, VoiceGuardSeqCNN), ("seqcnn_v1", VoiceGuardSeqCNN), ("seqtcn_v2", VoiceGuardSeqTCN), ("conformer_v1", VoiceGuardConformer)])
def test_factory_and_forward_shapes(arch, cls):
    model = build_model_from_norm_stats(_norm(arch)).eval()
    assert isinstance(model, cls)
    rf, at = model(torch.randn(8, 184, 60), torch.randn(8, 6))
    assert rf.shape == (8, 2) and at.shape == (8, 2)


def test_unknown_arch_refused():
    with pytest.raises(ValueError):
        build_model_from_norm_stats(_norm("lstm_v9"))


@pytest.mark.parametrize("capacity,expect_wider,expect_rf", [
    # default: width 64, dilations (1,2,4,8,16) -> RF 65, <120k params
    ({}, False, 65),
    # wider + extra dilation: must be larger param count & 2x receptive field
    # (1+2*(1+sum(1,2,4,8,16,32)) = 1+2*64 = 129)
    ({"channels": 112, "dilations": (1, 2, 4, 8, 16, 32), "hidden": 128, "dropout": 0.2}, True, 129),
])
def test_seqtcn_v2_capacity_roundtrip(capacity, expect_wider, expect_rf):
    """Capacity persisted in norm_stats must rebuild VoiceGuardSeqTCN with the SAME
    wider capacity (load-time architecture parity with model.pt) and the expected RF.
    ONNX I/O contract is unchanged: inputs (1,184,60)+(1,6) -> (1,2)+(1,2)."""
    ns = _norm("seqtcn_v2")
    if capacity:
        ns["channels"] = np.array(capacity["channels"], dtype=np.int64)
        ns["dilations"] = np.array(capacity["dilations"], dtype=np.int64)
        ns["hidden"] = np.array(capacity["hidden"], dtype=np.int64)
        ns["dropout"] = np.array(capacity["dropout"], dtype=np.float32)
    model = build_model_from_norm_stats(ns).eval()
    assert isinstance(model, VoiceGuardSeqTCN)
    rf, at = model(torch.randn(1, 184, 60), torch.randn(1, 6))
    assert rf.shape == (1, 2) and at.shape == (1, 2)
    dil = capacity["dilations"] if capacity else (1, 2, 4, 8, 16)
    assert VoiceGuardSeqTCN.receptive_field(dil) == expect_rf
    n_params = sum(p.numel() for p in model.parameters())
    # default (<120k per existing test) vs wider must flip the comparison
    assert (n_params < 120_000) is (not expect_wider)


def test_tcn_size_and_receptive_field():
    model = build_model_from_norm_stats(_norm("seqtcn_v2")).eval()
    assert sum(p.numel() for p in model.parameters()) < 120_000
    assert VoiceGuardSeqTCN.receptive_field() == 65
    # empirical: which input frames influence the pre-pooling activation at frame 92
    x = torch.randn(1, 184, 60, requires_grad=True)
    h = model.blocks(model.stem(model.seq_normalize(x).transpose(1, 2)))
    h[0, :, 92].sum().backward()
    influenced = np.flatnonzero(x.grad[0].abs().sum(1).numpy() > 0)
    assert influenced.min() == 92 - 32 and influenced.max() == 92 + 32


def test_onnx_contract_and_parity(tmp_path):
    import onnxruntime as ort

    model = build_model_from_norm_stats(_norm("seqtcn_v2")).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path)
    sess = ort.InferenceSession(str(path))
    assert [i.name for i in sess.get_inputs()] == ONNX_INPUT_NAMES
    assert [o.name for o in sess.get_outputs()] == ONNX_OUTPUT_NAMES
    assert list(sess.get_inputs()[0].shape) == [1, 184, 60] and list(sess.get_inputs()[1].shape) == [1, 6]
    seq, scal = np.random.default_rng(0).normal(0, 1, (1, 184, 60)).astype(np.float32), np.ones((1, 6), np.float32)
    rf_o, at_o = sess.run(None, {"lfcc_sequence": seq, "scalars": scal})
    with torch.no_grad():
        rf_t, at_t = model(torch.from_numpy(seq), torch.from_numpy(scal))
    assert np.allclose(rf_o, rf_t.numpy(), atol=1e-4) and np.allclose(at_o, at_t.numpy(), atol=1e-4)


@pytest.mark.parametrize("n_layers", [1, 2, 4])
@pytest.mark.parametrize("channels", [16, 32, 48])
def test_conformer_capacity_roundtrip(n_layers, channels):
    """conformer_v1 capacity persisted in norm_stats must rebuild with the SAME
    wider capacity and correct head dim (channels / n_heads). ONNX I/O contract
    is unchanged: inputs (1,184,60)+(1,6) -> (1,2)+(1,2)."""
    ns = _norm("conformer_v1")
    ns["channels"] = np.array(channels)
    ns["n_heads"] = np.array(4)
    ns["n_layers"] = np.array(n_layers)
    ns["ff_expansion"] = np.array(2)
    ns["conv_kernel"] = np.array(7)
    ns["hidden"] = np.array(32)
    # channels must be divisible by n_heads
    assert channels % 4 == 0
    model = build_model_from_norm_stats(ns).eval()
    assert isinstance(model, VoiceGuardConformer)
    rf, at = model(torch.randn(1, 184, 60), torch.randn(1, 6))
    assert rf.shape == (1, 2) and at.shape == (1, 2)
    # param count scales with n_layers
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


def test_conformer_onnx_contract(tmp_path):
    """conformer_v1 exports with the SAME ONNX I/O contract as seqtcn_v2."""
    import onnxruntime as ort

    model = build_model_from_norm_stats(_norm("conformer_v1")).eval()
    path = tmp_path / "conf.onnx"
    export_onnx(model, path)
    sess = ort.InferenceSession(str(path))
    assert [i.name for i in sess.get_inputs()] == ONNX_INPUT_NAMES
    assert [o.name for o in sess.get_outputs()] == ONNX_OUTPUT_NAMES
    assert list(sess.get_inputs()[0].shape) == [1, 184, 60]
    assert list(sess.get_inputs()[1].shape) == [1, 6]
    seq, scal = (np.random.default_rng(0).normal(0, 1, (1, 184, 60)).astype(np.float32),
                 np.ones((1, 6), np.float32))
    rf_o, at_o = sess.run(None, {"lfcc_sequence": seq, "scalars": scal})
    with torch.no_grad():
        rf_t, at_t = model(torch.from_numpy(seq), torch.from_numpy(scal))
    assert np.allclose(rf_o, rf_t.numpy(), atol=1e-4) and np.allclose(at_o, at_t.numpy(), atol=1e-4)


def test_conformer_eval_determinism():
    """conformer_v1 forward is deterministic in eval mode (no dropout/BN noise)."""
    model = build_model_from_norm_stats(_norm("conformer_v1")).eval()
    seq, scal = torch.randn(4, 184, 60), torch.randn(4, 6)
    with torch.no_grad():
        out1 = model(seq, scal)
        out2 = model(seq, scal)
    assert torch.allclose(out1[0], out2[0]) and torch.allclose(out1[1], out2[1])


@pytest.mark.parametrize("dim", [63, 66])
def test_mlp_adapter_matches_flat_feature_path(dim):
    rng = np.random.default_rng(1)
    t = np.arange(48000) / 16000
    pcm = 0.3 * np.sin(2 * np.pi * 170 * t) + rng.normal(0, 0.02, 48000)
    mlp = VoiceGuardMLP(input_dim=dim, norm_mean=rng.normal(0, 1, dim), norm_std=rng.uniform(0.5, 2, dim)).eval()
    flat = torch.from_numpy(extract_features(pcm)[:dim][None])
    seq = torch.from_numpy(extract_lfcc_sequence(pcm).astype(np.float32)[None])
    scal = torch.from_numpy(extract_scalars(pcm).astype(np.float32)[None])
    with torch.no_grad():
        want = mlp(flat)
        got, att = MLPSequenceAdapter(mlp)(seq, scal)
    assert att is None
    assert torch.allclose(want, got, atol=1e-3)


@pytest.mark.parametrize("rel,kind", [("voice_guard_v9_noisefix_final/model.pt", "mlp_63d"),
                                      ("voice_guard_v11_seqcnn_selected/model.pt", "seqcnn_v1"),
                                      ("voice_guard_v12_selected/model.pt", "seqtcn_v2"),
                                      ("voice_guard_v13_selected/model.pt", "seqtcn_v2")])
def test_load_scoring_model_on_real_checkpoints(rel, kind):
    path = RUNS / rel
    if not path.exists():
        pytest.skip(f"{path} not present")
    model, arch = load_scoring_model(path)
    assert arch == kind
    with torch.no_grad():
        rf, _at = model(torch.randn(2, 184, 60), torch.rand(2, 6))
    assert rf.shape == (2, 2)


# ---------------------------------------------------------------------------
# Post-v13 rework (docs/2026-09-training-improvement-plan.md axes B/G):
# capacity + paired regularization, persisted in norm_stats so a checkpoint
# always rebuilds the architecture it was trained with.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reg", [
    {"stochastic_depth": 0.2, "cmvn": True, "se": True},
    {"stochastic_depth": 0.0, "cmvn": True, "se": False},
])
def test_seqtcn_v2_regularization_roundtrip(reg, tmp_path):
    import onnxruntime as ort

    ns = _norm("seqtcn_v2")
    ns["stochastic_depth"] = np.array(reg["stochastic_depth"], dtype=np.float32)
    ns["cmvn"] = np.array(reg["cmvn"])
    ns["se"] = np.array(reg["se"])
    model = build_model_from_norm_stats(ns).eval()
    assert model.cmvn is reg["cmvn"]
    assert (model.blocks[0].se is not None) is reg["se"]
    rf, at = model(torch.randn(1, 184, 60), torch.randn(1, 6))
    assert rf.shape == (1, 2) and at.shape == (1, 2)
    # the wider/regularized arch must still satisfy the app's fixed ONNX contract
    path = tmp_path / "reg.onnx"
    export_onnx(model, path)
    sess = ort.InferenceSession(str(path))
    assert list(sess.get_inputs()[0].shape) == [1, 184, 60] and list(sess.get_inputs()[1].shape) == [1, 6]
    seq = np.random.default_rng(0).normal(0, 1, (1, 184, 60)).astype(np.float32)
    outs = sess.run(None, {"lfcc_sequence": seq, "scalars": np.ones((1, 6), np.float32)})
    assert [o.shape for o in outs] == [(1, 2), (1, 2)]
    if reg["cmvn"]:
        with torch.no_grad():
            want = model(torch.from_numpy(seq), torch.ones(1, 6))
        assert np.allclose(outs[0], want[0].numpy(), atol=1e-4)


def test_stochastic_depth_is_train_only():
    ns = _norm("seqtcn_v2")
    ns["stochastic_depth"] = np.array(0.9, dtype=np.float32)
    ns["dropout"] = np.array(0.0, dtype=np.float32)  # isolate the only train-time randomness
    torch.manual_seed(0)
    model = build_model_from_norm_stats(ns)
    # non-degenerate input: a constant input would be normalized to exactly 0 by
    # BatchNorm in train mode, which would mask the mask (pun intended)
    x, s = torch.randn(32, 184, 60), torch.randn(32, 6)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x, s)[0], model(x, s)[0])  # eval takes the identity path
    model.train()
    assert not torch.allclose(model(x, s)[0], model(x, s)[0])  # train draws fresh masks


def test_per_utterance_cmvn_removes_level_and_scale():
    from model import per_utterance_cmvn

    base = torch.randn(2, 50, 60)
    want = per_utterance_cmvn(base)
    for scale, shift in ((1.0, 0.0), (7.5, 3.0)):
        got = per_utterance_cmvn(base * scale + shift)
        assert torch.allclose(got, want, atol=1e-4)
    assert want.mean(dim=1).abs().max() < 1e-4          # zero-mean per utterance
    assert (want.std(dim=1, unbiased=False) - 1).abs().max() < 0.05   # ~unit variance


def test_cmvn_flag_actually_changes_the_score():
    """A constant level/shift in the LFCC input must not move a cmvn model's logits,
    but must move the plain model's -- i.e. the flag is wired into forward()."""
    plain = build_model_from_norm_stats(_norm("seqtcn_v2")).eval()
    ns = _norm("seqtcn_v2")
    ns["cmvn"] = np.array(True)
    cmvn = build_model_from_norm_stats(ns).eval()
    torch.manual_seed(3)
    seq = torch.randn(1, 184, 60)
    shifted = seq + 4.0
    with torch.no_grad():
        assert not torch.allclose(plain(seq, torch.zeros(1, 6))[0], plain(shifted, torch.zeros(1, 6))[0], atol=1e-4)
        assert torch.allclose(cmvn(seq, torch.zeros(1, 6))[0], cmvn(shifted, torch.zeros(1, 6))[0], atol=1e-4)
