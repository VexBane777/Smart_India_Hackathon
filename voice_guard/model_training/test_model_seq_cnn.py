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


@pytest.mark.parametrize("arch,cls", [(None, VoiceGuardSeqCNN), ("seqcnn_v1", VoiceGuardSeqCNN), ("seqtcn_v2", VoiceGuardSeqTCN)])
def test_factory_and_forward_shapes(arch, cls):
    model = build_model_from_norm_stats(_norm(arch)).eval()
    assert isinstance(model, cls)
    rf, at = model(torch.randn(8, 184, 60), torch.randn(8, 6))
    assert rf.shape == (8, 2) and at.shape == (8, 2)


def test_unknown_arch_refused():
    with pytest.raises(ValueError):
        build_model_from_norm_stats(_norm("lstm_v9"))


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
                                      ("voice_guard_v11_seqcnn_selected/model.pt", "seqcnn_v1")])
def test_load_scoring_model_on_real_checkpoints(rel, kind):
    path = RUNS / rel
    if not path.exists():
        pytest.skip(f"{path} not present")
    model, arch = load_scoring_model(path)
    assert arch == kind
    with torch.no_grad():
        rf, _at = model(torch.randn(2, 184, 60), torch.rand(2, 6))
    assert rf.shape == (2, 2)
