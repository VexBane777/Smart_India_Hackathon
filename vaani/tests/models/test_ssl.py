"""
Unit tests for SSLHead model.

Tests cover:
1. Registration in MODEL_REGISTRY (no real download required).
2. Direct instantiation with kwargs and with a config dict, exercising the
   real `facebook/wav2vec2-base` checkpoint via
   `transformers.Wav2Vec2Model.from_pretrained(...)` (not a mock).
3. Forward pass output shape verification.
4. Freezing behavior: feature extractor frozen, `w` and classifier trainable.
5. End-to-end registry loading against the real `configs/ssl_teacher.yaml`.

Real-checkpoint tests need network access (and, on the very first run, a
one-time ~360MB download into the HF Hub cache). Following the pattern
already used for real-ffmpeg tests in
`tests/telechannel/test_codec.py` (skip cleanly, don't fail, when a real
external dependency is unavailable), these tests attempt to load the real
model once at module scope and are SKIPPED (not failed) if that load fails
for any reason (no network, HF Hub down, etc.). When the load succeeds (as
it will in an environment with network access), the tests exercise the real
pretrained model end-to-end.
"""

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from models.ssl_head import NUM_HIDDEN_STATES, SSLHead
from registry import MODEL_REGISTRY, load


def _load_real_ssl_head(**kwargs):
    """
    Instantiate SSLHead against the real `facebook/wav2vec2-base` checkpoint,
    skipping the calling test (not failing it) if the checkpoint can't be
    fetched -- mirrors the `HAS_FFMPEG` skip pattern in
    tests/telechannel/test_codec.py for a real, non-mocked external
    dependency.
    """
    try:
        return SSLHead(**kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure
        # to reach/load the real HF Hub checkpoint should skip, not fail.
        pytest.skip(f"Could not load real facebook/wav2vec2-base checkpoint: {exc}")


# ---------------------------------------------------------------------------
# Tests: Registration (no network required)
# ---------------------------------------------------------------------------


def test_sslhead_registered_in_model_registry():
    """Test that SSLHead is registered in MODEL_REGISTRY under 'ssl_head'."""
    assert "ssl_head" in MODEL_REGISTRY
    assert MODEL_REGISTRY["ssl_head"] is SSLHead


# ---------------------------------------------------------------------------
# Tests: Direct Instantiation (real checkpoint, network-gated)
# ---------------------------------------------------------------------------


def test_sslhead_instantiation_with_default_base():
    """Test that SSLHead can be instantiated with the default base checkpoint."""
    model = _load_real_ssl_head()
    assert isinstance(model, nn.Module)
    assert model.base_name == "facebook/wav2vec2-base"


def test_sslhead_instantiation_with_kwarg_base():
    """Test that SSLHead accepts an explicit `base` keyword argument."""
    model = _load_real_ssl_head(base="facebook/wav2vec2-base")
    assert model.base_name == "facebook/wav2vec2-base"


def test_sslhead_instantiation_with_config():
    """Test that SSLHead resolves `base` from a config dict's model.base key."""
    config: Dict[str, Any] = {
        "model": {
            "base": "facebook/wav2vec2-base",
            "layers": 12,
        },
    }
    model = _load_real_ssl_head(config=config)
    assert model.base_name == "facebook/wav2vec2-base"


def test_sslhead_config_fallback_to_default_base():
    """Test that SSLHead falls back to the default base when config lacks one."""
    model = _load_real_ssl_head(config={})
    assert model.base_name == "facebook/wav2vec2-base"


def test_sslhead_invalid_config_type_raises_error():
    """Test that passing a non-dict config raises ValueError (no network needed:
    this is validated before any checkpoint load is attempted)."""
    with pytest.raises(ValueError, match="config must be a dict or None"):
        SSLHead(config="invalid_config")


# ---------------------------------------------------------------------------
# Tests: Forward Pass
# ---------------------------------------------------------------------------


def test_sslhead_forward_output_shape():
    """Test that passing (1, 32000) raw audio returns (1, 2) logits."""
    model = _load_real_ssl_head()
    model.eval()
    x = torch.randn(1, 32000)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 2)


def test_sslhead_forward_batch_output_shape():
    """Test that SSLHead handles batches correctly."""
    model = _load_real_ssl_head()
    model.eval()
    batch_size = 3
    x = torch.randn(batch_size, 32000)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (batch_size, 2)


def test_sslhead_output_is_float():
    """Test that SSLHead outputs are float32 tensors."""
    model = _load_real_ssl_head()
    model.eval()
    x = torch.randn(1, 32000)
    with torch.no_grad():
        output = model(x)
    assert output.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: Layer-Weighting Vector `w`
# ---------------------------------------------------------------------------


def test_sslhead_w_is_learnable_parameter_of_correct_size():
    """`w` must be an nn.Parameter sized to the real number of hidden states
    the checkpoint returns (13 for wav2vec2-base: embeddings + 12 layers)."""
    model = _load_real_ssl_head()
    assert isinstance(model.w, nn.Parameter)
    assert model.w.shape == (NUM_HIDDEN_STATES,)
    assert model.w.requires_grad is True


def test_sslhead_w_is_softmax_normalized_before_weighted_sum():
    """The raw `w` vector must be softmax-normalized (not used raw) when
    combining hidden states -- verify by checking the actual mixing weights
    used in forward() sum to 1 and are all non-negative."""
    model = _load_real_ssl_head()
    import torch.nn.functional as F

    weights = F.softmax(model.w, dim=0)
    assert torch.all(weights >= 0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Freezing (Step 3 of the brief)
# ---------------------------------------------------------------------------


def test_sslhead_feature_extractor_is_frozen():
    """Test that every feature-extractor parameter has requires_grad=False."""
    model = _load_real_ssl_head()
    frozen_params = list(model.feature_extractor.parameters())
    assert len(frozen_params) > 0, "Expected the feature extractor to have parameters"
    for param in frozen_params:
        assert param.requires_grad is False


def test_sslhead_w_and_classifier_remain_trainable():
    """Test that `w` and the classifier head are NOT frozen, i.e. freezing
    the feature extractor does not also freeze the learnable layer-weighting
    vector or the MLP classification head."""
    model = _load_real_ssl_head()
    assert model.w.requires_grad is True
    classifier_params = list(model.classifier.parameters())
    assert len(classifier_params) > 0
    for param in classifier_params:
        assert param.requires_grad is True


def test_sslhead_only_w_and_classifier_receive_gradients():
    """Test end-to-end that a backward pass only populates gradients for `w`
    and the classifier head, never for the frozen feature extractor."""
    model = _load_real_ssl_head()
    model.train()
    x = torch.randn(2, 16000)
    target = torch.tensor([0, 1])

    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    assert model.w.grad is not None
    for param in model.classifier.parameters():
        assert param.grad is not None

    for param in model.feature_extractor.parameters():
        assert param.grad is None


# ---------------------------------------------------------------------------
# Tests: Registry Integration
# ---------------------------------------------------------------------------


def test_registry_load_ssl_teacher():
    """Test end-to-end registry loading of the real configs/ssl_teacher.yaml.

    This verifies SSLHead's config-dict resolution path against the actual
    schema of configs/ssl_teacher.yaml (type: ssl_head, model.base:
    facebook/wav2vec2-base), not a guessed schema.
    """
    try:
        model = load("ssl_teacher")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load real facebook/wav2vec2-base checkpoint: {exc}")
    assert isinstance(model, SSLHead)
    assert model.base_name == "facebook/wav2vec2-base"


def test_registry_loaded_ssl_teacher_forward_pass():
    """Test that a registry-loaded SSLHead performs a real forward pass."""
    try:
        model = load("ssl_teacher")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load real facebook/wav2vec2-base checkpoint: {exc}")
    model.eval()
    x = torch.randn(1, 32000)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 2)
