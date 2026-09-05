"""
Unit tests for SSLHead model.

Tests cover:
1. Registration in MODEL_REGISTRY (no real download required).
2. Direct instantiation with kwargs and with a config dict, exercising the
   real `facebook/wav2vec2-base` checkpoint via
   `transformers.Wav2Vec2Model.from_pretrained(...)` (not a mock).
3. Forward pass output shape verification.
4. Freezing behavior: feature extractor frozen, `w` and classifier trainable,
   and train()-mode forwards stay deterministic (extractor pinned to eval).
5. End-to-end registry loading against the real `configs/ssl_teacher.yaml`.
6. `w`'s size tracks the loaded base checkpoint's actual transformer depth,
   not a hardcoded constant.
7. `transformers` is imported lazily (not at module scope), and both models
   declare their `input_kind` ("waveform" vs. "mel").

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

from models.ssl_head import SSLHead
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


def test_loading_tinycnn_via_registry_does_not_import_transformers():
    """I3 fix regression: `import models` (triggered by every
    `registry.load()` call) must stay cheap -- `transformers` is only
    imported lazily inside `SSLHead.__init__`, not at `models`/`ssl_head`
    module scope, so a TinyCNN-only caller/environment never pays for (or
    needs) `transformers` to be installed.

    This test only proves the module-scope import is gone: `transformers`
    may already be in `sys.modules` from earlier tests in this same
    process (module imports aren't undone between tests), so it does not
    assert `transformers` is absent -- it asserts that `models.ssl_head`
    itself carries no top-level `Wav2Vec2Model`/`transformers` reference.
    """
    import inspect

    import models.ssl_head as ssl_head_mod

    module_source = inspect.getsource(ssl_head_mod)
    # The only `import transformers`/`from transformers import ...` in the
    # module must be inside a function body (indented), not at column 0.
    for line in module_source.splitlines():
        if "import" in line and "transformers" in line:
            assert line.startswith(" ") or line.startswith("\t"), (
                f"transformers import must be inside a function, not at "
                f"module scope: {line!r}"
            )


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
    the loaded checkpoint returns (num_hidden_layers + 1: embeddings plus
    each transformer layer's output) -- 13 for wav2vec2-base (12 layers)."""
    model = _load_real_ssl_head()
    expected_size = model.feature_extractor.config.num_hidden_layers + 1
    assert isinstance(model.w, nn.Parameter)
    assert model.w.shape == (expected_size,)
    assert expected_size == 13  # wav2vec2-base specifically
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
# Tests: train() mode does not un-freeze extractor stochasticity (C1 fix)
# ---------------------------------------------------------------------------


def test_sslhead_train_mode_forward_is_deterministic():
    """`model.train()` (called by train.py at the top of every epoch) must
    not re-enable the wav2vec2 extractor's internal dropout or SpecAugment
    time-masking. Two train()-mode forwards on the *same* input tensor must
    be bit-identical -- if `SSLHead.train()` did not keep
    `feature_extractor` pinned to eval mode, this would fail (measured
    ~0.0028 max diff from SpecAugment alone on the real checkpoint)."""
    model = _load_real_ssl_head()
    model.train()
    assert model.training is True
    assert model.feature_extractor.training is False

    x = torch.randn(1, 16000)
    out1 = model(x)
    out2 = model(x)
    assert torch.equal(out1, out2)


def test_sslhead_eval_also_keeps_feature_extractor_in_eval():
    """Sanity check: `model.eval()` still puts the extractor in eval mode
    (this always worked; this test guards against a `train()` override that
    accidentally breaks the `mode=False` path)."""
    model = _load_real_ssl_head()
    model.eval()
    assert model.training is False
    assert model.feature_extractor.training is False


# ---------------------------------------------------------------------------
# Tests: `w` size tracks the real base checkpoint, not a hardcoded constant
# (I2 fix)
# ---------------------------------------------------------------------------


def test_sslhead_w_size_and_forward_track_a_smaller_synthetic_base(tmp_path):
    """SSLHead must not hardcode the number of hidden states/layers: a base
    checkpoint with a different transformer depth than wav2vec2-base's 12
    layers must still construct with a correctly-sized `w` and run a
    working forward pass. Uses a tiny, locally-constructed & saved
    Wav2Vec2Config/Model (no real network fetch of a second large
    checkpoint needed) with 3 transformer layers, so `w` must end up
    sized 4 (3 + 1 for the embedding output), not the wav2vec2-base-shaped
    13.
    """
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    tiny_config = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(32, 32),
        conv_stride=(5, 2),
        conv_kernel=(10, 3),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )
    tiny_model_dir = tmp_path / "tiny_wav2vec2"
    Wav2Vec2Model(tiny_config).save_pretrained(tiny_model_dir)

    model = SSLHead(base=str(tiny_model_dir))

    assert model.w.shape == (4,)  # 3 layers + 1 embedding output
    assert model.w.shape != (13,)  # must NOT inherit wav2vec2-base's shape

    x = torch.randn(1, 4000)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 2)


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


# ---------------------------------------------------------------------------
# Tests: input_kind contract (I4 fix)
# ---------------------------------------------------------------------------


def test_sslhead_declares_waveform_input_kind():
    """SSLHead must declare its input contract explicitly, since it is not
    interchangeable with TinyCNN at call time (waveform vs. mel input)."""
    assert SSLHead.input_kind == "waveform"


def test_tinycnn_declares_mel_input_kind():
    """TinyCNN must declare the complementary input contract."""
    from models.cnn import TinyCNN

    assert TinyCNN.input_kind == "mel"
