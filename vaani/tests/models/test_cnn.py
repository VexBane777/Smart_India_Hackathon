"""
Unit tests for TinyCNN model.

Tests cover:
1. Direct instantiation with keyword arguments
2. Config-based instantiation (from registry.load)
3. Forward pass output shape verification
4. End-to-end registry loading
"""

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from models.cnn import TinyCNN
from registry import MODEL_REGISTRY, load, register_model


# ---------------------------------------------------------------------------
# Tests: Direct Instantiation and Forward Pass
# ---------------------------------------------------------------------------


def test_tinycnn_instantiation_with_defaults():
    """Test that TinyCNN can be instantiated with default parameters."""
    model = TinyCNN()
    assert isinstance(model, nn.Module)
    assert model.n_mels == 64
    assert model.chs == (32, 64, 128, 256)


def test_tinycnn_instantiation_with_custom_params():
    """Test that TinyCNN can be instantiated with custom parameters."""
    model = TinyCNN(n_mels=80, chs=(16, 32, 64, 128))
    assert model.n_mels == 80
    assert model.chs == (16, 32, 64, 128)


def test_tinycnn_forward_output_shape():
    """Test that TinyCNN produces correct output shape (1, 2) for input (1, 1, 64, 200)."""
    model = TinyCNN()
    x = torch.randn(1, 1, 64, 200)
    output = model(x)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"


def test_tinycnn_forward_batch_output_shape():
    """Test that TinyCNN handles batches correctly."""
    model = TinyCNN()
    batch_size = 8
    x = torch.randn(batch_size, 1, 64, 200)
    output = model(x)
    assert output.shape == (batch_size, 2)


def test_tinycnn_forward_custom_n_mels():
    """Test that TinyCNN accepts different n_mels values."""
    model = TinyCNN(n_mels=80)
    x = torch.randn(4, 1, 80, 200)
    output = model(x)
    assert output.shape == (4, 2)


def test_tinycnn_forward_custom_channels():
    """Test that TinyCNN accepts custom channel configurations."""
    model = TinyCNN(chs=(16, 32, 64, 128))
    x = torch.randn(2, 1, 64, 200)
    output = model(x)
    assert output.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests: Config-Based Instantiation
# ---------------------------------------------------------------------------


def test_tinycnn_instantiation_with_config():
    """Test that TinyCNN can be instantiated from a config dict."""
    config = {
        "features": {
            "n_mels": 64,
        },
        "model": {
            "channels": [32, 64, 128, 256],
        },
    }
    model = TinyCNN(config)
    assert isinstance(model, nn.Module)
    assert model.n_mels == 64
    assert model.chs == (32, 64, 128, 256)


def test_tinycnn_instantiation_with_config_custom_values():
    """Test that TinyCNN config extraction works with custom values."""
    config = {
        "features": {
            "n_mels": 80,
            "n_fft": 512,
        },
        "model": {
            "channels": [16, 32, 64, 128],
        },
    }
    model = TinyCNN(config)
    assert model.n_mels == 80
    assert model.chs == (16, 32, 64, 128)


def test_tinycnn_config_fallback_to_defaults():
    """Test that TinyCNN uses defaults when config keys are missing."""
    config = {}  # Empty config
    model = TinyCNN(config)
    assert model.n_mels == 64  # Should use default
    assert model.chs == (32, 64, 128, 256)  # Should use default


def test_tinycnn_config_forward_pass():
    """Test that TinyCNN instantiated from config produces correct output shape."""
    config = {
        "features": {
            "n_mels": 64,
        },
        "model": {
            "channels": [32, 64, 128, 256],
        },
    }
    model = TinyCNN(config)
    x = torch.randn(1, 1, 64, 200)
    output = model(x)
    assert output.shape == (1, 2)


def test_tinycnn_invalid_config_type_raises_error():
    """Test that passing a non-dict config raises ValueError."""
    with pytest.raises(ValueError, match="config must be a dict or None"):
        TinyCNN(config="invalid_config")


# ---------------------------------------------------------------------------
# Tests: Registry Integration
# ---------------------------------------------------------------------------


def test_tinycnn_registered_in_model_registry():
    """Test that TinyCNN is registered in MODEL_REGISTRY."""
    assert "tinycnn" in MODEL_REGISTRY
    assert MODEL_REGISTRY["tinycnn"] is TinyCNN


def test_registry_load_cnn_week1():
    """Test end-to-end registry loading of cnn_week1 config.

    This test verifies that the registry.load("cnn_week1") call works
    correctly, loading the config and instantiating a TinyCNN model.
    This is the actual integration point between Task 1 (registry) and
    Task 2 (TinyCNN implementation).
    """
    model = load("cnn_week1")
    assert isinstance(model, TinyCNN)
    assert model.n_mels == 64
    assert model.chs == (32, 64, 128, 256)


def test_registry_loaded_model_forward_pass():
    """Test that a model loaded via registry.load() can do a forward pass."""
    model = load("cnn_week1")
    x = torch.randn(1, 1, 64, 200)
    output = model(x)
    assert output.shape == (1, 2)


def test_registry_loaded_model_batch_forward():
    """Test forward pass on batch for registry-loaded model."""
    model = load("cnn_week1")
    batch_size = 16
    x = torch.randn(batch_size, 1, 64, 200)
    output = model(x)
    assert output.shape == (batch_size, 2)


# ---------------------------------------------------------------------------
# Tests: Model Behavior
# ---------------------------------------------------------------------------


def test_tinycnn_produces_different_outputs_for_different_inputs():
    """Test that model produces different outputs for different inputs."""
    model = TinyCNN()
    x1 = torch.randn(1, 1, 64, 200)
    x2 = torch.randn(1, 1, 64, 200)

    out1 = model(x1)
    out2 = model(x2)

    # Outputs should be different (with very high probability)
    assert not torch.allclose(out1, out2)


def test_tinycnn_is_trainable():
    """Test that model parameters can be updated (model is trainable)."""
    model = TinyCNN()
    x = torch.randn(4, 1, 64, 200)
    loss_fn = nn.CrossEntropyLoss()
    target = torch.tensor([0, 1, 0, 1])

    # Check initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Forward and backward pass
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()

    # Verify gradients are computed
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0

    # Update parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()

    # Verify parameters changed
    params_changed = False
    for initial, current in zip(initial_params, model.parameters()):
        if not torch.allclose(initial, current):
            params_changed = True
            break

    assert params_changed, "Model parameters did not update during training"


def test_tinycnn_output_is_float():
    """Test that TinyCNN outputs are float32 tensors."""
    model = TinyCNN()
    x = torch.randn(1, 1, 64, 200)
    output = model(x)
    assert output.dtype == torch.float32
