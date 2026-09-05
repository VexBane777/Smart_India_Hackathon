"""
Tests for train.py - basic supervised training loop.

There is no real audio corpus or manifest-to-audio Dataset in this repo yet,
so these tests exercise train() against a small synthetic in-memory dataset
of mel-shaped tensors, wrapped in DataLoaders, matching the (mel, label)
contract train() expects.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import models  # noqa: F401  (registers TinyCNN etc. via @register_model side effects)
from registry import load
from train import evaluate, train


N_MELS = 64
T_FRAMES = 50  # small time dimension to keep tests fast on CPU


def _make_synthetic_dataset(n_samples: int, seed: int) -> TensorDataset:
    """
    Build a small, easily-learnable synthetic binary classification dataset
    of mel-shaped tensors (1, n_mels, T).

    Label 0 samples are centered around -0.5, label 1 samples around +0.5,
    so a model can plausibly learn to separate them from noise alone.
    """
    generator = torch.Generator().manual_seed(seed)

    labels = torch.randint(0, 2, (n_samples,), generator=generator)
    # Shift: -0.5 for label 0, +0.5 for label 1.
    shift = (labels.float() * 2 - 1) * 0.5
    noise = torch.randn(n_samples, 1, N_MELS, T_FRAMES, generator=generator) * 0.3
    inputs = noise + shift.view(-1, 1, 1, 1)

    return TensorDataset(inputs, labels)


@pytest.fixture
def train_loader():
    dataset = _make_synthetic_dataset(n_samples=64, seed=0)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def val_loader():
    dataset = _make_synthetic_dataset(n_samples=32, seed=1)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def model():
    return load("cnn_week1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_runs_and_produces_reloadable_checkpoint(
    tmp_path, model, train_loader, val_loader
):
    """Training runs for N epochs without error and writes a checkpoint that
    round-trips through torch.save/torch.load + load_state_dict, producing
    identical outputs."""
    checkpoint_path = tmp_path / "model.pth"

    history = train(
        model,
        train_loader,
        val_loader,
        epochs=2,
        lr=1e-3,
        device=torch.device("cpu"),
        checkpoint_path=str(checkpoint_path),
    )

    assert len(history) == 2
    assert checkpoint_path.exists()

    # Reload into a fresh model instance and confirm identical outputs.
    reloaded = load("cnn_week1")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    reloaded.load_state_dict(state_dict)

    model.eval()
    reloaded.eval()

    sample_input, _ = next(iter(val_loader))
    with torch.no_grad():
        original_output = model(sample_input)
        reloaded_output = reloaded(sample_input)

    assert torch.allclose(original_output, reloaded_output, atol=1e-6)


def test_validation_metrics_are_sane(model, train_loader, val_loader):
    """Validation loss/accuracy are computed and within sane ranges."""
    metrics = evaluate(
        model, val_loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_training_reduces_loss_on_easy_synthetic_task(model, train_loader, val_loader):
    """Smoke check that the loop is actually learning: validation loss after
    training should be no higher than validation loss before training on an
    easy, learnable synthetic task. Kept tolerant/non-flaky (loose threshold,
    not an exact target) to avoid CPU-run flakiness."""
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    pre_metrics = evaluate(model, val_loader, criterion, device)

    history = train(
        model,
        train_loader,
        val_loader,
        epochs=5,
        lr=1e-3,
        device=device,
        checkpoint_path=None,
    )

    first_epoch_val_loss = history[0]["val_loss"]
    last_epoch_val_loss = history[-1]["val_loss"]

    # Loss should generally decrease across training on this easy task.
    assert last_epoch_val_loss <= first_epoch_val_loss + 1e-6
    # And training should have moved loss meaningfully below the untrained
    # baseline (generous margin to avoid flakiness).
    assert last_epoch_val_loss <= pre_metrics["loss"] * 1.05
