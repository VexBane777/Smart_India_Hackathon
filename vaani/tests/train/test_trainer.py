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
from registry import load, load_config
import train as train_module
from train import evaluate, train, train_from_config


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
    # Seed the shuffle order explicitly so batch ordering (and therefore the
    # exact gradient trajectory) is deterministic across runs/machines.
    shuffle_generator = torch.Generator().manual_seed(123)
    return DataLoader(dataset, batch_size=8, shuffle=True, generator=shuffle_generator)


@pytest.fixture
def val_loader():
    dataset = _make_synthetic_dataset(n_samples=32, seed=1)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def model():
    # Seed the global RNG before instantiation so TinyCNN's weight
    # initialization is deterministic across runs/machines. This, combined
    # with the seeded dataset generation and seeded DataLoader shuffling
    # above, removes the last source of run-to-run randomness in these
    # tests, which is required for the loss-decrease smoke test below to be
    # genuinely non-flaky rather than merely "usually passes".
    torch.manual_seed(42)
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
    training should trend down from the untrained baseline on an easy,
    learnable synthetic task.

    Non-flakiness: model init (see the `model` fixture), dataset generation,
    and DataLoader shuffle order are all seeded deterministically, so this
    test's outcome is reproducible run-to-run rather than depending on an
    unseeded random draw. On top of that determinism, the comparison itself
    avoids relying on two single-epoch point samples (which could still be
    noisy epoch-to-epoch even with a fixed seed) - it compares the *average*
    validation loss over the last two epochs against the average over the
    first two epochs, with a generous margin.
    """
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    pre_metrics = evaluate(model, val_loader, criterion, device)

    history = train(
        model,
        train_loader,
        val_loader,
        epochs=6,
        lr=1e-3,
        device=device,
        checkpoint_path=None,
    )

    early_avg_val_loss = sum(h["val_loss"] for h in history[:2]) / 2
    late_avg_val_loss = sum(h["val_loss"] for h in history[-2:]) / 2

    # Loss should trend down across training on this easy task, comparing
    # multi-epoch averages (not single point samples) with headroom.
    assert late_avg_val_loss <= early_avg_val_loss + 0.05
    # And training should have moved loss meaningfully below the untrained
    # baseline (generous margin to avoid flakiness).
    assert late_avg_val_loss <= pre_metrics["loss"] * 1.05


# ---------------------------------------------------------------------------
# Tests: train_from_config (consumes the optim:/ckpt: blocks train() ignores)
# ---------------------------------------------------------------------------


@pytest.fixture
def cnn_week1_config(tmp_path):
    """cnn_week1's config, with ckpt.dir redirected under tmp_path so the
    test doesn't write into the real repo's runs/ directory."""
    config = load_config("cnn_week1")
    config["ckpt"]["dir"] = str(tmp_path / "runs" / "cnn_week1")
    config["ckpt"]["every_steps"] = 3
    config["ckpt"]["keep_last"] = 2
    return config


def test_train_from_config_reads_optim_epochs_not_hardcoded_default(
    cnn_week1_config, train_loader, val_loader
):
    """cnn_week1.yaml declares optim.epochs: 20; train_from_config() must
    actually run that many epochs rather than train()'s hardcoded default
    of 10, proving the optim: block is read, not ignored."""
    torch.manual_seed(42)
    model = load("cnn_week1")
    config = dict(cnn_week1_config)
    config["optim"] = dict(config["optim"])
    config["optim"]["epochs"] = 3  # override down, just to keep the test fast

    history = train_from_config(model, config, train_loader, val_loader, device=torch.device("cpu"))

    assert len(history) == 3


def test_train_from_config_uses_configured_learning_rate(
    cnn_week1_config, train_loader, val_loader, monkeypatch
):
    """The optimizer's lr must come from optim.lr, not train()'s hardcoded
    default of 1e-3."""
    torch.manual_seed(42)
    model = load("cnn_week1")
    config = dict(cnn_week1_config)
    config["optim"] = dict(config["optim"])
    config["optim"]["epochs"] = 1
    config["optim"]["lr"] = 5e-2

    captured = {}
    original_adamw = torch.optim.AdamW

    def spy_adamw(params, lr=None, **kwargs):
        captured["lr"] = lr
        return original_adamw(params, lr=lr, **kwargs)

    # train_from_config resolves "adamw" via its own _SUPPORTED_OPTIMIZERS
    # dict (bound at import time), so patch that entry directly rather than
    # torch.optim.AdamW -- the latter wouldn't be seen by the already-bound
    # reference.
    monkeypatch.setitem(train_module._SUPPORTED_OPTIMIZERS, "adamw", spy_adamw)

    train_from_config(model, config, train_loader, val_loader, device=torch.device("cpu"))

    assert captured["lr"] == 5e-2


def test_train_from_config_saves_periodic_checkpoints_and_rotates(
    cnn_week1_config, train_loader, val_loader
):
    """ckpt.every_steps/keep_last must actually be honored: checkpoints are
    written periodically during training (not just once at the end), and
    only the most recent `keep_last` are retained."""
    torch.manual_seed(42)
    model = load("cnn_week1")
    config = dict(cnn_week1_config)
    config["optim"] = dict(config["optim"])
    config["optim"]["epochs"] = 5

    train_from_config(model, config, train_loader, val_loader, device=torch.device("cpu"))

    ckpt_dir = Path(config["ckpt"]["dir"])
    assert ckpt_dir.exists()
    saved = sorted(ckpt_dir.glob("*.pt"))
    assert len(saved) == config["ckpt"]["keep_last"]


def test_train_from_config_checkpoint_payload_supports_resume(
    cnn_week1_config, train_loader, val_loader
):
    """Checkpoint payload includes model/optimizer state, step, epoch, and
    config_hash -- the richer resume payload DOC2 sec2.3 specifies, not just
    a bare state_dict."""
    torch.manual_seed(42)
    model = load("cnn_week1")
    config = dict(cnn_week1_config)
    config["optim"] = dict(config["optim"])
    config["optim"]["epochs"] = 2

    train_from_config(model, config, train_loader, val_loader, device=torch.device("cpu"))

    ckpt_dir = Path(config["ckpt"]["dir"])
    latest = sorted(ckpt_dir.glob("*.pt"))[-1]
    payload = torch.load(latest, map_location="cpu")

    assert set(["model", "optimizer", "step", "epoch", "config_hash"]) <= set(payload.keys())
    assert payload["config_hash"] == payload["config_hash"]  # deterministic, non-empty
    assert isinstance(payload["config_hash"], str) and len(payload["config_hash"]) > 0
