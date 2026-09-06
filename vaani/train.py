"""
Basic supervised training loop for VAANI Module B models.

This module is intentionally dataset-agnostic: it does not know how to build
a Dataset from a manifest (there is no settled manifest -> audio-file
interface anywhere in the repo yet, and no real audio corpus present). It
only knows how to train an already-instantiated `nn.Module` (e.g. one
produced by `registry.load(config)`) against already-constructed
`torch.utils.data.DataLoader`s.

Scope note: this implements Step 1 ("basic training loop") of the Module B
Task 4 plan. `train_from_config()` (added afterwards) closes some of the
config-consumption gaps Step 1 left open; see its own docstring. The
following remain explicitly out of scope and are NOT implemented here or
anywhere else in the repo yet:

  - Distillation loss.
  - MLflow logging.
  - HF Hub sync.
  - `base.yaml`'s entire `tracking:` block (`mlflow_uri`, `experiment`) --
    unused; see MLflow logging above. Wiring this up meaningfully needs a
    reachable MLflow server, which is a GPU-rota-machine concern (Doc 8),
    not a CPU-only gap like the ones `train_from_config()` closes.

Expected data contract:
    Each DataLoader yields batches of `(mel_tensor, label)` where:
      - mel_tensor has shape (B, 1, n_mels, T), matching TinyCNN's expected
        input (B, 1, 64, T).
      - label is an integer class tensor of shape (B,) with values in {0, 1}.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run a single evaluation pass over `data_loader` in eval mode, no grad.

    Returns:
        Dict with "loss" (mean per-sample loss) and "accuracy" (fraction of
        correctly classified samples).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run a standard supervised training loop: DataLoader -> Optimizer ->
    Epoch loop -> Validation.

    For each epoch, iterates `train_loader`, computes cross-entropy loss,
    backpropagates, and steps an Adam optimizer. After each epoch, runs a
    full validation pass over `val_loader` (model.eval() + torch.no_grad()).

    If `checkpoint_path` is given, saves `model.state_dict()` there (via
    `torch.save`) after training completes.

    Args:
        model: An already-instantiated nn.Module (e.g. from
            `registry.load(config)`) accepting (B, 1, n_mels, T) inputs and
            producing (B, num_classes) logits.
        train_loader: DataLoader yielding (mel_tensor, label) training batches.
        val_loader: DataLoader yielding (mel_tensor, label) validation batches.
        epochs: Number of training epochs.
        lr: Learning rate for the Adam optimizer.
        device: torch.device to train on. Defaults to CUDA if available, else CPU.
        checkpoint_path: Optional path to save the final model state_dict to.

    Returns:
        A list of per-epoch history dicts, one per epoch, each with keys:
        "epoch", "train_loss", "val_loss", "val_accuracy".
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

        train_loss = running_loss / running_samples if running_samples else 0.0

        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)

    return history


_SUPPORTED_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def _config_hash(config: Dict[str, Any]) -> str:
    """
    Deterministic short hash of a config dict, for the checkpoint payload's
    `config_hash` field (DOC2 sec2.1: "every run logs its config_hash").

    `json.dumps(..., sort_keys=True)` gives a stable serialization
    regardless of dict key insertion order; falls back to `str()` for any
    non-JSON-serializable value (there are none in this schema today, but a
    hash function shouldn't crash on a config it can't fully render).
    """
    try:
        serialized = json.dumps(config, sort_keys=True, default=str)
    except TypeError:
        serialized = str(config)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _build_optimizer(model: nn.Module, optim_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build an optimizer from an `optim:` config block (`opt`/`lr`), e.g.
    `cnn_week1.yaml`'s `optim: {opt: adamw, lr: 3.0e-4, ...}`.

    Only `adam`/`adamw` are supported (the only values used anywhere in
    `configs/*.yaml` today); an unrecognized `opt` raises rather than
    silently falling back to a default the config didn't ask for.
    """
    opt_name = str(optim_cfg.get("opt", "adam")).lower()
    if opt_name not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optim.opt '{opt_name}'. Supported: {sorted(_SUPPORTED_OPTIMIZERS)}"
        )
    lr = float(optim_cfg.get("lr", 1e-3))
    return _SUPPORTED_OPTIMIZERS[opt_name](model.parameters(), lr=lr)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "config_hash": config_hash,
        },
        path,
    )


def _rotate_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    """Delete all but the `keep_last` most recent (by step, encoded in the
    filename as `step_<8-digit step>.pt`) checkpoints in `ckpt_dir`.

    Sorts by the step number parsed from the filename rather than mtime:
    consecutive checkpoints can be written faster than filesystem mtime
    resolution, especially on a fast CPU with a tiny synthetic dataset.
    """
    checkpoints = sorted(
        ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    for stale in checkpoints[:-keep_last]:
        stale.unlink()


def train_from_config(
    model: nn.Module,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Config-driven training loop: reads hyperparameters from `config` (as
    produced by `registry.load_config()`) instead of hardcoding them as
    keyword arguments, closing the gaps `train()` (Step 1 of Module B Task
    4) documented as out of scope:

      - `optim.opt` / `optim.lr` select the optimizer (adam/adamw) and its
        learning rate, instead of `train()`'s hardcoded Adam + `lr` kwarg.
      - `optim.epochs` drives the epoch count, instead of `train()`'s
        hardcoded default of 10.
      - `ckpt.dir` / `ckpt.every_steps` / `ckpt.keep_last` (after
        `registry.load_config()`'s `${name}` interpolation has resolved
        `ckpt.dir`) drive periodic checkpointing with bounded retention,
        instead of `train()`'s single save-once-at-the-end behavior.
      - The checkpoint payload is the richer `{model, optimizer, step,
        epoch, config_hash}` dict DOC2 sec2.3 specifies (enabling resume),
        not a bare `model.state_dict()`.

    Still NOT implemented (see module docstring): `optim.sched` (LR
    scheduling), `optim.clip` (gradient clipping), `optim.batch` (the
    DataLoader's batch size is the caller's responsibility, same as
    `train()`), distillation, MLflow, HF Hub sync, and actually resuming
    from a saved checkpoint (this function only *writes* the resumable
    payload; reading one back in is a separate, not-yet-needed piece since
    no real training run exists to resume yet).

    Args:
        model: An already-instantiated nn.Module (e.g. from
            `registry.load(config)`).
        config: A merged config dict from `registry.load_config()`, expected
            to have `optim.opt`/`optim.lr`/`optim.epochs` and (optionally)
            `ckpt.dir`/`ckpt.every_steps`/`ckpt.keep_last`.
        train_loader: DataLoader yielding (mel_tensor, label) training batches.
        val_loader: DataLoader yielding (mel_tensor, label) validation batches.
        device: torch.device to train on. Defaults to CUDA if available, else CPU.

    Returns:
        A list of per-epoch history dicts, one per epoch (same shape as
        `train()`'s return value).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optim_cfg = config.get("optim", {})
    epochs = int(optim_cfg.get("epochs", 10))

    ckpt_cfg = config.get("ckpt", {})
    ckpt_dir = Path(ckpt_cfg["dir"]) if ckpt_cfg.get("dir") else None
    every_steps = int(ckpt_cfg.get("every_steps", 0)) if ckpt_dir is not None else 0
    keep_last = int(ckpt_cfg.get("keep_last", 0)) if ckpt_dir is not None else 0
    config_hash = _config_hash(config)

    model = model.to(device)
    optimizer = _build_optimizer(model, optim_cfg)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

            if every_steps and global_step % every_steps == 0:
                _save_checkpoint(
                    ckpt_dir / f"step_{global_step:08d}.pt",
                    model,
                    optimizer,
                    global_step,
                    epoch,
                    config_hash,
                )
                if keep_last:
                    _rotate_checkpoints(ckpt_dir, keep_last)

        train_loss = running_loss / running_samples if running_samples else 0.0

        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

    return history
