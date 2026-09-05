"""
Basic supervised training loop for VAANI Module B models.

This module is intentionally dataset-agnostic: it does not know how to build
a Dataset from a manifest (there is no settled manifest -> audio-file
interface anywhere in the repo yet, and no real audio corpus present). It
only knows how to train an already-instantiated `nn.Module` (e.g. one
produced by `registry.load(config)`) against already-constructed
`torch.utils.data.DataLoader`s.

Scope note: this implements Step 1 ("basic training loop") of the Module B
Task 4 plan only. Distillation loss, MLflow logging, and HF Hub sync are
explicitly out of scope for this slice and are NOT implemented here.

Expected data contract:
    Each DataLoader yields batches of `(mel_tensor, label)` where:
      - mel_tensor has shape (B, 1, n_mels, T), matching TinyCNN's expected
        input (B, 1, 64, T).
      - label is an integer class tensor of shape (B,) with values in {0, 1}.
"""

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
