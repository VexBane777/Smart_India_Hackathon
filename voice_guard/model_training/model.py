"""
Tiny MLP over the 63-d LFCC+prosody feature vector (see features.py).
Deliberately small: the feature vector is already mean-pooled (no time
axis survives), so a CNN/MobileNet buys nothing here — a small MLP is
the right-sized model for this input, sized for Android CPU inference
in low milliseconds.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

INPUT_DIM = 63  # 60 LFCC + 3 prosody, must match audio_processor.dart exactly
NUM_CLASSES = 2  # 0 = real, 1 = synthetic/cloned


class FixedNormalize(nn.Module):
    """Non-trainable (x - mean) / std, baked in as buffers so it travels with
    the exported graph. tflite_io.dart sends raw [...lfcc, ...prosody] straight
    off the phone — normalization MUST live inside the shipped model, not as a
    separate preprocessing step nothing on the Dart side would ever apply."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class VoiceGuardMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
        norm_mean: np.ndarray | None = None,
        norm_std: np.ndarray | None = None,
    ):
        super().__init__()
        self.normalize = FixedNormalize(
            norm_mean if norm_mean is not None else np.zeros(input_dim),
            norm_std if norm_std is not None else np.ones(input_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.normalize.mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.normalize.std.copy_(torch.from_numpy(std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))  # raw logits — Dart applies softmax itself
