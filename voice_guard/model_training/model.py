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

INPUT_DIM = 66  # 60 LFCC + 3 prosody + 3 physio (jitter/shimmer/HNR), must match audio_processor.dart exactly
NUM_CLASSES = 2  # 0 = real, 1 = synthetic/cloned


class FixedNormalize(nn.Module):
    """Non-trainable (x - mean) / std, baked in as buffers so it travels with
    the exported graph. tflite_io.dart sends raw [...lfcc, ...prosody] straight
    off the phone — normalization MUST live inside the shipped model, not as a
    separate preprocessing step nothing on the Dart side would ever apply."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        assert mean.shape == std.shape, f"mean/std shape mismatch: {mean.shape} vs {std.shape}"
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
        hidden_dims: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.normalize = FixedNormalize(
            norm_mean if norm_mean is not None else np.zeros(input_dim),
            norm_std if norm_std is not None else np.ones(input_dim),
        )
        actual_dim = self.normalize.mean.shape[0]
        assert actual_dim == input_dim, (
            f"input_dim={input_dim} but norm_mean/norm_std have {actual_dim} entries — "
            "train.py must pass input_dim= explicitly if features.py's output width changes"
        )
        dims = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.normalize.mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.normalize.std.copy_(torch.from_numpy(std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))  # raw logits — Dart applies softmax itself


class FixedNormalizeSeq(nn.Module):
    """Per-LFCC-channel (x - mean) / std, broadcast across the time axis.
    mean/std are (60,) — one pair of stats per LFCC coefficient index,
    shared across all frame positions (not per-frame-position stats; see
    track 3 plan §5's open question — this is the simpler of the two
    options, chosen as the starting point)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        assert mean.shape == std.shape
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std  # broadcasts over (batch, time, 60)


class VoiceGuardSeqCNN(nn.Module):
    """Frame-level LFCC sequence -> Conv1d stack -> pooled embedding,
    concatenated with normalized scalars, feeding two heads: real/fake
    (trained on every example) and attack-type (trained only on labeled
    fakes — see train_seq_cnn.py's masked loss). See track 3 plan §2.2 for
    the architecture rationale (CNN over GRU/LSTM: stateless, simpler ONNX
    export) and the track 4 design spec §4 for the dual-head design."""

    def __init__(
        self,
        n_frames: int,
        n_lfcc: int,
        n_scalars: int,
        seq_mean: np.ndarray,
        seq_std: np.ndarray,
        scalar_mean: np.ndarray,
        scalar_std: np.ndarray,
        conv_channels: tuple[int, ...] = (32, 16),
        num_classes: int = 2,
    ):
        super().__init__()
        self.seq_normalize = FixedNormalizeSeq(seq_mean, seq_std)
        self.scalar_normalize = FixedNormalize(scalar_mean, scalar_std)

        conv_layers: list[nn.Module] = []
        in_ch = n_lfcc
        for out_ch in conv_channels:
            conv_layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        embedding_dim = in_ch * 2  # avg-pool + max-pool concatenated

        trunk_in = embedding_dim + n_scalars
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 32), nn.ReLU(),
        )
        self.real_fake_head = nn.Linear(32, num_classes)
        self.attack_type_head = nn.Linear(32, num_classes)

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.seq_normalize(seq)  # (batch, time, 60)
        seq = seq.transpose(1, 2)  # -> (batch, 60, time) for Conv1d
        conv_out = self.conv(seq)  # (batch, channels, time)
        avg_pool = conv_out.mean(dim=2)
        max_pool = conv_out.amax(dim=2)
        embedding = torch.cat([avg_pool, max_pool], dim=1)

        scalars = self.scalar_normalize(scalars)
        trunk_in = torch.cat([embedding, scalars], dim=1)
        trunk_out = self.trunk(trunk_in)

        return self.real_fake_head(trunk_out), self.attack_type_head(trunk_out)
