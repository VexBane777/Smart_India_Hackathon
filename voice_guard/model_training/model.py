"""
VoiceGuard model architectures, and the one place that knows how to rebuild
any of them from a saved run.

- VoiceGuardMLP (v3-v10): 63/66-d mean-pooled LFCC + prosody(+physio) vector.
  Retired for training, kept so v9_noisefix can be re-scored as a baseline
  (through MLPSequenceAdapter, from the same cached sequences).
- VoiceGuardSeqCNN ("seqcnn_v1", v11): 2 Conv1d layers, k=3, so a receptive
  field of 5 frames (~130 ms at hop 256), too little temporal context.
- VoiceGuardSeqTCN ("seqtcn_v2", v12): dilated residual TCN, receptive
  field 65 frames (~1.1 s), mean+std+max pooling, ~87k params.

All sequence models share one ONNX I/O contract (what lib/services/src/
tflite_io.dart sends and reads):
  inputs  lfcc_sequence (1, 184, 60) float32, scalars (1, 6) float32
  outputs real_fake_logits (1, 2), attack_type_logits (1, 2)  (raw logits)
Normalization lives inside the graph (FixedNormalize*); the phone sends raw
features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_DIM = 66  # MLP-era flat vector: 60 LFCC + 3 prosody + 3 physio
NUM_CLASSES = 2  # 0 = real, 1 = synthetic/cloned
N_FRAMES = 184
N_LFCC = 60
N_SCALARS = 6
ONNX_INPUT_NAMES = ["lfcc_sequence", "scalars"]
ONNX_OUTPUT_NAMES = ["real_fake_logits", "attack_type_logits"]
DEFAULT_ARCH = "seqcnn_v1"  # norm_stats without an "arch" key predate v12 = v11


class FixedNormalize(nn.Module):
    """Non-trainable (x - mean) / std baked in as buffers so it travels with
    the exported graph."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        assert mean.shape == std.shape, f"mean/std shape mismatch: {mean.shape} vs {std.shape}"
        self.register_buffer("mean", torch.from_numpy(np.asarray(mean, dtype=np.float32)))
        self.register_buffer("std", torch.from_numpy(np.asarray(std, dtype=np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class FixedNormalizeSeq(FixedNormalize):
    """Per-LFCC-coefficient (x - mean) / std, broadcast over (batch, time, 60)."""


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
        assert self.normalize.mean.shape[0] == input_dim
        dims = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))


class MLPSequenceAdapter(nn.Module):
    """Scores an MLP baseline from (lfcc_sequence, scalars): rebuilds its
    flat input exactly as features.extract_lfcc/extract_features do (mean
    over frames, then per-window standardization across the 60
    coefficients), appends prosody (and physio if the MLP is 66-d).
    Returns (logits, None): MLPs have no attack-type head."""

    def __init__(self, mlp: VoiceGuardMLP):
        super().__init__()
        self.mlp = mlp
        self.n_scalar_inputs = mlp.normalize.mean.shape[0] - N_LFCC

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor):
        pooled = seq.mean(dim=1)
        m = pooled.mean(dim=1, keepdim=True)
        std = torch.sqrt(((pooled - m) ** 2).mean(dim=1, keepdim=True) + 1e-8)
        x = torch.cat([(pooled - m) / std, scalars[:, : self.n_scalar_inputs]], dim=1)
        return self.mlp(x), None


class VoiceGuardSeqCNN(nn.Module):
    """v11 ("seqcnn_v1"): LFCC sequence -> Conv1d(60->32->16, k3) -> avg+max
    pool, concatenated with normalized scalars -> Linear 32 -> two heads."""

    def __init__(self, n_frames: int, n_lfcc: int, n_scalars: int, seq_mean: np.ndarray, seq_std: np.ndarray,
                 scalar_mean: np.ndarray, scalar_std: np.ndarray, conv_channels: tuple[int, ...] = (32, 16),
                 num_classes: int = 2):
        super().__init__()
        self.seq_normalize = FixedNormalizeSeq(seq_mean, seq_std)
        self.scalar_normalize = FixedNormalize(scalar_mean, scalar_std)
        conv_layers: list[nn.Module] = []
        in_ch = n_lfcc
        for out_ch in conv_channels:
            conv_layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.trunk = nn.Sequential(nn.Linear(in_ch * 2 + n_scalars, 32), nn.ReLU())
        self.real_fake_head = nn.Linear(32, num_classes)
        self.attack_type_head = nn.Linear(32, num_classes)

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(self.seq_normalize(seq).transpose(1, 2))
        embedding = torch.cat([conv_out.mean(dim=2), conv_out.amax(dim=2)], dim=1)
        trunk_out = self.trunk(torch.cat([embedding, self.scalar_normalize(scalars)], dim=1))
        return self.real_fake_head(trunk_out), self.attack_type_head(trunk_out)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.bn(self.conv(x))))


class VoiceGuardSeqTCN(nn.Module):
    """v12 ("seqtcn_v2"): stem Conv1d(60->64, k3)+BN+ReLU, then residual
    blocks Conv1d(64, k3, dilation 1/2/4/8/16)+BN+ReLU+Dropout, then
    mean+std+max pooling concatenated with normalized scalars ->
    Linear 64 + ReLU + Dropout -> real/fake and attack-type heads.
    Receptive field 1 + 2*(1+1+2+4+8+16) = 65 frames (~1.1 s)."""

    def __init__(self, n_lfcc: int, n_scalars: int, seq_mean: np.ndarray, seq_std: np.ndarray,
                 scalar_mean: np.ndarray, scalar_std: np.ndarray, channels: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.1, hidden: int = 64,
                 num_classes: int = 2):
        super().__init__()
        self.seq_normalize = FixedNormalizeSeq(seq_mean, seq_std)
        self.scalar_normalize = FixedNormalize(scalar_mean, scalar_std)
        self.stem = nn.Sequential(nn.Conv1d(n_lfcc, channels, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResidualBlock(channels, d, dropout) for d in dilations])
        self.trunk = nn.Sequential(nn.Linear(3 * channels + n_scalars, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.real_fake_head = nn.Linear(hidden, num_classes)
        self.attack_type_head = nn.Linear(hidden, num_classes)

    @staticmethod
    def receptive_field(dilations: tuple[int, ...] = (1, 2, 4, 8, 16)) -> int:
        return 1 + 2 * (1 + sum(dilations))

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(self.seq_normalize(seq).transpose(1, 2)))
        mean = h.mean(dim=2)
        std = torch.sqrt(((h - mean.unsqueeze(2)) ** 2).mean(dim=2) + 1e-5)
        pooled = torch.cat([mean, std, h.amax(dim=2)], dim=1)
        trunk_out = self.trunk(torch.cat([pooled, self.scalar_normalize(scalars)], dim=1))
        return self.real_fake_head(trunk_out), self.attack_type_head(trunk_out)


ARCHS = ("seqcnn_v1", "seqtcn_v2")


def build_model_from_norm_stats(norm_stats: dict) -> nn.Module:
    """Rebuilds an (untrained) sequence model from a run's norm_stats.npz."""
    arch = str(norm_stats["arch"]) if "arch" in norm_stats else DEFAULT_ARCH
    common = dict(n_lfcc=int(norm_stats["n_lfcc"]), n_scalars=int(norm_stats["n_scalars"]),
                  seq_mean=norm_stats["seq_mean"], seq_std=norm_stats["seq_std"],
                  scalar_mean=norm_stats["scalar_mean"], scalar_std=norm_stats["scalar_std"])
    if arch == "seqcnn_v1":
        return VoiceGuardSeqCNN(n_frames=int(norm_stats["n_frames"]), **common)
    if arch == "seqtcn_v2":
        return VoiceGuardSeqTCN(**common)
    raise ValueError(f"unknown arch {arch!r}; known: {ARCHS}")


def load_scoring_model(model_path: Path | str, norm_stats_path: Path | str | None = None) -> tuple[nn.Module, str]:
    """Any saved VoiceGuard model as a module with forward(seq, scalars) ->
    (real_fake_logits, attack_type_logits or None), in eval mode.
    Returns (module, arch name). MLP checkpoints (v3-v10) are detected by
    their state-dict keys and wrapped in MLPSequenceAdapter. Sequence models
    need norm_stats.npz (default: next to model_path)."""
    model_path = Path(model_path)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if "net.0.weight" in state:
        input_dim = int(state["normalize.mean"].shape[0])
        hidden = []
        i = 0
        while f"net.{i}.weight" in state:
            hidden.append(int(state[f"net.{i}.weight"].shape[0]))
            i += 2
        mlp = VoiceGuardMLP(input_dim=input_dim, hidden_dims=tuple(hidden[:-1]))
        mlp.load_state_dict(state)
        return MLPSequenceAdapter(mlp).eval(), f"mlp_{input_dim}d"
    ns_path = Path(norm_stats_path) if norm_stats_path else model_path.parent / "norm_stats.npz"
    norm_stats = dict(np.load(ns_path))
    model = build_model_from_norm_stats(norm_stats)
    model.load_state_dict(state)
    arch = str(norm_stats["arch"]) if "arch" in norm_stats else DEFAULT_ARCH
    return model.eval(), arch


def export_onnx(model: nn.Module, path: Path | str, n_frames: int = N_FRAMES, n_lfcc: int = N_LFCC,
                n_scalars: int = N_SCALARS) -> None:
    """Exports with the app's fixed I/O contract (batch 1)."""
    model = model.cpu().eval()
    dummy = (torch.zeros(1, n_frames, n_lfcc), torch.zeros(1, n_scalars))
    torch.onnx.export(model, dummy, str(path), input_names=ONNX_INPUT_NAMES, output_names=ONNX_OUTPUT_NAMES,
                      opset_version=13, dynamo=False)
