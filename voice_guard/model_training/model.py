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
- VoiceGuardConformer ("conformer_v1", v14): Conformer encoder (MHSA + Conv
  module), same ONNX I/O contract as seqtcn_v2 (rework axis C/4).

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


class _SqueezeExcite(nn.Module):
    """Channel attention (SE): global-avg-pool -> 1x1 bottleneck -> sigmoid gate.

    Implemented with kernel-size-1 Conv1d so it maps cleanly to ONNX ops (no
    flatten/unsqueeze tricks). Adds ~2*C*C/reduction params.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)
        self.act = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.gate(self.fc2(self.act(self.fc1(x.mean(dim=2, keepdim=True)))))
        return x * s


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float,
                 stochastic_depth: float = 0.0, se: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.se = _SqueezeExcite(channels) if se else None
        # per-block drop probability (the caller ramps it with depth); eval takes the
        # identity path, so an exported graph never sees the mask
        self.stochastic_depth = float(stochastic_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn(self.conv(x)))
        if self.se is not None:
            h = self.se(h)
        h = self.drop(h)
        if self.training and self.stochastic_depth > 0.0:
            keep = 1.0 - self.stochastic_depth
            mask = torch.empty(x.shape[0], 1, 1, dtype=h.dtype, device=h.device).bernoulli_(keep)
            h = h * mask / keep  # survival scaling keeps E[h] unchanged
        return x + h


def per_utterance_cmvn(seq: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-utterance, per-coefficient zero-mean/unit-variance on (B, T, C).

    Removes the level/coloration differences an unseen channel or a speaker->mic
    loop imposes, which the confound analysis (`docs/CRITICAL-entity-vs-style-confound.md`)
    identifies as the cue the model keys on instead of speaker identity. Fully
    deterministic, so it is identical in PyTorch and ONNX.
    """
    mean = seq.mean(dim=1, keepdim=True)
    var = ((seq - mean) ** 2).mean(dim=1, keepdim=True)
    return (seq - mean) / torch.sqrt(var + eps)



class VoiceGuardSeqTCN(nn.Module):
    """v12 ("seqtcn_v2"): stem Conv1d(60->64, k3)+BN+ReLU, then residual
    blocks Conv1d(64, k3, dilation 1/2/4/8/16)+BN+ReLU+Dropout, then
    mean+std+max pooling concatenated with normalized scalars ->
    Linear 64 + ReLU + Dropout -> real/fake and attack-type heads.
    Receptive field 1 + 2*(1+1+2+4+8+16) = 65 frames (~1.1 s).

    Capacity/regularization kwargs (all persisted in norm_stats.npz by
    train_seq_cnn.py and read back by build_model_from_norm_stats, so a
    checkpoint always rebuilds the architecture it was trained with):
      channels, dilations, hidden, dropout  -- size (post-v13 rework axis B)
      stochastic_depth, cmvn, se            -- paired regularization (axis G)
    Defaults reproduce the v12/v13 architecture bit-for-bit.
    """

    def __init__(self, n_lfcc: int, n_scalars: int, seq_mean: np.ndarray, seq_std: np.ndarray,
                 scalar_mean: np.ndarray, scalar_std: np.ndarray, channels: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.1, hidden: int = 64,
                 stochastic_depth: float = 0.0, cmvn: bool = False, se: bool = False,
                 num_classes: int = 2):
        super().__init__()
        self.seq_normalize = FixedNormalizeSeq(seq_mean, seq_std)
        self.scalar_normalize = FixedNormalize(scalar_mean, scalar_std)
        self.cmvn = bool(cmvn)
        self.stem = nn.Sequential(nn.Conv1d(n_lfcc, channels, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(channels), nn.ReLU())
        # linear (dense) stochastic-depth rule from the paper: block l of L gets
        # p_l = p * (l + 1) / L, so shallow blocks are rarely skipped
        n_blocks = len(dilations)
        self.blocks = nn.Sequential(*[
            _ResidualBlock(channels, d, dropout,
                           stochastic_depth=stochastic_depth * (i + 1) / n_blocks, se=se)
            for i, d in enumerate(dilations)])
        self.trunk = nn.Sequential(nn.Linear(3 * channels + n_scalars, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.real_fake_head = nn.Linear(hidden, num_classes)
        self.attack_type_head = nn.Linear(hidden, num_classes)

    @staticmethod
    def receptive_field(dilations: tuple[int, ...] = (1, 2, 4, 8, 16)) -> int:
        return 1 + 2 * (1 + sum(dilations))

    def normalize_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, T, C): sequence normalization (and optional in-graph CMVN)."""
        x = self.seq_normalize(seq)
        return per_utterance_cmvn(x) if self.cmvn else x

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(self.normalize_sequence(seq).transpose(1, 2)))
        mean = h.mean(dim=2)
        std = torch.sqrt(((h - mean.unsqueeze(2)) ** 2).mean(dim=2) + 1e-5)
        pooled = torch.cat([mean, std, h.amax(dim=2)], dim=1)
        trunk_out = self.trunk(torch.cat([pooled, self.scalar_normalize(scalars)], dim=1))
        return self.real_fake_head(trunk_out), self.attack_type_head(trunk_out)


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.). Registered as a
    buffer so it bakes into the ONNX graph. max_len must be >= the sequence
    length used at export time (184 frames)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        assert d_model % 2 == 0, f"d_model {d_model} must be even for sinusoidal PE"
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish: x * sigmoid(x). Decomposes to Sigmoid + Mul for ONNX opset 13."""
    return x * torch.sigmoid(x)


class _ConformerFFN(nn.Module):
    """Pre-norm feed-forward (Macaron): LayerNorm -> Linear(d, d*e) -> SiLU ->
    Dropout -> Linear(d*e, d) -> Dropout. Caller applies the residual."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(_silu(self.fc1(x)))
        return self.drop(self.fc2(x))


class _ConformerSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention with manual softmax (ONNX-safe, opset 13).

    No nn.MultiheadAttention — explicit matmul + softmax so the graph traces
    cleanly under dynamo=False export. Operates on (B, T, C)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
        self.norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class _ConformerConv(nn.Module):
    """Pre-norm convolution module:
    LayerNorm -> 1x1 Conv1d(d, 2*d) -> GLU -> depthwise Conv1d(k) -> BatchNorm
    -> SiLU -> 1x1 Conv1d(d, d) -> Dropout. (B, T, C) in/out."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size {kernel_size} must be odd for symmetric padding"
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.glu(self.pw1(x))
        x = _silu(self.bn(self.dw(x)))
        x = self.pw2(x)
        x = self.drop(x)
        return x.transpose(1, 2)


class _ConformerBlock(nn.Module):
    """Single Conformer encoder block (pre-norm residual).

    x = x + 0.5 * FFN1(x)     # Macaron, half-step
    x = x +       MHSA(x)
    x = x +       Conv(x)
    x = x + 1.0 * FFN2(x)
    x = LayerNorm(x)          # output norm
    """

    def __init__(self, d_model: int, n_heads: int, expansion: int = 4,
                 kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = _ConformerFFN(d_model, expansion, dropout)
        self.mhsa = _ConformerSelfAttention(d_model, n_heads, dropout)
        self.conv = _ConformerConv(d_model, kernel_size, dropout)
        self.ffn2 = _ConformerFFN(d_model, expansion, dropout)
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.drop(self.ffn1(x))
        x = x + self.drop(self.mhsa(x))
        x = x + self.conv(x)
        x = x + 1.0 * self.drop(self.ffn2(x))
        return self.out_norm(x)


class VoiceGuardConformer(nn.Module):
    """v14 ("conformer_v1"): Conformer encoder (MHSA + Conv module) with the
    SAME ONNX I/O contract as VoiceGuardSeqTCN.

    inputs:  lfcc_sequence (B, 184, 60) float32, scalars (B, 6) float32
    outputs: real_fake_logits (B, 2), attack_type_logits (B, 2)  (raw logits)

    Normalization (FixedNormalize* + optional per-utterance CMVN) lives inside
    the graph. Capacity kwargs persisted in norm_stats.npz — see
    build_model_from_norm_stats / train_seq_cnn.py.
    """

    def __init__(self, n_lfcc: int, n_scalars: int, seq_mean: np.ndarray, seq_std: np.ndarray,
                 scalar_mean: np.ndarray, scalar_std: np.ndarray,
                 channels: int = 48, n_heads: int = 4, n_layers: int = 2,
                 ff_expansion: int = 4, conv_kernel: int = 31, dropout: float = 0.1,
                 hidden: int = 64, cmvn: bool = False, num_classes: int = 2):
        super().__init__()
        assert channels % 2 == 0, "channels (d_model) must be even for sinusoidal PE"
        assert channels % n_heads == 0, f"channels {channels} not divisible by n_heads {n_heads}"
        self.seq_normalize = FixedNormalizeSeq(seq_mean, seq_std)
        self.scalar_normalize = FixedNormalize(scalar_mean, scalar_std)
        self.cmvn = bool(cmvn)
        self.pos_enc = _SinusoidalPositionalEncoding(channels)
        self.input_proj = nn.Linear(n_lfcc, channels)
        self.blocks = nn.Sequential(*[
            _ConformerBlock(channels, n_heads, ff_expansion, conv_kernel, dropout)
            for _ in range(n_layers)])
        self.trunk = nn.Sequential(
            nn.Linear(3 * channels + n_scalars, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.real_fake_head = nn.Linear(hidden, num_classes)
        self.attack_type_head = nn.Linear(hidden, num_classes)

    def normalize_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, T, C): sequence normalization (and optional in-graph CMVN)."""
        x = self.seq_normalize(seq)
        return per_utterance_cmvn(x) if self.cmvn else x

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pos_enc(self.input_proj(self.normalize_sequence(seq)))  # (B, T, C)
        x = self.blocks(x)
        mean = x.mean(dim=1)
        std = torch.sqrt(((x - mean.unsqueeze(1)) ** 2).mean(dim=1) + 1e-5)
        pooled = torch.cat([mean, std, x.amax(dim=1)], dim=1)
        trunk_out = self.trunk(torch.cat([pooled, self.scalar_normalize(scalars)], dim=1))
        return self.real_fake_head(trunk_out), self.attack_type_head(trunk_out)


ARCHS = ("seqcnn_v1", "seqtcn_v2", "conformer_v1")


def build_model_from_norm_stats(norm_stats: dict) -> nn.Module:
    """Rebuilds an (untrained) sequence model from a run's norm_stats.npz."""
    arch = str(norm_stats["arch"]) if "arch" in norm_stats else DEFAULT_ARCH
    common = dict(n_lfcc=int(norm_stats["n_lfcc"]), n_scalars=int(norm_stats["n_scalars"]),
                  seq_mean=norm_stats["seq_mean"], seq_std=norm_stats["seq_std"],
                  scalar_mean=norm_stats["scalar_mean"], scalar_std=norm_stats["scalar_std"])
    if arch == "seqcnn_v1":
        return VoiceGuardSeqCNN(n_frames=int(norm_stats["n_frames"]), **common)
    if arch == "seqtcn_v2":
        # capacity/regularization the run was trained with (absent for v12/v13:
        # those predate the kwargs, and the defaults reproduce them exactly)
        channels = int(norm_stats["channels"]) if "channels" in norm_stats else 64
        dilations = (tuple(int(d) for d in norm_stats["dilations"])
                     if "dilations" in norm_stats else (1, 2, 4, 8, 16))
        dropout = float(norm_stats["dropout"]) if "dropout" in norm_stats else 0.1
        hidden = int(norm_stats["hidden"]) if "hidden" in norm_stats else 64
        stochastic_depth = float(norm_stats["stochastic_depth"]) if "stochastic_depth" in norm_stats else 0.0
        cmvn = bool(np.asarray(norm_stats["cmvn"]).item()) if "cmvn" in norm_stats else False
        se = bool(np.asarray(norm_stats["se"]).item()) if "se" in norm_stats else False
        return VoiceGuardSeqTCN(**common, channels=channels, dilations=dilations,
                                dropout=dropout, hidden=hidden,
                                stochastic_depth=stochastic_depth, cmvn=cmvn, se=se)
    if arch == "conformer_v1":
        channels = int(norm_stats["channels"]) if "channels" in norm_stats else 48
        n_heads = int(norm_stats["n_heads"]) if "n_heads" in norm_stats else 4
        n_layers = int(norm_stats["n_layers"]) if "n_layers" in norm_stats else 2
        ff_expansion = int(norm_stats["ff_expansion"]) if "ff_expansion" in norm_stats else 4
        conv_kernel = int(norm_stats["conv_kernel"]) if "conv_kernel" in norm_stats else 31
        dropout = float(norm_stats["dropout"]) if "dropout" in norm_stats else 0.1
        hidden = int(norm_stats["hidden"]) if "hidden" in norm_stats else 64
        cmvn = bool(np.asarray(norm_stats["cmvn"]).item()) if "cmvn" in norm_stats else False
        return VoiceGuardConformer(**common, channels=channels, n_heads=n_heads,
                                   n_layers=n_layers, ff_expansion=ff_expansion,
                                   conv_kernel=conv_kernel, dropout=dropout,
                                   hidden=hidden, cmvn=cmvn)
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
