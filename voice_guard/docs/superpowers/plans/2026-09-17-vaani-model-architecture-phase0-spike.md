# VAANI Model Architecture — Phase 0 Feasibility Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the combined Idea-1/Idea-3 feasibility spike — a small-scale raw-waveform AASIST-lite trunk with three depth-staged auxiliary heads (VAD / human-fake / language) plus a gradient-reversal adversarial style-suppression head — and score it against the same `measure_confound.py`-lineage gate the deployed model has failed three times, so the project can decide (data-backed, not felt) whether to commit the rest of the 3-month architecture budget or pivot to corpus/hard-negative work instead. Also closes the one loose end on the already-shipped per-speaker calibration track (Idea 1, Track 1): its on-device manual check.

**Architecture:** Everything in this plan is additive and spike-scoped: new files only, nothing in the production LFCC pipeline (`dataset.py`, `feature_cache.py`, `train_seq_cnn.py`, `evaluate.py`, `model.py`) is modified. A parallel raw-PCM cache (`raw_pcm_cache.py`) reuses the existing windowing primitives (`dataset.load_audio`, `trim_edge_silence`, `apply_channel`, `pad_to_window`, `stable_seed`) without touching `dataset.Window` or its 20+ existing tests. A new model file (`spike_model.py`) builds the AASIST-lite trunk (SincNet-style frontend + residual blocks banded into early/mid/late) and a `MultiHeadSpike` wrapper with the VAD, human-fake, language and GRL-style-adversarial heads. The confound *gate itself* is reused unmodified from `evaluate.py` (`confound_table`) — this plan does not reimplement the statistics, only feeds them new numbers.

**Tech Stack:** Python, PyTorch, numpy, soundfile/librosa (existing deps only — no new packages).

**Spec:** `voice_guard/docs/superpowers/specs/2026-09-17-vaani-model-architecture-brainstorm.md` (Ideas 1–3) and the consolidated recommendation given in this conversation (backbone = Idea 1's gated phased plan; Phase 0's spike = Idea 3 Approach C trained on an AASIST-lite trunk; three-signal gate; calibration ships independently). Executors should read Idea 1 and Idea 3 in full before starting — this plan implements their "Phase 0" and "Track 1 verification" only, not Phases 1–5 (see the Roadmap section at the end, which is deliberately NOT a task list).

## Global Constraints

- **Channel policy (`eval_protocol.py`, user directive 2026-09-11): no eval or training run may be clean-audio-only unless `--application bank`.** Every new script that builds a cache or scores a model must resolve its channel list through `eval_protocol.resolve_channels` (or hard-code a channel list that already includes at least one of `eval_protocol.PHONE_CHANNELS`) — never silently default to `[None]`.
- **Determinism:** every random draw must be seeded from `dataset.stable_seed(...)`/`dataset.stable_unit(...)`, matching the existing codebase convention, so cache contents and pseudo-labels are reproducible run-to-run without being committed to disk.
- **This spike's raw-PCM cache is disposable, not production infra.** No feature-version fingerprinting, no `revalidate()`, no multi-recipe scoping like `feature_cache.py` — if code changes, delete and rebuild the cache directory. State this in every new module's docstring so nobody later assumes it has `feature_cache.py`'s guarantees.
- **`float32` throughout** (existing convention — `dataset.py`'s comment: float64 sequences doubled peak RAM on a full-corpus run).
- **Existing test suites (`test_dataset_integrity.py`, `test_feature_cache.py`, `test_train_seq_cnn.py`, `test_evaluate.py`, etc.) must stay green** — this plan touches none of their subject files, so a regression there means a task broke an import path or shared constant; treat that as a bug in the task, not an acceptable side effect.
- Run all new tests with `pytest <file> -v` from `voice_guard/model_training/` (matches the existing suite's working directory).

## File Structure

| File | Responsibility |
|---|---|
| `model_training/grl.py` | Gradient-reversal layer (forward = identity, backward = negate-and-scale). |
| `model_training/ramp_schedule.py` | DANN-style sigmoid ramp for `λ_style` over training progress. |
| `model_training/spike_model.py` | `AASISTLiteSpike` (raw-waveform SincNet + banded residual trunk) and `MultiHeadSpike` (VAD/human-fake/language/style-adversarial heads on top of it). |
| `model_training/gradient_diagnostics.py` | Per-band gradient-cosine-similarity diagnostic (Idea 3's "gradient analysis hook"). |
| `model_training/raw_pcm_cache.py` | Raw-waveform windowing (mirrors `dataset.process_file` without touching it), VAD pseudo-labels, language labels from `source_set`, on-disk cache + `RawPCMCollection`. |
| `model_training/train_spike_aasist.py` | Training entrypoint: builds the model, runs the joint loss, logs the gradient diagnostic, checkpoints. |
| `model_training/score_spike_confound.py` | Scores a spike checkpoint on the held-out split and runs it through `evaluate.confound_table` (imported, not reimplemented). |
| `model_training/export_spike_onnx.py` | Best-effort ONNX export of a spike checkpoint + a CPU latency proxy benchmark. |
| `model_training/docs/2026-09-XX-phase0-spike-decision.md` | The recorded go/no-go decision (Task 10's deliverable). |

---

### Task 1: Gradient Reversal Layer

**Files:**
- Create: `model_training/grl.py`
- Test: `model_training/test_grl.py`

**Interfaces:**
- Produces: `GradientReversalLayer(nn.Module)` with a settable `.lambda_: float` attribute and `forward(x: torch.Tensor) -> torch.Tensor` (identity in the forward direction; backward multiplies the incoming gradient by `-self.lambda_`).

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_grl.py
from __future__ import annotations

import torch

from grl import GradientReversalLayer


def test_forward_is_identity():
    grl = GradientReversalLayer(lambda_=0.7)
    x = torch.randn(4, 8, requires_grad=True)
    out = grl(x)
    assert torch.equal(out, x)


def test_backward_negates_and_scales_gradient():
    grl = GradientReversalLayer(lambda_=0.5)
    x = torch.randn(4, 8, requires_grad=True)
    out = grl(x)
    out.backward(torch.ones_like(out))
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))


def test_lambda_is_mutable_without_rebuilding_the_layer():
    grl = GradientReversalLayer(lambda_=0.0)
    x = torch.randn(2, 2, requires_grad=True)
    grl(x).backward(torch.ones(2, 2))
    assert torch.allclose(x.grad, torch.zeros(2, 2))
    x.grad = None
    grl.lambda_ = 2.0
    grl(x).backward(torch.ones(2, 2))
    assert torch.allclose(x.grad, torch.full((2, 2), -2.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_grl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grl'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/grl.py
"""Gradient Reversal Layer (Ganin & Lempitsky, 2015 — DANN).

Forward pass is the identity. Backward pass multiplies the incoming
gradient by -lambda_, so a loss computed on top of this layer trains its
OWN parameters normally (their gradient never crosses this boundary) while
training whatever feeds INTO this layer adversarially — to make that
upstream representation *uninformative* about what the downstream loss
predicts.

Idea 3 (2026-09-17-vaani-model-architecture-brainstorm.md) uses this to
suppress style-cue information (pauseRatio/energyVariance/zcrVariance) at
the mid-band tap of the AASIST-lite trunk, targeting the entity-vs-style
confound directly rather than hoping it falls out of task separation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """`lambda_` is a plain mutable float attribute (not a buffer/parameter)
    so a training loop can ramp it every step without touching the
    optimizer or the autograd graph structure."""

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_grl.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model_training/grl.py model_training/test_grl.py
git commit -m "feat(voice_guard): add gradient reversal layer for style-adversarial training"
```

---

### Task 2: DANN-style λ_style ramp schedule

**Files:**
- Create: `model_training/ramp_schedule.py`
- Test: `model_training/test_ramp_schedule.py`

**Interfaces:**
- Produces: `dann_ramp(progress: float, max_lambda: float, steepness: float = 10.0) -> float`, `progress` in `[0, 1]` (fraction of total training steps elapsed).

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_ramp_schedule.py
from __future__ import annotations

import math

from ramp_schedule import dann_ramp


def test_starts_at_zero():
    assert dann_ramp(0.0, max_lambda=1.0) == 0.0


def test_approaches_max_lambda_at_progress_one():
    assert math.isclose(dann_ramp(1.0, max_lambda=1.0), 1.0, abs_tol=1e-3)


def test_monotonically_increasing():
    xs = [dann_ramp(p / 10, max_lambda=2.0) for p in range(11)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def test_scales_with_max_lambda():
    assert math.isclose(dann_ramp(0.5, max_lambda=4.0), 4.0 * dann_ramp(0.5, max_lambda=1.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_ramp_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ramp_schedule'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/ramp_schedule.py
"""DANN-style (Ganin et al. 2016, eq. 15) sigmoid ramp for an adversarial
loss weight: starts at 0 (no adversarial pressure while the trunk is
finding its footing), approaches max_lambda smoothly. `steepness` controls
how early the ramp bites, per Idea 3's note that ramp SHAPE (not just the
final value) changes convergence stability."""
from __future__ import annotations

import math


def dann_ramp(progress: float, max_lambda: float, steepness: float = 10.0) -> float:
    """progress in [0, 1]: fraction of total training steps elapsed."""
    progress = min(max(progress, 0.0), 1.0)
    return max_lambda * (2.0 / (1.0 + math.exp(-steepness * progress)) - 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_ramp_schedule.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model_training/ramp_schedule.py model_training/test_ramp_schedule.py
git commit -m "feat(voice_guard): add DANN-style ramp schedule for style-adversarial lambda"
```

---

### Task 3: AASIST-lite trunk + multi-head spike model

**Files:**
- Create: `model_training/spike_model.py`
- Test: `model_training/test_spike_model.py`

**Interfaces:**
- Consumes: `GradientReversalLayer` from `grl.py` (Task 1).
- Produces:
  - `SPIKE_SAMPLE_RATE = 16000`, `SPIKE_WINDOW_SAMPLES = 48000` (must match `dataset.SAMPLE_RATE`/`dataset.WINDOW_SAMPLES` — asserted in a test, not hard-imported, to keep this file import-independent of the production dataset module).
  - `class AASISTLiteSpike(nn.Module)`: `forward(self, pcm: torch.Tensor) -> dict[str, torch.Tensor]` where `pcm` is `(B, 48000)`. Returns `{"early_band": (B, C_e, T_e), "mid_band": (B, C_m, T_m), "late_band": (B, C_l, T_l)}`.
  - `class MultiHeadSpike(nn.Module)`: `forward(self, pcm: torch.Tensor) -> dict[str, torch.Tensor]` returning `{"vad_logits": (B, T_e), "human_fake_logits": (B, 2), "language_logits": (B, 2), "style_pred": (B, 3)}`. Has a `.style_grl: GradientReversalLayer` attribute a training loop sets `.lambda_` on directly.
  - `IGNORE_LANGUAGE = -100` (mirrors `dataset.IGNORE_ATTACK_TYPE`'s masking convention).
  - `STYLE_TARGET_COLUMNS = (0, 1, 2)` — indices into the existing 6-column `scalars` array (`pauseRatio, energyVariance, zcrVariance`; see `evaluate.CONFOUND_FEATURES`), the three continuous style-adversarial regression targets.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_spike_model.py
"""Shape/gradient-flow tests for the Phase-0 spike architecture (Idea 3
Approach C on an AASIST-lite trunk). Not a claim these numbers are good —
only that the graph is wired correctly (right shapes, GRL actually
reverses gradient into the trunk, masked language loss ignores -100)."""
from __future__ import annotations

import torch
from torch import nn

from spike_model import IGNORE_LANGUAGE, STYLE_TARGET_COLUMNS, AASISTLiteSpike, MultiHeadSpike


def test_trunk_returns_three_bands_with_decreasing_time_resolution():
    trunk = AASISTLiteSpike()
    bands = trunk(torch.randn(2, 48000))
    assert set(bands) == {"early_band", "mid_band", "late_band"}
    t_early, t_mid, t_late = (bands[k].shape[-1] for k in ("early_band", "mid_band", "late_band"))
    assert t_early >= t_mid >= t_late > 0


def test_multihead_output_shapes():
    model = MultiHeadSpike()
    out = model(torch.randn(3, 48000))
    assert out["human_fake_logits"].shape == (3, 2)
    assert out["language_logits"].shape == (3, 2)
    assert out["style_pred"].shape == (3, len(STYLE_TARGET_COLUMNS))
    assert out["vad_logits"].dim() == 2 and out["vad_logits"].shape[0] == 3


def test_grl_lambda_zero_means_style_loss_does_not_move_trunk():
    model = MultiHeadSpike()
    model.style_grl.lambda_ = 0.0
    pcm = torch.randn(2, 48000, requires_grad=False)
    out = model(pcm)
    style_target = torch.zeros_like(out["style_pred"])
    loss_style = nn.functional.mse_loss(out["style_pred"], style_target)
    trunk_param = next(model.trunk.mid_blocks.parameters())
    trunk_param.grad = None
    loss_style.backward()
    assert trunk_param.grad is None or torch.allclose(trunk_param.grad, torch.zeros_like(trunk_param.grad), atol=1e-6)


def test_grl_nonzero_lambda_pushes_trunk_gradient_opposite_to_head_gradient():
    model = MultiHeadSpike()
    model.style_grl.lambda_ = 1.0
    pcm = torch.randn(2, 48000)
    out = model(pcm)
    style_target = torch.ones_like(out["style_pred"])
    loss_style = nn.functional.mse_loss(out["style_pred"], style_target)
    loss_style.backward()
    # head parameters got a normal (non-reversed) gradient: they exist and are nonzero
    head_grad = next(model.style_head.parameters()).grad
    assert head_grad is not None and head_grad.abs().sum() > 0
    # trunk parameters upstream of the GRL also received a (reversed) gradient
    trunk_grad = next(model.trunk.mid_blocks.parameters()).grad
    assert trunk_grad is not None and trunk_grad.abs().sum() > 0


def test_language_loss_ignores_unlabeled_examples():
    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([0, 1, IGNORE_LANGUAGE, IGNORE_LANGUAGE])
    loss = nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE_LANGUAGE)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[2:].abs().sum() == 0  # ignored rows carry no gradient
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_spike_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spike_model'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/spike_model.py
"""Phase-0 feasibility-spike architecture: a small raw-waveform, AASIST-
FAMILY trunk (SincNet-style learnable bandpass frontend + banded residual
Conv1d blocks — NOT the full published AASIST graph-attention model; that
is Phase 1's job if this spike clears its gate) with three depth-staged
auxiliary heads and a gradient-reversal style-adversarial head (Idea 3
Approach C, docs/superpowers/specs/2026-09-17-vaani-model-architecture-
brainstorm.md).

Deliberately independent of dataset.py/model.py: consumes raw PCM
(B, 48000) directly, not the LFCC-sequence ONNX contract the deployed
model uses. This is a spike, not a deployment candidate — if it clears
the Phase 0 gate, Phase 1 decides the real production architecture.

IMPORTANT — the doc's loss formula `L = ... - lambda_style * L_style_adv`
describes the NET EFFECT on the trunk, not literal code. The training
script (train_spike_aasist.py) must ADD loss_style_adv like every other
term; GradientReversalLayer is what negates the gradient specifically on
the path back into the trunk. Subtracting the term in code would also
flip the sign of the style head's OWN parameter gradient, training it to
get WORSE at predicting style and defeating the point of having a real
adversary.
"""
from __future__ import annotations

import torch
from torch import nn

from grl import GradientReversalLayer

IGNORE_LANGUAGE = -100
STYLE_TARGET_COLUMNS = (0, 1, 2)  # pauseRatio, energyVariance, zcrVariance (evaluate.CONFOUND_FEATURES order)


class _SincConv1d(nn.Module):
    """Learnable bandpass filterbank frontend (Ravanelli & Bengio, SincNet).
    Each output channel is a sinc-based bandpass filter parameterized by a
    learnable (low, band) cutoff pair in normalized frequency, windowed
    with a Hamming window. Operates on raw (B, 1, N) audio."""

    def __init__(self, out_channels: int, kernel_size: int = 251, stride: int = 10, sample_rate: int = 16000):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size {kernel_size} must be odd"
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        # Initialize filters evenly spaced over the mel-ish range 0..sample_rate/2.
        low_hz = torch.linspace(30, sample_rate / 2 - 30, out_channels)
        band_hz = torch.full((out_channels,), (sample_rate / 2 - 60) / out_channels)
        self.low_hz_ = nn.Parameter(low_hz)
        self.band_hz_ = nn.Parameter(band_hz)
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("n_", n)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * torch.pi * (n + kernel_size / 2) / kernel_size))

    def _filters(self) -> torch.Tensor:
        low = self.low_hz_.abs() + 30.0
        high = torch.clamp(low + self.band_hz_.abs(), 30.0, self.sample_rate / 2.0)
        n = self.n_ / self.sample_rate
        # sinc(2*f*n) for f=high minus f=low, safe at n=0
        def sinc_term(f):
            x = 2 * f.unsqueeze(1) * n.unsqueeze(0)
            out = torch.sin(x) / x.clamp_min(1e-8)
            out[:, self.kernel_size // 2] = 1.0
            return out

        band_pass = 2 * high.unsqueeze(1) * sinc_term(high) - 2 * low.unsqueeze(1) * sinc_term(low)
        band_pass = band_pass * self.window_.unsqueeze(0)
        band_pass = band_pass / band_pass.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        return band_pass.unsqueeze(1)  # (out_channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv1d(x, self._filters(), stride=self.stride, padding=self.kernel_size // 2)


class _Band(nn.Module):
    """One residual band: N Conv1d(k=3)+BN+ReLU blocks at constant width,
    optionally downsampling by 2 at the first block (stride)."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int, downsample: bool, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_blocks):
            stride = 2 if (downsample and i == 0) else 1
            c_in = in_ch if i == 0 else out_ch
            layers += [nn.Conv1d(c_in, out_ch, kernel_size=3, stride=stride, padding=1),
                       nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AASISTLiteSpike(nn.Module):
    """Raw waveform (B, 48000) -> {early_band, mid_band, late_band}, each
    (B, C, T) with decreasing T (each band downsamples by 2 vs. the last).
    """

    def __init__(self, sinc_channels: int = 32, early_channels: int = 32, mid_channels: int = 64,
                 late_channels: int = 64, n_blocks_per_band: int = 2, dropout: float = 0.1):
        super().__init__()
        self.frontend = _SincConv1d(sinc_channels)
        self.frontend_act = nn.Sequential(nn.BatchNorm1d(sinc_channels), nn.ReLU())
        self.early_band = _Band(sinc_channels, early_channels, n_blocks_per_band, downsample=False, dropout=dropout)
        self.mid_band = _Band(early_channels, mid_channels, n_blocks_per_band, downsample=True, dropout=dropout)
        self.late_band = _Band(mid_channels, late_channels, n_blocks_per_band, downsample=True, dropout=dropout)

    def forward(self, pcm: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.frontend_act(self.frontend(pcm.unsqueeze(1)))
        early = self.early_band(x)
        mid = self.mid_band(early)
        late = self.late_band(mid)
        return {"early_band": early, "mid_band": mid, "late_band": late}


def _pool(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) -> (B, 2C): mean+max pooled over time."""
    return torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)


class MultiHeadSpike(nn.Module):
    """Idea 3 Approach C: shared AASISTLiteSpike trunk, three depth-tapped
    auxiliary heads (VAD at early band, human/fake at mid band, language
    at late band) trained jointly, plus a GRL-adversarial style predictor
    tapped at the SAME mid-band point as the human/fake head."""

    def __init__(self, trunk: AASISTLiteSpike | None = None, n_languages: int = 2, dropout: float = 0.1):
        super().__init__()
        self.trunk = trunk or AASISTLiteSpike()
        early_ch = self.trunk.early_band.net[-4].out_channels  # last Conv1d's out_channels
        mid_ch = self.trunk.mid_band.net[-4].out_channels
        late_ch = self.trunk.late_band.net[-4].out_channels

        self.vad_head = nn.Conv1d(early_ch, 1, kernel_size=1)  # per-timestep frame VAD logit
        self.human_fake_head = nn.Sequential(nn.Linear(2 * mid_ch, 32), nn.ReLU(), nn.Dropout(dropout),
                                              nn.Linear(32, 2))
        self.language_head = nn.Sequential(nn.Linear(2 * late_ch, 32), nn.ReLU(), nn.Dropout(dropout),
                                            nn.Linear(32, n_languages))
        self.style_grl = GradientReversalLayer(lambda_=0.0)
        self.style_head = nn.Sequential(nn.Linear(2 * mid_ch, 32), nn.ReLU(),
                                         nn.Linear(32, len(STYLE_TARGET_COLUMNS)))

    def forward(self, pcm: torch.Tensor) -> dict[str, torch.Tensor]:
        bands = self.trunk(pcm)
        mid_pooled = _pool(bands["mid_band"])
        return {
            "vad_logits": self.vad_head(bands["early_band"]).squeeze(1),
            "human_fake_logits": self.human_fake_head(mid_pooled),
            "language_logits": self.language_head(_pool(bands["late_band"])),
            "style_pred": self.style_head(self.style_grl(mid_pooled)),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_spike_model.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model_training/spike_model.py model_training/test_spike_model.py
git commit -m "feat(voice_guard): add AASIST-lite trunk + depth-staged multi-head spike model"
```

---

### Task 4: Gradient-correlation diagnostic

**Files:**
- Create: `model_training/gradient_diagnostics.py`
- Test: `model_training/test_gradient_diagnostics.py`

**Interfaces:**
- Consumes: any two scalar loss tensors sharing part of the autograd graph, and a list of `nn.Parameter` to probe (e.g. `list(model.trunk.mid_band.parameters())`).
- Produces: `param_gradient_vector(loss, params, retain_graph=True) -> torch.Tensor` (flat, concatenated, zero-filled for unused params) and `gradient_cosine_similarity(loss_a, loss_b, params, retain_graph=True) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_gradient_diagnostics.py
from __future__ import annotations

import torch
from torch import nn

from gradient_diagnostics import gradient_cosine_similarity, param_gradient_vector


def _tiny_net():
    return nn.Linear(4, 4, bias=False)


def test_identical_losses_have_cosine_similarity_one():
    net = _tiny_net()
    x = torch.randn(3, 4)
    y = net(x).sum()
    sim = gradient_cosine_similarity(y, y, list(net.parameters()), retain_graph=True)
    assert sim > 0.999


def test_opposite_losses_have_cosine_similarity_minus_one():
    net = _tiny_net()
    x = torch.randn(3, 4)
    out = net(x)
    loss_a = out.sum()
    loss_b = -out.sum()
    sim = gradient_cosine_similarity(loss_a, loss_b, list(net.parameters()), retain_graph=True)
    assert sim < -0.999


def test_param_gradient_vector_has_expected_length():
    net = _tiny_net()
    loss = net(torch.randn(2, 4)).sum()
    vec = param_gradient_vector(loss, list(net.parameters()), retain_graph=True)
    n_params = sum(p.numel() for p in net.parameters())
    assert vec.shape == (n_params,)


def test_unused_param_contributes_zero_not_an_error():
    net = _tiny_net()
    unused = nn.Linear(4, 4, bias=False)
    loss = net(torch.randn(2, 4)).sum()
    vec = param_gradient_vector(loss, list(net.parameters()) + list(unused.parameters()), retain_graph=True)
    assert torch.allclose(vec[-unused.weight.numel():], torch.zeros(unused.weight.numel()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_gradient_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gradient_diagnostics'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/gradient_diagnostics.py
"""Idea 3's "gradient analysis hook": quantifies, mid-training, whether the
human/fake decision and the style-adversarial signal are pulling on the
SAME trunk parameters in the SAME direction. High cosine similarity early
that drops as lambda_style ramps is quantitative evidence the confound is
unwinding — cheaper and earlier than waiting for evaluate.py's end-of-run
confound rows. Near-zero correlation even pre-ramp is evidence the
confound is not a simple linear entanglement at this depth (see the
brainstorm doc's "What would change this recommendation").

Uses parameter-space gradients (not activation-space): simpler to wire up
correctly (no retain_grad bookkeeping on an intermediate tensor) and
directly answers "are these two losses pulling the same trunk weights the
same way," which is what the diagnostic needs.
"""
from __future__ import annotations

import torch
from torch import nn


def param_gradient_vector(loss: torch.Tensor, params: list[nn.Parameter], retain_graph: bool = True) -> torch.Tensor:
    """Flat, concatenated gradient of `loss` w.r.t. `params`, in `params`
    order. A parameter loss doesn't depend on contributes zeros (allow_unused),
    never an error -- callers may probe params spanning multiple heads."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def gradient_cosine_similarity(loss_a: torch.Tensor, loss_b: torch.Tensor, params: list[nn.Parameter],
                                retain_graph: bool = True) -> float:
    va = param_gradient_vector(loss_a, params, retain_graph=retain_graph)
    vb = param_gradient_vector(loss_b, params, retain_graph=retain_graph)
    return torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_gradient_diagnostics.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model_training/gradient_diagnostics.py model_training/test_gradient_diagnostics.py
git commit -m "feat(voice_guard): add gradient-correlation diagnostic for the confound spike"
```

---

### Task 5: Raw-window helpers — VAD pseudo-labels and language labels

**Files:**
- Create: `model_training/raw_pcm_cache.py` (this task: the pure-function half only — `raw_process_file`, `vad_pseudo_labels`, `language_for_source_set`; Task 6 adds the on-disk cache/collection to the same file)
- Test: `model_training/test_raw_pcm_cache.py`

**Interfaces:**
- Consumes: `dataset.FileSpec`, `dataset.load_audio`, `dataset.trim_edge_silence`, `dataset.apply_channel`, `dataset.pad_to_window`, `dataset.stable_seed`, `dataset.channel_name`, `dataset.noise_floor`, `dataset.WINDOW_SAMPLES`, `dataset.SAMPLE_RATE`, `dataset.MIN_CLIP_SECONDS`, `dataset.STATUS_OK`/`STATUS_TOO_SHORT`/`STATUS_SILENT`/`STATUS_LOAD_ERROR` (all existing, read-only imports — nothing in `dataset.py` changes).
- Produces:
  - `IGNORE_LANGUAGE = -100`, `LANGUAGE_BY_SOURCE_SET: dict[str, int]` (`{"noiseaug_train_en": 0, "accent_real_en_native": 0, "accent_fake_en_native": 0, "noiseaug_train_hi": 1, "accent_real_hi_native": 1, "accent_fake_hi_native": 1}`), `language_for_source_set(source_set: str) -> int`.
  - `VAD_FRAME_HOP = 320` (20 ms at 16 kHz, matches the noise-floor frame size `dataset.noise_floor` already uses), `vad_pseudo_labels(pcm: np.ndarray, threshold_db: float = 12.0) -> np.ndarray` — `(len(pcm) // VAD_FRAME_HOP,)` uint8 array, 1 = frame RMS more than `threshold_db` dB above `dataset.noise_floor(pcm)`, else 0.
  - `@dataclass RawWindow`: `pcm: np.ndarray` (48000,), `label: int`, `file_id: str`, `source_set: str`, `channel: str`, `window_index: int`, `pad_fraction: float`.
  - `raw_process_file(spec, recipe, base_seed: int = 0) -> tuple[list[RawWindow], str, float]` — same return contract and windowing behavior as `dataset.process_file`, minus feature extraction (returns raw `pcm` instead of `lfcc_seq`/`scalars`). Must produce IDENTICAL segment boundaries and pad/crop decisions as `dataset.process_file` for the same inputs (same seeds, same primitives) — verified by the parity test below, since a spike whose windows don't match the production windowing would not be a fair test of the architecture.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_raw_pcm_cache.py
from __future__ import annotations

import numpy as np

from dataset import FileSpec, WINDOW_SAMPLES
from raw_pcm_cache import (
    IGNORE_LANGUAGE,
    VAD_FRAME_HOP,
    language_for_source_set,
    raw_process_file,
    vad_pseudo_labels,
)


def test_language_for_known_source_set():
    assert language_for_source_set("noiseaug_train_en") == 0
    assert language_for_source_set("noiseaug_train_hi") == 1


def test_language_for_unlabeled_source_set_is_ignored():
    assert language_for_source_set("real") == IGNORE_LANGUAGE
    assert language_for_source_set("fake2021") == IGNORE_LANGUAGE


def test_vad_pseudo_labels_flags_loud_region_as_speech():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1e-4, WINDOW_SAMPLES // 2).astype(np.float32)
    loud = rng.normal(0, 0.3, WINDOW_SAMPLES // 2).astype(np.float32)
    pcm = np.concatenate([quiet, loud])
    labels = vad_pseudo_labels(pcm)
    n_frames = WINDOW_SAMPLES // VAD_FRAME_HOP
    assert labels.shape == (n_frames,)
    first_half_speech_rate = labels[: n_frames // 2].mean()
    second_half_speech_rate = labels[n_frames // 2 :].mean()
    assert second_half_speech_rate > first_half_speech_rate


def test_raw_process_file_matches_lfcc_process_file_window_count(tmp_path):
    import soundfile as sf

    from dataset import process_file

    rng = np.random.default_rng(1)
    pcm = rng.normal(0, 0.2, WINDOW_SAMPLES * 2).astype(np.float32)
    path = tmp_path / "clip.wav"
    sf.write(str(path), pcm, 16000)
    spec = FileSpec(path=str(path), file_id="clip", label=0, source_set="real")

    lfcc_windows, lfcc_status, lfcc_dur = process_file(spec, None, base_seed=0)
    raw_windows, raw_status, raw_dur = raw_process_file(spec, None, base_seed=0)

    assert raw_status == lfcc_status
    assert raw_dur == lfcc_dur
    assert len(raw_windows) == len(lfcc_windows)
    for rw, lw in zip(raw_windows, lfcc_windows):
        assert rw.pad_fraction == lw.pad_fraction
        assert rw.pcm.shape == (WINDOW_SAMPLES,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_raw_pcm_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raw_pcm_cache'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/raw_pcm_cache.py
"""Spike-scoped raw-waveform cache for the Phase-0 AASIST-lite spike
(Idea 1 / Idea 3, docs/superpowers/specs/2026-09-17-vaani-model-
architecture-brainstorm.md). DISPOSABLE: no feature-version fingerprinting
like feature_cache.py's production LFCC cache -- delete and rebuild this
cache directory if this file or dataset.py's windowing primitives change.

Mirrors dataset.process_file's windowing control flow using the SAME
primitives (load_audio/trim_edge_silence/apply_channel/pad_to_window/
stable_seed) so spike windows are boundary-for-boundary identical to what
the production LFCC pipeline would window, without importing or modifying
dataset.Window/dataset.process_file (kept untouched: 20+ existing tests
depend on their exact behavior, and this file must never risk them).

Labels this spike needs that don't exist as ground truth in the corpus:
- VAD: no frame-level ground truth exists anywhere in this project. Uses
  an energy-threshold pseudo-label relative to dataset.noise_floor (the
  same quietest-10%-of-20ms-frames measure the production pad/noise code
  already computes) -- a reasonable bootstrap, explicitly NOT ground
  truth (see this plan's Task 10 risk notes).
- language: only a minority of the corpus has a language tag at all
  (noiseaug_train_{en,hi} and the accent_* held-out cells, per corpus.py's
  SetDef list) -- everything else (real, real2021, fake2021, ...) is
  unlabeled. Masked out via IGNORE_LANGUAGE=-100 fed to
  cross_entropy(..., ignore_index=IGNORE_LANGUAGE), the exact pattern
  dataset.IGNORE_ATTACK_TYPE already uses for leave-out attacks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset import (
    MIN_CLIP_SECONDS,
    STATUS_LOAD_ERROR,
    STATUS_OK,
    STATUS_SILENT,
    STATUS_TOO_SHORT,
    WINDOW_SAMPLES,
    FileSpec,
    apply_channel,
    channel_name,
    load_audio,
    noise_floor,
    pad_to_window,
    stable_seed,
    trim_edge_silence,
)

IGNORE_LANGUAGE = -100
LANGUAGE_BY_SOURCE_SET: dict[str, int] = {
    "noiseaug_train_en": 0, "accent_real_en_native": 0, "accent_fake_en_native": 0,
    "noiseaug_train_hi": 1, "accent_real_hi_native": 1, "accent_fake_hi_native": 1,
}
VAD_FRAME_HOP = 320  # 20 ms @ 16 kHz, matches dataset.noise_floor's frame size

SILENCE_TRIM_THRESHOLD = 1e-4  # dataset.py's own module-level constant, duplicated here (read-only reuse)


def language_for_source_set(source_set: str) -> int:
    return LANGUAGE_BY_SOURCE_SET.get(source_set, IGNORE_LANGUAGE)


def vad_pseudo_labels(pcm: np.ndarray, threshold_db: float = 12.0) -> np.ndarray:
    """Per-20ms-frame binary speech/non-speech, thresholded against this
    clip's own noise floor (not a fixed absolute level, so it adapts per
    channel/recipe same as the rest of this project's noise-aware code)."""
    n_frames = len(pcm) // VAD_FRAME_HOP
    floor = max(noise_floor(pcm), 1e-6)
    thresh = floor * (10 ** (threshold_db / 20))
    frames = pcm[: n_frames * VAD_FRAME_HOP].reshape(n_frames, VAD_FRAME_HOP)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return (rms > thresh).astype(np.uint8)


def _fit_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x[:WINDOW_SAMPLES], dtype=np.float32)
    if len(x) < WINDOW_SAMPLES:
        x = np.concatenate([x, np.zeros(WINDOW_SAMPLES - len(x), dtype=np.float32)])
    return x


@dataclass
class RawWindow:
    pcm: np.ndarray  # (48000,) float32, post-channel
    label: int
    file_id: str
    source_set: str
    channel: str
    window_index: int
    pad_fraction: float


def raw_process_file(spec: FileSpec, recipe: str | None, base_seed: int = 0) -> tuple[list[RawWindow], str, float]:
    """Raw-PCM analogue of dataset.process_file: same windowing decisions
    (seeded identically), no LFCC/scalar extraction. Returns
    (windows, status, trimmed_duration_seconds)."""
    chan = channel_name(recipe)
    try:
        pcm = load_audio(spec.path)
    except Exception:
        return [], STATUS_LOAD_ERROR, 0.0
    if not (np.abs(pcm) > SILENCE_TRIM_THRESHOLD).any():
        return [], STATUS_SILENT, len(pcm) / 16000

    file_rng = np.random.default_rng(stable_seed(spec.file_id, base_seed, "trim"))
    pcm = trim_edge_silence(pcm, rng=file_rng)
    duration = len(pcm) / 16000
    if duration < MIN_CLIP_SECONDS:
        return [], STATUS_TOO_SHORT, duration

    chan_rng = np.random.default_rng(stable_seed(spec.file_id, base_seed, "channel", chan))
    from dataset import WINDOW_SECONDS

    segments: list[tuple[np.ndarray, float]] = []
    if duration < WINDOW_SECONDS:
        if spec.keep_short_prob < 1.0:
            from dataset import stable_unit

            if stable_unit(spec.file_id, base_seed, "keep_short") >= spec.keep_short_prob:
                return [], "balanced_out", duration
        padded, n_pad = pad_to_window(pcm, file_rng)
        segments.append((_fit_window(apply_channel(padded, recipe, chan_rng)), n_pad / WINDOW_SAMPLES))
    else:
        degraded = apply_channel(pcm, recipe, chan_rng)
        n_full = max(1, len(degraded) // WINDOW_SAMPLES)
        starts = [i * WINDOW_SAMPLES for i in range(n_full)]
        from dataset import TAIL_MIN_SECONDS

        if len(degraded) - n_full * WINDOW_SAMPLES >= TAIL_MIN_SECONDS * 16000:
            starts.append(len(degraded) - WINDOW_SAMPLES)
        for wi, s in enumerate(starts):
            segments.append((_fit_window(degraded[s:s + WINDOW_SAMPLES]), 0.0))

    windows = [
        RawWindow(pcm=seg, label=spec.label, file_id=spec.file_id, source_set=spec.source_set,
                  channel=chan, window_index=wi, pad_fraction=float(pad_frac))
        for wi, (seg, pad_frac) in enumerate(segments)
    ]
    return windows, STATUS_OK, duration
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_raw_pcm_cache.py -v`
Expected: PASS (4 tests)

**Note for the implementer:** the parity test (`test_raw_process_file_matches_lfcc_process_file_window_count`) only checks window *count* and `pad_fraction`, not sample-for-sample PCM equality with what `extract_lfcc_sequence` was computed from inside `dataset.process_file` — that's intentional (this file doesn't import `dataset.process_file`'s internals to avoid coupling), but if it fails, the most likely cause is the `convert_prob` crop-window branch (Idea 1's training pad-balancing) not being reproduced here. This spike's training run (Task 7) does not need `convert_prob` support (`keep_short_prob`-only balancing is enough for a small-scale spike) — if a future task needs it, add it to `raw_process_file` following `dataset.process_file`'s crop branch exactly, seeded the same way.

- [ ] **Step 5: Commit**

```bash
git add model_training/raw_pcm_cache.py model_training/test_raw_pcm_cache.py
git commit -m "feat(voice_guard): add raw-waveform windowing, VAD pseudo-labels and language labels for the spike"
```

---

### Task 6: Raw-PCM on-disk cache + collection

**Files:**
- Modify: `model_training/raw_pcm_cache.py` (add to the file created in Task 5)
- Test: `model_training/test_raw_pcm_cache.py` (add to the file created in Task 5)

**Interfaces:**
- Consumes: `RawWindow`, `raw_process_file`, `vad_pseudo_labels`, `language_for_source_set` (Task 5); `dataset.FileSpec`.
- Produces:
  - `build_raw_unit(name: str, specs: list[FileSpec], recipe: str | None, cache_root, seed: int = 0) -> Path` — writes `<cache_root>/<name>__<channel>/{pcm.npy, scalars_style.npy, vad.npy, language.npy, meta.npz}` (no manifest/fingerprint machinery — Global Constraints already documents this cache as disposable).
    - `pcm.npy`: `(N, 48000)` float32.
    - `scalars_style.npy`: `(N, 3)` float32 — `pauseRatio, energyVariance, zcrVariance` via `features.extract_scalars` computed on each window's `pcm` (reuses the existing, unmodified `features.py` function — only the 3 style columns are kept, to avoid this spike cache depending on the full 6-column physio extraction it doesn't need).
    - `vad.npy`: object array of per-window `vad_pseudo_labels(pcm)` arrays (ragged only in principle — every window is exactly `WINDOW_SAMPLES` long, so every VAD array is the same length; stored as a regular `(N, WINDOW_SAMPLES // VAD_FRAME_HOP)` uint8 array).
    - `language.npy`: `(N,)` int64.
    - `meta.npz`: `label (N,) int64`, `file_id (N,) <U64`, `source_set (N,) <U64`, `pad_fraction (N,) float32`.
  - `class RawPCMCollection`: `__init__(self, dirs: list[Path])`; attributes `.label`, `.file_id`, `.source_set`, `.pad_fraction`, `.language`, `.n: int` (all in RAM — small); `.get_pcm(idx)` / `.get_style_scalars(idx)` / `.get_vad(idx)` (memory-mapped); `.iter_batches(idx: np.ndarray, batch: int = 256)` yielding `(batch_idx, pcm_batch, style_scalars_batch, vad_batch, language_batch)`.

- [ ] **Step 1: Write the failing tests (append to `test_raw_pcm_cache.py`)**

```python
def test_build_raw_unit_and_collection_roundtrip(tmp_path):
    import soundfile as sf

    from raw_pcm_cache import RawPCMCollection, build_raw_unit

    rng = np.random.default_rng(2)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    for i in range(3):
        pcm = rng.normal(0, 0.2, WINDOW_SAMPLES).astype(np.float32)
        sf.write(str(real_dir / f"f{i}.wav"), pcm, 16000)
    specs = [FileSpec(path=str(p), file_id=p.stem, label=0, source_set="noiseaug_train_en")
             for p in sorted(real_dir.glob("*.wav"))]

    cache_root = tmp_path / "cache"
    unit_dir = build_raw_unit("real_test", specs, None, cache_root, seed=0)

    coll = RawPCMCollection([unit_dir])
    assert coll.n == 3
    assert (coll.label == 0).all()
    assert (coll.language == 0).all()  # noiseaug_train_en -> language 0

    batches = list(coll.iter_batches(np.arange(coll.n), batch=2))
    total = sum(len(b_idx) for b_idx, *_ in batches)
    assert total == coll.n
    _b_idx, pcm_batch, style_batch, vad_batch, lang_batch = batches[0]
    assert pcm_batch.shape[1] == WINDOW_SAMPLES
    assert style_batch.shape[1] == 3
    assert vad_batch.ndim == 2
    assert lang_batch.ndim == 1


def test_build_raw_unit_is_idempotent(tmp_path):
    import soundfile as sf

    from raw_pcm_cache import build_raw_unit

    pcm = np.random.default_rng(3).normal(0, 0.2, WINDOW_SAMPLES).astype(np.float32)
    p = tmp_path / "f.wav"
    sf.write(str(p), pcm, 16000)
    specs = [FileSpec(path=str(p), file_id="f", label=1, source_set="fake")]
    cache_root = tmp_path / "cache"

    d1 = build_raw_unit("fake_test", specs, None, cache_root, seed=0)
    d2 = build_raw_unit("fake_test", specs, None, cache_root, seed=0)
    assert d1 == d2
    assert (d1 / "pcm.npy").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_raw_pcm_cache.py -v`
Expected: FAIL — `build_raw_unit`/`RawPCMCollection` not defined

- [ ] **Step 3: Write the implementation (append to `raw_pcm_cache.py`)**

```python
from pathlib import Path

from features import extract_scalars

STYLE_SCALAR_COLUMNS = (0, 1, 2)  # pauseRatio, energyVariance, zcrVariance out of extract_scalars' 3


def unit_dir(cache_root: Path, name: str, recipe: str | None) -> Path:
    return Path(cache_root) / f"{name}__{channel_name(recipe)}"


def build_raw_unit(name: str, specs: list[FileSpec], recipe: str | None, cache_root: Path, seed: int = 0) -> Path:
    """Builds (if missing) or returns the existing disposable raw-PCM cache
    unit for `specs` under `recipe`. No fingerprinting: delete the
    directory to force a rebuild after any code change."""
    d = unit_dir(cache_root, name, recipe)
    if (d / "meta.npz").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)

    pcm_list, style_list, vad_list, lang_list = [], [], [], []
    labels, file_ids, source_sets, pad_fracs = [], [], [], []
    for spec in specs:
        windows, _status, _dur = raw_process_file(spec, recipe, seed)
        for w in windows:
            pcm_list.append(w.pcm)
            style_list.append(extract_scalars(w.pcm)[list(STYLE_SCALAR_COLUMNS)])
            vad_list.append(vad_pseudo_labels(w.pcm))
            lang_list.append(language_for_source_set(w.source_set))
            labels.append(w.label)
            file_ids.append(w.file_id)
            source_sets.append(w.source_set)
            pad_fracs.append(w.pad_fraction)

    np.save(d / "pcm.npy", np.stack(pcm_list).astype(np.float32) if pcm_list else np.zeros((0, WINDOW_SAMPLES), np.float32))
    np.save(d / "scalars_style.npy", np.stack(style_list).astype(np.float32) if style_list else np.zeros((0, 3), np.float32))
    np.save(d / "vad.npy", np.stack(vad_list).astype(np.uint8) if vad_list else np.zeros((0, WINDOW_SAMPLES // VAD_FRAME_HOP), np.uint8))
    np.save(d / "language.npy", np.asarray(lang_list, dtype=np.int64))
    np.savez(d / "meta.npz", label=np.asarray(labels, dtype=np.int64),
             file_id=np.asarray(file_ids, dtype="<U64"), source_set=np.asarray(source_sets, dtype="<U64"),
             pad_fraction=np.asarray(pad_fracs, dtype=np.float32))
    return d


class RawPCMCollection:
    """Several build_raw_unit directories viewed as one window table.
    Metadata/labels in RAM; pcm/vad/style stay memory-mapped."""

    def __init__(self, dirs: list[Path]):
        self.dirs = [Path(d) for d in dirs]
        pcms, styles, vads, langs = [], [], [], []
        labels, file_ids, source_sets, pad_fracs = [], [], [], []
        for d in self.dirs:
            pcms.append(np.load(d / "pcm.npy", mmap_mode="r"))
            styles.append(np.load(d / "scalars_style.npy", mmap_mode="r"))
            vads.append(np.load(d / "vad.npy", mmap_mode="r"))
            langs.append(np.load(d / "language.npy"))
            with np.load(d / "meta.npz") as z:
                labels.append(z["label"]); file_ids.append(z["file_id"])
                source_sets.append(z["source_set"]); pad_fracs.append(z["pad_fraction"])
        self._pcms, self._styles, self._vads = pcms, styles, vads
        self._offsets = np.cumsum([0] + [len(p) for p in pcms])
        self.language = np.concatenate(langs) if langs else np.zeros(0, np.int64)
        self.label = np.concatenate(labels) if labels else np.zeros(0, np.int64)
        self.file_id = np.concatenate(file_ids) if file_ids else np.zeros(0, dtype="<U64")
        self.source_set = np.concatenate(source_sets) if source_sets else np.zeros(0, dtype="<U64")
        self.pad_fraction = np.concatenate(pad_fracs) if pad_fracs else np.zeros(0, np.float32)
        self.n = int(self._offsets[-1])

    def _locate(self, i: int) -> tuple[int, int]:
        unit = int(np.searchsorted(self._offsets, i, side="right") - 1)
        return unit, i - self._offsets[unit]

    def get_pcm(self, idx: np.ndarray) -> np.ndarray:
        return np.stack([self._pcms[u][j] for u, j in map(self._locate, idx)])

    def get_style_scalars(self, idx: np.ndarray) -> np.ndarray:
        return np.stack([self._styles[u][j] for u, j in map(self._locate, idx)])

    def get_vad(self, idx: np.ndarray) -> np.ndarray:
        return np.stack([self._vads[u][j] for u, j in map(self._locate, idx)])

    def iter_batches(self, idx: np.ndarray, batch: int = 256):
        for start in range(0, len(idx), batch):
            b_idx = idx[start:start + batch]
            yield b_idx, self.get_pcm(b_idx), self.get_style_scalars(b_idx), self.get_vad(b_idx), self.language[b_idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_raw_pcm_cache.py -v`
Expected: PASS (all 6 tests in the file)

- [ ] **Step 5: Commit**

```bash
git add model_training/raw_pcm_cache.py model_training/test_raw_pcm_cache.py
git commit -m "feat(voice_guard): add disposable raw-PCM cache and collection for the spike"
```

---

### Task 7: Spike training entrypoint

**Files:**
- Create: `model_training/train_spike_aasist.py`
- Test: `model_training/test_train_spike_aasist.py`

**Interfaces:**
- Consumes: `MultiHeadSpike`, `IGNORE_LANGUAGE`, `STYLE_TARGET_COLUMNS` (Task 3); `dann_ramp` (Task 2); `gradient_cosine_similarity` (Task 4); `RawPCMCollection`, `build_raw_unit`, `VAD_FRAME_HOP` (Tasks 5–6).
- Produces: `compute_losses(out: dict, y: torch.Tensor, vad_target: torch.Tensor, language: torch.Tensor, style_target: torch.Tensor, lambda_lang: float) -> dict[str, torch.Tensor]` (keys `vad`, `human_fake`, `language`, `style_adv`, `total` — `total` is the SUM, per Task 3's docstring warning, not a difference); a `main()` CLI (`--out`, `--cache-root`, `--epochs`, `--batch-size`, `--lr`, `--lambda-lang`, `--style-lambda-max`, `--grad-diag-every`, `--device`) that trains and checkpoints per epoch to `<out>/epoch_NN.pt` plus `<out>/gradient_diagnostics.jsonl`.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_train_spike_aasist.py
from __future__ import annotations

import torch

from spike_model import IGNORE_LANGUAGE, MultiHeadSpike
from train_spike_aasist import compute_losses


def _fake_output(batch=4, t_early=300):
    return {
        "vad_logits": torch.randn(batch, t_early, requires_grad=True),
        "human_fake_logits": torch.randn(batch, 2, requires_grad=True),
        "language_logits": torch.randn(batch, 2, requires_grad=True),
        "style_pred": torch.randn(batch, 3, requires_grad=True),
    }


def test_total_loss_is_the_sum_of_terms_not_a_difference():
    out = _fake_output()
    y = torch.tensor([0, 1, 0, 1])
    vad_target = torch.randint(0, 2, (4, out["vad_logits"].shape[1])).float()
    language = torch.tensor([0, IGNORE_LANGUAGE, 1, IGNORE_LANGUAGE])
    style_target = torch.randn(4, 3)
    losses = compute_losses(out, y, vad_target, language, style_target, lambda_lang=0.5)
    expected_total = losses["vad"] + losses["human_fake"] + 0.5 * losses["language"] + losses["style_adv"]
    assert torch.allclose(losses["total"], expected_total)


def test_language_loss_is_finite_when_all_examples_unlabeled():
    out = _fake_output()
    y = torch.tensor([0, 1, 0, 1])
    vad_target = torch.zeros(4, out["vad_logits"].shape[1])
    language = torch.full((4,), IGNORE_LANGUAGE)
    style_target = torch.zeros(4, 3)
    losses = compute_losses(out, y, vad_target, language, style_target, lambda_lang=0.5)
    # cross_entropy with every target ignored is documented NaN (dataset.py's own
    # compute_masked_attack_type_loss docstring notes this for the identical pattern)
    assert torch.isnan(losses["language"]) or losses["language"] == 0.0


def test_train_spike_aasist_end_to_end_smoke(tmp_path):
    """Full loop on a synthetic 2-file corpus: must run without error and
    write a checkpoint. Not a claim about model quality."""
    import numpy as np
    import soundfile as sf

    from dataset import WINDOW_SAMPLES
    from raw_pcm_cache import build_raw_unit
    from dataset import FileSpec
    from train_spike_aasist import run_training

    rng = np.random.default_rng(0)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    for i in range(2):
        sf.write(str(real_dir / f"r{i}.wav"), rng.normal(0, 0.2, WINDOW_SAMPLES).astype("float32"), 16000)
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    for i in range(2):
        sf.write(str(fake_dir / f"k{i}.wav"), rng.normal(0, 0.2, WINDOW_SAMPLES).astype("float32"), 16000)

    real_specs = [FileSpec(path=str(p), file_id=p.stem, label=0, source_set="noiseaug_train_en")
                  for p in sorted(real_dir.glob("*.wav"))]
    fake_specs = [FileSpec(path=str(p), file_id=p.stem, label=1, source_set="fake")
                  for p in sorted(fake_dir.glob("*.wav"))]
    cache_root = tmp_path / "cache"
    real_unit = build_raw_unit("real", real_specs, None, cache_root)
    fake_unit = build_raw_unit("fake", fake_specs, None, cache_root)

    out_dir = tmp_path / "run"
    run_training(unit_dirs=[real_unit, fake_unit], out_dir=out_dir, epochs=1, batch_size=2,
                  lr=1e-3, lambda_lang=0.5, style_lambda_max=0.5, grad_diag_every=1, device="cpu")

    assert (out_dir / "epoch_01.pt").exists()
    assert (out_dir / "gradient_diagnostics.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_train_spike_aasist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_spike_aasist'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/train_spike_aasist.py
"""Phase-0 spike training entrypoint (Idea 1 + Idea 3 combined, see
docs/superpowers/specs/2026-09-17-vaani-model-architecture-brainstorm.md
and model_training/docs/2026-09-XX-phase0-spike-decision.md for the
result). Small-scale by design -- this is the feasibility spike, not a
production training run.

Usage:
    python train_spike_aasist.py --out runs/spike_aasist_v1 \\
        --cache-root runs/spike_raw_cache --epochs 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VAD_POOL_STRIDE = None  # set at runtime once the trunk's early-band stride is known


def compute_losses(out: dict, y, vad_target, language, style_target, lambda_lang: float) -> dict:
    import torch
    from torch import nn

    from spike_model import IGNORE_LANGUAGE

    t_pred = out["vad_logits"].shape[1]
    if vad_target.shape[1] != t_pred:
        # pool the frame-level pseudo-labels down to the trunk's early-band
        # time resolution by max-pooling (a window counts as "speech" if any
        # sub-frame in it is)
        factor = vad_target.shape[1] // t_pred
        vad_target = vad_target[:, : factor * t_pred].reshape(vad_target.shape[0], t_pred, factor).amax(dim=2)

    loss_vad = nn.functional.binary_cross_entropy_with_logits(out["vad_logits"], vad_target)
    loss_human_fake = nn.functional.cross_entropy(out["human_fake_logits"], y)
    loss_language = nn.functional.cross_entropy(out["language_logits"], language, ignore_index=IGNORE_LANGUAGE)
    loss_style_adv = nn.functional.mse_loss(out["style_pred"], style_target)

    lang_term = torch.nan_to_num(loss_language, nan=0.0)
    total = loss_vad + loss_human_fake + lambda_lang * lang_term + loss_style_adv
    return {"vad": loss_vad, "human_fake": loss_human_fake, "language": loss_language,
            "style_adv": loss_style_adv, "total": total}


def run_training(unit_dirs: list[Path], out_dir: Path, epochs: int, batch_size: int, lr: float,
                  lambda_lang: float, style_lambda_max: float, grad_diag_every: int, device: str) -> None:
    import torch
    from torch import optim

    from gradient_diagnostics import gradient_cosine_similarity
    from ramp_schedule import dann_ramp
    from raw_pcm_cache import RawPCMCollection, VAD_FRAME_HOP
    from spike_model import MultiHeadSpike

    out_dir.mkdir(parents=True, exist_ok=True)
    coll = RawPCMCollection(unit_dirs)
    model = MultiHeadSpike().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    idx_all = np.arange(coll.n)
    total_steps = max(1, epochs * (len(idx_all) // max(batch_size, 1) + 1))
    step = 0
    diag_path = out_dir / "gradient_diagnostics.jsonl"
    with diag_path.open("w") as diag_f:
        for epoch in range(1, epochs + 1):
            rng = np.random.default_rng(epoch)
            order = rng.permutation(idx_all)
            for b_idx, pcm, _style, vad, lang in coll.iter_batches(order, batch_size):
                if len(b_idx) == 0:
                    continue
                progress = step / total_steps
                model.style_grl.lambda_ = dann_ramp(progress, style_lambda_max)

                pcm_t = torch.from_numpy(pcm).float().to(device)
                y_t = torch.from_numpy(coll.label[b_idx]).long().to(device)
                vad_t = torch.from_numpy(vad).float().to(device)
                lang_t = torch.from_numpy(lang).long().to(device)
                style_t = torch.from_numpy(coll.get_style_scalars(b_idx)).float().to(device)

                out = model(pcm_t)
                losses = compute_losses(out, y_t, vad_t, lang_t, style_t, lambda_lang)

                if grad_diag_every and step % grad_diag_every == 0:
                    mid_params = list(model.trunk.mid_band.parameters())
                    cos = gradient_cosine_similarity(losses["human_fake"], losses["style_adv"], mid_params,
                                                      retain_graph=True)
                    diag_f.write(json.dumps({"epoch": epoch, "step": step, "cosine_human_fake_vs_style": cos,
                                              "style_lambda": model.style_grl.lambda_}) + "\n")

                opt.zero_grad()
                losses["total"].backward()
                opt.step()
                step += 1

            torch.save({"model_state": model.state_dict(), "epoch": epoch}, out_dir / f"epoch_{epoch:02d}.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-lang", type=float, default=0.5)
    ap.add_argument("--style-lambda-max", type=float, default=1.0)
    ap.add_argument("--grad-diag-every", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    unit_dirs = [d for d in Path(args.cache_root).iterdir() if d.is_dir()]
    run_training(unit_dirs, args.out, args.epochs, args.batch_size, args.lr, args.lambda_lang,
                 args.style_lambda_max, args.grad_diag_every, args.device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_train_spike_aasist.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model_training/train_spike_aasist.py model_training/test_train_spike_aasist.py
git commit -m "feat(voice_guard): add Phase-0 spike training entrypoint with gradient diagnostic logging"
```

---

### Task 8: Score a spike checkpoint through the existing confound gate

**Files:**
- Create: `model_training/score_spike_confound.py`
- Test: `model_training/test_score_spike_confound.py`

**Interfaces:**
- Consumes: `evaluate.confound_table`, `evaluate.CONFOUND_FEATURES` (imported unmodified — the gate math is never reimplemented); `eval_stats.compute_eer_threshold`; `RawPCMCollection`, `build_raw_unit`; `MultiHeadSpike`.
- Produces: `score_spike(model, coll, idx, device, batch=256) -> np.ndarray` (p_fake only — mirrors `train_seq_cnn.score`'s batching pattern, adapted to raw PCM); a `main()` CLI (`--checkpoint`, `--cache-root`, `--out`) writing `<out>/spike_confound_report.json` with the same `{"passed": bool, "failed": [...], "rows": [...]}` shape `evaluate.confound_table` already returns.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_score_spike_confound.py
from __future__ import annotations

import numpy as np
import torch

from score_spike_confound import score_spike
from spike_model import MultiHeadSpike


class _FakeCollection:
    """Minimal stand-in for RawPCMCollection, just enough for score_spike."""

    def __init__(self, n):
        self.n = n

    def iter_batches(self, idx, batch=256):
        for start in range(0, len(idx), batch):
            b_idx = idx[start:start + batch]
            pcm = np.random.default_rng(0).normal(0, 0.1, (len(b_idx), 48000)).astype(np.float32)
            yield b_idx, pcm, None, None, None


def test_score_spike_returns_one_probability_per_window():
    model = MultiHeadSpike().eval()
    coll = _FakeCollection(5)
    p = score_spike(model, coll, np.arange(5), device="cpu")
    assert p.shape == (5,)
    assert ((p >= 0) & (p <= 1)).all()


def test_confound_gate_is_imported_not_reimplemented():
    import score_spike_confound
    from evaluate import confound_table

    assert score_spike_confound.confound_table is confound_table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_score_spike_confound.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'score_spike_confound'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/score_spike_confound.py
"""Scores a Phase-0 spike checkpoint on the held-out split and runs it
through evaluate.py's EXISTING confound gate (imported directly -- the
14-row median-split statistics are never reimplemented here, so a PASS/
FAIL means exactly what it means for v11/v12/v13).

Usage:
    python score_spike_confound.py --checkpoint runs/spike_aasist_v1/epoch_10.pt \\
        --cache-root runs/spike_raw_cache_heldout --out runs/spike_aasist_v1_confound
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_stats import compute_eer_threshold
from evaluate import confound_table


def score_spike(model, coll, idx: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    import torch

    model.eval()
    p_fake = []
    with torch.no_grad():
        for b_idx, pcm, *_ in coll.iter_batches(idx, batch):
            if len(b_idx) == 0:
                continue
            out = model(torch.from_numpy(pcm).float().to(device))
            p_fake.append(torch.softmax(out["human_fake_logits"], -1)[:, 1].float().cpu().numpy())
    return np.concatenate(p_fake) if p_fake else np.zeros(0)


def main() -> None:
    import torch

    from raw_pcm_cache import RawPCMCollection
    from spike_model import MultiHeadSpike

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True, help="raw-PCM cache root for the held-out split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = MultiHeadSpike().to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])

    unit_dirs = [d for d in Path(args.cache_root).iterdir() if d.is_dir()]
    coll = RawPCMCollection(unit_dirs)
    idx = np.arange(coll.n)

    p = score_spike(model, coll, idx, args.device)
    y = coll.label
    # pad_fraction from meta, style scalars (pauseRatio/energyVariance/zcrVariance)
    # from the cache's scalars_style.npy; confound_table's CONFOUND_FEATURES
    # indexes 6 columns + pad_fraction -- this spike cache only has the 3 style
    # columns, so score against those 3 plus pad_fraction (6 of the 14 rows);
    # the remaining jitter/shimmer/hnr_db rows require the physio extractor,
    # out of scope for this spike (see this plan's Task 10 risk notes).
    scalars_3col = coll.get_style_scalars(idx)
    scalars_6col = np.zeros((len(idx), 6), dtype=np.float32)
    scalars_6col[:, :3] = scalars_3col
    threshold_eer, threshold = compute_eer_threshold(p, y)
    mask = np.ones(len(y), dtype=bool)
    result = confound_table(p, y, scalars_6col, coll.pad_fraction, mask, threshold, fid=coll.file_id)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "spike_confound_report.json").write_text(json.dumps(result, indent=1, default=float))
    n_pass = sum(1 for r in result["rows"] if r["passed"])
    print(f"confound gate: {n_pass}/{len(result['rows'])} rows pass (threshold={threshold:.3f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_score_spike_confound.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model_training/score_spike_confound.py model_training/test_score_spike_confound.py
git commit -m "feat(voice_guard): score the spike checkpoint through evaluate.py's existing confound gate"
```

---

### Task 9: ONNX export + CPU latency proxy

**Files:**
- Create: `model_training/export_spike_onnx.py`
- Test: `model_training/test_export_spike_onnx.py`

**Interfaces:**
- Consumes: `MultiHeadSpike`.
- Produces: `export_spike_onnx(model, out_path: Path) -> None` (single input `pcm` `(1, 48000)` float32, outputs `human_fake_logits` `(1, 2)` only — VAD/language/style heads are training-time-only, not exported); `benchmark_cpu_latency_ms(onnx_path: Path, n_runs: int = 50) -> float` (mean wall-clock ms per window on CPU, via `onnxruntime`).

- [ ] **Step 1: Add the missing dependency.** `model_training/requirements.txt` lists `onnx` (the format library) but not `onnxruntime` (the inference engine this task actually needs to run an exported graph) — `backend/requirements.txt` has `onnxruntime>=1.17` for a different venv, model_training's does not. Add it:

```
# model_training/requirements.txt — add this line (see export_spike_onnx.py, Task 9):
onnxruntime>=1.17
```

Then `pip install -r requirements.txt` (or `pip install onnxruntime>=1.17` directly) before running this task's tests.

- [ ] **Step 2: Write the failing tests**

```python
# model_training/test_export_spike_onnx.py
from __future__ import annotations

from export_spike_onnx import benchmark_cpu_latency_ms, export_spike_onnx
from spike_model import MultiHeadSpike


def test_export_produces_a_loadable_onnx_model(tmp_path):
    import onnxruntime as ort

    model = MultiHeadSpike().eval()
    onnx_path = tmp_path / "spike.onnx"
    export_spike_onnx(model, onnx_path)
    assert onnx_path.exists()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["pcm"]
    assert [o.name for o in sess.get_outputs()] == ["human_fake_logits"]


def test_benchmark_cpu_latency_returns_a_positive_number(tmp_path):
    model = MultiHeadSpike().eval()
    onnx_path = tmp_path / "spike.onnx"
    export_spike_onnx(model, onnx_path)
    ms = benchmark_cpu_latency_ms(onnx_path, n_runs=5)
    assert ms > 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest test_export_spike_onnx.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_spike_onnx'`

- [ ] **Step 4: Write the implementation**

```python
# model_training/export_spike_onnx.py
"""Best-effort ONNX export of a Phase-0 spike checkpoint, and a CPU
latency proxy. This is a PROXY gate only: EVAL-PROTOCOL.md's windowing
rule requires real-time on target ANDROID hardware (a 3 s window scored
every second, so inference must clear well under 1 s per window) -- a
desktop-CPU onnxruntime number is a useful early filter (if this fails,
Android will fail too) but passing it is NOT the Phase 0 gate's mobile
criterion by itself. See this plan's Task 10 for the actual gate and the
manual on-device follow-up it still requires.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch


def export_spike_onnx(model, out_path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 48000)

    class _HumanFakeOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pcm):
            return self.inner(pcm)["human_fake_logits"]

    wrapped = _HumanFakeOnly(model)
    torch.onnx.export(wrapped, dummy, str(out_path), input_names=["pcm"], output_names=["human_fake_logits"],
                       opset_version=13, dynamic_axes=None)


def benchmark_cpu_latency_ms(onnx_path: Path, n_runs: int = 50) -> float:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pcm = np.random.default_rng(0).normal(0, 0.1, (1, 48000)).astype(np.float32)
    for _ in range(5):  # warmup
        sess.run(None, {"pcm": pcm})
    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {"pcm": pcm})
    return (time.perf_counter() - start) * 1000 / n_runs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest test_export_spike_onnx.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add model_training/requirements.txt model_training/export_spike_onnx.py model_training/test_export_spike_onnx.py
git commit -m "feat(voice_guard): add ONNX export and CPU latency proxy for the spike"
```

---

### Task 10: Run the spike end-to-end and record the go/no-go decision

This task is the actual experiment, not more library code — it has no new source file to TDD, but it is where Tasks 1–9 pay off. Do not skip it or treat Task 9 passing its unit tests as equivalent to running it.

**Files:**
- Create: `model_training/docs/2026-09-XX-phase0-spike-decision.md` (replace `XX` with the actual date)

- [ ] **Step 1: Build the training cache** on a real (not synthetic-test) subset. Reuse `corpus.TRAIN_SETS_V12` for real/fake dirs and `eval_protocol.TRAIN_CHANNELS` for channels — same corpus definitions the production trainer uses, so this spike is trained on data comparable to v11–v13, not a different distribution. A few thousand files per class is enough for a feasibility read (this is explicitly NOT a full training run).

- [ ] **Step 2: Train** via `train_spike_aasist.py`, e.g.:

```bash
python train_spike_aasist.py --out runs/spike_aasist_v1 --cache-root runs/spike_raw_cache --epochs 10 --batch-size 64
```

- [ ] **Step 3: Build the held-out raw-PCM cache** for `corpus.CORE_EVAL_SETS` on the `select` split (via `corpus.eval_specs("select", eval_protocol.PHONE_CHANNELS, ...)` and `raw_pcm_cache.build_raw_unit`), then run:

```bash
python score_spike_confound.py --checkpoint runs/spike_aasist_v1/epoch_10.pt \
    --cache-root runs/spike_raw_cache_heldout --out runs/spike_aasist_v1_confound
```

- [ ] **Step 4: Export + CPU latency proxy:**

```bash
python export_spike_onnx.py  # or a short inline script calling export_spike_onnx + benchmark_cpu_latency_ms
```

- [ ] **Step 5: Read `runs/spike_aasist_v1/gradient_diagnostics.jsonl`** and plot/inspect `cosine_human_fake_vs_style` over `step`, alongside `style_lambda`. Note whether the correlation drops as `style_lambda` ramps (evidence the confound is unwinding) or stays flat/near-zero even pre-ramp (evidence against a simple linear entanglement at the mid band).

- [ ] **Step 6: Write the decision doc** (`model_training/docs/2026-09-XX-phase0-spike-decision.md`), recording, per the three-signal gate agreed in this plan's spec:
  1. Confound-gate rows passed (out of the 6 rows this spike's 3-style-scalar cache can score — note explicitly that jitter/shimmer/hnr_db rows need the physio extractor and are out of this spike's scope, so "6/6" here is not directly comparable to v13's "5/14"; a full comparison needs Phase 1's full feature set) vs. the current deployed model's rows.
  2. Gradient-correlation trend (Step 5's reading).
  3. CPU latency proxy from Task 9, with the explicit caveat that it is not the real mobile gate.
  - State the decision: proceed to Phase 1 (full AASIST-family training + full feature confound scoring), or pivot toward corpus/hard-negative work, per the "what would change this recommendation" section of Idea 1/Idea 3 in the brainstorm doc.
  - Cross-link this file from `voice_guard/state.md` (new dated session entry, per that file's own maintenance convention) and from the brainstorm doc's Idea 1/Idea 3 sections.

- [ ] **Step 7: Commit**

```bash
git add model_training/docs/2026-09-XX-phase0-spike-decision.md model_training/runs/spike_aasist_v1_confound/spike_confound_report.json voice_guard/state.md
git commit -m "docs(voice_guard): Phase 0 spike result and go/no-go decision"
```

---

### Task 11: Verify the already-shipped per-speaker calibration on-device (Idea 1, Track 1)

**Context — do not rebuild this feature.** Per `voice_guard/state.md`'s 2026-09-11 session entry, `CalibrationProvider`, `AudioService.captureCalibrationSample`, and the "Voice Calibration" settings-screen UI are already implemented and unit-tested (`test/calibration_provider_test.dart`, `test/audio_service_calibration_test.dart`, 15/15 passing). The brainstorm doc's Idea 1 table entry marking Track 1 "Not attempted" is **stale** — flag this correction wherever Idea 1 is referenced next. The only thing not done is the on-device manual check, because no physical/emulated Android device was available in that session.

**Files:**
- No new files. Follow `docs/superpowers/plans/2026-09-11-per-speaker-calibration-plan.md`, Task 4, Step 3 (the already-written manual check this plan never got to run).
- Modify: `voice_guard/state.md` (record the result — proven working, or what broke).

- [ ] **Step 1:** Confirm a physical or emulated Android device is available (`adb devices`). If not, stop here and note the blocker in `state.md` rather than skipping silently.
- [ ] **Step 2:** Build and install the app (`flutter run` or the project's existing build path — check `voice_guard/README.md` for the current device-install command).
- [ ] **Step 3:** Open Settings → Voice Calibration, tap "Calibrate My Voice," read a short deliberately-monotone passage into the mic, confirm the status text updates from "not calibrated" to a calibrated state with a stored baseline.
- [ ] **Step 4:** Start a Live Mic Test / call-detection session reading the same monotone register. Confirm via `debugPrint`/logcat that `effectiveThreshold` differs meaningfully from `settings.sensitivity` (the uncalibrated value) — i.e., calibration is actually shifting the alert line, not silently returning the uncalibrated default.
- [ ] **Step 5:** Tap "Reset," confirm the status text returns to "not calibrated" and `effectiveThreshold` returns to `settings.sensitivity`.
- [ ] **Step 6:** Record the result in `voice_guard/state.md` under a new dated session entry — proven-working with the observed threshold values, or the specific failure if one occurs. Do not mark it "done" on partial evidence (this file's own stated convention).
- [ ] **Step 7: Commit**

```bash
git add voice_guard/state.md
git commit -m "test(voice_guard): on-device verification of per-speaker calibration (Track 1)"
```

---

## Roadmap (NOT part of this plan — do not execute as tasks)

Phases 1–5 of the consolidated recommendation (full AASIST-family training, mobile distillation to RawNet2/RawNet3 or LCNN, `vaani_laptop`/`vaani/mobile` app integration) are gated on Task 10's decision and on choices not yet made (band boundaries, final λ values, which mobile architecture). Per the "No Placeholders" rule, writing detailed TDD steps for them now would mean inventing specifics that Task 10 hasn't determined yet. Once Task 10's decision doc exists, write Phase 1's plan as its own document, using this plan's Task 10 result as its spec input.
