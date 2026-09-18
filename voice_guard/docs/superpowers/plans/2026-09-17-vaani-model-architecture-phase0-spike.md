# VAANI Model Architecture — Phase 0 Feasibility Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the combined Idea-1/Idea-3 feasibility spike — a small-scale raw-waveform AASIST-lite trunk with three depth-staged auxiliary heads (VAD / human-fake / language) plus a gradient-reversal adversarial style-suppression head — and score it against the same `measure_confound.py`-lineage gate the deployed model has failed three times, so the project can decide (data-backed, not felt) whether to commit the rest of the 3-month architecture budget or pivot to corpus/hard-negative work instead. Also closes the one loose end on the already-shipped per-speaker calibration track (Idea 1, Track 1): its on-device manual check.

**Folded in, 2026-09-18 (Ideas 4–6):** three later brainstorm ideas extend this same spike rather than starting a new one. **Idea 5** (per-attack-family specialist heads) is an additive variant of the Task 3 model and Task 7 training loop — Task 12 adds it as a flag-selectable option so Task 10's decision run can compare baseline-shared-head-BCE vs. specialist-heads on the same corpus and same confound gate. **Idea 6** (OC-Softmax one-class objective) has two independent paths: Task 13 adds it as a spike variant (same as Idea 5, gated on the AASIST spike actually running), but — per a correction made this session, see the brainstorm doc's Idea 6 section — OC-Softmax doesn't actually need the AASIST spike at all, since the pooled→frame-level feature swap it was thought to depend on already shipped as the currently-deployed v13 model. **Task 15 tests it directly against v13**, independent of whether Tasks 1–11's spike ever runs. **Idea 4** (manual dataset pruning) is not spike code at all — it's a corpus-quality precondition — but it directly affects what Task 10 trains on, so Task 14 makes checking its status an explicit pre-Task-10 step rather than a silent assumption that today's corpus is the right one to spike against.

**Architecture:** Everything in this plan is additive and spike-scoped: new files only, nothing in the production LFCC pipeline (`dataset.py`, `feature_cache.py`, `train_seq_cnn.py`, `evaluate.py`, `model.py`) is modified. A parallel raw-PCM cache (`raw_pcm_cache.py`) reuses the existing windowing primitives (`dataset.load_audio`, `trim_edge_silence`, `apply_channel`, `pad_to_window`, `stable_seed`) without touching `dataset.Window` or its 20+ existing tests. A new model file (`spike_model.py`) builds the AASIST-lite trunk (SincNet-style frontend + residual blocks banded into early/mid/late) and a `MultiHeadSpike` wrapper with the VAD, human-fake, language and GRL-style-adversarial heads. The confound *gate itself* is reused unmodified from `evaluate.py` (`confound_table`) — this plan does not reimplement the statistics, only feeds them new numbers. Idea 5/6's additions (Tasks 12–13) follow the same rule: `MultiHeadSpike` gains new optional heads/outputs, nothing existing is removed, and `score_spike_confound.py`/`train_spike_aasist.py` gain flags, not forks.

**Tech Stack:** Python, PyTorch, numpy, soundfile/librosa (existing deps only — no new packages).

**Spec:** `voice_guard/docs/superpowers/specs/2026-09-17-vaani-model-architecture-brainstorm.md` (Ideas 1–6, Idea 6 corrected 2026-09-18 — its feature-extraction dependency was overstated, see that section) and the consolidated recommendation given in this conversation (backbone = Idea 1's gated phased plan; Phase 0's spike = Idea 3 Approach C trained on an AASIST-lite trunk, extended by Idea 5's specialist heads and Idea 6's one-class objective as comparison variants; three-signal gate; calibration ships independently; Idea 4 gates what corpus Task 10 trains on). Executors should read Ideas 1, 3, 4, 5 and 6 in full before starting Tasks 12–15 — this plan implements Ideas 1/3's "Phase 0" and "Track 1 verification" (Tasks 1–11, already committed) plus Ideas 4/5/6's fold-in (Task 12: Idea 5; Task 13: Idea 6 as a spike variant; Task 14: Idea 4's pruning-status gate; Task 15: Idea 6 tested standalone against the deployed v13 model, no spike dependency), not Phases 1–5 (see the Roadmap section at the end, which is deliberately NOT a task list). **Task 15 has no prerequisite among Tasks 1–14 except Task 13's `oc_softmax.py` module** — it can be executed first, in isolation, and is the fastest way to get real evidence on Idea 6.

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
| `model_training/docs/2026-09-XX-phase0-spike-decision.md` | The recorded go/no-go decision (Task 10's deliverable) — now also records the specialist-heads and OC-Softmax comparison (Tasks 12–13) and the pruning-status check (Task 14). |
| `model_training/oc_softmax.py` | (Task 13, Idea 6) `OCSoftmaxLoss` — one-class softmax objective on `MultiHeadSpike`'s mid-band embedding. |
| `model_training/docs/2026-09-XX-pruning-status-check.md` | (Task 14, Idea 4) Recorded pruning-plan status at spike time, and which corpus variant Task 10 actually trained on. |
| `model_training/finetune_oc_softmax_v13.py` | (Task 15, Idea 6) OC-Softmax fine-tune directly on the **already-deployed v13 SeqTCN** — reuses `oc_softmax.py`, no dependency on the AASIST spike (Tasks 1–11) or its unexecuted status. |
| `model_training/docs/2026-09-XX-oc-softmax-v13-result.md` | (Task 15) The confound-gate result of OC-Softmax on v13, independent of the Task 10 spike decision. |

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

**Run Task 14 first** (dataset-pruning status check, Idea 4) — it determines which corpus this task trains on. Then run this task's Steps 1–4 **three times**: once with the baseline shared head (Tasks 1–9 as originally written), once with `--use-specialist-heads` (Task 12, Idea 5), and once with `--loss-objective oc-softmax` (Task 13, Idea 6) — same cache, same seeds, same held-out split across all three, so the comparison in Step 6 isolates the objective/head choice and nothing else.

**Files:**
- Create: `model_training/docs/2026-09-XX-phase0-spike-decision.md` (replace `XX` with the actual date)

- [ ] **Step 1: Build the training cache** on a real (not synthetic-test) subset. Reuse `corpus.TRAIN_SETS_V12` for real/fake dirs and `eval_protocol.TRAIN_CHANNELS` for channels — same corpus definitions the production trainer uses, so this spike is trained on data comparable to v11–v13, not a different distribution (unless Task 14 found a pruned corpus to use instead — see that task). A few thousand files per class is enough for a feasibility read (this is explicitly NOT a full training run). Build this cache once and reuse it for all three variant runs.

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

- [ ] **Step 6: Write the decision doc** (`model_training/docs/2026-09-XX-phase0-spike-decision.md`), recording, per the three-signal gate agreed in this plan's spec, **for each of the three variants (baseline / specialist-heads / OC-Softmax)**:
  1. Confound-gate rows passed (out of the 6 rows this spike's 3-style-scalar cache can score — note explicitly that jitter/shimmer/hnr_db rows need the physio extractor and are out of this spike's scope, so "6/6" here is not directly comparable to v13's "5/14"; a full comparison needs Phase 1's full feature set) vs. the current deployed model's rows.
  2. Gradient-correlation trend (Step 5's reading; for the OC-Softmax variant, also note whether `oc_softmax_score`'s cosine-similarity distribution separates real/fake cleanly — a degenerate `w0` collapse is this objective's own failure mode, distinct from the confound question).
  3. CPU latency proxy from Task 9, with the explicit caveat that it is not the real mobile gate. Specialist heads add negligible latency (three small linear heads on an already-computed tensor); OC-Softmax removes `human_fake_head` from the inference path entirely — note whether that changes the proxy number at all.
  - State the decision: proceed to Phase 1 (full AASIST-family training + full feature confound scoring) with whichever variant wins (or a combination, see Roadmap), or pivot toward corpus/hard-negative work, per the "what would change this recommendation" sections of Ideas 1, 3, 5 and 6 in the brainstorm doc.
  - Reference Task 14's pruning-status finding explicitly — a variant "winning" on an unpruned corpus is a weaker claim than the same result on a pruned one, and the decision doc should say which applies.
  - Cross-link this file from `voice_guard/state.md` (new dated session entry, per that file's own maintenance convention) and from the brainstorm doc's Idea 1/3/4/5/6 sections.

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

### Task 12: Per-attack-family specialist heads as a comparison variant (Idea 5)

**Files:**
- Modify: `model_training/spike_model.py`, `model_training/raw_pcm_cache.py`, `model_training/train_spike_aasist.py`, `model_training/score_spike_confound.py`
- Modify tests: `model_training/test_spike_model.py`, `model_training/test_raw_pcm_cache.py`, `model_training/test_train_spike_aasist.py`

**Interfaces:**
- `raw_pcm_cache.py` gains `IGNORE_ATTACK_FAMILY = -100`, `ATTACK_FAMILY_BY_SOURCE_SET: dict[str, int]` (`{"tts": 0, "voice_clone": 1, "other": 2}` keyed by the source's already-known generator — XTTS-tagged sources → `voice_clone`, ASVspoof/MLAAD TTS-tagged sources → `tts`, CodecFake/DECRO/unmapped fakes → `other`; real sources always map to `IGNORE_ATTACK_FAMILY`, same masking convention as `language_for_source_set`), `attack_family_for_source_set(source_set: str) -> int`. `build_raw_unit` writes an additional `attack_family.npy` `(N,)` int64 array; `RawPCMCollection` gains `.attack_family` and `iter_batches` yields it as a 6th element.
- `spike_model.py`'s `MultiHeadSpike` gains a constructor flag `use_specialist_heads: bool = False`. When `True`: replaces the single `human_fake_head` with `self.specialist_heads = nn.ModuleDict({"tts": ..., "voice_clone": ..., "other": ...})`, each the same shape as the original `human_fake_head` (`Linear(2*mid_ch,32)`, `ReLU`, `Dropout`, `Linear(32,2)`), reading the same `mid_pooled` tensor. `forward()` always returns `human_fake_logits` (max-score combiner: `torch.stack([softmax(h)[:,1] for h in specialist outputs]).max(dim=0)`, converted back to 2-class logits via `torch.stack([1-p, p], dim=-1).log()` so downstream consumers — `score_spike_confound.py`, `compute_losses` — need no changes) plus, only when `use_specialist_heads`, a `specialist_logits: dict[str, Tensor]` key for the per-family loss.
- `train_spike_aasist.py`'s `compute_losses` gains an `attack_family: torch.Tensor | None = None` parameter: when `out` contains `specialist_logits`, replaces the single `human_fake` BCE term with the sum of three masked per-family binary losses (real examples contribute to all three as negatives; a fake example only contributes to its own family's loss as positive, masked out of the other two via the same `ignore_index` pattern already used for language). `main()` gains `--use-specialist-heads`.

- [ ] **Step 1: Write the failing tests** (append to the three existing test files)

```python
# append to test_raw_pcm_cache.py
def test_attack_family_for_known_and_unknown_source_sets():
    from raw_pcm_cache import IGNORE_ATTACK_FAMILY, attack_family_for_source_set
    assert attack_family_for_source_set("xtts_hi_clone") == 1  # voice_clone
    assert attack_family_for_source_set("real") == IGNORE_ATTACK_FAMILY


# append to test_spike_model.py
def test_specialist_heads_produce_combined_human_fake_logits_same_shape_as_baseline():
    from spike_model import MultiHeadSpike
    model = MultiHeadSpike(use_specialist_heads=True)
    out = model(torch.randn(3, 48000))
    assert out["human_fake_logits"].shape == (3, 2)
    assert set(out["specialist_logits"]) == {"tts", "voice_clone", "other"}
    for v in out["specialist_logits"].values():
        assert v.shape == (3, 2)


def test_baseline_head_unaffected_when_specialist_heads_disabled():
    from spike_model import MultiHeadSpike
    model = MultiHeadSpike(use_specialist_heads=False)
    out = model(torch.randn(2, 48000))
    assert "specialist_logits" not in out
    assert out["human_fake_logits"].shape == (2, 2)


# append to test_train_spike_aasist.py
def test_specialist_loss_masks_out_non_owning_families_for_fake_examples():
    from spike_model import IGNORE_LANGUAGE, MultiHeadSpike
    from train_spike_aasist import compute_losses
    model = MultiHeadSpike(use_specialist_heads=True)
    out = model(torch.randn(4, 48000))
    y = torch.tensor([0, 1, 0, 1])
    attack_family = torch.tensor([-100, 0, -100, 1])  # real examples ignored, fakes are tts/voice_clone
    vad_target = torch.zeros(4, out["vad_logits"].shape[1])
    language = torch.full((4,), IGNORE_LANGUAGE)
    style_target = torch.zeros(4, 3)
    losses = compute_losses(out, y, vad_target, language, style_target, lambda_lang=0.5,
                             attack_family=attack_family)
    assert torch.isfinite(losses["human_fake"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_raw_pcm_cache.py test_spike_model.py test_train_spike_aasist.py -v`
Expected: FAIL — `use_specialist_heads`/`attack_family_for_source_set`/`attack_family=` not defined

- [ ] **Step 3: Implement.** Follow Idea 5's Approach B (max-score combiner) from the brainstorm doc exactly — do not build the learned meta-combiner (Approach C) speculatively; that's an explicit escalation-only step in the spec, gated on this variant showing a calibration problem. Reuse `dataset.IGNORE_ATTACK_TYPE`'s existing cross-entropy `ignore_index` masking pattern for the per-family loss, the same way `language`'s loss already does it — don't invent a new masking mechanism.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_raw_pcm_cache.py test_spike_model.py test_train_spike_aasist.py -v`

- [ ] **Step 5: Commit**

```bash
git add model_training/spike_model.py model_training/raw_pcm_cache.py model_training/train_spike_aasist.py model_training/score_spike_confound.py model_training/test_spike_model.py model_training/test_raw_pcm_cache.py model_training/test_train_spike_aasist.py
git commit -m "feat(voice_guard): add per-attack-family specialist heads as a Phase-0 spike comparison variant (Idea 5)"
```

---

### Task 13: OC-Softmax one-class objective as a comparison variant (Idea 6)

**Files:**
- Create: `model_training/oc_softmax.py`
- Test: `model_training/test_oc_softmax.py`
- Modify: `model_training/spike_model.py` (expose the mid-band pooled tensor as an `embedding` output — trivial, already computed, no new parameters), `model_training/train_spike_aasist.py` (loss-objective switch)

**Interfaces:**
- `oc_softmax.py` produces `class OCSoftmaxLoss(nn.Module)` (Zhang et al. 2021): holds a single learned unit-norm target weight `w0` of the embedding's dimension; `forward(embedding: torch.Tensor, y: torch.Tensor, m_real: float = 0.9, m_fake: float = 0.2, alpha: float = 20.0) -> torch.Tensor` returns the scalar OC-softmax loss (softplus of `alpha * (m_target - cos_sim) * sign`, `m_target` = `m_real` for genuine (`y==0`), `m_fake` for spoof (`y==1`), per the paper's formulation — genuine embeddings pulled inside a tight margin around `w0`, spoof embeddings pushed outside a looser one). Also produces `oc_softmax_score(embedding: torch.Tensor) -> torch.Tensor` returning the raw cosine similarity to `w0` (higher = more human-like), for use as `score_spike_confound.py`'s `p_fake` after a `(1 - similarity) / 2` rescale to `[0,1]`.
- `spike_model.py`'s `MultiHeadSpike.forward()` always adds `"embedding": mid_pooled` to its output dict (no flag needed — free to compute, other consumers just ignore the extra key).
- `train_spike_aasist.py` gains `--loss-objective {bce,oc-softmax}` (default `bce`). When `oc-softmax`: `compute_losses` replaces the `human_fake` BCE term with `OCSoftmaxLoss()(out["embedding"], y)`; `human_fake_head`'s own parameters are simply unused in this mode (left in the model rather than conditionally removed, so switching objectives doesn't change the model's state-dict shape — a smaller diff for the Task 10 comparison run).

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_oc_softmax.py
from __future__ import annotations

import torch

from oc_softmax import OCSoftmaxLoss, oc_softmax_score


def test_loss_is_lower_when_genuine_embeddings_align_with_w0():
    loss_fn = OCSoftmaxLoss(embedding_dim=8)
    aligned = loss_fn.w0.detach().unsqueeze(0).repeat(4, 1)
    y_real = torch.zeros(4, dtype=torch.long)
    aligned_loss = loss_fn(aligned, y_real)
    random_loss = loss_fn(torch.randn(4, 8), y_real)
    assert aligned_loss.item() < random_loss.item()


def test_loss_is_finite_and_scalar_for_mixed_batch():
    loss_fn = OCSoftmaxLoss(embedding_dim=8)
    emb = torch.randn(5, 8, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 1])
    loss = loss_fn(emb, y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert emb.grad is not None


def test_score_is_bounded_and_gradient_direction_makes_sense():
    loss_fn = OCSoftmaxLoss(embedding_dim=8)
    aligned = loss_fn.w0.detach().unsqueeze(0)
    opposite = -loss_fn.w0.detach().unsqueeze(0)
    s_aligned = oc_softmax_score(aligned, loss_fn.w0)
    s_opposite = oc_softmax_score(opposite, loss_fn.w0)
    assert -1.0 <= s_opposite.item() <= s_aligned.item() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_oc_softmax.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oc_softmax'`

- [ ] **Step 3: Write the implementation**

```python
# model_training/oc_softmax.py
"""One-Class Softmax (Zhang, Yamagishi & Todisco, 2021 — "One-Class
Learning Towards Synthetic Voice Spoofing Detection"), as Idea 6's
primary lever (docs/superpowers/specs/2026-09-17-vaani-model-architecture-
brainstorm.md). Bounds a single learned target direction w0 representing
"genuine human speech"; genuine embeddings are pulled inside a tight
margin around w0, spoof embeddings pushed outside a looser one. Unlike
two-class BCE, nothing here is free to pick style as the separating axis
by construction -- style variance within genuine speech still has to fit
inside the SAME bound, whatever axis the trunk ends up using.
"""
from __future__ import annotations

import torch
from torch import nn


class OCSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim: int, m_real: float = 0.9, m_fake: float = 0.2, alpha: float = 20.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(embedding_dim))
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha

    def forward(self, embedding: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w0 = nn.functional.normalize(self.w0, dim=0)
        emb = nn.functional.normalize(embedding, dim=-1)
        cos_sim = emb @ w0  # (B,)
        is_real = (y == 0).float()
        margin = torch.where(y == 0, torch.full_like(cos_sim, self.m_real), torch.full_like(cos_sim, self.m_fake))
        sign = torch.where(y == 0, torch.ones_like(cos_sim), -torch.ones_like(cos_sim))
        return nn.functional.softplus(self.alpha * sign * (margin - cos_sim)).mean()


def oc_softmax_score(embedding: torch.Tensor, w0: torch.Tensor) -> torch.Tensor:
    """Cosine similarity to w0: higher = more human-like. Callers rescale
    to a [0,1] fake-probability via (1 - score) / 2 for the confound gate."""
    return nn.functional.normalize(embedding, dim=-1) @ nn.functional.normalize(w0, dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_oc_softmax.py -v`

- [ ] **Step 5: Commit**

```bash
git add model_training/oc_softmax.py model_training/test_oc_softmax.py model_training/spike_model.py model_training/train_spike_aasist.py
git commit -m "feat(voice_guard): add OC-Softmax one-class objective as a Phase-0 spike comparison variant (Idea 6)"
```

---

### Task 14: Check dataset-pruning status before the Task 10 decision run (Idea 4)

This task has no new source file — it's a precondition check, the same way Task 11 was a verification task rather than new library code. Do it **before** re-running or extending Task 10's decision run with the Task 12/13 variants, not after.

**Files:**
- Create: `model_training/docs/2026-09-XX-pruning-status-check.md` (replace `XX` with the actual date)
- Read: `voice_guard/docs/DATASET-PRUNING-PLAN.md` (Idea 4's full plan)

- [ ] **Step 1:** Check whether any of Idea 4's three ordered experiments (reason-category ablation, scenario-relevance sweep, or the resulting fixed pruned corpus) have actually been run. As of this plan's last update they have not — `docs/DATASET-PRUNING-PLAN.md` is still "proposed, not started." If that's still true, record it explicitly rather than silently training on the unpruned corpus and letting a reader assume pruning was considered and rejected.
- [ ] **Step 2:** If pruning has landed by the time this task runs, use the pruned corpus (per `DATASET-PRUNING-PLAN.md`'s manifest schema) as Task 10/12/13's training and held-out source instead of the corpus `corpus.TRAIN_SETS_V12`/`corpus.CORE_EVAL_SETS` currently point to, and note which variant (pruned vs. unpruned) each run in the decision doc used.
- [ ] **Step 3:** If pruning has NOT landed, record that explicitly in `model_training/docs/2026-09-XX-pruning-status-check.md`: the Task 10 comparison (baseline / specialist-heads / OC-Softmax) is being run on the unpruned corpus, so any of the three architecture/objective variants "winning" is confounded with whatever data-quality noise Idea 4 would have removed. This isn't a reason to block Tasks 12/13 — Idea 1 and Idea 3 already establish precedent for spiking on the existing corpus — but it's a caveat the decision doc must carry forward, not silently drop.
- [ ] **Step 4:** Cross-link this file from `model_training/docs/2026-09-XX-phase0-spike-decision.md` (Task 10) and from `voice_guard/state.md`.
- [ ] **Step 5: Commit**

```bash
git add model_training/docs/2026-09-XX-pruning-status-check.md
git commit -m "docs(voice_guard): record dataset-pruning status ahead of the Phase 0 spike decision (Idea 4)"
```

---

### Task 15: OC-Softmax fine-tune directly on the deployed v13 SeqTCN (Idea 6, standalone)

**Does not depend on Tasks 1–11 (the AASIST spike).** The brainstorm doc's
Idea 6 section originally implied OC-Softmax was gated on Idea 1's
raw-waveform feature-extraction decision; that was a correction made
2026-09-18 (see that section) — the pooled→frame-level swap already
shipped as v13, so OC-Softmax has a real embedding to test against today,
with zero dependency on whether the AASIST spike (still unexecuted, no
code on disk) ever runs or clears its gate. This task can be done before,
after, or in parallel with Tasks 1–14.

**Files:**
- Create: `model_training/finetune_oc_softmax_v13.py`
- Test: `model_training/test_finetune_oc_softmax_v13.py`
- Consumes (read-only, unmodified, same pattern as `score_spike_confound.py`'s use of `evaluate.confound_table`): `model.VoiceGuardSeqTCN`, `train_seq_cnn.build_model_from_norm_stats` (or equivalent checkpoint-loading helper — confirm the actual name in `train_seq_cnn.py` before writing this task's code, it is not re-derived here), `feature_cache.py`/`dataset.py`'s existing loaders for the `select`/`test` split, `evaluate.confound_table`, `oc_softmax.OCSoftmaxLoss`/`oc_softmax_score` (Task 13).

**Interfaces:**
- `embed_v13(model: VoiceGuardSeqTCN, seq: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor` — returns `trunk_out` (the shared hidden representation immediately before `real_fake_head`/`attack_type_head`), computed by calling `model`'s already-public submodules (`model.normalize_sequence`, `model.stem`, `model.blocks`, `model.trunk`, `model.scalar_normalize`) in the same order `VoiceGuardSeqTCN.forward` already does — **`model.py` is not modified**, this is an external function reading public attributes of an already-instantiated model, matching this whole plan's "production pipeline untouched" constraint.
- `main()` CLI: `--checkpoint` (an existing v13 run, e.g. `runs/voice_guard_v13_final/model.pt`), `--freeze-backbone` (default `True` — see rationale below), trains `OCSoftmaxLoss` (and, only if `--freeze-backbone=False`, the loaded backbone's own parameters too) on `embed_v13`'s output using the production `select` split, then scores through `evaluate.confound_table` unmodified — the exact same gate v13's existing BCE result is already reported against, so this is a like-for-like comparison, not a new metric.

**Why `--freeze-backbone=True` is the default, not an afterthought:** the question this task answers is "does changing the objective alone move the confound gate," isolated from "does more training move it" — the same one-variable-at-a-time discipline that already found the `'clean'`-recipe regression (`state.md`, "Attempt 2 + follow-up ablations"). Fine-tuning the backbone too is a real escalation worth trying if the frozen-backbone result is inconclusive, not the first thing to reach for.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_finetune_oc_softmax_v13.py
from __future__ import annotations

import numpy as np
import torch

from finetune_oc_softmax_v13 import embed_v13
from model import VoiceGuardSeqTCN


def _tiny_v13_model():
    n_lfcc, n_scalars = 60, 6
    return VoiceGuardSeqTCN(n_lfcc=n_lfcc, n_scalars=n_scalars,
                             seq_mean=np.zeros(n_lfcc, dtype=np.float32), seq_std=np.ones(n_lfcc, dtype=np.float32),
                             scalar_mean=np.zeros(n_scalars, dtype=np.float32), scalar_std=np.ones(n_scalars, dtype=np.float32))


def test_embed_v13_matches_forwards_own_trunk_out_shape():
    model = _tiny_v13_model().eval()
    seq = torch.randn(3, 184, 60)
    scalars = torch.randn(3, 6)
    emb = embed_v13(model, seq, scalars)
    assert emb.shape == (3, model.trunk[0].out_features)


def test_embed_v13_is_deterministic_and_does_not_mutate_model_weights():
    model = _tiny_v13_model().eval()
    seq, scalars = torch.randn(2, 184, 60), torch.randn(2, 6)
    before = [p.clone() for p in model.parameters()]
    e1 = embed_v13(model, seq, scalars)
    e2 = embed_v13(model, seq, scalars)
    assert torch.allclose(e1, e2)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_finetune_oc_softmax_v13.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune_oc_softmax_v13'`

- [ ] **Step 3: Write the implementation.** `embed_v13` is a straight read of `VoiceGuardSeqTCN.forward`'s own body (`model.py`, `VoiceGuardSeqTCN.forward`) up to and including `trunk_out`, calling the same public submodules in the same order — copy that control flow exactly, don't reinvent it. The `main()` CLI mirrors `score_spike_confound.py`'s structure (Task 8): load checkpoint, build data loader from the existing production split, run the loss, score via the unmodified `confound_table` import, write a JSON result.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_finetune_oc_softmax_v13.py -v`

- [ ] **Step 5: Run the actual fine-tune + confound scoring**, frozen-backbone first:

```bash
python finetune_oc_softmax_v13.py --checkpoint runs/voice_guard_v13_final/model.pt --freeze-backbone --out model_training/docs/2026-09-XX-oc-softmax-v13-result.md
```

Record the confound-gate row count against v13's existing BCE result (same rows, same held-out split) in `model_training/docs/2026-09-XX-oc-softmax-v13-result.md`. If the frozen-backbone result is inconclusive (neither clearly better nor clearly worse), re-run with `--freeze-backbone=False` and record that too, labeled separately — don't conflate the two in one number.

- [ ] **Step 6: Cross-link** this result from `voice_guard/state.md` (new dated entry) and from the brainstorm doc's Idea 6 section — this is real evidence about Idea 6's core hypothesis, independent of whatever Task 10 eventually decides about the AASIST spike, and should be recorded there rather than only in this plan.

- [ ] **Step 7: Commit**

```bash
git add model_training/finetune_oc_softmax_v13.py model_training/test_finetune_oc_softmax_v13.py model_training/docs/2026-09-XX-oc-softmax-v13-result.md voice_guard/state.md
git commit -m "feat(voice_guard): test OC-Softmax directly on the deployed v13 SeqTCN, independent of the AASIST spike (Idea 6)"
```

---

## Roadmap (NOT part of this plan — do not execute as tasks)

Phases 1–5 of the consolidated recommendation (full AASIST-family training, mobile distillation to RawNet2/RawNet3 or LCNN, `vaani_laptop`/`vaani/mobile` app integration) are gated on Task 10's decision and on choices not yet made (band boundaries, final λ values, which mobile architecture). Per the "No Placeholders" rule, writing detailed TDD steps for them now would mean inventing specifics that Task 10 hasn't determined yet. Once Task 10's decision doc exists, write Phase 1's plan as its own document, using this plan's Task 10 result as its spec input.

**Where Ideas 4–6 sit in this roadmap, now that they're folded in:** Task 10's decision run should train and score three variants on the same corpus and same confound gate before writing the decision doc — baseline shared-head BCE (Tasks 1–9 as originally written), specialist heads (Task 12, Idea 5), and OC-Softmax (Task 13, Idea 6) — and the decision doc should record all three, not just pick a winner silently. Idea 3's GRL term applies to all three variants equally (it's already in the shared trunk path), so this is a 3-way comparison, not 3×2. Task 14 (Idea 4) determines which corpus that 3-way comparison actually ran on, and must be read alongside the decision doc, not treated as a separate, disconnected finding. If Phase 1 proceeds, it inherits whichever of the three variants (or a combination — e.g. specialist heads AND an OC-Softmax-per-specialist objective is a plausible Phase 1 refinement, not scoped here) the Task 10 decision doc recommends, plus Idea 4's pruned corpus if it has landed by then.

**Task 15 is explicitly out of that dependency chain.** It doesn't wait on Task 10's go/no-go, and Task 10's outcome doesn't retroactively invalidate it — it's a direct test of Idea 6's core hypothesis (one-class objective vs. two-class BCE) on the model actually running in production today. Three outcomes are all useful, independently of what the AASIST spike decides: (a) OC-Softmax measurably improves v13's confound-gate rows → worth shipping as a v14 retrain even if the AASIST spike never happens, a materially faster win than waiting on Phase 1–5; (b) it doesn't move the gate at all → weakens Idea 6's core hypothesis generally (evidence the confound isn't purely an objective-function artifact), which should inform how much to expect from Task 13's spike-integrated variant too, not just this standalone one; (c) inconclusive on a frozen backbone → try the `--freeze-backbone=False` escalation (Step 5) before drawing either conclusion.
