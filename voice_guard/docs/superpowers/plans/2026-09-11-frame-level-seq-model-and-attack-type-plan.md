# Frame-level sequence model + attack-type differentiator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mean-pooled 66-feature MLP with a small per-frame
LFCC sequence CNN (fixing the entity-vs-style representational ceiling —
remediation track 3), fold in physio scalars (track 2) and a masked
TTS-vs-voice-conversion auxiliary head (remediation track 4) into the same
architecture, and ship all three as **one** consolidated retrain — not
three separate model swaps.

**Architecture:** `features.py`/`audio_processor.dart` stop mean-pooling
LFCC and emit the full `(184, 60)` per-frame matrix instead; a new
`VoiceGuardSeqCNN` runs a 3-layer Conv1d stack over that sequence, pools to
a fixed-size embedding, concatenates 6 scalars (3 prosody + 3 physio), and
feeds two small linear heads: real/fake (trained on everything) and
attack-type (trained only on labeled TTS/VC fakes, masked out elsewhere via
`ignore_index`). Chunking already guarantees fixed-length windows, so no
padding/masking is needed on the sequence branch itself — only on the
attack-type loss.

**Tech Stack:** Python/numpy/PyTorch (training), Dart (on-device
extraction), ONNX Runtime (mobile inference, now 2 inputs / 2 outputs).

**Spec:**
`voice_guard/docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md`
(track 4) and
`voice_guard/docs/superpowers/plans/2026-09-11-frame-level-sequence-model-plan.md`
(track 3's original scope-only doc — this plan supersedes it as the actual
task breakdown; read that doc's §5 "Open risks" too, they still apply).

## Global Constraints

- Chunking already guarantees constant-length windows (`chunk_audio`,
  `CHUNK_SAMPLES = 48000`) — no padding/masking needed on the sequence
  branch. Only the attack-type head's loss needs masking (real examples
  and unlabeled fakes get `ignore_index=-100`).
- Dart and Python implementations MUST stay numerically equivalent
  (existing project rule, `features.py:1-14`) — every new Python
  extraction function needs a mirrored Dart function and a parity test,
  same discipline as track 2's `extractPhysio`.
- No retrain is production-ready without beating the currently-deployed
  model's noise-FPR AND ITW-held-out-EER within statistical noise (use
  `eval_stats.py::bootstrap_eer_ci`, never a raw point estimate) — this
  applies to the final Task 12 gate, same as every prior track.
- Track 3's own acceptance criterion for the confound itself: the
  median-split correlation analysis
  (`energyVariance`/`pauseRatio`/`zcrVariance`, real clips split at the
  median, compare mean fake-probability between halves) must show a
  **much weaker** split than the currently-deployed model's measured
  25.1%-vs-12.9% (`energyVariance`), 14.7%-vs-23.6% (`pauseRatio`),
  13.1%-vs-24.9% (`zcrVariance`) gaps. ITW EER improving alone is NOT
  sufficient evidence this is fixed.
- The `v10_physio` run (mean-pooled MLP + physio features, in flight as of
  this plan's writing) is diagnostic-only — do not treat its `model.pt` as
  a candidate for deployment; this plan's Task 12 produces the actual
  deployment candidate.
- Do not touch `assets/models/voice_detector.onnx` until Task 12's gate
  passes.

---

### Task 1: Per-frame LFCC extraction (Python side, TDD)

**Files:**
- Modify: `model_training/features.py`
- Create: `model_training/test_features_sequence.py`

**Interfaces:**
- Produces: `extract_lfcc_sequence(pcm: np.ndarray) -> np.ndarray` (shape
  `(184, 60)`, one row per frame, zeros if input too short for even one
  frame — mirrors `extract_lfcc`'s existing degenerate-input convention).
- Produces: `extract_scalars(pcm: np.ndarray) -> np.ndarray` (shape `(6,)`,
  order `[pauseRatio, energyVariance, zcrVariance, jitter_local,
  shimmer_local, hnr_db]` — prosody then physio, reusing
  `extract_prosody`/`extract_physio` unchanged).
- Consumes: `_frame_signal`, `_hamming`, `_magnitude_spectrum`,
  `_linear_filterbank`, `_dct2_orthonormal` (all already in `features.py`,
  unchanged — this task only stops the mean-pool step, everything upstream
  of it is untouched).

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_features_sequence.py
"""Tests for the per-frame LFCC sequence extraction added for the frame-
level CNN (remediation track 3, see
docs/superpowers/plans/2026-09-11-frame-level-seq-model-and-attack-type-plan.md)."""
from __future__ import annotations

import numpy as np

from features import (
    CHUNK_SAMPLES,
    FFT_SIZE,
    HOP_LENGTH,
    N_LFCC,
    extract_lfcc,
    extract_lfcc_sequence,
    extract_physio,
    extract_prosody,
    extract_scalars,
)


def test_extract_lfcc_sequence_shape():
    pcm = np.random.default_rng(0).normal(0, 0.1, CHUNK_SAMPLES).astype(np.float64)
    seq = extract_lfcc_sequence(pcm)
    expected_frames = 1 + (CHUNK_SAMPLES - FFT_SIZE) // HOP_LENGTH
    assert seq.shape == (expected_frames, N_LFCC)


def test_extract_lfcc_sequence_short_input_returns_zeros():
    pcm = np.zeros(100, dtype=np.float64)
    seq = extract_lfcc_sequence(pcm)
    assert seq.shape == (0, N_LFCC)


def test_extract_lfcc_sequence_mean_matches_pooled_extract_lfcc():
    # extract_lfcc's own mean/std normalization happens AFTER pooling, so
    # this checks the pre-normalization DCT coefficients agree, not the
    # final normalized vectors directly comparable value-for-value.
    pcm = np.random.default_rng(1).normal(0, 0.1, CHUNK_SAMPLES).astype(np.float64)
    seq = extract_lfcc_sequence(pcm)
    pooled = extract_lfcc(pcm)
    raw_mean = seq.mean(axis=0)
    m = raw_mean.mean()
    std = np.sqrt(((raw_mean - m) ** 2).mean() + 1e-8)
    normalized = (raw_mean - m) / std
    np.testing.assert_allclose(normalized, pooled, rtol=1e-5)


def test_extract_scalars_is_prosody_then_physio_in_order():
    pcm = np.random.default_rng(2).normal(0, 0.1, CHUNK_SAMPLES).astype(np.float64)
    scalars = extract_scalars(pcm)
    assert scalars.shape == (6,)
    prosody = extract_prosody(pcm)
    physio = extract_physio(pcm)
    np.testing.assert_allclose(scalars[:3], prosody, rtol=1e-5)
    np.testing.assert_allclose(scalars[3:], physio, rtol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `model_training/`, `.venv313` active):
`python -m pytest test_features_sequence.py -v`
Expected: FAIL — `extract_lfcc_sequence`/`extract_scalars` don't exist.

- [ ] **Step 3: Implement `extract_lfcc_sequence` and `extract_scalars`**

Add to `model_training/features.py`, right after `extract_lfcc`:

```python
def extract_lfcc_sequence(pcm: np.ndarray) -> np.ndarray:
    """3s (or shorter) PCM float buffer -> (n_frames, 60) per-frame LFCC
    matrix, unpooled — same math as extract_lfcc up to (and not including)
    the final mean-pool + normalize step. Used by the frame-level CNN
    (VoiceGuardSeqCNN); extract_lfcc is kept for anything not yet migrated
    to the sequence model."""
    if len(pcm) < FFT_SIZE:
        return np.zeros((0, N_LFCC), dtype=np.float64)
    frames = _hamming(_frame_signal(pcm))
    mag = _magnitude_spectrum(frames)
    energies = _linear_filterbank(mag)
    log_e = np.log(energies + 1e-10)
    return _dct2_orthonormal(log_e)[:, :N_LFCC]


def extract_scalars(pcm: np.ndarray) -> np.ndarray:
    """3s PCM float buffer -> 6 scalars: 3 prosody + 3 physio, in that
    order. Used as the scalar branch input to VoiceGuardSeqCNN."""
    return np.concatenate([extract_prosody(pcm), extract_physio(pcm)]).astype(np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_features_sequence.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add model_training/features.py model_training/test_features_sequence.py
git commit -m "feat(voice_guard): add per-frame LFCC sequence + scalar extraction for track 3 CNN"
```

---

### Task 2: Attack-type labeling (Python side, TDD)

**Files:**
- Create: `model_training/attack_labels.py`
- Create: `model_training/test_attack_labels.py`

**Interfaces:**
- Produces: `ATTACK_ID_TO_TYPE: dict[str, str]` (module constant, values
  `"tts"` or `"vc"`).
- Produces: `load_asvspoof_attack_map(protocol_path: Path) -> dict[str, str]`
  — parses a `trial_metadata.txt`-format file into `{utt_id: "tts"|"vc"}`.
- Produces: `DIRECTORY_DEFAULT_ATTACK_TYPE: dict[str, str]` (module
  constant, resolved absolute directory path string -> `"tts"`, for
  directories where every fake file is TTS by construction).
- Produces: `attack_type_for_file(wav_path: Path, source_dir: str, per_file_map: dict[str, str] | None) -> str` — returns `"tts"`, `"vc"`, or
  `"unknown"`.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_attack_labels.py
"""Tests for attack-type labeling (remediation track 4). See
docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §3
for the labeling coverage table this implements."""
from __future__ import annotations

from pathlib import Path

from attack_labels import (
    ATTACK_ID_TO_TYPE,
    attack_type_for_file,
    load_asvspoof_attack_map,
)


def test_attack_id_to_type_covers_a01_through_a19():
    for i in range(1, 20):
        assert f"A{i:02d}" in ATTACK_ID_TO_TYPE
        assert ATTACK_ID_TO_TYPE[f"A{i:02d}"] in ("tts", "vc")


def test_attack_id_to_type_known_vc_ids():
    # Empirically confirmed via model_training/test_assets/
    # voice_conversion_asvspoof_a17.wav (see that dir's README).
    assert ATTACK_ID_TO_TYPE["A17"] == "vc"
    assert ATTACK_ID_TO_TYPE["A18"] == "vc"
    assert ATTACK_ID_TO_TYPE["A19"] == "vc"


def test_load_asvspoof_attack_map_parses_trial_metadata_format(tmp_path: Path):
    protocol = tmp_path / "trial_metadata.txt"
    protocol.write_text(
        "LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval\n"
        "LA_0009 LA_E_1428848 alaw ita_tx A17 spoof notrim eval\n"
        "LA_0009 LA_E_0000001 alaw ita_tx - bonafide notrim eval\n"
    )
    result = load_asvspoof_attack_map(protocol)
    assert result["LA_E_9332881"] == "tts"
    assert result["LA_E_1428848"] == "vc"
    assert "LA_E_0000001" not in result  # bonafide rows carry no attack type


def test_attack_type_for_file_uses_per_file_map_first():
    assert attack_type_for_file(
        Path("/x/LA_E_1428848.wav"), "/x", {"LA_E_1428848": "vc"}
    ) == "vc"


def test_attack_type_for_file_falls_back_to_unknown_with_no_map_or_default():
    assert attack_type_for_file(Path("/some/unlabeled/dir/1.wav"), "/some/unlabeled/dir", None) == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_attack_labels.py -v`
Expected: FAIL — `attack_labels` module doesn't exist.

- [ ] **Step 3: Implement `attack_labels.py`**

```python
# model_training/attack_labels.py
"""Attack-type ({tts, vc}) labeling for remediation track 4. See
docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §3
for the per-source-directory coverage this implements and the caveat on
the ASVspoof2019 attack-ID table below.

NOTE: the table below is a draft based on the well-established convention
that A05/A06 and A17-A19 are voice-conversion systems in the ASVspoof2019
LA taxonomy, with A13 a TTS+VC hybrid bucketed here as VC (its VC stage is
the final signal-transforming step). It has NOT been independently
verified against the official ASVspoof2019 evaluation plan table — do not
treat this as authoritative without that check; an error here silently
mislabels training examples. A17-A19 are the one part of this table
empirically confirmed (via model_training/test_assets/
voice_conversion_asvspoof_a17.wav actually being generated by a
conversion, not synthesis, process)."""
from __future__ import annotations

from pathlib import Path

ATTACK_ID_TO_TYPE: dict[str, str] = {
    "A01": "tts", "A02": "tts", "A03": "tts", "A04": "tts",
    "A05": "vc", "A06": "vc",
    "A07": "tts", "A08": "tts", "A09": "tts", "A10": "tts",
    "A11": "tts", "A12": "tts",
    "A13": "vc",  # TTS+VC hybrid, bucketed as vc — see module docstring
    "A14": "vc", "A15": "vc",
    "A16": "tts",
    "A17": "vc", "A18": "vc", "A19": "vc",
}

# Directories where every fake file is TTS by construction (verified
# against each source's own generation script docstring — see the design
# spec §3 table). Keyed by resolved absolute path string.
DIRECTORY_DEFAULT_ATTACK_TYPE: dict[str, str] = {}


def register_tts_only_directory(path: Path) -> None:
    """Call once per known-all-TTS directory (MLAAD, XTTS/YourTTS/MMS
    accent-expansion fakes) during corpus setup, so attack_type_for_file
    can resolve it without a per-file map."""
    DIRECTORY_DEFAULT_ATTACK_TYPE[str(Path(path).resolve())] = "tts"


def load_asvspoof_attack_map(protocol_path: Path) -> dict[str, str]:
    """Parses a trial_metadata.txt-format ASVspoof2019/2021 protocol file
    (columns: speaker_id utt_id codec tx attack_id key trim subset) into
    {utt_id: "tts"|"vc"}. Bonafide rows (key == "bonafide", attack_id "-")
    carry no attack type and are omitted, not mapped to a placeholder."""
    result: dict[str, str] = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            utt_id, attack_id, key = parts[1], parts[4], parts[5]
            if key != "spoof" or attack_id not in ATTACK_ID_TO_TYPE:
                continue
            result[utt_id] = ATTACK_ID_TO_TYPE[attack_id]
    return result


def attack_type_for_file(
    wav_path: Path, source_dir: str, per_file_map: dict[str, str] | None
) -> str:
    """Resolves one file's attack type: per-file map first (e.g.
    ASVspoof2021's trial_metadata.txt, keyed by filename stem), then the
    directory-level default (e.g. MLAAD, accent-expansion TTS dirs),
    else "unknown" (In-the-Wild, CodecFake — no ground truth available)."""
    if per_file_map is not None:
        stem = wav_path.stem
        if stem in per_file_map:
            return per_file_map[stem]
    resolved_dir = str(Path(source_dir).resolve())
    return DIRECTORY_DEFAULT_ATTACK_TYPE.get(resolved_dir, "unknown")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_attack_labels.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add model_training/attack_labels.py model_training/test_attack_labels.py
git commit -m "feat(voice_guard): add attack-type ({tts,vc}) labeling for track 4"
```

---

### Task 3: Wire sequence + scalars + attack-type through `dataset.py`

**Files:**
- Modify: `model_training/dataset.py`
- Create: `model_training/test_dataset_sequence.py`

**Interfaces:**
- Modifies: `Example` gains `lfcc_seq: np.ndarray` (was `features`),
  `scalars: np.ndarray`, `attack_type: int` (`0`=tts, `1`=vc,
  `-100`=unknown/real — PyTorch `CrossEntropyLoss`'s default
  `ignore_index`, chosen so no `ignore_index=` kwarg is needed downstream).
- Modifies: `build_examples(..., attack_type_maps: dict[str, dict[str, str] | None] | None = None)`
  — `attack_type_maps` keyed by resolved source directory path, value is
  either a per-file map (from `load_asvspoof_attack_map`) or `None` (use
  the directory default / unknown fallback). Directories not present in
  the dict behave exactly as `attack_type_for_file` with `per_file_map=None`.
- Modifies: `to_arrays(examples) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  — `(X_seq, X_scalars, y, attack_type_y)`, where `X_seq` is
  `(N, n_frames, 60)`.
- Consumes: `extract_lfcc_sequence`, `extract_scalars` (Task 1),
  `attack_type_for_file` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_dataset_sequence.py
"""Tests for dataset.py's sequence/scalar/attack-type Example fields
(remediation tracks 3+4)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from attack_labels import register_tts_only_directory
from dataset import build_examples, to_arrays
from features import CHUNK_SAMPLES, N_LFCC, SAMPLE_RATE


def _write_tone(path, seconds=3.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    sig = 0.3 * np.sin(2 * np.pi * 150 * t) + rng.normal(0, 0.01, n)
    sf.write(str(path), sig.astype(np.float32), SAMPLE_RATE)


def test_build_examples_populates_sequence_scalars_and_attack_type(tmp_path):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    _write_tone(real_dir / "r1.wav", seed=1)
    _write_tone(fake_dir / "f1.wav", seed=2)
    register_tts_only_directory(fake_dir)

    examples = build_examples(real_dir, fake_dir, channel_recipes=[None])
    assert len(examples) > 0

    real_ex = [e for e in examples if e.label == 0][0]
    fake_ex = [e for e in examples if e.label == 1][0]

    assert real_ex.lfcc_seq.shape[1] == N_LFCC
    assert real_ex.scalars.shape == (6,)
    assert real_ex.attack_type == -100  # real examples are never attack-typed

    assert fake_ex.attack_type == 0  # "tts", via the registered directory default


def test_to_arrays_returns_four_arrays_with_matching_lengths(tmp_path):
    real_dir = tmp_path / "real2"
    fake_dir = tmp_path / "fake2"
    real_dir.mkdir()
    fake_dir.mkdir()
    _write_tone(real_dir / "r1.wav", seed=3)
    _write_tone(fake_dir / "f1.wav", seed=4)

    examples = build_examples(real_dir, fake_dir, channel_recipes=[None])
    X_seq, X_scalars, y, attack_y = to_arrays(examples)
    n = len(examples)
    assert X_seq.shape[0] == n
    assert X_scalars.shape == (n, 6)
    assert y.shape == (n,)
    assert attack_y.shape == (n,)


def test_unlabeled_fake_directory_gets_unknown_attack_type(tmp_path):
    real_dir = tmp_path / "real3"
    fake_dir = tmp_path / "fake3_unlabeled"
    real_dir.mkdir()
    fake_dir.mkdir()
    _write_tone(real_dir / "r1.wav", seed=5)
    _write_tone(fake_dir / "f1.wav", seed=6)
    # deliberately NOT registered as tts-only, and no attack_type_maps entry

    examples = build_examples(real_dir, fake_dir, channel_recipes=[None])
    fake_ex = [e for e in examples if e.label == 1][0]
    assert fake_ex.attack_type == -100  # unknown -> masked out, same sentinel as real
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_dataset_sequence.py -v`
Expected: FAIL — `Example` has no `lfcc_seq`/`scalars`/`attack_type`
fields yet, `to_arrays` still returns 2 arrays.

- [ ] **Step 3: Implement the changes in `dataset.py`**

Replace the `Example` dataclass:

```python
ATTACK_TYPE_TO_INT = {"tts": 0, "vc": 1}  # "unknown" and real both map to IGNORE_ATTACK_TYPE
IGNORE_ATTACK_TYPE = -100  # PyTorch CrossEntropyLoss's default ignore_index


@dataclass
class Example:
    lfcc_seq: np.ndarray  # (n_frames, 60), unpooled
    scalars: np.ndarray  # (6,) = 3 prosody + 3 physio
    label: int
    attack_type: int  # ATTACK_TYPE_TO_INT value, or IGNORE_ATTACK_TYPE
    source_file: str
    source_dir: str  # which --real/--fake/--real-clean/--fake-clean dir this came from
```

Update the import line:

```python
from attack_labels import attack_type_for_file
from features import SAMPLE_RATE, chunk_audio, extract_lfcc_sequence, extract_scalars
```

Update `_process_file` (replace its body from `pcm = trim_edge_silence` onward):

```python
def _process_file(
    args: tuple[Path, int, str, list[str | None], np.random.SeedSequence, dict[str, str] | None],
) -> list[Example]:
    wav_path, label, source_dir, channel_recipes, seed_seq, per_file_attack_map = args
    rng = np.random.default_rng(seed_seq)
    pcm = _load_mono_16k(wav_path)
    pcm = trim_edge_silence(pcm, rng=rng)
    if label == 1:
        attack_type_str = attack_type_for_file(wav_path, source_dir, per_file_attack_map)
        attack_type = ATTACK_TYPE_TO_INT.get(attack_type_str, IGNORE_ATTACK_TYPE)
    else:
        attack_type = IGNORE_ATTACK_TYPE  # real examples are never attack-typed
    out: list[Example] = []
    for recipe in channel_recipes:
        degraded = _maybe_channel(pcm, recipe, rng)
        for chunk in chunk_audio(degraded):
            out.append(
                Example(
                    lfcc_seq=extract_lfcc_sequence(chunk),
                    scalars=extract_scalars(chunk),
                    label=label,
                    attack_type=attack_type,
                    source_file=str(wav_path.resolve()),
                    source_dir=source_dir,
                )
            )
    return out
```

Update `build_examples`'s signature and task-building loop:

```python
def build_examples(
    real_dir: Path | list[Path],
    fake_dir: Path | list[Path],
    channel_recipes: list[str | None] = (None,),
    workers: int | None = None,
    seed: int = 0,
    attack_type_maps: dict[str, dict[str, str] | None] | None = None,
) -> list[Example]:
    """... (docstring unchanged) ...
    attack_type_maps: optional {resolved_source_dir: per_file_map_or_None}
    — see attack_labels.attack_type_for_file. Directories absent from this
    dict fall back to the directory-level default / "unknown", same as
    passing an explicit None entry."""
    attack_type_maps = attack_type_maps or {}
    tasks: list[tuple[Path, int, str, list[str | None], dict[str, str] | None]] = []
    for label, dirs in ((0, real_dir), (1, fake_dir)):
        for directory in _as_dir_list(dirs):
            source_dir = str(Path(directory).resolve())
            per_file_map = attack_type_maps.get(source_dir)
            for wav_path in sorted(Path(directory).glob("*.wav")):
                tasks.append((wav_path, label, source_dir, list(channel_recipes), per_file_map))

    seed_seqs = np.random.SeedSequence(seed).spawn(len(tasks))
    tasks = [(*t, ss) for t, ss in zip(tasks, seed_seqs)]
    # re-order so seed_seq is 5th positional arg (matches _process_file's tuple order)
    tasks = [(t[0], t[1], t[2], t[3], t[5], t[4]) for t in tasks]
    ...
```

(The rest of `build_examples` — the `workers`/`ProcessPoolExecutor`
dispatch loop and the `report_yield` call — is unchanged; it already
treats `tasks` opaquely via `_process_file`.)

Update `to_arrays`:

```python
def to_arrays(examples: list[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_seq = np.stack([e.lfcc_seq for e in examples]).astype(np.float32)
    X_scalars = np.stack([e.scalars for e in examples]).astype(np.float32)
    y = np.array([e.label for e in examples], dtype=np.int64)
    attack_y = np.array([e.attack_type for e in examples], dtype=np.int64)
    return X_seq, X_scalars, y, attack_y
```

`report_yield` is unchanged (it only reads `source_dir`/`source_file`,
unaffected by the `features`->`lfcc_seq`/`scalars` rename).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_dataset_sequence.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the existing dataset-integrity suite and fix fallout**

Run: `python -m pytest test_dataset_integrity.py test_pipeline_smoke.py -v`
Expected: failures anywhere still referencing `Example(features=...)` or
`to_arrays`'s old 2-tuple return — fix those call sites to match the new
`Example`/`to_arrays` signatures (mechanical, not conceptual: e.g.
`test_dataset_integrity.py`'s `Example(features=np.zeros(63,...), ...)`
synthetic fixture becomes
`Example(lfcc_seq=np.zeros((10, 60)), scalars=np.zeros(6), ..., attack_type=-100, ...)`).

- [ ] **Step 6: Commit**

```bash
git add model_training/dataset.py model_training/test_dataset_sequence.py model_training/test_dataset_integrity.py model_training/test_pipeline_smoke.py
git commit -m "feat(voice_guard): thread per-frame sequence, scalars, and attack-type through dataset.py"
```

---

### Task 4: `VoiceGuardSeqCNN` model (TDD)

**Files:**
- Modify: `model_training/model.py`
- Create: `model_training/test_model_seq_cnn.py`

**Interfaces:**
- Produces: `class FixedNormalizeSeq(nn.Module)` — normalizes a `(batch,
  n_frames, 60)` sequence per-LFCC-channel (broadcast across the time
  axis), analogous to `FixedNormalize` but for the sequence branch.
- Produces: `class VoiceGuardSeqCNN(nn.Module)` —
  `forward(seq: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]`
  returning `(real_fake_logits, attack_type_logits)`, each `(batch, 2)`.
- Consumes: nothing from earlier tasks except the shapes established
  there (`(n_frames, 60)` sequence, `(6,)` scalars).

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_model_seq_cnn.py
"""Tests for VoiceGuardSeqCNN (remediation tracks 3+4)."""
from __future__ import annotations

import numpy as np
import torch

from model import FixedNormalizeSeq, VoiceGuardSeqCNN


def test_fixed_normalize_seq_broadcasts_across_time():
    mean = np.zeros(60, dtype=np.float32)
    std = np.ones(60, dtype=np.float32)
    norm = FixedNormalizeSeq(mean, std)
    x = torch.randn(4, 184, 60)
    out = norm(x)
    assert out.shape == (4, 184, 60)


def test_seq_cnn_forward_shapes():
    model = VoiceGuardSeqCNN(
        n_frames=184,
        n_lfcc=60,
        n_scalars=6,
        seq_mean=np.zeros(60, dtype=np.float32),
        seq_std=np.ones(60, dtype=np.float32),
        scalar_mean=np.zeros(6, dtype=np.float32),
        scalar_std=np.ones(6, dtype=np.float32),
    )
    seq = torch.randn(8, 184, 60)
    scalars = torch.randn(8, 6)
    real_fake_logits, attack_type_logits = model(seq, scalars)
    assert real_fake_logits.shape == (8, 2)
    assert attack_type_logits.shape == (8, 2)


def test_seq_cnn_param_count_is_small():
    model = VoiceGuardSeqCNN(
        n_frames=184, n_lfcc=60, n_scalars=6,
        seq_mean=np.zeros(60, dtype=np.float32), seq_std=np.ones(60, dtype=np.float32),
        scalar_mean=np.zeros(6, dtype=np.float32), scalar_std=np.ones(6, dtype=np.float32),
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000  # per track 3 plan §2.3's "under 50-100K" budget, generous margin
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_model_seq_cnn.py -v`
Expected: FAIL — `FixedNormalizeSeq`/`VoiceGuardSeqCNN` don't exist.

- [ ] **Step 3: Implement in `model.py`**

Add after the existing `VoiceGuardMLP` class (keep `VoiceGuardMLP`,
`FixedNormalize`, `INPUT_DIM` as-is — nothing currently deployed uses the
new classes yet, and `select_best_checkpoint.py`/`eval_held_out_dirs.py`
still need to load old-style checkpoints for baseline comparison in
Task 7):

```python
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
    fakes — see train.py's masked loss). See track 3 plan §2.2 for the
    architecture rationale (CNN over GRU/LSTM: stateless, simpler ONNX
    export) and track 4 spec §4 for the dual-head design."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_model_seq_cnn.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add model_training/model.py model_training/test_model_seq_cnn.py
git commit -m "feat(voice_guard): add VoiceGuardSeqCNN with real/fake + attack-type heads"
```

---

### Task 5: `train_seq_cnn.py` — training loop with masked multi-task loss

**Files:**
- Create: `model_training/train_seq_cnn.py`
- Create: `model_training/test_train_seq_cnn.py`

**Interfaces:**
- Produces: `compute_masked_attack_type_loss(logits: Tensor, targets: Tensor) -> Tensor`
  — thin wrapper around `nn.CrossEntropyLoss(ignore_index=-100)`, exists as
  its own tested function so Task's masking behavior has a direct unit
  test independent of the full training loop.
- Produces: a `main()` CLI, modeled on `train.py`'s existing argument
  surface (`--real`, `--fake`, `--real-clean`, `--fake-clean`, `--channel`,
  `--epochs`, `--weight-decay`, `--label-smoothing`,
  `--save-every-epoch-checkpoints`, `--out`, `--attack-type-loss-weight`
  new), reusing `train.py::compute_eer`/`parse_channel_arg` rather than
  duplicating them.
- Consumes: `build_examples`, `to_arrays` (Task 3), `VoiceGuardSeqCNN`
  (Task 4).

- [ ] **Step 1: Write the failing test for the masked loss**

```python
# model_training/test_train_seq_cnn.py
"""Tests for train_seq_cnn.py's masked multi-task loss."""
from __future__ import annotations

import torch

from train_seq_cnn import compute_masked_attack_type_loss


def test_masked_loss_ignores_ignore_index_examples():
    # 2 labeled (tts=0, vc=1) + 2 unlabeled (-100) examples.
    logits = torch.tensor([
        [5.0, -5.0],  # confidently "tts" -> should contribute ~0 loss for target 0
        [-5.0, 5.0],  # confidently "vc" -> should contribute ~0 loss for target 1
        [0.0, 0.0],   # ignored regardless of logits
        [0.0, 0.0],   # ignored regardless of logits
    ])
    targets = torch.tensor([0, 1, -100, -100])
    loss = compute_masked_attack_type_loss(logits, targets)
    assert loss.item() < 0.01


def test_masked_loss_all_ignored_does_not_crash():
    logits = torch.zeros(3, 2)
    targets = torch.tensor([-100, -100, -100])
    loss = compute_masked_attack_type_loss(logits, targets)
    assert torch.isfinite(loss) or torch.isnan(loss)  # CrossEntropyLoss returns NaN when the whole batch is ignored — documented PyTorch behavior, caller must guard (see Step 3's train loop)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_train_seq_cnn.py -v`
Expected: FAIL — `train_seq_cnn` module doesn't exist.

- [ ] **Step 3: Implement `train_seq_cnn.py`**

```python
# model_training/train_seq_cnn.py
"""Trains VoiceGuardSeqCNN: frame-level LFCC sequence + prosody/physio
scalars -> real/fake (all examples) + attack-type (labeled fakes only,
masked loss). Consolidates remediation tracks 2 (physio), 3 (sequence
CNN), and 4 (attack-type head) into one training pass — see
docs/superpowers/plans/2026-09-11-frame-level-seq-model-and-attack-type-plan.md
Global Constraints for why this replaces three separate retrains.

Usage:
    python train_seq_cnn.py \
        --real data/real data/real2021 data/real_itw_train \
        --fake data/fake data/fake2021 data/fake_itw_train \
        --real-clean data/real_noise_aug_split/train/en_native data/real_noise_aug_split/train/hi_native \
        --channel whatsapp volte none \
        --weight-decay 1e-4 --label-smoothing 0.05 --attack-type-loss-weight 1.0 \
        --save-every-epoch-checkpoints \
        --out runs/voice_guard_v11_seqcnn --epochs 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import build_examples, split_by_source, to_arrays
from train import compute_eer, parse_channel_arg


def compute_masked_attack_type_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropyLoss with ignore_index=-100 (dataset.py's
    IGNORE_ATTACK_TYPE sentinel) — real examples and unlabeled fakes
    contribute zero gradient to the attack-type head. Returns NaN if
    EVERY example in the batch is ignored (documented PyTorch behavior)
    — callers must skip adding this term to the total loss in that case
    (see main()'s training loop below)."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits, targets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, required=True, nargs="+")
    ap.add_argument("--fake", type=Path, required=True, nargs="+")
    ap.add_argument("--real-clean", type=Path, default=[], nargs="*")
    ap.add_argument("--fake-clean", type=Path, default=[], nargs="*")
    ap.add_argument("--channel", nargs="*", default=[None])
    ap.add_argument("--out", type=Path, default=Path("runs/voice_guard_seqcnn"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--attack-type-loss-weight", type=float, default=1.0)
    ap.add_argument("--save-every-epoch-checkpoints", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    from model import VoiceGuardSeqCNN  # local import: keeps torch off the ProcessPoolExecutor workers' import path

    recipes = parse_channel_arg(args.channel)
    print(f"Building examples (channels={recipes})...")
    examples = build_examples(args.real, args.fake, channel_recipes=recipes, workers=args.workers, seed=args.seed)
    if args.real_clean or args.fake_clean:
        examples += build_examples(args.real_clean, args.fake_clean, channel_recipes=[None], workers=args.workers, seed=args.seed)

    train_ex, val_ex = split_by_source(examples)
    X_seq_train, X_scalar_train, y_train, attack_train = to_arrays(train_ex)
    X_seq_val, X_scalar_val, y_val, attack_val = to_arrays(val_ex)
    print(f"{len(train_ex)} train windows / {len(val_ex)} val windows from "
          f"{len({e.source_file for e in examples})} source files")

    seq_mean = X_seq_train.reshape(-1, X_seq_train.shape[-1]).mean(axis=0)
    seq_std = X_seq_train.reshape(-1, X_seq_train.shape[-1]).std(axis=0) + 1e-8
    scalar_mean = X_scalar_train.mean(axis=0)
    scalar_std = X_scalar_train.std(axis=0) + 1e-8

    model = VoiceGuardSeqCNN(
        n_frames=X_seq_train.shape[1], n_lfcc=X_seq_train.shape[2], n_scalars=X_scalar_train.shape[1],
        seq_mean=seq_mean, seq_std=seq_std, scalar_mean=scalar_mean, scalar_std=scalar_std,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    real_fake_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_seq_train), torch.from_numpy(X_scalar_train),
            torch.from_numpy(y_train), torch.from_numpy(attack_train),
        ),
        batch_size=args.batch_size, shuffle=True,
    )
    X_seq_val_t = torch.from_numpy(X_seq_val).to(device)
    X_scalar_val_t = torch.from_numpy(X_scalar_val).to(device)

    checkpoint_dir = args.out / "checkpoints"
    if args.save_every_epoch_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for seq_b, scalar_b, y_b, attack_b in train_loader:
            seq_b, scalar_b, y_b, attack_b = seq_b.to(device), scalar_b.to(device), y_b.to(device), attack_b.to(device)
            opt.zero_grad()
            real_fake_logits, attack_logits = model(seq_b, scalar_b)
            loss = real_fake_loss_fn(real_fake_logits, y_b)
            if (attack_b != -100).any():  # skip the attack-type term entirely if the whole batch is unlabeled (avoids NaN, see compute_masked_attack_type_loss's docstring)
                loss = loss + args.attack_type_loss_weight * compute_masked_attack_type_loss(attack_logits, attack_b)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(seq_b)
        total_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_real_fake_logits, val_attack_logits = model(X_seq_val_t, X_scalar_val_t)
            val_probs = torch.softmax(val_real_fake_logits, dim=-1)[:, 1].cpu().numpy()
        eer = compute_eer(val_probs, y_val)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss:.4f}  val_eer={eer:.4f}")
        if args.save_every_epoch_checkpoints:
            torch.save(model.cpu().state_dict(), checkpoint_dir / f"epoch_{epoch + 1:02d}.pt")
            model = model.to(device)

    model = model.cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    np.savez(
        args.out / "norm_stats.npz",
        seq_mean=seq_mean, seq_std=seq_std, scalar_mean=scalar_mean, scalar_std=scalar_std,
        n_frames=X_seq_train.shape[1], n_lfcc=X_seq_train.shape[2], n_scalars=X_scalar_train.shape[1],
    )  # ONNX export (Task 9) needs these shapes/stats without re-running feature extraction

    dummy_seq = torch.zeros(1, X_seq_train.shape[1], X_seq_train.shape[2])
    dummy_scalars = torch.zeros(1, X_scalar_train.shape[1])
    torch.onnx.export(
        model, (dummy_seq, dummy_scalars), str(args.out / "model.onnx"),
        input_names=["lfcc_sequence", "scalars"], output_names=["real_fake_logits", "attack_type_logits"],
        opset_version=13, dynamo=False,
    )
    print(f"Saved model.pt / model.onnx / norm_stats.npz -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the masked-loss test to verify it passes**

Run: `python -m pytest test_train_seq_cnn.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Smoke-test the full training loop on synthetic data**

```python
# append to model_training/test_train_seq_cnn.py
def test_train_seq_cnn_end_to_end_smoke(tmp_path, monkeypatch):
    import subprocess
    import sys

    import numpy as np
    import soundfile as sf

    real_dir, fake_dir = tmp_path / "real", tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    rng = np.random.default_rng(0)
    sr = 16000
    for i in range(6):
        t = np.arange(int(3.5 * sr)) / sr
        sig = (0.3 * np.sin(2 * np.pi * 150 * t) + rng.normal(0, 0.02, len(t))).astype(np.float32)
        sf.write(str(real_dir / f"r{i}.wav"), sig, sr)
        sig2 = (0.3 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.02, len(t))).astype(np.float32)
        sf.write(str(fake_dir / f"f{i}.wav"), sig2, sr)

    out_dir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
         "--out", str(out_dir), "--epochs", "1", "--workers", "1"],
        cwd=Path(__file__).parent, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "model.onnx").exists()
    assert (out_dir / "norm_stats.npz").exists()
```

Run: `python -m pytest test_train_seq_cnn.py -v`
Expected: PASS (3 tests). If the ONNX export step fails, check the opset
version and `dynamo=False` flag first (matches the existing `train.py`
pattern exactly) before assuming a new bug.

- [ ] **Step 6: Commit**

```bash
git add model_training/train_seq_cnn.py model_training/test_train_seq_cnn.py
git commit -m "feat(voice_guard): add train_seq_cnn.py with masked multi-task loss"
```

---

### Task 6: Attack-type label wiring for the real corpus + coverage reporting

**Files:**
- Modify: `model_training/check_corpus.py`
- Modify: `model_training/dataset_audit.py`
- Create: `model_training/build_attack_type_maps.py`

**Interfaces:**
- Produces: `build_attack_type_maps() -> dict[str, dict[str, str] | None]`
  — the concrete `attack_type_maps` argument for `build_examples`,
  constructed from this project's actual corpus layout (hardcodes the
  known directory roles from the design spec's §3 table). Kept as its own
  script/function (not inlined into `train_seq_cnn.py`) so it can be
  tested and reused by `check_corpus.py`'s coverage report without
  duplicating the directory list.
- Modifies: `check_corpus.py` gains an attack-type coverage line per
  `--pair`, reusing `build_attack_type_maps()`.

- [ ] **Step 1: Implement `build_attack_type_maps.py`**

```python
# model_training/build_attack_type_maps.py
"""Builds the attack_type_maps argument for dataset.build_examples from
this project's actual corpus layout — see
docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §3
for the source-by-source rationale. Kept separate from train_seq_cnn.py
so check_corpus.py can report coverage without running a training job."""
from __future__ import annotations

from pathlib import Path

from attack_labels import load_asvspoof_attack_map, register_tts_only_directory

HERE = Path(__file__).resolve().parent


def build_attack_type_maps() -> dict[str, dict[str, str] | None]:
    maps: dict[str, dict[str, str] | None] = {}

    # data/fake2021: per-file labels via ASVspoof2021's trial_metadata.txt
    protocol = HERE / "data" / "asvspoof2021_la" / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
    if protocol.exists():
        per_file_map = load_asvspoof_attack_map(protocol)
        maps[str((HERE / "data" / "fake2021").resolve())] = per_file_map

    # Directories where every fake file is TTS by construction (verified
    # against each generation script's own docstring, spec §3).
    tts_only_dirs = [
        HERE / "data" / "mlaad_en500" / "fake" / "en",
        HERE / "data" / "accents" / "fake" / "en_foreign",
        HERE / "data" / "accents" / "fake" / "hi_native",
        HERE / "data" / "accents" / "fake" / "hi_foreign",
    ]
    for d in tts_only_dirs:
        if d.exists():
            register_tts_only_directory(d)
            maps[str(d.resolve())] = None  # None = use the directory default just registered

    # data/fake (ASVspoof2019 LA train): protocol not yet recovered — see
    # spec §3's "Recoverable, not yet recovered" row. Left unmapped here
    # (falls back to "unknown") until that protocol is re-downloaded; this
    # function does not silently fabricate labels for it.

    return maps
```

- [ ] **Step 2: Test it against the real corpus (not a synthetic fixture — this function's whole job is real-path wiring)**

```python
# append to model_training/test_attack_labels.py
def test_build_attack_type_maps_finds_fake2021_if_protocol_present():
    from build_attack_type_maps import build_attack_type_maps
    from pathlib import Path

    maps = build_attack_type_maps()
    fake2021 = str((Path(__file__).parent / "data" / "fake2021").resolve())
    protocol_path = (
        Path(__file__).parent / "data" / "asvspoof2021_la" / "LA-keys-full"
        / "keys" / "LA" / "CM" / "trial_metadata.txt"
    )
    if protocol_path.exists():
        assert fake2021 in maps
        assert len(maps[fake2021]) > 0
    else:
        assert fake2021 not in maps  # nothing to assert if the real corpus isn't present on this machine
```

Run: `python -m pytest test_attack_labels.py -v`
Expected: PASS (6 tests total in this file now).

- [ ] **Step 3: Extend `check_corpus.py` with attack-type coverage reporting**

Find `check_corpus.py`'s per-pair reporting loop (the `for real_dir, fake_dir in args.pair:` block or equivalent — read the file first, its current structure was shown in this session's own preflight run) and add, right after each pair's existing yield/shortcut output:

```python
    from build_attack_type_maps import build_attack_type_maps
    from attack_labels import attack_type_for_file

    attack_maps = build_attack_type_maps()
    fake_resolved = str(fake_dir.resolve())
    per_file_map = attack_maps.get(fake_resolved)
    fake_files = sorted(fake_dir.glob("*.wav"))
    if fake_files:
        labeled = sum(
            1 for f in fake_files
            if attack_type_for_file(f, fake_resolved, per_file_map) != "unknown"
        )
        print(f"  attack-type coverage: {labeled}/{len(fake_files)} fake files labeled "
              f"({labeled / len(fake_files):.0%})")
```

- [ ] **Step 4: Run it against the real corpus to see current coverage**

Run: `python check_corpus.py --pair data/real data/fake --pair data/real2021 data/fake2021 --pair data/real_itw_train data/fake_itw_train`
Expected: prints an attack-type coverage line per pair — `data/fake2021`
should show high coverage (per-file map), `data/fake` should show ~0%
(protocol not recovered yet, Task 7 addresses this), `data/fake_itw_train`
should show 0% (no ground truth exists, by design — not a bug to fix).

- [ ] **Step 5: Commit**

```bash
git add model_training/build_attack_type_maps.py model_training/check_corpus.py model_training/test_attack_labels.py
git commit -m "feat(voice_guard): wire real corpus attack-type maps + coverage reporting"
```

---

### Task 7: Recover `data/fake`'s ASVspoof2019 attack-type labels

**Files:**
- Modify: `model_training/data/README.md`
- Modify: `model_training/build_attack_type_maps.py`

**Interfaces:**
- Modifies: `build_attack_type_maps()` to also map `data/fake` once its
  protocol file is present.

- [ ] **Step 1: Document and run the protocol-only re-download**

Add to `model_training/data/README.md`, in the ASVspoof2019 LA section:

```markdown
**Attack-type protocol only** (for track 4's TTS/VC labeling — does NOT
need the multi-GB audio archive, `data/real`/`data/fake` are already
extracted): re-download just the protocol file from the same Kaggle
source used originally:

```bash
kaggle datasets download -d anishsarkar22/asvpoof-2019-dataset-la -p data/asvspoof2019_la_protocol_only --unzip -f "ASVspoof2019.LA.cm.train.trn.txt"
```

(If `-f` isn't supported by the installed `kaggle` CLI version, download
the full dataset to a throwaway directory and delete everything except
the protocol file afterward — it's a few hundred KB, the audio is
multiple GB; do not leave the audio archive on disk.)

The train protocol's column layout differs slightly from
`trial_metadata.txt` (columns: `speaker_id utt_id - attack_id key`, no
codec/tx/trim/subset columns) — `load_asvspoof_attack_map` handles this
via its `len(parts) < 6: continue` guard for the 2021 format; the 2019
train format needs its own thin loader, see
`build_attack_type_maps.py`.
```

- [ ] **Step 2: Add a 2019-train-protocol loader and wire it in**

```python
# add to model_training/attack_labels.py, after load_asvspoof_attack_map
def load_asvspoof_2019_train_attack_map(protocol_path: Path) -> dict[str, str]:
    """Parses the ASVspoof2019 LA train protocol format (columns:
    speaker_id utt_id - attack_id key — no codec/tx/trim/subset columns,
    unlike the 2021 trial_metadata.txt format load_asvspoof_attack_map
    handles) into {utt_id: "tts"|"vc"}."""
    result: dict[str, str] = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            utt_id, attack_id, key = parts[1], parts[3], parts[4]
            if key != "spoof" or attack_id not in ATTACK_ID_TO_TYPE:
                continue
            result[utt_id] = ATTACK_ID_TO_TYPE[attack_id]
    return result
```

```python
# add a corresponding test to model_training/test_attack_labels.py
def test_load_asvspoof_2019_train_attack_map(tmp_path: Path):
    from attack_labels import load_asvspoof_2019_train_attack_map

    protocol = tmp_path / "train.trn.txt"
    protocol.write_text(
        "LA_0079 LA_T_1234567 - A01 spoof\n"
        "LA_0079 LA_T_7654321 - A17 spoof\n"
        "LA_0079 LA_T_0000000 - - bonafide\n"
    )
    result = load_asvspoof_2019_train_attack_map(protocol)
    assert result["LA_T_1234567"] == "tts"
    assert result["LA_T_7654321"] == "vc"
    assert "LA_T_0000000" not in result
```

Update `build_attack_type_maps()`'s `data/fake` comment block:

```python
    # data/fake (ASVspoof2019 LA train): per-file labels via the
    # protocol-only re-download (see data/README.md).
    train_protocol = HERE / "data" / "asvspoof2019_la_protocol_only" / "ASVspoof2019.LA.cm.train.trn.txt"
    if train_protocol.exists():
        maps[str((HERE / "data" / "fake").resolve())] = load_asvspoof_2019_train_attack_map(train_protocol)
```

- [ ] **Step 3: Run tests, then re-run the coverage check**

Run: `python -m pytest test_attack_labels.py -v`
Expected: PASS (7 tests).

Run: `python check_corpus.py --pair data/real data/fake` (after actually
performing Step 1's download)
Expected: `data/fake`'s attack-type coverage now near 100% (every spoof
file in ASVspoof2019 LA train has a documented attack ID).

- [ ] **Step 4: Commit**

```bash
git add model_training/attack_labels.py model_training/build_attack_type_maps.py model_training/test_attack_labels.py model_training/data/README.md
git commit -m "feat(voice_guard): recover data/fake's ASVspoof2019 attack-type labels"
```

---

### Task 8: Phase 1 validation — offline retrain + confound/regression gate

**Files:** none (training-run + analysis task) — outputs land in
`model_training/runs/voice_guard_v11_seqcnn/`

- [ ] **Step 1: Preflight**

Run `check_corpus.py` against the same directory pairs used for every
prior track's retrain (`data/real`+`fake`, `data/real2021`+`fake2021`,
`data/real_itw_train`+`fake_itw_train`), confirming attack-type coverage
from Tasks 6-7 looks right before committing to a full run.

- [ ] **Step 2: Retrain with `train_seq_cnn.py`, same corpus scope as the currently-deployed model**

```bash
python train_seq_cnn.py \
  --real data/real data/real2021 data/real_itw_train \
  --fake data/fake data/fake2021 data/fake_itw_train \
  --real-clean data/real_noise_aug_split/train/en_native data/real_noise_aug_split/train/hi_native \
  --channel whatsapp volte none \
  --weight-decay 1e-4 --label-smoothing 0.05 --attack-type-loss-weight 1.0 \
  --workers 8 \
  --save-every-epoch-checkpoints \
  --out runs/voice_guard_v11_seqcnn --epochs 25
```

Launch detached (`nohup ... &` + `disown`) per this project's documented
memory-constraint workaround, with `--workers` sized to actual free RAM at
launch time (see `state.md`'s running note on this — verify current free
memory before picking a worker count, don't copy a stale number).

- [ ] **Step 3: Reproduce the median-split confound analysis against the new model**

Adapt `measure_confound.py` for the two-input model (it currently assumes
a single flat feature vector and a `VoiceGuardMLP`) — the median-split
logic itself is unchanged, only the model-loading and feature-indexing
need updating:

```python
# model_training/measure_confound_seqcnn.py
"""measure_confound.py's analysis, adapted for VoiceGuardSeqCNN's two
inputs. See docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md
and the Global Constraints section of this plan for the acceptance bar."""
import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import build_examples, to_arrays
from model import VoiceGuardSeqCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--norm-stats", type=Path, required=True)
    ap.add_argument("--real", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    examples = build_examples(args.real, [], channel_recipes=[None])
    X_seq, X_scalars, _y, _attack = to_arrays(examples)

    stats = np.load(args.norm_stats)
    model = VoiceGuardSeqCNN(
        n_frames=int(stats["n_frames"]), n_lfcc=int(stats["n_lfcc"]), n_scalars=int(stats["n_scalars"]),
        seq_mean=stats["seq_mean"], seq_std=stats["seq_std"],
        scalar_mean=stats["scalar_mean"], scalar_std=stats["scalar_std"],
    )
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        real_fake_logits, _attack_logits = model(torch.from_numpy(X_seq), torch.from_numpy(X_scalars))
        probs = torch.softmax(real_fake_logits, dim=-1)[:, 1].numpy()

    # scalars order is [pauseRatio, energyVariance, zcrVariance, jitter, shimmer, hnr]
    names = {"pauseRatio": 0, "energyVariance": 1, "zcrVariance": 2}
    for name, idx in names.items():
        col = X_scalars[:, idx]
        median = np.median(col)
        low_mean = probs[col < median].mean()
        high_mean = probs[col >= median].mean()
        print(f"{name}: low-half mean fake-prob={low_mean:.3f}  high-half={high_mean:.3f}  gap={abs(high_mean - low_mean):.3f}")


if __name__ == "__main__":
    main()
```

Run against both the currently-deployed model and the new checkpoint:

```bash
python measure_confound.py --model runs/voice_guard_v9_noisefix_final/model.pt --real data/real_itw_held
python measure_confound_seqcnn.py --model runs/voice_guard_v11_seqcnn/model.pt --norm-stats runs/voice_guard_v11_seqcnn/norm_stats.npz --real data/real_itw_held
```

**Pass condition** (per this plan's Global Constraints, restated from
track 3's own plan): the new model's gap for `energyVariance` and
`pauseRatio` must be *much* smaller than the old model's 25.1%-vs-12.9%
and 14.7%-vs-23.6% gaps — not marginally smaller. Use
`eval_stats.py::bootstrap_eer_ci`'s bootstrap approach adapted to this gap
statistic if the shrink looks borderline, same discipline as every prior
track.

- [ ] **Step 4: Check ITW EER and noise-FPR didn't regress, and check attack-type accuracy**

Adapt `eval_held_out_dirs.py` similarly (two-input model loading, same
pattern as Step 3) to report both the real/fake EER and a simple
attack-type accuracy on the labeled subset of the held-out fakes (report
only — no baseline to compare against yet, this is the first model with
this head).

- [ ] **Step 5: Record results, whether or not this passes**

Update `voice_guard/state.md` with a dated section: the measured confound
gaps (old vs. new), ITW EER, noise-FPR, and attack-type accuracy — same
"report honest negative results plainly" discipline as every prior track,
per this project's own established practice. If the confound gap and
EER/FPR gates both pass, proceed to Task 9 (Dart/mobile port); if not,
stop here and report the negative result rather than porting a model that
doesn't yet meet the bar.

---

### Task 9: Dart mirror — per-frame LFCC + scalars (Phase 2 start)

**Files:**
- Modify: `lib/utils/audio_processor.dart`
- Create: `test/audio_processor_sequence_parity_test.dart`
- Create: `model_training/dump_sequence_parity_fixture.py`

**Interfaces:**
- Produces: `AudioProcessor.extractLfccSequence(List<double> pcm) -> List<List<double>>`
  (one inner list per frame, each length 60), mirroring
  `extract_lfcc_sequence` exactly.
- Produces: `AudioProcessor.extractScalars(List<double> pcm) -> List<double>`
  (length 6, prosody then physio), mirroring `extract_scalars` exactly
  (trivial — just concatenates the existing `extractProsody`/
  `extractPhysio` outputs, no new math).

- [ ] **Step 1: Generate the parity fixture from Python**

```python
# model_training/dump_sequence_parity_fixture.py
"""Same approach as dump_parity_fixture.py (track 2), extended to the
per-frame sequence. Prints the first and last frame's LFCC coefficients
(not all 184 frames — too much to hand-paste) plus the 6 scalars, so the
Dart parity test can assert against Python-computed ground truth without
shipping a large fixture file."""
import numpy as np
from features import extract_lfcc_sequence, extract_scalars, SAMPLE_RATE

rng = np.random.default_rng(7)
n = SAMPLE_RATE * 3
t = np.arange(n) / SAMPLE_RATE
f0 = 140.0 + 6.0 * np.sin(2 * np.pi * 4.0 * t)
phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
sig = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
sig = sig + rng.normal(0, 0.02, size=n)
sig = (sig / np.abs(sig).max()).astype(np.float64)

seq = extract_lfcc_sequence(sig)
scalars = extract_scalars(sig)
print("n_frames =", seq.shape[0])
print("first_frame =", repr(seq[0].tolist()))
print("last_frame =", repr(seq[-1].tolist()))
print("scalars =", repr(scalars.tolist()))
```

Run: `python model_training/dump_sequence_parity_fixture.py` and record
the printed values.

- [ ] **Step 2: Write the failing Dart parity test**

```dart
// test/audio_processor_sequence_parity_test.dart
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/utils/audio_processor.dart';

/// Same deterministic-signal approach as audio_processor_parity_test.dart
/// (track 2) — see that file's comment for why exact PRNG parity isn't
/// expected/required, only frame-aggregate-statistic-level closeness.
List<double> _fixtureSignal() {
  const sampleRate = 16000;
  const n = sampleRate * 3;
  final rng = math.Random(7);
  final sig = List<double>.filled(n, 0);
  double phase = 0;
  for (int i = 0; i < n; i++) {
    final t = i / sampleRate;
    final f0 = 140.0 + 6.0 * math.sin(2 * math.pi * 4.0 * t);
    phase += 2 * math.pi * f0 / sampleRate;
    double s = 0.6 * math.sin(phase) + 0.25 * math.sin(2 * phase) + 0.1 * math.sin(3 * phase);
    s += (rng.nextDouble() - 0.5) * 2 * 0.02;
    sig[i] = s;
  }
  final maxAbs = sig.map((v) => v.abs()).reduce(math.max);
  return sig.map((v) => v / maxAbs).toList();
}

void main() {
  test('extractLfccSequence matches Python within tolerance', () {
    final pcm = _fixtureSignal();
    final seq = AudioProcessor.extractLfccSequence(pcm);

    // PASTE the printed n_frames / first_frame / last_frame from Step 1 here.
    const pythonNFrames = 0; // REPLACE
    const List<double> pythonFirstFrame = []; // REPLACE
    const List<double> pythonLastFrame = []; // REPLACE

    expect(seq.length, pythonNFrames);
    for (int i = 0; i < 60; i++) {
      expect(seq.first[i], closeTo(pythonFirstFrame[i], 0.1));
      expect(seq.last[i], closeTo(pythonLastFrame[i], 0.1));
    }
  });

  test('extractScalars matches Python within tolerance', () {
    final pcm = _fixtureSignal();
    final scalars = AudioProcessor.extractScalars(pcm);

    const List<double> pythonScalars = []; // REPLACE with printed `scalars` from Step 1
    expect(scalars.length, 6);
    for (int i = 0; i < 6; i++) {
      expect(scalars[i], closeTo(pythonScalars[i], 0.05));
    }
  });
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `flutter test test/audio_processor_sequence_parity_test.dart`
Expected: FAIL — `extractLfccSequence`/`extractScalars` don't exist yet.

- [ ] **Step 4: Implement in `audio_processor.dart`**

Add near the existing `extractLfcc` (the per-frame math already exists
inside that function's loop — this factors it out instead of duplicating
it; read `extractLfcc`'s current body first, since this step restructures
it rather than adding fully independent code):

```dart
  /// Per-frame LFCC matrix (unpooled) — one inner list per frame, each
  /// length 60. Mirrors features.py::extract_lfcc_sequence exactly.
  static List<List<double>> extractLfccSequence(List<double> pcm) {
    if (pcm.length < fftSize) return [];
    final frames = _frameSignal(pcm);
    return frames.map((frame) {
      final windowed = _hamming(frame);
      final spectrum = _magnitudeSpectrum(windowed);
      final energies = _linearFilterbank(spectrum);
      final logE = energies.map((v) => math.log(v + 1e-10)).toList();
      return _dct(logE).sublist(0, nLfcc);
    }).toList();
  }

  /// [pauseRatio, energyVariance, zcrVariance, jitterLocal, shimmerLocal,
  /// hnrDb] — prosody then physio. Mirrors features.py::extract_scalars.
  static List<double> extractScalars(List<double> pcm) {
    return [...extractProsody(pcm), ...extractPhysio(pcm)];
  }
```

Then simplify `extractLfcc` to reuse the new function (removes
duplication, matches this project's existing DRY convention):

```dart
  /// 3s PCM16 buffer → 60 LFCC coefficients (mean-pooled over frames).
  static List<double> extractLfcc(List<double> pcm) {
    if (pcm.length < fftSize) return List.filled(nLfcc, 0);
    final lfccFrames = extractLfccSequence(pcm);

    final mean = List<double>.filled(nLfcc, 0);
    for (final f in lfccFrames) {
      for (int i = 0; i < nLfcc; i++) { mean[i] += f[i]; }
    }
    for (int i = 0; i < nLfcc; i++) { mean[i] /= lfccFrames.length; }
    final m = mean.reduce((a, b) => a + b) / mean.length;
    final variance = mean.map((v) => (v - m) * (v - m)).reduce((a, b) => a + b) / mean.length;
    final std = math.sqrt(variance + 1e-8);
    return mean.map((v) => (v - m) / std).toList();
  }
```

- [ ] **Step 5: Paste the real Python values, run the test**

Replace the `REPLACE` placeholders with Step 1's actual output.

Run: `flutter test test/audio_processor_sequence_parity_test.dart`
Expected: PASS. Also run `flutter test test/audio_processor_parity_test.dart`
(track 2's existing test) to confirm the `extractLfcc` refactor didn't
change its output — that test's fixed expected values must still match,
since `extractLfcc`'s math is unchanged, only restructured.

- [ ] **Step 6: Commit**

```bash
git add lib/utils/audio_processor.dart test/audio_processor_sequence_parity_test.dart model_training/dump_sequence_parity_fixture.py
git commit -m "feat(voice_guard): mirror per-frame LFCC sequence + scalars in Dart, refactor extractLfcc to reuse it"
```

---

### Task 10: ONNX Runtime inference path — two inputs, two outputs

**Files:**
- Modify: `lib/services/src/tflite_io.dart`
- Modify: `lib/services/src/tflite_interface.dart`
- Modify: `lib/services/src/tflite_stub.dart`
- Modify: `lib/services/audio_service.dart`
- Modify: `model_training/README.md`

**Interfaces:**
- Modifies: `TFLiteService.infer` signature becomes
  `infer(List<List<double>> lfccSequence, List<double> scalars) -> Future<(double, String?, double)>`
  — returns `(realFakeProb, attackTypeLabel, attackTypeConfidence)`;
  `attackTypeLabel` is `null` when the model has no ONNX session loaded
  (heuristic fallback) or when confidence doesn't clear the UI threshold
  (Task 11 reads the raw confidence and label separately from this,
  Task 11's own threshold-gating happens in `audio_service.dart`, not
  here — this layer reports the raw model output, unfiltered).
- Modifies: `scoreChunk` calls `AudioProcessor.extractLfccSequence`/
  `extractScalars` instead of `extractLfcc`/`extractProsody`, and
  `extractPhysio` is no longer called directly here (folded into
  `extractScalars`).

- [ ] **Step 1: Verify `flutter_onnxruntime`'s multi-input/multi-output support before writing code**

Read `flutter_onnxruntime`'s pubspec-pinned version's own API docs/example
(check `.pub-cache` or the package's README via `flutter pub deps` output)
for `session.run` with more than one named input and more than one named
output tensor — track 3's own plan §5 flagged this as an open risk
(`Conv1d` op coverage) alongside this multi-I/O question. If either is
unsupported on the actual mobile ONNX Runtime build, this task blocks —
report that finding plainly rather than guessing around it.

- [ ] **Step 2: Update `tflite_interface.dart`**

```dart
abstract class TFLiteServiceBase {
  bool get isReady;
  Future<void> init();
  Future<(double, String?, double)> infer(List<List<double>> lfccSequence, List<double> scalars);
  Future<(double, String?, double)> scoreChunk(List<double> pcm);
  void dispose();
}
```

- [ ] **Step 3: Update `tflite_io.dart`**

```dart
  Future<(double, String?, double)> infer(List<List<double>> lfccSequence, List<double> scalars) async {
    final session = _session;
    if (_ready && session != null) {
      try {
        final flatSeq = lfccSequence.expand((frame) => frame).toList();
        final seqInput = await OrtValue.fromList(
          flatSeq.map((e) => e.toDouble()).toList(),
          [1, lfccSequence.length, nLfcc],
        );
        final scalarInput = await OrtValue.fromList(
          scalars.map((e) => e.toDouble()).toList(),
          [1, scalars.length],
        );
        final outputs = await session.run({'lfcc_sequence': seqInput, 'scalars': scalarInput});

        final realFakeOut = (await outputs['real_fake_logits']!.asFlattenedList())
            .map((e) => (e as num).toDouble()).toList();
        final attackOut = (await outputs['attack_type_logits']!.asFlattenedList())
            .map((e) => (e as num).toDouble()).toList();

        final realFakeProb = _softmaxSecond(realFakeOut);
        final attackProbs = _softmax(attackOut);
        final attackLabel = attackProbs[1] > attackProbs[0] ? 'vc' : 'tts';
        final attackConfidence = attackProbs.reduce(math.max);

        return (realFakeProb, attackLabel, attackConfidence);
      } catch (e) {
        debugPrint('ONNX inference failed: $e');
      }
    }
    final heuristic = _heuristic(lfccSequence, scalars);
    return (heuristic, null, 0.0);  // heuristic fallback never claims an attack type
  }

  double _softmaxSecond(List<double> logits) {
    final probs = _softmax(logits);
    return probs[1].clamp(0.0, 1.0);
  }

  List<double> _softmax(List<double> logits) {
    final maxL = logits.reduce(math.max);
    final exps = logits.map((v) => math.exp(v - maxL)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  double _heuristic(List<List<double>> lfccSequence, List<double> scalars) {
    if (lfccSequence.isEmpty) return 0.15;
    final pooled = List<double>.filled(nLfcc, 0);
    for (final frame in lfccSequence) {
      for (int i = 0; i < nLfcc; i++) { pooled[i] += frame[i]; }
    }
    for (int i = 0; i < nLfcc; i++) { pooled[i] /= lfccSequence.length; }
    final high = pooled.sublist((pooled.length * 0.6).floor());
    final meanH = high.reduce((a, b) => a + b) / high.length;
    final varH = high.map((v) => (v - meanH) * (v - meanH)).reduce((a, b) => a + b) / high.length;
    final pauseRatio = scalars.isNotEmpty ? scalars[0] : 0.2;
    double raw = (varH * 0.9 + pauseRatio * 0.25 + (pooled[0].abs() * 0.05)).clamp(0.0, 1.0);
    raw = 0.08 + raw * 0.78;
    return raw;
  }
```

Update `_inputDim`-style constants: remove `_inputDim` (no longer a single
flat vector), rely on the shapes passed explicitly above instead.

Update `scoreChunk`:

```dart
  Future<(double, String?, double)> scoreChunk(List<double> pcm) {
    final lfccSequence = AudioProcessor.extractLfccSequence(pcm);
    final scalars = AudioProcessor.extractScalars(pcm);
    return infer(lfccSequence, scalars);
  }
```

- [ ] **Step 4: Update `tflite_stub.dart`** (web stub, heuristic-only path)

```dart
  Future<(double, String?, double)> infer(List<List<double>> lfccSequence, List<double> scalars) async =>
      (_heuristic(lfccSequence, scalars), null, 0.0);

  double _heuristic(List<List<double>> lfccSequence, List<double> scalars) {
    // Kept independent from tflite_io.dart's copy per this project's
    // existing convention (tflite_stub.dart has never imported from
    // tflite_io.dart) — identical body, duplicated deliberately.
    if (lfccSequence.isEmpty) return 0.15;
    final pooled = List<double>.filled(60, 0);
    for (final frame in lfccSequence) {
      for (int i = 0; i < 60; i++) { pooled[i] += frame[i]; }
    }
    for (int i = 0; i < 60; i++) { pooled[i] /= lfccSequence.length; }
    final high = pooled.sublist((pooled.length * 0.6).floor());
    final meanH = high.reduce((a, b) => a + b) / high.length;
    final varH = high.map((v) => (v - meanH) * (v - meanH)).reduce((a, b) => a + b) / high.length;
    final pauseRatio = scalars.isNotEmpty ? scalars[0] : 0.2;
    double raw = (varH * 0.9 + pauseRatio * 0.25 + (pooled[0].abs() * 0.05)).clamp(0.0, 1.0);
    raw = 0.08 + raw * 0.78;
    return raw;
  }

  Future<(double, String?, double)> scoreChunk(List<double> pcm) {
    final lfccSequence = AudioProcessor.extractLfccSequence(pcm);
    final scalars = AudioProcessor.extractScalars(pcm);
    return infer(lfccSequence, scalars);
  }
```

- [ ] **Step 5: Update `audio_service.dart`'s call sites**

Find both call sites of `tflite.scoreChunk(chunk)` (`grep -n "scoreChunk"
lib/services/audio_service.dart` — this session already located them at
two lines) and update them to destructure the new 3-tuple return, e.g.:

```dart
final (score, attackType, attackConfidence) = await tflite.scoreChunk(chunk);
```

then thread `attackType`/`attackConfidence` through to wherever `score`
already flows (the alert-gating state machine and whatever notifies the UI
layer) — the exact downstream field names depend on `audio_service.dart`'s
current structure; read it fully before this step, since this plan's
earlier tasks didn't need to touch its internals and this is the first
one that does.

- [ ] **Step 6: Update `model_training/README.md`'s feature contract**

Replace the "Feature contract" paragraph to describe two ONNX inputs
(`lfcc_sequence` shape `(1, n_frames, 60)`, `scalars` shape `(1, 6)`) and
two outputs (`real_fake_logits`, `attack_type_logits`, both `(1, 2)`),
cross-referencing this plan and the track 4 design spec.

- [ ] **Step 7: `flutter analyze` and `flutter test`**

Run: `flutter analyze && flutter test`
Expected: clean analyze, all tests pass (existing tests exercising
`infer`/`scoreChunk`'s old signature need their call sites updated to the
new tuple return — fix any that break, same mechanical pattern as Task 3
Step 5).

- [ ] **Step 8: Commit**

```bash
git add lib/services/src/tflite_io.dart lib/services/src/tflite_interface.dart lib/services/src/tflite_stub.dart lib/services/audio_service.dart model_training/README.md
git commit -m "feat(voice_guard): wire two-input/two-output ONNX inference (sequence CNN + attack-type head)"
```

---

### Task 11: UI — gated attack-type sub-label

**Files:**
- Modify: `lib/screens/call_screen.dart`

**Interfaces:**
- Consumes: `attackType`/`attackConfidence` as threaded through by
  Task 10 Step 5.

- [ ] **Step 1: Read `call_screen.dart`'s current alert-display logic**

The 2026-09-11 shadcn redesign (merged into `vaani`) rewrote most of this
file — read its current state fully before editing, do not assume this
plan's earlier description of the old UI still matches. Locate wherever
the "AI DETECTED" / `ALERT` state renders its label text or badge (likely
a `ShadBadge` per the new design system).

- [ ] **Step 2: Add the gated sub-label**

Wherever the primary alert label renders, add (adapt exact widget/variable
names to what Step 1 found):

```dart
String? _attackTypeSubLabel(bool isAlertFiring, String? attackType, double attackConfidence) {
  const confidenceThreshold = 0.70; // draft value from the design spec §6, tune during on-device testing
  if (!isAlertFiring || attackType == null || attackConfidence < confidenceThreshold) {
    return null;  // never shown standalone, never below threshold
  }
  return attackType == 'vc' ? 'Voice conversion' : 'Synthetic voice';
}
```

Render it as a secondary line/badge beneath the primary "AI DETECTED"
label only when `_attackTypeSubLabel(...)` returns non-null — use
`ShadBadge` (or whatever secondary-status widget the new design system
uses elsewhere in this same screen, for visual consistency) rather than
introducing a new widget style.

- [ ] **Step 3: Manual on-device check (no automated test — this is a live visual/behavioral check)**

Once Task 10's model is deployed (after Task 12's gate passes): run Live
Mic Test against all four `test_assets/` clips
(`ai_clone_test_clip.wav`, `tts_elevenlabs_sample.wav`,
`tts_chattts_sample.wav`, `voice_conversion_asvspoof_a17.wav`) through an
external speaker, confirm the sub-label reads "Synthetic voice" for the
TTS clips and "Voice conversion" for the VC clip when confidence clears
the threshold, and confirm it's absent (not a wrong label) when confidence
doesn't clear it — a missing sub-label is the correct degraded behavior,
not a bug.

- [ ] **Step 4: Commit**

```bash
git add lib/screens/call_screen.dart
git commit -m "feat(voice_guard): add gated attack-type sub-label to call screen"
```

---

### Task 12: Final consolidated retrain + deploy gate

**Files:**
- Modify (conditionally): `assets/models/voice_detector.onnx`
- Modify: `voice_guard/state.md`
- Modify: `voice_guard/docs/CRITICAL-entity-vs-style-confound.md`

- [ ] **Step 1: Re-run Task 8's retrain with the FULL corpus (including the newly-recovered `data/fake` labels from Task 7), if Task 8's own run used a partial label set**

If Task 7 landed after Task 8's Phase 1 validation run, retrain once more
with `data/fake`'s labels now included — otherwise Task 8's run already
reflects full labeling and this step is a no-op (check `state.md`'s Task 8
entry before deciding).

- [ ] **Step 2: Deploy gate**

Only replace `assets/models/voice_detector.onnx` if ALL of: (a) Task 8's
confound-gap criterion passed, (b) ITW EER and noise-FPR didn't regress
beyond bootstrap-CI noise vs. the currently-deployed model, (c)
`flutter_onnxruntime` actually runs the two-input/two-output model
correctly on-device (Task 10 Step 1's risk didn't block). If any fail: do
not deploy, say so plainly in `state.md` (same discipline as every prior
track), and leave the currently-deployed model in place.

- [ ] **Step 3: On-device verification**

Same protocol as every prior model swap: install a build with the new
`.onnx`, run Live Mic Test with a deliberately monotone/controlled human
reading (the exact register that triggered the original false positive in
the CRITICAL doc), confirm it scores appropriately, and run Task 11 Step 3's
four-clip attack-type check.

- [ ] **Step 4: Update `state.md` and the CRITICAL doc**

Record: what was implemented (tracks 2+3+4 together), Task 8's measured
numbers, attack-type accuracy, whether deployed, on-device verification
status. Update the CRITICAL doc's remediation-track table to reflect
tracks 2, 3, and 4's outcomes.

- [ ] **Step 5: Commit**

```bash
git add voice_guard/state.md voice_guard/docs/CRITICAL-entity-vs-style-confound.md
# add assets/models/voice_detector.onnx too, only if Step 2 deployed it
git commit -m "feat(voice_guard): deploy consolidated sequence-CNN + physio + attack-type model"
```
