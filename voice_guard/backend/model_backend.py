"""
model_backend.py — the deployed ONNX model, exposed as a scoring backend.

Why this exists
---------------
`main.py` has carried the note "neither side has a real trained model yet"
since the integration API was written, and scored every request with
`_heuristic`. That is no longer true: `assets/models/voice_detector.onnx` is a
real trained artifact (the v11_seqcnn epoch_22 checkpoint — see
`voice_guard/state.md`). This module is the single place both the integration
API and the laptop console (`voice_guard/laptop_monitor/`) score a window with
it, so a judge, an agent, or the console all get their number from the exact
file that ships in the phone's APK.

Verified I/O contract (introspected 2026-09-15 against the deployed file):

    lfcc_sequence (1, 184, 60) float32   <- 3 s @ 16 kHz, 1024-pt frame / 256 hop
    scalars       (1, 6)       float32   <- [pauseRatio, energyVar, zcrVar,
                                             jitter_local, shimmer_local, hnr_db]
 -> real_fake_logits (1, 2), attack_type_logits (1, 2)

Normalization is baked into the exported graph (`FixedNormalize*`, see
`model_training/model.py:17`), so raw features go in — byte-for-byte what
`lib/services/src/tflite_io.dart` sends. That is what makes a laptop number
comparable to a phone number.

Honest-degradation rule
-----------------------
If onnxruntime is missing, the asset is absent, or the session raises, this
module records *why* and reports `loaded == False`. Callers then fall back to
the heuristic path and label the response `modelBackend="heuristic"`. It never
returns a number that did not come from the model, and it never takes the API
down with it.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

N_FRAMES = 184      # 3 s @ 16 kHz through 1024-pt frames with a 256 hop
N_LFCC = 60
N_SCALARS = 6

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "voice_detector.onnx"

ATTACK_LABELS = ("tts", "vc")  # index 0 / 1 of attack_type_logits, per tflite_io.dart


class ModelUnavailable(RuntimeError):
    """Raised when scoring is attempted without a loaded session."""


@dataclass(frozen=True)
class ScoreResult:
    """One window's model output, with the raw logits kept for diagnostics."""

    p_fake: float
    p_real: float
    attack_type: Optional[str]
    attack_confidence: float
    infer_ms: float
    real_fake_logits: tuple[float, float]
    attack_type_logits: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_fake": self.p_fake,
            "p_real": self.p_real,
            "attack_type": self.attack_type,
            "attack_confidence": self.attack_confidence,
            "infer_ms": self.infer_ms,
            "real_fake_logits": list(self.real_fake_logits),
            "attack_type_logits": list(self.attack_type_logits),
        }


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax — same math as `tflite_io.dart`'s `_softmax`."""
    top = max(logits)
    exps = [math.exp(float(v) - top) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def validate_shapes(lfcc_sequence: Sequence[Sequence[float]], scalars: Sequence[float]) -> None:
    """Check a caller's window against the deployed graph's fixed contract.

    Raises ValueError with a precise, actionable message — the API turns this
    into a 422, so a wrong-shaped window is a clear client error rather than an
    opaque ONNX Runtime crash.
    """
    n_frames = len(lfcc_sequence)
    if n_frames != N_FRAMES:
        raise ValueError(
            f"lfcc_sequence must have {N_FRAMES} frames (3 s @ 16 kHz), got {n_frames}"
        )
    widths = {len(frame) for frame in lfcc_sequence}
    if widths != {N_LFCC}:
        raise ValueError(
            f"each lfcc_sequence frame must have {N_LFCC} coefficients, got widths {sorted(widths)}"
        )
    if len(scalars) != N_SCALARS:
        raise ValueError(
            f"scalars must have {N_SCALARS} values "
            "[pauseRatio, energyVar, zcrVar, jitter, shimmer, hnr_db], got {len(scalars)}"
        )


class OnnxModelBackend:
    """Loads the deployed ONNX detector once and scores windows with it."""

    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._session: Any = None
        self.load_error: Optional[str] = None
        self.load_ms: Optional[float] = None
        self.sha256_short: Optional[str] = None
        self.input_specs: list[tuple[str, Any, str]] = []
        self.output_specs: list[tuple[str, Any, str]] = []
        self.providers: list[str] = []
        self.infer_count = 0

    # ------------------------------------------------------------------ state
    @property
    def loaded(self) -> bool:
        return self._session is not None

    def identity(self) -> dict[str, Any]:
        """Everything needed to say *which artifact* produced a number."""
        info: dict[str, Any] = {
            "model_path": str(self.model_path),
            "expected_model_path": str(DEFAULT_MODEL_PATH),
            "exists": self.model_path.exists(),
            "loaded": self.loaded,
            "load_error": self.load_error,
            "load_ms": self.load_ms,
            "sha256_short": self.sha256_short,
            "input_names": [n for (n, _s, _t) in self.input_specs],
            "output_names": [n for (n, _s, _t) in self.output_specs],
            "inputs": [{"name": n, "shape": s, "type": t} for (n, s, t) in self.input_specs],
            "outputs": [{"name": n, "shape": s, "type": t} for (n, s, t) in self.output_specs],
            "providers": self.providers,
            "infer_count": self.infer_count,
            "contract": {
                "n_frames": N_FRAMES,
                "n_lfcc": N_LFCC,
                "n_scalars": N_SCALARS,
                "attack_labels": list(ATTACK_LABELS),
            },
        }
        if self.model_path.exists():
            stat = self.model_path.stat()
            info["size_bytes"] = stat.st_size
            info["mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        return info

    # ------------------------------------------------------------- lifecycle
    def load(self) -> bool:
        """Load the session. Never raises: on failure, records the reason and returns False."""
        started = time.perf_counter()
        self.load_error = None
        try:
            import onnxruntime as ort  # imported here so an ort-less env still imports this module
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._session = None
            self.load_error = f"onnxruntime import failed: {exc}"
            return False
        if not self.model_path.exists():
            self._session = None
            self.load_error = f"model asset not found: {self.model_path}"
            return False
        try:
            self.sha256_short = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self.input_specs = [(i.name, i.shape, i.type) for i in self._session.get_inputs()]
            self.output_specs = [(o.name, o.shape, o.type) for o in self._session.get_outputs()]
            self.providers = list(self._session.get_providers())
        except Exception as exc:
            self._session = None
            self.load_error = f"session creation failed: {exc}"
            return False
        finally:
            self.load_ms = round((time.perf_counter() - started) * 1000, 3)
        return True

    def reload(self) -> bool:
        """Re-read the asset from disk — lets a diagnostics run swap the ONNX in place."""
        self._session = None
        return self.load()

    # --------------------------------------------------------------- scoring
    def score_sequence(
        self, lfcc_sequence: Sequence[Sequence[float]], scalars: Sequence[float]
    ) -> ScoreResult:
        """Score one window from raw features. Raises ModelUnavailable if not loaded."""
        if self._session is None:
            raise ModelUnavailable(self.load_error or "model not loaded")
        validate_shapes(lfcc_sequence, scalars)
        import numpy as np  # numpy is already a hard dependency of both callers

        seq = np.asarray(lfcc_sequence, dtype=np.float32).reshape(1, N_FRAMES, N_LFCC)
        scal = np.asarray(scalars, dtype=np.float32).reshape(1, N_SCALARS)
        started = time.perf_counter()
        real_fake, attack = self._session.run(
            None, {"lfcc_sequence": seq, "scalars": scal}
        )
        infer_ms = round((time.perf_counter() - started) * 1000, 3)
        self.infer_count += 1

        rf = [float(v) for v in real_fake[0]]
        at = [float(v) for v in attack[0]]
        rf_probs = softmax(rf)
        at_probs = softmax(at)
        # Index 1 = synthetic (label convention in dataset.py: 0 real, 1 fake),
        # matching tflite_io.dart's _softmaxSecond.
        return ScoreResult(
            p_fake=rf_probs[1],
            p_real=rf_probs[0],
            attack_type=ATTACK_LABELS[1] if at_probs[1] > at_probs[0] else ATTACK_LABELS[0],
            attack_confidence=max(at_probs),
            infer_ms=infer_ms,
            real_fake_logits=(rf[0], rf[1]),
            attack_type_logits=(at[0], at[1]),
        )