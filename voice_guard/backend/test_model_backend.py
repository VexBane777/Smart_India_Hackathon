"""Tests for the real-model scoring path and the shared monitor surface.

Two things matter most here, and both are about not regressing what already
worked:

1. `POST /v1/analyze-chunk` with the *original* body shape (60-d `lfcc` and
   prosody only, no sequence) must behave exactly as it did before this
   change — that is the contract `lib/services/api_service.dart` and
   `backend/README.md` publish.
2. The new model path must either return a real ONNX score, or refuse
   loudly. It must never return a heuristic number under an `onnx` label.

Run:  cd voice_guard/backend && python -m pytest -q
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import model_backend
from main import API_KEY, app
from model_backend import (
    N_FRAMES,
    N_LFCC,
    N_SCALARS,
    ModelUnavailable,
    OnnxModelBackend,
    softmax,
    validate_shapes,
)

HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def client():
    """TestClient inside its context manager, so the app's lifespan actually runs.

    Without entering the context the startup hook never fires, the model is
    never loaded, and every model-path assertion would test the 503 branch
    instead of the real one.
    """
    with TestClient(app) as c:
        yield c


def window(fill: float = 0.0, scalars: tuple = (0.2, 0.01, 0.01, 0.0, 0.0, 0.0)) -> dict:
    """A well-formed model-path request body (184x60 + 6 scalars)."""
    return {
        "lfcc_sequence": np.full((N_FRAMES, N_LFCC), fill, dtype=float).tolist(),
        "scalars": list(scalars),
        "session_id": "test-model-path",
    }


# Measured 2026-09-15 against the deployed artifact: with an all-zero LFCC
# window, pauseRatio=1.0 scores 0.9978 while pauseRatio=0.2 scores 0.2946. Used
# as a deterministic *high* input below, with an assertion on the raw score so
# the test cannot quietly stop testing the alert policy if the model changes.
HIGH_SCALARS = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------- pure helpers
def test_softmax_is_normalized_and_stable():
    probs = softmax([1000.0, 1001.0])  # would overflow a naive exp()
    assert sum(probs) == pytest.approx(1.0)
    assert probs[1] > probs[0]
    # Index 1 is "fake" — the label convention dataset.py and tflite_io.dart share.
    assert softmax([0.0, 0.0])[1] == pytest.approx(0.5)


def test_validate_shapes_rejects_wrong_frame_count():
    with pytest.raises(ValueError, match="must have 184 frames"):
        validate_shapes(np.zeros((10, N_LFCC)).tolist(), [0.0] * N_SCALARS)


def test_validate_shapes_rejects_wrong_coefficient_count():
    with pytest.raises(ValueError, match="coefficients"):
        validate_shapes(np.zeros((N_FRAMES, 12)).tolist(), [0.0] * N_SCALARS)


def test_validate_shapes_rejects_wrong_scalar_count():
    with pytest.raises(ValueError, match="scalars must have 6"):
        validate_shapes(np.zeros((N_FRAMES, N_LFCC)).tolist(), [0.0, 0.0])


def test_scoring_without_a_session_raises_model_unavailable(tmp_path):
    backend = OnnxModelBackend(tmp_path / "nope.onnx")
    assert backend.load() is False
    assert backend.load_error and "not found" in backend.load_error
    with pytest.raises(ModelUnavailable):
        backend.score_sequence(np.zeros((N_FRAMES, N_LFCC)).tolist(), [0.0] * N_SCALARS)


# ----------------------------------------------------------- the real artifact
@pytest.fixture(scope="module")
def real_backend():
    backend = OnnxModelBackend()
    if not backend.load():
        pytest.skip(f"deployed ONNX unavailable in this environment: {backend.load_error}")
    return backend


def test_real_model_identity_matches_the_contract(real_backend):
    info = real_backend.identity()
    assert info["loaded"] is True
    assert info["input_names"] == ["lfcc_sequence", "scalars"]
    assert info["output_names"] == ["real_fake_logits", "attack_type_logits"]
    assert info["size_bytes"] > 0 and info["sha256_short"]
    assert info["contract"]["n_frames"] == N_FRAMES


def test_real_model_scores_a_window_in_range(real_backend):
    result = real_backend.score_sequence(
        np.zeros((N_FRAMES, N_LFCC)).tolist(), [0.2, 0.01, 0.01, 0.0, 0.0, 0.0]
    )
    assert 0.0 <= result.p_fake <= 1.0
    assert result.p_fake + result.p_real == pytest.approx(1.0)
    assert result.attack_type in model_backend.ATTACK_LABELS
    assert len(result.real_fake_logits) == 2 and len(result.attack_type_logits) == 2
    assert result.infer_ms >= 0.0


def test_real_model_is_deterministic(real_backend):
    args = (np.zeros((N_FRAMES, N_LFCC)).tolist(), [0.2, 0.01, 0.01, 0.0, 0.0, 0.0])
    first, second = real_backend.score_sequence(*args), real_backend.score_sequence(*args)
    assert first.p_fake == pytest.approx(second.p_fake, abs=1e-6)


# ------------------------------------------------------------ API: SDK contract
def test_legacy_body_shape_still_works_unchanged(client):
    """The pre-2026-09-15 request shape must keep working, labelled as heuristic."""
    r = client.post(
        "/v1/analyze-chunk",
        headers=HEADERS,
        json={"lfcc": [0.1] * 60, "prosody": {"pauseRatio": 0.2}, "session_id": "legacy"},
    )
    assert r.status_code == 200
    body = r.json()
    # The original five fields are all present with their original meanings.
    assert set(["riskScore", "verdict", "confidence", "latencyMs", "modelVersion"]).issubset(body)
    assert body["verdict"] in {"VERIFIED_HUMAN", "SUSPICIOUS", "AI_DETECTED"}
    # ...and the new fields say which scorer ran.
    assert body["modelBackend"] == "heuristic"
    assert body["attackType"] is None


def test_legacy_body_requires_an_api_key(client):
    r = client.post("/v1/analyze-chunk", json={"lfcc": [0.1] * 60})
    assert r.status_code == 401


def test_empty_body_is_a_clear_422(client):
    r = client.post("/v1/analyze-chunk", headers=HEADERS, json={})
    assert r.status_code == 422
    assert "lfcc" in r.json()["detail"]


# ------------------------------------------------------------ API: model path
def test_model_path_scores_with_onnx(client):
    r = client.post("/v1/analyze-chunk", headers=HEADERS, json=window())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modelBackend"] == "onnx"
    assert body["modelVersion"].startswith("voice_detector.onnx@")
    assert body["attackType"] in model_backend.ATTACK_LABELS
    assert body["inferMs"] is not None and body["inferMs"] >= 0
    assert 0.0 <= body["rawScore"] <= 1.0
    # First window of a fresh session: EMA seeds from raw, so they agree.
    assert body["riskScore"] == pytest.approx(body["rawScore"], abs=1e-4)


def test_model_path_rejects_a_half_supplied_window(client):
    r = client.post("/v1/analyze-chunk", headers=HEADERS,
                    json={"lfcc_sequence": np.zeros((N_FRAMES, N_LFCC)).tolist()})
    assert r.status_code == 422
    assert "both" in r.json()["detail"]


def test_model_path_rejects_wrong_shapes_with_422_not_500(client):
    bad = window()
    bad["lfcc_sequence"] = np.zeros((5, N_LFCC)).tolist()
    r = client.post("/v1/analyze-chunk", headers=HEADERS, json=bad)
    assert r.status_code == 422
    assert "184" in r.json()["detail"]


def test_two_high_windows_trigger_the_shared_alert_policy(client):
    """EMA + 2-consecutive rule is shared by both paths — one window is never an alert.

    Uses a window the deployed model scores high (all-zero LFCC with
    pauseRatio=1.0; measured 0.9978). The raw score is asserted first, so this
    test fails loudly rather than silently passing if the model or its
    threshold behaviour ever changes.
    """
    body = window(scalars=HIGH_SCALARS)
    body["session_id"] = "alert-policy"
    first = client.post("/v1/analyze-chunk", headers=HEADERS, json=body).json()
    assert first["rawScore"] > 0.6, "window is no longer a high-scoring input — test premise broke"
    assert first["state"] in {"normal", "warn"}  # one window is never enough
    second = client.post("/v1/analyze-chunk", headers=HEADERS, json=body).json()
    assert second["state"] == "alert"
    assert second["verdict"] == "AI_DETECTED"
    assert second["confidence"] == pytest.approx(abs(second["riskScore"] - 0.5) * 2, abs=1e-3)


def test_reset_clears_the_session_state(client):
    body = window()
    body["session_id"] = "reset-me"
    client.post("/v1/analyze-chunk", headers=HEADERS, json=body)
    assert client.post("/v1/reset/reset-me", headers=HEADERS).json() == {"reset": "reset-me"}
    after = client.post("/v1/analyze-chunk", headers=HEADERS, json=body).json()
    assert after["state"] != "alert"


# ----------------------------------------------------------------- the monitor
def test_model_info_reports_the_artifact(client):
    info = client.get("/v1/model-info").json()
    assert info["loaded"] is True
    assert info["sha256_short"]
    assert info["inputs"][0]["name"] == "lfcc_sequence"


def test_monitor_json_carries_the_last_scored_numbers(client):
    body = window()
    body["session_id"] = "monitor-read"
    scored = client.post("/v1/analyze-chunk", headers=HEADERS, json=body).json()
    snap = client.get("/monitor.json").json()
    assert snap["ok"] is True
    scoring = snap["state"]["scoring"]
    # The number an agent reads here is the number the caller got.
    assert scoring["raw_score"] == pytest.approx(scored["rawScore"], abs=1e-6)
    assert scoring["ema"] == pytest.approx(scored["riskScore"], abs=1e-6)
    assert scoring["backend"] == "onnx"
    assert snap["log"]["newest_seq"] is not None


def test_monitor_txt_is_one_greppable_line(client):
    client.post("/v1/analyze-chunk", headers=HEADERS, json=window())
    text = client.get("/monitor.txt").text
    assert text.startswith("Monitor: ")
    assert "\n" not in text.strip()
    assert "scoring.backend" in text


def test_ndjson_tail_is_incremental_and_gap_free(client):
    client.post("/v1/analyze-chunk", headers=HEADERS, json=window())
    first = client.get("/logs.ndjson").text.strip().splitlines()
    assert first, "expected at least one log line"
    seqs = [json.loads(line)["seq"] for line in first]
    assert seqs == sorted(seqs)
    cursor = seqs[-1]
    # Tailing from the newest seq yields nothing new...
    assert client.get(f"/logs.ndjson?since={cursor}").text.strip() == ""
    # ...and one more call yields exactly the new lines, no repeats.
    client.post("/v1/analyze-chunk", headers=HEADERS, json=window())
    tail = client.get(f"/logs.ndjson?since={cursor}").text.strip().splitlines()
    assert tail and all(json.loads(line)["seq"] > cursor for line in tail)


def test_ndjson_reports_a_cursor_that_fell_behind(client):
    r = client.get("/logs.ndjson?since=0")
    assert "X-Monitor-Oldest-Seq" in r.headers
    assert r.headers["X-Monitor-Missed"] == "0"  # from seq 0 nothing was missed


def test_alert_is_written_to_the_monitor(client):
    client.post("/v1/alert", headers=HEADERS,
                json={"callerId": "+91-90000", "riskScore": 0.83, "verdict": "AI_DETECTED"})
    snap = client.get("/monitor.json").json()
    assert snap["log"]["stage_counts"].get("alert") == 1


def test_control_clear_and_reload(client):
    assert client.post("/v1/control", headers=HEADERS, json={"action": "clear-monitor"}).json()["cleared"] is True
    reloaded = client.post("/v1/control", headers=HEADERS, json={"action": "reload-model"}).json()
    assert reloaded["loaded"] is True
    bad = client.post("/v1/control", headers=HEADERS, json={"action": "nonsense"})
    assert bad.status_code == 400


def test_root_advertises_the_monitor_surface(client):
    body = client.get("/").json()
    assert body["monitor"]["snapshot"] == "GET /monitor.json"
    assert body["monitor"]["tail"].startswith("GET /logs.ndjson")