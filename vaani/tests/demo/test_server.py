"""
tests for app/server.py — WebSocket streaming contract (Task 4 Step 1).

Uses FastAPI's TestClient WebSocket transport, with `pace=0` so the whole
90 s call streams instantly.
"""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.server import app, available_calls


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestEndpoints:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["backend"] == "mock"

    def test_calls_listed(self, client):
        r = client.get("/api/calls")
        assert r.status_code == 200
        calls = r.json()
        assert "call_A" in calls
        assert "call_N" in calls
        assert "call_B" in calls


class TestWebSocketStream:
    def test_stream_shape(self, client):
        with client.websocket_connect("/ws/stream/call_A?pace=0") as ws:
            meta = ws.receive_json()
            assert meta["type"] == "meta"
            assert meta["sr"] == 16000
            assert meta["backend"] == "mock"
            assert abs(meta["duration_s"] - 90.0) < 0.5

            n_chunks = 0
            last = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "end":
                    break
                assert msg["type"] == "chunk"
                # audio decodes to valid PCM
                pcm = np.frombuffer(base64.b64decode(msg["audio_b64"]), dtype=np.int16)
                assert len(pcm) > 0
                assert 0.0 <= msg["raw_score"] <= 1.0
                assert msg["state"] in ("normal", "warn", "alert")
                last = msg
                n_chunks += 1
            # 90 s at 0.5 s hop -> ~180 chunks
            assert 170 <= n_chunks <= 190
            assert abs(last["t"] - 89.5) < 0.6

    def test_scores_rise_after_clone_entry(self, client):
        with client.websocket_connect("/ws/stream/call_A?pace=0") as ws:
            ws.receive_json()  # meta
            early, late = [], []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "end":
                    break
                if msg["type"] != "chunk":
                    continue
                (early if msg["t"] < 20.0 else late).append(msg["raw_score"])
        assert np.mean(late) > np.mean(early) + 0.3

    def test_call_N_control_never_alerts(self, client):
        # Regression test (2026-09-06): server used to hardcode MockBackend()
        # with its default clone_entry_s=22.0 regardless of call_key, so the
        # all-real-voice control (call_N) would falsely spike after 22s too.
        with client.websocket_connect("/ws/stream/call_N?pace=0") as ws:
            ws.receive_json()  # meta
            while True:
                msg = ws.receive_json()
                if msg["type"] == "end":
                    break
                assert msg["type"] == "chunk"
                assert msg["state"] != "alert"

    def test_unknown_call_errors(self, client):
        with client.websocket_connect("/ws/stream/nope?pace=0") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"


class TestChanneledDemoAssets:
    """Module C Task 2 Steps 2-3: engine behavior on the WhatsApp-channeled audio."""

    def test_call_A_channeled_alerts(self, client):
        with client.websocket_connect("/ws/stream/call_A?pace=0") as ws:
            ws.receive_json()  # meta
            saw_alert = False
            while True:
                msg = ws.receive_json()
                if msg["type"] == "end":
                    break
                if msg["state"] == "alert":
                    saw_alert = True
            assert saw_alert, "channeled call_A must still alert"

    def test_call_N_channeled_never_alerts(self, client):
        with client.websocket_connect("/ws/stream/call_N?pace=0") as ws:
            ws.receive_json()  # meta
            while True:
                msg = ws.receive_json()
                if msg["type"] == "end":
                    break
                assert msg["state"] != "alert"


def test_available_calls_no_crash():
    calls = available_calls()
    assert isinstance(calls, dict)
