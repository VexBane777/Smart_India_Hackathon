"""
tests/demo/test_offline.py — Module C Task 6: airplane-mode / zero-network
verification (automated form).

Global Constraint (module C plan): "100% Offline (Airplane Mode). Zero
external API calls." This test enforces that guarantee at the socket layer:
it patches `socket.socket.connect`/`connect_ex` to raise for any connection
attempt to a *non-loopback* address, then runs the full demo call flow
(stream call_A and call_N end-to-end via FastAPI's TestClient, plus the
bank-sim workflow) and asserts it completes with zero exceptions and zero
non-loopback connection attempts. Loopback (127.0.0.1/::1) is allowed
through since that's how the local demo (browser <-> localhost:8000) and
this test's own WebSocket transport work — neither is an external call.

This automates plan Steps 2-4 (kill network, full run, verify no
"connecting/timeout" errors) as a CI-runnable check. It does NOT replace a
real physical rehearsal (Step 2's literal "disable WiFi/Ethernet" on the
actual demo machine, with a human watching the actual UI) — that manual
pass is still required before the real pitch, since a monkeypatched socket
can't catch every path a library might use to reach the network (e.g. a
vendored DNS resolver). This test is the CPU-only, automatable slice of
Task 6; the physical rehearsal is out of scope for an autonomous session.
"""

import socket

import pytest
from fastapi.testclient import TestClient

from app.server import app


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def no_network(monkeypatch):
    """
    Raise on any attempt to open a socket connection to a non-loopback
    address. Loopback (127.0.0.1/::1) is allowed through: it's how
    TestClient's own WebSocket transport talks to the in-process ASGI app
    (a real local socket under the hood, not "the internet"), and how a
    real rehearsal's browser<->localhost:8000 traffic works too — neither
    is an "external API call" per the Global Constraints.
    """
    attempts = []
    _real_connect = socket.socket.connect
    _real_connect_ex = socket.socket.connect_ex

    def _check(address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            attempts.append(address)
            raise OSError(f"non-loopback network access blocked in airplane-mode test: {address!r}")

    def _guarded_connect(self, address, *a, **kw):
        _check(address)
        return _real_connect(self, address, *a, **kw)

    def _guarded_connect_ex(self, address, *a, **kw):
        _check(address)
        return _real_connect_ex(self, address, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    return attempts


class TestAirplaneModeStream:
    def test_call_A_streams_offline(self, no_network):
        # FastAPI's TestClient runs the ASGI app in-process (ASGITransport),
        # not over a real socket — ports open here are the local WS test
        # transport, and this asserts nothing tries to open one *outbound*.
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stream/call_A?pace=0") as ws:
                msgs = []
                while True:
                    msg = ws.receive_json()
                    msgs.append(msg)
                    if msg["type"] == "end":
                        break
        assert any(m["type"] == "chunk" for m in msgs)
        assert not no_network, f"unexpected outbound connection attempts: {no_network}"

    def test_call_N_streams_offline(self, no_network):
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stream/call_N?pace=0") as ws:
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "end":
                        break
        assert not no_network


class TestAirplaneModeBankSim:
    def test_bank_workflow_offline(self, no_network):
        from streamlit.testing.v1 import AppTest

        def script():
            import streamlit as st

            from app.components.audit_log import AuditLog
            from app.components.bank_modal import render_bank_panel

            st.session_state.setdefault("audit_log", AuditLog())
            render_bank_panel("alert", ema=0.9, audit_log=st.session_state.audit_log)

        at = AppTest.from_function(script)
        at.run()
        at.button[0].click().run()
        assert "TRANSACTION HELD" in at.error[0].value
        assert not no_network
