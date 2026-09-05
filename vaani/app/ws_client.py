"""
ws_client.py — WebSocket client for the Streamlit dashboard (Task 4).

Consumes the FastAPI `/ws/stream/{call_key}` JSON stream and yields parsed
messages. Uses the `websockets` sync client so the Streamlit fragment can
poll new chunks between reruns without an event loop.
"""

import json
import queue
from typing import Any, Dict, List, Optional

import websockets


def build_uri(host: str, port: int, call_key: str, pace: float = 1.0) -> str:
    return f"ws://{host}:{port}/ws/stream/{call_key}?pace={pace}"


class StreamClient:
    """
    Collects chunk messages from one WS session on a background thread.

    Designed for the Streamlit fragment pattern: `connect()` once, then
    `poll()` inside st.fragment every 0.5 s and render the accumulated
    history. The server's default realtime pacing would lag a poll-poll UI,
    so the client never reads audio faster than it is produced — but it also
    supports `pace=0` (fast mode) for tests.
    """

    def __init__(self, uri: str):
        self.uri = uri
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.meta: Optional[dict] = None
        self.history: List[Dict[str, Any]] = []  # chunk dicts (t, ema, state...)
        self.done = False
        self.error: Optional[str] = None
        self._thread = None

    def connect(self) -> None:
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            with websockets.connect(self.uri) as ws:
                for raw in ws:
                    msg = json.loads(raw)
                    self._absorb(msg)
                    if msg.get("type") in ("end", "error"):
                        break
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI
            self.error = str(exc)
        finally:
            self.done = True
            self._queue.put({"type": "__closed__"})

    def _absorb(self, msg: dict) -> None:
        if msg.get("type") == "meta":
            self.meta = msg
        elif msg.get("type") == "chunk":
            self.history.append(msg)
        elif msg.get("type") == "end":
            self.done = True
        elif msg.get("type") == "error":
            self.error = msg.get("detail", "unknown error")
            self.done = True
        self._queue.put(msg)

    def poll(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Return the next message, or None if none arrived within `timeout`."""
        try:
            msg = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if msg.get("type") == "__closed__":
            return None
        return msg

    def drain(self) -> List[Dict[str, Any]]:
        """Return all messages currently queued (non-blocking)."""
        out = []
        while True:
            msg = self.poll(timeout=0)
            if msg is None:
                return out
            out.append(msg)

