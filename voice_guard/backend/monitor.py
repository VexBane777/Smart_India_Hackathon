"""
monitor.py — the shared, agent-readable monitor surface.

Why this exists
---------------
Both the judge-facing integration API (`main.py`) and the laptop diagnostic
console (`voice_guard/laptop_monitor/`) have to be readable by *any* coding
agent, not just by a human looking at a browser: an agent needs to watch the
numbers the pipeline is producing and correlate each one with the stage that
produced it. Rather than invent two formats, both processes mount this one
module — the same event record, the same bounded ring buffer, the same four
serializations:

    GET /monitor.json          full snapshot: latest event + live state + counts
    GET /monitor.txt           the same numbers as one `Monitor:`-prefixed line
                               (the prefix `call_screen.dart` already uses)
    GET /logs.ndjson?since=N   append-only log with a monotonic cursor, so an
                               agent can tail it with no duplicates and no gaps
    GET /events                text/event-stream fan-out
    WS  /ws/monitor            the same fan-out for the browser UI

`/monitor.json` and `/monitor.txt` are derived from the *same* flattened
mapping, so an agent parses one schema two ways: structured (JSON) or as a
single grep-able line (text).

Concurrency note
----------------
`log()` is called from the console's async pipeline task *and* from
`main.py`'s request handlers — which FastAPI runs in a worker thread, because
those endpoints are sync `def`s. So the buffer, counters and state are guarded
by a lock, and fan-out to subscribers goes through
`loop.call_soon_threadsafe`. Subscriber queues are bounded and drop their
*oldest* line when full: a slow or dead reader can never stall, block or OOM
the pipeline it is monitoring.
"""
import asyncio
import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

MAX_LOG_LINES = 5000
MAX_SUBSCRIBER_BACKLOG = 1000
SEVERITIES = ("debug", "info", "warn", "error")

TEXT_PREFIX = "Monitor:"


def iso_now() -> str:
    """Wall-clock UTC ISO-8601 with milliseconds — the timestamp readers sort by."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Flatten nested mappings into `a.b.c` keys so text and JSON agree field-for-field."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = value
    else:
        out[prefix] = value


@dataclass
class MonitorEvent:
    """One diagnostic line: what happened, at which stream time, with which numbers."""

    seq: int
    ts: str
    mono_ms: float
    stage: str
    severity: str
    message: str
    t: Optional[float] = None          # stream/call time in seconds, when known
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seq": self.seq,
            "ts": self.ts,
            "mono_ms": self.mono_ms,
            "stage": self.stage,
            "severity": self.severity,
            "message": self.message,
        }
        if self.t is not None:
            d["t"] = self.t
        if self.data:
            d["data"] = self.data
        return d


def _offer(queue: "asyncio.Queue[MonitorEvent]", event: MonitorEvent) -> None:
    """Enqueue without ever blocking; drop the oldest line if the reader is behind.

    Runs on the event loop (see the module docstring), so it is the only place
    a subscriber queue is touched.
    """
    while True:
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            try:
                queue.get_nowait()  # discard the stalest line, keep the newest
            except asyncio.QueueEmpty:  # pragma: no cover - lost the race only
                return


class MonitorLog:
    """Bounded, thread-safe event log with a live state block and fan-out."""

    def __init__(self, title: str, maxlen: int = MAX_LOG_LINES, echo_stdout: bool = True):
        self.title = title
        self.maxlen = maxlen
        self.echo_stdout = echo_stdout
        self._lock = threading.Lock()
        self._events: deque[MonitorEvent] = deque(maxlen=maxlen)
        self._next_seq = 1
        self._state: dict[str, Any] = {}
        self._stage_counts: Counter[str] = Counter()
        self._sev_counts: Counter[str] = Counter()
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[MonitorEvent]]] = []
        self._t0_mono = time.monotonic()

    # ---------------------------------------------------------------- writing
    def log(
        self,
        stage: str,
        message: str,
        *,
        severity: str = "info",
        t: Optional[float] = None,
        **data: Any,
    ) -> MonitorEvent:
        """Record one diagnostic event and fan it out to every subscriber."""
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
        with self._lock:
            event = MonitorEvent(
                seq=self._next_seq,
                ts=iso_now(),
                mono_ms=round((time.monotonic() - self._t0_mono) * 1000, 3),
                stage=stage,
                severity=severity,
                message=message,
                t=None if t is None else round(float(t), 3),
                data=data,
            )
            self._next_seq += 1
            self._events.append(event)
            self._stage_counts[stage] += 1
            self._sev_counts[severity] += 1
            subs = list(self._subscribers)

        if self.echo_stdout:
            # Same line the phone prints (`Monitor: raw=… ema=…` in
            # call_screen.dart), so a terminal-watching agent sees the pipeline
            # without opening the page.
            print(self.format_line(event), flush=True)
        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(_offer, queue, event)
            except RuntimeError:  # loop closed (client went away) — drop silently
                self._drop_subscriber(loop, queue)
        return event

    def set_state(self, **kv: Any) -> None:
        """Merge live values into the `/monitor.json` `state` block."""
        with self._lock:
            self._state.update(kv)

    def clear(self) -> None:
        """Drop the log and counters (keeps the live state — it is still current)."""
        with self._lock:
            self._events.clear()
            self._stage_counts.clear()
            self._sev_counts.clear()

    # ---------------------------------------------------------------- reading
    @property
    def latest(self) -> Optional[MonitorEvent]:
        with self._lock:
            return self._events[-1] if self._events else None

    @property
    def oldest_seq(self) -> Optional[int]:
        with self._lock:
            return self._events[0].seq if self._events else None

    def since(self, seq: int) -> list[MonitorEvent]:
        """Events with `seq` strictly greater than `seq` — the incremental-tail call."""
        with self._lock:
            return [e for e in self._events if e.seq > seq]

    def snapshot(self, **extra: Any) -> dict[str, Any]:
        """Everything a reader needs in one object: identity, latest, live state, counts."""
        with self._lock:
            events = list(self._events)
            state = dict(self._state)
            stage_counts = dict(self._stage_counts)
            sev_counts = dict(self._sev_counts)
            oldest = events[0].seq if events else None
            newest = events[-1].seq if events else None
        snap: dict[str, Any] = {
            "ok": True,
            "title": self.title,
            "generated_at": iso_now(),
            "log": {
                "lines": len(events),
                "oldest_seq": oldest,
                "newest_seq": newest,
                "max_lines": self.maxlen,
                "stage_counts": stage_counts,
                "severity_counts": sev_counts,
            },
            "state": state,
            "latest": events[-1].to_dict() if events else None,
        }
        snap.update(extra)
        return snap

    @staticmethod
    def format_line(event: MonitorEvent) -> str:
        """One `k=v` line for `/monitor.txt` and stdout — stable key order."""
        parts = [
            f"seq={event.seq}",
            f"ts={event.ts}",
            f"sev={event.severity}",
            f"stage={event.stage}",
        ]
        if event.t is not None:
            parts.append(f"t={event.t:.3f}")
        parts.append(f"msg={json.dumps(event.message)}")
        flat: dict[str, Any] = {}
        _flatten("", event.data, flat)
        for k in sorted(flat):
            parts.append(f"{k}={json.dumps(flat[k], default=str)}")
        return f"{TEXT_PREFIX} " + " ".join(parts)

    def live_text(self, **extra: Any) -> str:
        """The current state as one `k=v` line — the numbers, without one event's noise."""
        flat: dict[str, Any] = {}
        _flatten("", self.snapshot(**extra), flat)
        return f"{TEXT_PREFIX} " + " ".join(
            f"{k}={json.dumps(flat[k], default=str)}" for k in sorted(flat)
        )

    def ndjson(self, since: int = 0) -> str:
        """The incremental tail, one JSON object per line."""
        return "".join(json.dumps(e.to_dict()) + "\n" for e in self.since(since))

    # ---------------------------------------------------------------- fan-out
    def subscribe(self, replay: int = 0) -> "asyncio.Queue[MonitorEvent]":
        """Create a subscriber queue; `replay` pre-fills it with the last N lines."""
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[MonitorEvent]" = asyncio.Queue(maxsize=MAX_SUBSCRIBER_BACKLOG)
        for event in self.since(max(0, (self.latest.seq if self.latest else 0) - replay)):
            queue.put_nowait(event)
        with self._lock:
            self._subscribers.append((loop, queue))
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[MonitorEvent]") -> None:
        with self._lock:
            self._subscribers = [(l, q) for (l, q) in self._subscribers if q is not queue]

    def _drop_subscriber(
        self, loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[MonitorEvent]"
    ) -> None:
        with self._lock:
            self._subscribers = [
                (l, q) for (l, q) in self._subscribers if not (l is loop and q is queue)
            ]

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


def attach_monitor_routes(app: Any, log: MonitorLog, *, tag: str = "Monitor") -> None:
    """Mount the five read-only monitor routes on a FastAPI app.

    Imported lazily so `monitor.py`'s pure logic (ring buffer, serializers) can
    be used and tested without FastAPI in the picture.

    Every route is read-only on purpose: an agent must be able to observe the
    pipeline without being able to perturb it. The one exception is each app's
    own `/control` (documented in that app), which is explicit and auditable
    because it logs its own event.
    """
    import asyncio
    import json as _json

    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.responses import PlainTextResponse, StreamingResponse

    @app.get("/monitor.json", tags=[tag], summary="Full snapshot: latest event, live state, counts")
    async def monitor_json() -> dict[str, Any]:
        return log.snapshot()

    @app.get(f"/monitor.txt", tags=[tag], response_class=PlainTextResponse,
             summary="The same numbers as one Monitor:-prefixed line")
    async def monitor_txt() -> PlainTextResponse:
        return PlainTextResponse(log.live_text())

    @app.get("/logs.ndjson", tags=[tag], response_class=PlainTextResponse,
             summary="Append-only log; pass ?since=<seq> to tail without gaps or duplicates")
    async def logs_ndjson(since: int = 0) -> PlainTextResponse:
        headers: dict[str, str] = {}
        oldest = log.oldest_seq
        if oldest is not None:
            headers["X-Monitor-Oldest-Seq"] = str(oldest)
            # Honour a cursor that fell behind the ring buffer by telling the
            # reader how many lines it lost, instead of silently pretending the
            # tail is complete. Headers are set on the returned response
            # directly: a `Response`-typed parameter with a default is not
            # injected by FastAPI, which silently dropped these.
            headers["X-Monitor-Missed"] = str(max(0, oldest - since - 1))
        return PlainTextResponse(log.ndjson(since), headers=headers)

    @app.get("/events", tags=[tag], summary="Server-sent event stream of every log line")
    async def events() -> StreamingResponse:
        queue = log.subscribe(replay=0)
        heartbeat_s = 15.0

        async def stream():
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"  # keeps proxies/clients from timing out
                        continue
                    yield f"data: {_json.dumps(event.to_dict())}\n\n"
            finally:
                log.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.websocket("/ws/monitor")
    async def ws_monitor(ws: WebSocket, replay: int = 200) -> None:
        await ws.accept()
        queue = log.subscribe(replay=replay)
        try:
            while True:
                event = await queue.get()
                await ws.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            log.unsubscribe(queue)