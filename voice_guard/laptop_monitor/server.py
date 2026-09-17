"""
server.py — the laptop diagnostic console's HTTP surface.

Three jobs, and nothing else:

1. Serve the page (`static/index.html`) that a human reads.
2. Expose the *same* machine-readable monitor surface the backend does
   (`monitor.py`: `/monitor.json`, `/monitor.txt`, `/logs.ndjson`, `/events`,
   `/ws/monitor`), so any coding agent can watch the numbers without a browser
   and without screen-scraping.
3. Accept `/control`, so that watching can be interactive: start a file, pause,
   single-step one window, move the threshold, switch the backend comparison
   mode.

The page deliberately consumes the public endpoints rather than a private
channel — if the agent-readable surface were insufficient to drive the UI, it
would not really be agent-readable.

Binding: localhost only by default. `/control` can start a run over any path the
user can read and mutate live state, which is appropriate for a diagnostics
console on your own laptop and inappropriate on a network — so it is not bound
to one.

Run:
    cd voice_guard/laptop_monitor
    python -m uvicorn server:app --port 8010
    # page       -> http://127.0.0.1:8010
    # agent view -> curl http://127.0.0.1:8010/monitor.txt
"""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import pipeline as P  # also puts voice_guard/backend on sys.path (see its header)
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from monitor import MonitorLog, attach_monitor_routes  # needs pipeline's sys.path first
from model_backend import OnnxModelBackend
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
VOICE_GUARD = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

SUPPORTED_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
MAX_SOURCES = 200

# Where "Import audio" uploads land. A browser file picker never hands the page
# an absolute filesystem path (security), so the only way to score an arbitrary
# local file the user just chose is to actually upload its bytes here first —
# same 25 MB cap as the phone's own file-scan path (AudioProcessor, ~25 MB).
UPLOAD_DIR = Path(tempfile.gettempdir()) / "voiceguard_laptop_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Shown in the picker by default. Deliberately the repo's own diagnostic assets:
# the four model_training/test_assets clips have known, recorded scores (see
# voice_guard/state.md), which makes them useful references, and the vaani demo
# calls are 48 kHz — so they exercise the resample stage on purpose.
#
# UPLOAD_DIR comes first so a file just imported through the browser (or in a
# past session — it survives until the OS cleans %TEMP%) shows up in the
# picker instead of only being reachable via the path the upload response
# returned once, for this page load, and never again.
SOURCE_DIRS = [
    UPLOAD_DIR,
    VOICE_GUARD / "model_training" / "test_assets",
    VOICE_GUARD / "assets",
    ROOT / "vaani" / "assets" / "demo",
    ROOT / "vaani" / "assets" / "raw",
]

_monitor = MonitorLog(title="voiceguard-laptop-console", echo_stdout=True)
_model = OnnxModelBackend()
_backend = P.BackendClient(
    base_url=os.environ.get("VOICEGUARD_BACKEND_URL", "http://127.0.0.1:8001"),
    api_key=os.environ.get("VOICEGUARD_API_KEY", "vg_demo_key"),
)
_runner = P.PipelineRunner(_monitor, _model, _backend)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the deployed model and record exactly what happened, as the backend does."""
    ok = _model.load()
    info = _model.identity()
    if ok:
        _monitor.log("model_load", f"deployed ONNX loaded in {info['load_ms']} ms",
                     sha256_short=info["sha256_short"], size_bytes=info.get("size_bytes"),
                     mtime=info.get("mtime"), inputs=info["input_names"],
                     outputs=info["output_names"], providers=info["providers"])
    else:
        _monitor.log("model_load", "ONNX model NOT loaded — the console cannot score",
                     severity="error", reason=_model.load_error, path=str(_model.model_path))
    _monitor.set_state(
        model={"loaded": _model.loaded, "path": str(_model.model_path),
               "sha256_short": _model.sha256_short, "load_error": _model.load_error,
               "contract": info["contract"], "inputs": info["inputs"], "outputs": info["outputs"]},
        backend={"base_url": _backend.base_url, "mode": _backend.mode},
        console={"static_dir": str(STATIC_DIR), "source_dirs": [str(d) for d in SOURCE_DIRS]},
    )
    _runner.publish_state()  # initialise the state block the page reads on load
    yield
    _monitor.log("shutdown", "console stopping")


app = FastAPI(title="VoiceGuard Laptop Diagnostic Console", version="0.1.0", lifespan=lifespan)
attach_monitor_routes(app, _monitor, tag="Monitor")


@app.middleware("http")
async def no_cache(request, call_next):
    """This page and its assets change under active development on the same
    machine that's viewing them. Without an explicit Cache-Control, browsers
    apply heuristic freshness to static files (Last-Modified-based) and can
    silently keep serving an old index.html/app.js without even a conditional
    request — a plain refresh then shows stale UI. Never worth the tradeoff here.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


# A file started via the raw-path field is often outside every SOURCE_DIRS
# entry (e.g. straight from Downloads) — nothing copies it anywhere, so unlike
# an upload there is no on-disk trace for list_sources() to rediscover. Without
# this, "started once" and "in the picker" only agreed for as long as the
# browser tab stayed open: a reload lost it, even though it had appeared to
# get added.
MAX_RECENT_PATHS = 50
_recent_paths: list[str] = []


def _remember_path(path: Path) -> None:
    s = str(path)
    if s in _recent_paths:
        _recent_paths.remove(s)
    _recent_paths.insert(0, s)
    del _recent_paths[MAX_RECENT_PATHS:]


def list_sources() -> list[dict[str, Any]]:
    """Enumerate scoreable files, plus any directories named in the env var.

    `VOICEGUARD_MONITOR_SOURCES` (path-separator separated) lets a session point
    the console at a corpus directory without editing code. The list is capped
    and the cap is *reported*, so a truncated list never looks like a complete one.
    """
    dirs = list(SOURCE_DIRS)
    extra = os.environ.get("VOICEGUARD_MONITOR_SOURCES")
    if extra:
        dirs += [Path(p) for p in extra.split(os.pathsep) if p]
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    for s in _recent_paths:
        path = Path(s)
        if s in seen or not path.is_file():
            continue
        seen.add(s)
        found.append({"name": path.name, "path": s, "dir": "recent", "size_bytes": path.stat().st_size})

    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
                continue
            s = str(path)
            if s in seen:
                continue
            seen.add(s)
            found.append({
                "name": path.name,
                "path": s,
                "dir": directory.name,
                "size_bytes": path.stat().st_size,
            })
            if len(found) >= MAX_SOURCES:
                found.append({"name": f"(list truncated at {MAX_SOURCES})", "path": "",
                              "dir": str(directory), "truncated": True, "cap": MAX_SOURCES})
                return found
    return found


def status() -> dict[str, Any]:
    """Everything a control caller — or an agent — needs to confirm what just happened."""
    snapshot = _monitor.snapshot()
    return {
        "ok": True,
        "state": snapshot["state"],
        "latest": snapshot["latest"],
        "model": {k: v for k, v in _model.identity().items()
                  if k in ("loaded", "sha256_short", "load_ms", "load_error", "model_path")},
    }


@app.get("/api/health", tags=["Console"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "voiceguard-laptop-console",
        "model_loaded": _model.loaded,
        "model_sha256_short": _model.sha256_short,
        "backend_url": _backend.base_url,
        "backend_mode": _backend.mode,
        "runner_active": _runner.is_active,
    }


@app.post("/api/upload", tags=["Console"])
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a file picked in the browser and land it where `/control start` can reach it.

    Returns the saved path; the page follows up with the normal `start` action
    rather than this endpoint starting the run itself, so a failed/rejected
    start still shows up as its own logged `control` event.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=422, detail=(
            f"unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        ))
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=(
            f"file too large ({len(data)} bytes) — max {MAX_UPLOAD_BYTES} bytes"
        ))
    target = UPLOAD_DIR / f"{int(time.time() * 1000)}_{Path(file.filename).name}"
    target.write_bytes(data)
    _monitor.log("control", f"uploaded {file.filename!r} -> {target}",
                 path=str(target), size_bytes=len(data), original_name=file.filename)
    return {"ok": True, "path": str(target), "name": file.filename, "size_bytes": len(data)}


@app.get("/api/sources", tags=["Console"])
def sources() -> dict[str, Any]:
    items = list_sources()
    return {"count": len(items), "source_dirs": [str(d) for d in SOURCE_DIRS], "items": items}


@app.get("/api/model-info", tags=["Console"])
def model_info() -> dict[str, Any]:
    return _model.identity()


@app.get("/api/backend-info", tags=["Console"])
async def backend_info() -> dict[str, Any]:
    """Ask the integration API which artifact *it* is scoring with (None if it is down)."""
    info = await _backend.model_info()
    return {"base_url": _backend.base_url, "reachable": info is not None,
            "mode": _backend.mode, "model": info}


@app.get("/api/config", tags=["Console"])
def config() -> dict[str, Any]:
    return {
        "window_samples": _runner.config.window_samples,
        "window_s": _runner.config.window_samples / P.SR,
        "hop_samples": _runner.config.hop_samples,
        "speed": _runner.config.speed,
        "threshold": _runner.config.threshold,
        "loop": _runner.config.loop,
        "backend_enabled": _runner.config.backend_enabled,
        "backend_mode": _backend.mode,
        "default_hop_samples": P.DEFAULT_HOP_SAMPLES,
        "min_hop_samples": 1,
        "max_hop_samples": _runner.config.window_samples,
    }


class ControlIn(BaseModel):
    action: str = Field(..., description=(
        "start | stop | pause | resume | step | threshold | hop | speed | loop | "
        "backend | reload-model | clear-monitor"
    ))
    path: Optional[str] = Field(None, description="start: which file to stream")
    value: Optional[float] = Field(None, description="threshold / hop / speed value")
    enabled: Optional[bool] = Field(None, description="backend: enable the comparison")
    mode: Optional[str] = Field(None, description="backend: 'model' or 'heuristic'")


def _require(value: Optional[float], name: str) -> float:
    if value is None:
        raise HTTPException(status_code=422, detail=f"action needs `value` ({name})")
    return value


@app.post("/control", tags=["Console"])
async def control(body: ControlIn) -> dict[str, Any]:
    """The console's only mutating endpoint. Every action logs itself."""
    action = body.action
    try:
        if action == "start":
            if not body.path:
                raise HTTPException(status_code=422, detail="start needs `path`")
            raw_path = body.path.strip()
            # Windows Explorer's "Copy as path" wraps the result in double
            # quotes (e.g. `"C:\Users\me\Downloads\clip.wav"`); pasted
            # verbatim into the raw-path field that quoted string is not a
            # path that exists, so it 404ed even though the real file did.
            if len(raw_path) >= 2 and raw_path[0] == '"' and raw_path[-1] == '"':
                raw_path = raw_path[1:-1]
            target = Path(raw_path)
            if not target.is_file():
                raise HTTPException(status_code=404, detail=f"no such file: {target}")
            if target.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise HTTPException(
                    status_code=422,
                    detail=f"unsupported file type {target.suffix!r}; expected one of "
                           f"{sorted(SUPPORTED_SUFFIXES)}",
                )
            _monitor.log("control", f"start requested: {target}",
                         path=str(target), size_bytes=target.stat().st_size)
            _remember_path(target)
            _runner.start(target)
        elif action == "stop":
            _runner.stop()
        elif action == "pause":
            _runner.pause()
        elif action == "resume":
            _runner.resume()
        elif action == "step":
            _runner.step()
        elif action == "threshold":
            _runner.set_threshold(_require(body.value, "threshold"))
        elif action == "hop":
            _runner.set_hop(int(_require(body.value, "hop")))
        elif action == "speed":
            _runner.set_speed(_require(body.value, "speed"))
        elif action == "loop":
            _runner.set_loop(bool(body.enabled))
        elif action == "backend":
            _runner.set_backend(True if body.enabled is None else body.enabled, mode=body.mode)
        elif action == "reload-model":
            ok = _model.reload()
            info = _model.identity()
            _monitor.log("model_reload", f"model reload -> loaded={ok}",
                         severity="info" if ok else "error",
                         sha256_short=info["sha256_short"], load_ms=info["load_ms"],
                         reason=info["load_error"])
        elif action == "clear-monitor":
            _monitor.clear()
            _monitor.log("monitor_clear", "diagnostic log cleared via /control")
        else:
            raise HTTPException(status_code=400, detail=(
                f"unknown action {action!r}; known: start, stop, pause, resume, step, threshold, "
                "hop, speed, loop, backend, reload-model, clear-monitor"
            ))
    except ValueError as exc:  # the setters validate their own ranges
        raise HTTPException(status_code=422, detail=str(exc))
    return {**status(), "action": action}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Mounted last: StaticFiles at "/" would otherwise shadow the API routes above.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")