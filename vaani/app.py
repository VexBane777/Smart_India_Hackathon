"""
app.py — VAANI live demo dashboard (Module C Tasks 4-5).

Streamlit frontend consuming the FastAPI WebSocket stream (app/server.py):
live risk gauge, spectrogram, EMA risk curve, and the mock-bank
transfer -> HOLD -> OTP workflow with a hash-chained audit log
(app/components/bank_modal.py, app/components/audit_log.py).

Currently driven by the MOCK score backend (app/engine_mock.py) — clearly
labeled as such. Once Module B Task 2 delivers a loadable checkpoint, swap
MockBackend in server.py; this file is unchanged.

Run:
    python -m uvicorn app.server:app --port 8000     (terminal 1)
    streamlit run app.py                              (terminal 2, from vaani/)

The risk curve, gauge, and spectrogram update inside st.fragment with a 0.5 s
refresh cadence, matching the server's chunk cadence.
"""

from collections import deque

import numpy as np
import streamlit as st

from app.components.audit_log import AuditLog
from app.components.bank_modal import render_bank_panel
from app.components.gauge import render_gauge
from app.components.spectrogram import SR, render_spectrogram
from app.ws_client import StreamClient, build_uri

HOST = "127.0.0.1"
PORT = 8000

st.set_page_config(page_title="VAANI — live call monitor", layout="wide")

# --- session state ---------------------------------------------------------
ss = st.session_state
ss.setdefault("client", None)
ss.setdefault("current_call", None)
ss.setdefault("latest_chunk", None)               # last chunk dict (holds audio_b64)
ss.setdefault("score_history", deque(maxlen=600))  # ~5 min at 0.5s hop
ss.setdefault("audit_log", AuditLog())

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("VAANI")
    st.caption("Real-time voice-clone detection — demo skeleton")
    st.markdown(
        "**Backend: MOCK** — Module B model pending. Scores are synthetic "
        "placeholders implementing the real decision logic (EMA + 2-window "
        "rule). Not for pitch use until swapped."
    )
    import json
    from pathlib import Path

    manifest_path = Path(__file__).resolve().parent / "assets" / "manifest.json"
    call_keys = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        call_keys = sorted(manifest.get("assets", {}).keys())
    call_key = st.selectbox("Demo call", call_keys or ["(no assets — run assets/scripts/build_demo_placeholders.py)"])
    pace = st.select_slider("Stream pacing", options=[0.0, 1.0, 2.0, 4.0], value=1.0,
                            help="1.0 = realtime; 0 = as fast as possible (dev)")

    if st.button("Start stream", disabled=call_key is None or "call" not in call_key):
        if ss.client is not None:
            ss.client.done = True
        ss.client = StreamClient(build_uri(HOST, PORT, call_key, pace))
        ss.client.connect()
        ss.current_call = call_key
        ss.score_history.clear()
        ss.latest_chunk = None
    if st.button("Stop") and ss.client is not None:
        ss.client.done = True
        ss.client = None

# --- main panel ------------------------------------------------------------
st.subheader("Live call monitor")

col_gauge, col_spec = st.columns([1, 1])

# Poll the client from the fragment loop
client: StreamClient = ss.client
if client is not None:
    for msg in client.drain():
        if msg.get("type") == "chunk":
            ss.latest_chunk = msg
            ss.score_history.append((msg["t"], msg["raw_score"], msg["ema"], msg["state"]))

latest = ss.latest_chunk
ema = latest.get("ema") if latest else None
raw = latest.get("raw_score") if latest else None
state = latest.get("state", "normal") if latest else "normal"

with col_gauge:
    render_gauge(ema, state, raw, backend="mock")

with col_spec:
    if latest is not None and latest.get("audio_b64"):
        import base64

        pcm = np.frombuffer(base64.b64decode(latest["audio_b64"]), dtype=np.int16)
        render_spectrogram(pcm.astype(np.float64) / 32768.0, SR)
    else:
        st.info("No audio yet — start the FastAPI server (`python -m uvicorn app.server:app --port 8000`) and press Start stream.")

# --- risk curve (Task 4 Step 4: smoothed EMA over call duration) -----------
st.subheader("Risk curve (EMA-smoothed, 0.5 s cadence)")
if ss.score_history:
    ts = [t for (t, _r, _e, _s) in ss.score_history]
    emas = [e for (_t, _r, e, _s) in ss.score_history]
    raws = [r for (_t, r, _e, _s) in ss.score_history]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=100)
    ax.plot(ts, raws, color="#94a3b8", linewidth=1, label="raw window score")
    ax.plot(ts, emas, color="#ef4444", linewidth=2, label="EMA")
    ax.axhline(0.6, color="#f59e0b", linestyle="--", linewidth=1, label="threshold")
    # shade alert regions (2+ consecutive above threshold)
    states = [s for (_t, _r, _e, s) in ss.score_history]
    in_alert = False
    for i, s in enumerate(states):
        if s == "alert" and not in_alert:
            start_t = ts[i]
            in_alert = True
        elif s != "alert" and in_alert:
            ax.axvspan(start_t, ts[i], color="#ef4444", alpha=0.15)
            in_alert = False
    if in_alert:
        ax.axvspan(start_t, ts[-1], color="#ef4444", alpha=0.15)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("call time (s)")
    ax.set_ylabel("risk score")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No score history yet — start the stream.")

st.divider()
render_bank_panel(state, ema, ss.audit_log)

st.caption(
    "VAANI demo skeleton · mock backend · scores are placeholders, decision "
    "logic (EMA + 2-consecutive-window rule) is final-form per master plan §6."
)
