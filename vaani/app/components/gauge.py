"""
gauge.py — Live risk gauge component (Module C Task 4 Step 2).

Custom HTML/CSS gauge (plan option B) rendered via st.markdown — smoother and
cheaper than st.metric for the 0.5 s refresh cadence, and it shows the EMA
score, the alert state colour, and the honest "mock backend" banner.

All styling is inline; no external assets, so it works in airplane mode.
"""

from typing import Optional

import streamlit as st

STATE_COLORS = {
    "normal": "#22c55e",   # green
    "warn": "#f59e0b",     # amber
    "alert": "#ef4444",    # red
}
STATE_LABELS = {
    "normal": "NORMAL",
    "warn": "WATCH",
    "alert": "ALERT — 2+ consecutive windows above threshold",
}


def render_gauge(ema: Optional[float], state: str, raw: Optional[float] = None,
                 backend: str = "mock") -> None:
    """
    Draw the risk gauge for the current EMA score and alert state.

    `backend` is shown verbatim so the demo never implies real inference
    while Module B's model is pending ("honest numbers only").
    """
    score = 0.0 if ema is None else float(ema)
    color = STATE_COLORS.get(state, "#64748b")
    label = STATE_LABELS.get(state, state.upper())
    pct = int(round(score * 100))

    raw_html = ""
    if raw is not None:
        raw_html = (
            f'<span style="font-size:0.9rem;color:#94a3b8;">'
            f'raw window: {float(raw):.2f}</span>'
        )

    html = f"""
    <div style="border:2px solid {color}; border-radius:16px; padding:16px 20px;
                max-width:520px; background:#0f172a; font-family:sans-serif;">
      <div style="display:flex; justify-content:space-between; align-items:baseline;">
        <span style="font-size:1.1rem; color:#e2e8f0; font-weight:600;">
          Synthetic-voice risk</span>
        <span style="font-size:0.8rem; padding:2px 10px; border-radius:999px;
              border:1px solid #475569; color:#94a3b8;">backend: {backend}</span>
      </div>
      <div style="font-size:3.2rem; font-weight:800; color:{color}; line-height:1.1;">
        {pct}%
      </div>
      <div style="background:#1e293b; border-radius:8px; height:14px; overflow:hidden;">
        <div style="background:{color}; width:{pct}%; height:100%;"></div>
      </div>
      <div style="margin-top:8px; font-size:0.95rem; color:{color}; font-weight:600;">
        {label}
      </div>
      <div style="margin-top:2px;">{raw_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
