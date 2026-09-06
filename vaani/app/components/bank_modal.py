"""
bank_modal.py — Mock bank transfer -> HOLD -> OTP workflow (Module C Task 5).

Simulated end-to-end: "Initiate Transfer" -> if the engine's current alert
state is "alert", the transfer is intercepted ("TRANSACTION HELD") and a
simulated OTP step-up is required before release. Every transition is
appended to a hash-chained AuditLog (app/components/audit_log.py) — master
plan §1 item 5.

The OTP itself is a UI simulation only: no real SMS is sent (disclosed in
the UI copy — real bulk SMS needs TRAI registration, out of scope at ₹0
budget per master plan §2). The code accepted is a fixed demo value, shown
on-screen, never actually delivered anywhere — this is intentional and
must stay disclosed, not hidden as if it were real.
"""

from typing import Optional

import streamlit as st

from app.components.audit_log import AuditLog

DEMO_OTP_CODE = "123456"  # simulated; shown on-screen, not sent anywhere


def decide_transfer_outcome(state: str) -> str:
    """
    Pure decision function (master plan §1 item 4: "holds transactions and
    triggers step-up verification; never silently blocks").

    Returns "held" if the current engine alert state is "alert", else
    "approved". Never returns a third value — the workflow either lets the
    transfer through or explicitly holds it for step-up, no silent drop.
    """
    return "held" if state == "alert" else "approved"


def render_bank_panel(state: str, ema: Optional[float], audit_log: AuditLog) -> None:
    """
    Render the transfer trigger + HOLD/OTP workflow. Call once per rerun;
    all workflow state (stage, entered OTP) lives in st.session_state so it
    survives the 0.5 s fragment refresh.
    """
    ss = st.session_state
    ss.setdefault("bank_stage", "idle")  # idle -> held -> released
    ss.setdefault("bank_otp_input", "")
    ss.setdefault("bank_release_reason", "approved")  # "approved" | "otp_verified"

    st.subheader("Mock bank: vendor transfer")
    st.caption(
        "Simulation only. OTP is never actually sent — shown on-screen for "
        "the demo (real bulk SMS needs TRAI registration, out of scope)."
    )

    if ss.bank_stage == "idle":
        if st.button("Initiate Transfer (₹40,00,000 → new remittance account)"):
            outcome = decide_transfer_outcome(state)
            if outcome == "held":
                ss.bank_stage = "held"
                audit_log.append(
                    "transfer_held",
                    {"amount_inr": 4_000_000, "ema_at_hold": ema, "state": state},
                )
            else:
                ss.bank_stage = "released"
                ss.bank_release_reason = "approved"
                audit_log.append(
                    "transfer_approved",
                    {"amount_inr": 4_000_000, "ema_at_approval": ema, "state": state},
                )
            st.rerun()

    elif ss.bank_stage == "held":
        st.error("🛑 TRANSACTION HELD — synthetic-voice risk detected on this call.")
        st.text_input("Enter 6-digit code sent to your device", key="bank_otp_input", max_chars=6)
        st.caption(f"(Demo code: {DEMO_OTP_CODE})")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Verify code"):
                if ss.bank_otp_input == DEMO_OTP_CODE:
                    ss.bank_stage = "released"
                    ss.bank_release_reason = "otp_verified"
                    audit_log.append("otp_verified", {"amount_inr": 4_000_000})
                    st.rerun()
                else:
                    audit_log.append("otp_failed", {"attempted": ss.bank_otp_input})
                    st.warning("Incorrect code — transfer remains held.")
        with col2:
            if st.button("Cancel transfer"):
                ss.bank_stage = "idle"
                audit_log.append("transfer_cancelled", {"amount_inr": 4_000_000})
                st.rerun()

    elif ss.bank_stage == "released":
        if ss.bank_release_reason == "otp_verified":
            st.success("✅ Transfer released after step-up verification.")
        else:
            st.success("✅ Transfer approved — no synthetic-voice risk detected.")
        if st.button("Reset demo"):
            ss.bank_stage = "idle"
            ss.bank_otp_input = ""
            st.rerun()

    with st.expander(f"Audit log ({len(audit_log)} entries, hash-chained)"):
        ok, bad_seq = audit_log.verify_chain()
        if ok:
            st.caption("✅ Chain verified — no tampering detected.")
        else:
            st.caption(f"⚠️ Chain broken at entry {bad_seq} — tampering detected.")
        st.json(audit_log.entries())
