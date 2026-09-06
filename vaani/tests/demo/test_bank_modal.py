"""tests for app/components/bank_modal.py — Module C Task 5 workflow."""

from streamlit.testing.v1 import AppTest

from app.components.bank_modal import DEMO_OTP_CODE, decide_transfer_outcome


def _panel_script(state: str):
    import streamlit as st

    from app.components.audit_log import AuditLog
    from app.components.bank_modal import render_bank_panel

    st.session_state.setdefault("audit_log", AuditLog())
    render_bank_panel(state, ema=0.9 if state == "alert" else 0.1, audit_log=st.session_state.audit_log)


class TestBankPanelWorkflow:
    def test_alert_call_holds_then_releases_on_correct_otp(self):
        at = AppTest.from_function(_panel_script, kwargs={"state": "alert"})
        at.run()
        at.button[0].click().run()  # Initiate Transfer
        assert "TRANSACTION HELD" in at.error[0].value

        at.text_input(key="bank_otp_input").set_value(DEMO_OTP_CODE).run()
        at.button(key=None)  # no-op access to keep buttons indexed after rerun
        verify_btn = [b for b in at.button if b.label == "Verify code"][0]
        verify_btn.click().run()
        assert "released" in at.success[0].value.lower()

    def test_normal_call_transfer_approved_immediately(self):
        at = AppTest.from_function(_panel_script, kwargs={"state": "normal"})
        at.run()
        at.button[0].click().run()
        assert "approved" in at.success[0].value.lower()
        assert len(at.error) == 0


class TestDecideTransferOutcome:
    def test_alert_state_holds(self):
        assert decide_transfer_outcome("alert") == "held"

    def test_normal_state_approves(self):
        assert decide_transfer_outcome("normal") == "approved"

    def test_warn_state_approves(self):
        # only a confirmed "alert" (2+ consecutive windows) holds a transfer —
        # a single-window "warn" must not block money movement (no false positives)
        assert decide_transfer_outcome("warn") == "approved"

    def test_outcome_is_never_a_third_value(self):
        for state in ("normal", "warn", "alert", "unknown"):
            assert decide_transfer_outcome(state) in ("held", "approved")


def test_demo_otp_code_is_disclosed_constant():
    # the OTP must be a fixed, on-screen demo value — never a real generated/sent code
    assert DEMO_OTP_CODE == "123456"
