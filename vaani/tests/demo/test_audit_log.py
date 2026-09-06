"""tests for app/components/audit_log.py — hash-chained tamper-evident log."""

import pytest

from app.components.audit_log import AuditLog, GENESIS_HASH


class TestAuditLog:
    def test_first_entry_links_to_genesis(self):
        log = AuditLog()
        e = log.append("transfer_initiated", {"amount": 4000000})
        assert e.prev_hash == GENESIS_HASH
        assert len(e.entry_hash) == 64

    def test_entries_chain(self):
        log = AuditLog()
        e1 = log.append("transfer_initiated", {"amount": 1})
        e2 = log.append("hold_triggered", {"reason": "risk"})
        assert e2.prev_hash == e1.entry_hash

    def test_verify_chain_intact(self):
        log = AuditLog()
        log.append("a", {})
        log.append("b", {})
        log.append("c", {})
        ok, bad_seq = log.verify_chain()
        assert ok is True
        assert bad_seq is None

    def test_verify_chain_detects_tampering(self):
        log = AuditLog()
        log.append("a", {"amount": 100})
        log.append("b", {"amount": 200})
        log.append("c", {"amount": 300})
        # tamper with an earlier entry's payload after the fact
        list(log)[1].payload["amount"] = 999999
        ok, bad_seq = log.verify_chain()
        assert ok is False
        assert bad_seq == 1

    def test_empty_log_verifies(self):
        log = AuditLog()
        assert log.verify_chain() == (True, None)

    def test_to_json_roundtrips_entry_count(self):
        import json

        log = AuditLog()
        log.append("a", {})
        log.append("b", {})
        parsed = json.loads(log.to_json())
        assert len(parsed) == 2
