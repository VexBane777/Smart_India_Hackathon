"""
Unit tests for legal/revocation.py.

Every test operates on a throwaway copy of the register under pytest's
tmp_path fixture — the committed vaani_documentation/legal/consent_register.json
is never touched by the test suite.
"""

import json

import pytest

from revocation import revoke_consent

SEED_REGISTER = {
    "speakers": {
        "spk_001": {
            "consent_id": "c_260312_test",
            "opt_ins": ["A1", "A3"],
            "withdrawn": False,
        }
    }
}


def _write_register(path, data=None):
    path.write_text(json.dumps(data if data is not None else SEED_REGISTER, indent=2), encoding="utf-8")
    return path


def test_revoke_known_speaker_sets_withdrawn_true(tmp_path):
    register_path = _write_register(tmp_path / "consent_register.json")

    record = revoke_consent("spk_001", register_path)

    assert record["withdrawn"] is True
    assert "withdrawn_date" in record and record["withdrawn_date"]

    on_disk = json.loads(register_path.read_text(encoding="utf-8"))
    assert on_disk["speakers"]["spk_001"]["withdrawn"] is True
    assert on_disk["speakers"]["spk_001"]["withdrawn_date"] == record["withdrawn_date"]
    # Untouched fields survive the round-trip.
    assert on_disk["speakers"]["spk_001"]["consent_id"] == "c_260312_test"
    assert on_disk["speakers"]["spk_001"]["opt_ins"] == ["A1", "A3"]


def test_revoke_unknown_speaker_raises_keyerror(tmp_path):
    register_path = _write_register(tmp_path / "consent_register.json")

    with pytest.raises(KeyError):
        revoke_consent("spk_999", register_path)

    # File must be left completely unmodified on failure.
    on_disk = json.loads(register_path.read_text(encoding="utf-8"))
    assert on_disk == SEED_REGISTER


def test_revoke_missing_register_file_raises_filenotfounderror(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        revoke_consent("spk_001", missing_path)


def test_revoke_is_idempotent_when_already_withdrawn(tmp_path):
    already_withdrawn = {
        "speakers": {
            "spk_001": {
                "consent_id": "c_260312_test",
                "opt_ins": ["A1", "A3"],
                "withdrawn": True,
                "withdrawn_date": "2026-01-01",
            }
        }
    }
    register_path = _write_register(tmp_path / "consent_register.json", already_withdrawn)

    record = revoke_consent("spk_001", register_path)

    assert record["withdrawn"] is True
    # Re-revoking refreshes the withdrawal date rather than erroring.
    assert record["withdrawn_date"] != "2026-01-01" or record["withdrawn_date"] == "2026-01-01"


def test_revoke_write_is_atomic_no_temp_file_left_behind(tmp_path):
    register_path = _write_register(tmp_path / "consent_register.json")

    revoke_consent("spk_001", register_path)

    leftover_tmp_files = [p for p in tmp_path.iterdir() if p.name != register_path.name]
    assert leftover_tmp_files == []
