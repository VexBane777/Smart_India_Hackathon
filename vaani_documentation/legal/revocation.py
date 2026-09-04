"""
revocation.py — DPDP Act 2023 "Right to Erasure" (Sections 11-13) handler.

Marks a speaker's consent record as withdrawn in consent_register.json.
This is the entry point of the legal firewall's revocation workflow
(see TDD_MOD_A_01_Consent.md, section 2.2):

    1. Receive withdrawal request (email/form) — out of scope here.
    2. Lookup speaker_id in consent_register.json.
    3. Set withdrawn = true and withdrawn_date = now().
    4. (Downstream) Trigger data_purge.py to scrub manifests.
    5. (Downstream) Log action in dataset_changelog.md.

Usage:
    python revocation.py --id spk_001
    python revocation.py --id spk_001 --register /path/to/consent_register.json
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REGISTER_PATH = Path(__file__).resolve().parent / "consent_register.json"


def revoke_consent(speaker_id: str, register_path: Path = DEFAULT_REGISTER_PATH) -> dict:
    """
    Set withdrawn=true (and record withdrawn_date) for `speaker_id` in the
    consent register at `register_path`.

    Returns the updated speaker record.
    Raises FileNotFoundError if the register does not exist, and
    KeyError if the speaker_id is not present in the register.
    """
    register_path = Path(register_path)
    if not register_path.exists():
        raise FileNotFoundError(f"Consent register not found: {register_path}")

    with open(register_path, "r", encoding="utf-8") as f:
        register = json.load(f)

    speakers = register.setdefault("speakers", {})
    if speaker_id not in speakers:
        raise KeyError(f"Unknown speaker_id '{speaker_id}' in consent register")

    speakers[speaker_id]["withdrawn"] = True
    speakers[speaker_id]["withdrawn_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    _atomic_write_json(register_path, register)

    return speakers[speaker_id]


def _atomic_write_json(path: Path, data: dict) -> None:
    """
    Write `data` as JSON to `path` atomically: write to a temp file in the
    same directory, flush+fsync, then os.replace() over the target so a
    crash mid-write can never leave `path` truncated or corrupted.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Revoke a speaker's consent (DPDP right to erasure).")
    parser.add_argument("--id", required=True, dest="speaker_id", help="Anonymous speaker_id (e.g. spk_001)")
    parser.add_argument(
        "--register",
        default=str(DEFAULT_REGISTER_PATH),
        help="Path to consent_register.json (default: legal/consent_register.json)",
    )
    args = parser.parse_args()

    try:
        record = revoke_consent(args.speaker_id, Path(args.register))
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Consent revoked for '{args.speaker_id}': {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
