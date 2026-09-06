"""
audit_log.py — Tamper-evident hash-chained decision log (master plan §1,
item 5: "our honest answer to the Blockchain theme"; Module C Task 5).

Each entry embeds the SHA-256 fingerprint of the previous entry, so any
edit to an earlier entry breaks every hash after it — `verify_chain()`
detects tampering by recomputing the chain and comparing.

This is a decision-log data structure, not a distributed ledger: single
writer, in-process (or file-persisted for the demo), no consensus. That
scope match is deliberate — the point is tamper-evidence for the honest
bank-sim workflow, not a blockchain reimplementation.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    seq: int
    ts: float
    event: str
    payload: dict
    prev_hash: str
    entry_hash: str = field(init=False)

    def __post_init__(self):
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        body = json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts,
                "event": self.event,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "event": self.event,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.entry_hash,
        }


class AuditLog:
    """In-memory hash-chained log. `append()` is the only mutator."""

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def append(self, event: str, payload: Optional[dict] = None, ts: Optional[float] = None) -> AuditEntry:
        prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        entry = AuditEntry(
            seq=len(self._entries),
            ts=time.time() if ts is None else ts,
            event=event,
            payload=payload or {},
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def entries(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        """
        Recompute every entry's hash and check the prev_hash links.

        Returns (True, None) if the chain is intact, else
        (False, <seq of the first broken/tampered entry>).
        """
        prev_hash = GENESIS_HASH
        for entry in self._entries:
            if entry.prev_hash != prev_hash:
                return False, entry.seq
            if entry._compute_hash() != entry.entry_hash:
                return False, entry.seq
            prev_hash = entry.entry_hash
        return True, None

    def to_json(self) -> str:
        return json.dumps(self.entries(), indent=2)
