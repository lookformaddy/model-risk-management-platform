"""SHA-256 hash-chained audit log.

Every governance event (registration, validation, approval, promotion,
drift alert, revalidation) is appended as an entry whose hash commits to
the previous entry's hash. Any later edit, deletion or reordering breaks
the chain, and :meth:`AuditLog.verify` reports the first broken index.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    event_type: str
    actor: str
    payload: dict
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "actor": self.actor,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


def _entry_hash(seq: int, timestamp: str, event_type: str, actor: str, payload: dict, prev_hash: str) -> str:
    material = "|".join(
        [str(seq), timestamp, event_type, actor, _canonical(payload), prev_hash]
    )
    return hashlib.sha256(material.encode()).hexdigest()


@dataclass
class AuditLog:
    entries: list[AuditEntry] = field(default_factory=list)

    @property
    def head_hash(self) -> str:
        return self.entries[-1].entry_hash if self.entries else "0" * 64

    def append(self, event_type: str, payload: dict, actor: str = "system") -> AuditEntry:
        # Snapshot the payload: callers may keep mutating their dicts/lists
        # (e.g. a model's approved_by list), and the hash must commit to
        # the payload as it was at append time.
        payload = copy.deepcopy(payload)
        timestamp = datetime.now(timezone.utc).isoformat()
        prev = self.head_hash
        seq = len(self.entries)
        entry_hash = _entry_hash(seq, timestamp, event_type, actor, payload, prev)
        entry = AuditEntry(seq, timestamp, event_type, actor, payload, prev, entry_hash)
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, int | None]:
        """Recompute the chain. Returns (ok, first_bad_index or None)."""
        prev = "0" * 64
        for i, entry in enumerate(self.entries):
            if entry.prev_hash != prev:
                return (False, i)
            expected = _entry_hash(
                entry.seq, entry.timestamp, entry.event_type, entry.actor,
                entry.payload, entry.prev_hash,
            )
            if expected != entry.entry_hash:
                return (False, i)
            prev = entry.entry_hash
        return (True, None)

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries], "head_hash": self.head_hash}
