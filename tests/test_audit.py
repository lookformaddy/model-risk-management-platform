"""Tests for the hash-chained audit log."""

from mrmp.audit import AuditLog


def test_append_and_verify_clean_chain():
    log = AuditLog()
    log.append("model_registered", {"model_id": "m"}, actor="owner")
    log.append("validation_completed", {"passed": True}, actor="validator")
    ok, bad = log.verify()
    assert ok and bad is None
    assert log.entries[0].prev_hash == "0" * 64
    assert log.entries[1].prev_hash == log.entries[0].entry_hash


def test_tampered_payload_detected():
    log = AuditLog()
    log.append("approval_signed", {"approver": "ana"}, actor="ana")
    log.append("promoted_to_production", {"version": "v1"}, actor="owner")
    log.entries[0].payload["approver"] = "mallory"  # tamper
    ok, bad = log.verify()
    assert not ok and bad == 0


def test_tampered_hash_detected():
    log = AuditLog()
    log.append("a", {}, actor="x")
    log.append("b", {}, actor="x")
    log.entries[1].entry_hash = "f" * 64
    ok, bad = log.verify()
    assert not ok and bad == 1


def test_empty_log_verifies():
    ok, bad = AuditLog().verify()
    assert ok and bad is None


def test_head_hash_tracks_last_entry():
    log = AuditLog()
    assert log.head_hash == "0" * 64
    e = log.append("a", {}, actor="x")
    assert log.head_hash == e.entry_hash


def test_mutating_caller_payload_after_append_does_not_break_chain():
    """The log snapshots payloads: later mutation must not look like tampering."""
    log = AuditLog()
    payload = {"approvers": ["ana"]}
    log.append("approval_signed", payload, actor="ana")
    payload["approvers"].append("omar")  # caller keeps mutating its own dict
    ok, bad = log.verify()
    assert ok and bad is None
