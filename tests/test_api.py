"""API tests (FastAPI TestClient). These run only when fastapi and
httpx are installed; otherwise they are skipped, not failed.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from mrmp.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["stage"] == "PRODUCTION"
    assert data["tier"] == "CRITICAL"
    assert data["validation_passed"] is True
    assert data["audit_ok"] is True


def test_list_models():
    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) == 1
    assert models[0]["model_id"] == "credit-decision-logreg"
    assert models[0]["stage"] == "PRODUCTION"


def test_get_model_and_unknown():
    resp = client.get("/models/credit-decision-logreg/v1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tier"]["tier"] == "CRITICAL"
    assert len(data["approvals"]) == 2
    assert client.get("/models/nope/v9").status_code == 404


def test_validation_endpoint():
    resp = client.get("/validation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert len(data["checks"]) >= 10


def test_challenger_endpoint_rejects():
    resp = client.get("/challenger")
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "REJECT"
    assert data["overall_delta"] > 0
    assert data["slice_deltas"]["C"] < -0.02


def test_drift_check_endpoint():
    resp = client.post("/drift-check", json={"shift": 0.0, "n_rows": 1000})
    assert resp.status_code == 200
    assert resp.json()["breached"] is False
    resp = client.post("/drift-check", json={"shift": 0.6, "n_rows": 1000})
    assert resp.status_code == 200
    assert resp.json()["breached"] is True


def test_revalidation_and_audit_endpoints():
    resp = client.get("/revalidation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["revalidation_triggered"] is True
    assert data["due_on"] == "2027-02-16"
    resp = client.get("/audit/verify")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
