"""FastAPI service for the model risk management platform.

Runs the full governed pipeline once at import (seeded, deterministic)
and exposes the inventory, validation reports, approvals, challenger
comparison, drift screens and the audit log:

* GET  /health — status, champion stage, audit-chain health
* GET  /models — registered model versions
* GET  /models/{model_id}/{version} — one version + manifest
* GET  /validation — latest validation report
* GET  /challenger — champion vs challenger comparison
* POST /drift-check — PSI/KS screen against a supplied shift
* GET  /revalidation — revalidation calendar status
* GET  /audit — hash-chained audit log
* GET  /audit/verify — tamper check on the chain
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mrmp.config import Config
from mrmp.data import generate_dataset
from mrmp.monitoring import screen_drift
from mrmp.pipeline import run_pipeline

_CONFIG = Config.from_env()
_RESULT = run_pipeline(_CONFIG)

app = FastAPI(title="Model Risk Management Platform", version="1.0.0")


class DriftCheckRequest(BaseModel):
    shift: float = Field(0.6, ge=-2.0, le=2.0, description="Mean shift applied to fresh synthetic data")
    n_rows: int = Field(1500, ge=100, le=10000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "champion": "credit-decision-logreg:v1",
        "stage": _RESULT.stage,
        "tier": _RESULT.tier.tier.value,
        "validation_passed": _RESULT.validation.passed,
        "audit_ok": _RESULT.audit_ok,
        "audit_entries": _RESULT.audit_entries,
    }


@app.get("/models")
def list_models() -> dict:
    return {
        "models": [
            {
                "model_id": "credit-decision-logreg",
                "version": "v1",
                "owner": "model-owner@example.com",
                "purpose": "Generic consumer credit decision support (synthetic demo model)",
                "stage": _RESULT.stage,
                "tier": _RESULT.tier.tier.value,
                "tier_score": _RESULT.tier.score,
                "manifest_hash": _RESULT.details["manifest_hash"],
            }
        ]
    }


@app.get("/models/{model_id}/{version}")
def get_model(model_id: str, version: str) -> dict:
    if model_id != "credit-decision-logreg" or version != "v1":
        raise HTTPException(status_code=404, detail="model version not found")
    return {
        "model_id": model_id,
        "version": version,
        "stage": _RESULT.stage,
        "tier": _RESULT.tier.to_dict(),
        "validation": _RESULT.validation.to_dict(),
        "approvals": _RESULT.approvals,
        "revalidation_due": _RESULT.revalidation_due,
        "manifest_hash": _RESULT.details["manifest_hash"],
    }


@app.get("/validation")
def validation() -> dict:
    return _RESULT.validation.to_dict()


@app.get("/challenger")
def challenger() -> dict:
    return _RESULT.challenger.to_dict()


@app.post("/drift-check")
def drift_check(req: DriftCheckRequest) -> dict:
    from mrmp.data import BASE_FEATURE_NAMES

    reference = generate_dataset(n=2000, seed=_CONFIG.seed)
    current = generate_dataset(n=req.n_rows, seed=_CONFIG.seed + 200, shift=req.shift)
    screen = screen_drift(reference.X, current.X, BASE_FEATURE_NAMES,
                          psi_threshold=_CONFIG.psi_threshold,
                          ks_threshold=_CONFIG.ks_threshold)
    return {"shift": req.shift, **screen.to_dict()}


@app.get("/revalidation")
def revalidation() -> dict:
    return {
        "model_id": "credit-decision-logreg",
        "version": "v1",
        "tier": _RESULT.tier.tier.value,
        "due_on": _RESULT.revalidation_due,
        "drift_breached": _RESULT.drift.breached,
        "decay_level": _RESULT.decay_level,
        "revalidation_triggered": _RESULT.drift.breached or _RESULT.decay_level == "ALERT",
    }


@app.get("/audit")
def audit() -> dict:
    return {
        "entries": _RESULT.audit_entries,
        "head_hash": _RESULT.details["audit_head"],
        "ok": _RESULT.audit_ok,
    }


@app.get("/audit/verify")
def audit_verify() -> dict:
    return {"ok": _RESULT.audit_ok, "entries": _RESULT.audit_entries}
