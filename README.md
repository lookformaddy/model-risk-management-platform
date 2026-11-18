# Model Risk Management Platform

A model risk management platform in the spirit of SR 11-7: a governed
model inventory, rule-based risk tiering, tier-driven validation suites,
champion/challenger comparison with a per-slice regression rule, a signed
staged approval workflow, ongoing drift and performance monitoring with a
revalidation calendar, and a SHA-256 hash-chained audit log — all built
from scratch on the Python standard library (no sklearn, numpy or torch)
and running on seeded synthetic data only.

## The enterprise problem

In a regulated enterprise, a model is not "done" when it trains well.
Before it touches a customer decision it must be inventoried, tiered by
risk, validated by an independent suite, approved by the roles its tier
requires, and then watched in production — with revalidation on a
schedule, not on vibes. The failure mode this platform is built around
is the one averages hide: a challenger that is better overall and worse
for one group of customers. The demo shows exactly that model being
rejected, for a named reason, by a rule — not by a meeting.

## Demo results (computed live, `scripts/demo.py`)

A from-scratch logistic regression (`credit-decision-logreg:v1`) trained
on 1,000 synthetic rows and validated on 2,000 held-out rows, seed 42:

| Step | Result |
|---|---|
| Champion validation accuracy | **0.7420** (CRITICAL floor 0.72) |
| Champion ROC-AUC | **0.8159** (floor 0.75) |
| Must-beat-baseline delta | **+0.1725** over the majority-class rate (floor 0.10) |
| Bootstrap 95% CI width (accuracy) | **0.0400** (cap 0.06) |
| Cross-fold accuracy std | **0.0176** (cap 0.03) |
| Demographic parity gap | **0.0517** (cap 0.12) |
| Equal-opportunity gap | **0.0892** (cap 0.12) |
| ECE / Brier | **0.0239** / **0.1729** (caps 0.08 / 0.25) |
| **Validation** | **PASSED — 14/14 checks** |

**Tier:** CRITICAL (score 2.55 — max decision impact + full customer
exposure + protected-class scope), requiring two signed approvals
(model validator + risk officer) and revalidation every 90 days.

**Approvals:** `ana.validator@example.com` (model_validator) and
`omar.risk@example.com` (risk_officer) signed the report hash; the model
promoted to PRODUCTION with revalidation due **2027-02-16**.

**Challenger (v2, adds a `channel_score` feature): REJECTED.**
It wins overall — 0.7420 → 0.7910 (+0.0490, McNemar p=0.0000) — but
regresses on group C from 0.7306 to 0.5389 (**−0.1917**, limit −0.02),
while groups A and B improve by +0.11 each. The slice rule is a hard
gate: a model that is better on average and worse for one group does not
ship.

**Monitoring:** fresh production data with a +0.6 mean shift breaches
every drift screen (PSI 0.30–0.42 > 0.20; KS 0.24–0.27 > 0.15 on all
four features) and triggers a revalidation, even though accuracy on the
drifted data only slips to 0.7287 (decay level OK) — drift is caught
before performance visibly decays.

**Audit:** 11 entries, SHA-256 hash chain verifies OK.

## Architecture

```
scripts/demo.py            end-to-end governed lifecycle
src/mrmp/
  data.py                  deterministic synthetic data (planted channel trap)
  trainer.py               from-scratch logistic regression (stdlib only)
  metrics.py               accuracy/AUC/Brier/ECE, bootstrap CI, k-fold,
                           McNemar, PSI, KS
  tiering.py               rule-based CRITICAL/HIGH/MEDIUM/LOW tiering
  registry.py              versioned inventory + reproducibility manifests
  validation.py            tier-driven validation suites (5 check families)
  challenger.py            champion vs challenger + slice regression rule
  workflow.py              staged approvals, signed + report-bound
  monitoring.py            drift screens, decay alerts, revalidation calendar
  audit.py                 SHA-256 hash-chained audit log
  pipeline.py              orchestration: train -> tier -> validate ->
                           approve -> promote -> challenge -> monitor
  service.py               FastAPI service
  cli.py                   CLI (demo / tier / audit-verify / serve)
  config.py                env-based configuration
tests/                     pytest suite (see below)
```

See `docs/architecture.md` for the full design, including the threat
model for the audit log and the tier-threshold table.

## How to run

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q   # 66 tests (+ API suite with FastAPI)
PYTHONPATH=src python scripts/demo.py        # full governed lifecycle
PYTHONPATH=src python -m mrmp.cli tier --impact 3 --sensitivity 2 --autonomy 2 --exposure 3
PYTHONPATH=src python -m mrmp.cli serve      # FastAPI on :8000
```

Docker: `docker build -t mrmp . && docker run -p 8000:8000 mrmp`

API: `GET /health`, `GET /models`, `GET /models/{id}/{version}`,
`GET /validation`, `GET /challenger`, `POST /drift-check`,
`GET /revalidation`, `GET /audit`, `GET /audit/verify`.

## Design decisions and trade-offs

- **Pure stdlib core.** No sklearn/numpy/torch anywhere in the library,
  so the platform has zero heavyweight dependencies and every statistic
  (including the McNemar p-value and the bootstrap CIs) is inspectable.
  FastAPI is an optional serving layer only; the core runs without it.
- **Fail-closed gates, everywhere.** No approval without a passing
  validation report; no promotion without every required signature;
  approvals are bound to one report hash, so a re-run report cannot
  inherit an old approval. Illegal stage transitions raise.
- **The slice rule is a hard gate, not a score.** A challenger that wins
  overall but regresses any protected slice by more than 0.02 is
  rejected outright. Averaging away a group's harm is the failure the
  platform exists to prevent.
- **Small-sample guards, not small-sample failures.** A fairness slice
  below 50 rows is reported as insufficient data, never failed on noise.
- **Audit log snapshots payloads.** Entries deep-copy their payload at
  append time, so later mutation of a caller's object cannot silently
  change what the hash committed to (caught by a regression test).
- **Signatures are HMAC-SHA-256, not PKI.** The demo secret stands in for
  a real signing key; the binding property (signature verifies only for
  the exact payload) is the same. Production should use per-approver keys.
- **In-memory stores.** The registry, workflow and audit log are
  in-memory by design — this is a reference implementation of the
  governance logic, not a persistence layer. Swap in a database behind
  the same interfaces for production use.

## What it demonstrates

Governed model lifecycle end to end: inventory with reproducibility
manifests, risk tiering that drives validation depth and approval
quorum, statistical validation (bootstrap CIs, cross-fold variance,
fairness slices, calibration, error concentration), paired
champion/challenger testing with a slice-regression veto, signed
run-bound approvals with expiry, drift and decay monitoring with a
revalidation calendar and overdue escalation, and a tamper-evident
audit trail — all deterministic and reproducible from a single seed.
