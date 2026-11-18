# Architecture — Model Risk Management Platform

## Overview

The platform implements the governed model lifecycle as a pipeline of
small, independently testable modules. Data flows in one direction:

```
synthetic data ──> trainer ──> registry (manifest)
                          │
                          ├─> tiering ──> validation suite ──> workflow
                          │                                    (approvals)
                          ├─> challenger comparison
                          └─> monitoring (drift / decay / calendar)
   every step ──> hash-chained audit log
```

The FastAPI service (`service.py`) and the CLI (`cli.py`) are thin
shells over `pipeline.run_pipeline()`, so the demo, the tests and the
API always report the same numbers.

## Modules

| Module | Responsibility |
|---|---|
| `data.py` | Seeded synthetic credit-style data; groups A/B/C; the planted `channel_score` (predictive for A/B, anti-predictive for C) that makes the challenger story real. |
| `trainer.py` | From-scratch logistic regression: standardization + full-batch gradient descent, L2, deterministic (zero init, fixed epochs). |
| `metrics.py` | Accuracy, rank AUC, Brier, log-loss, ECE, confusion, percentile bootstrap CIs, k-fold variance, McNemar (continuity-corrected, chi-square p via `erfc`), PSI (decile bins), two-sample KS. |
| `tiering.py` | Weighted score over decision impact (0.35), data sensitivity (0.20), autonomy (0.25), exposure (0.20); thresholds at 2.5 / 1.8 / 1.0; auto-escalation to CRITICAL when decision impact and autonomy are both maxed; every assignment carries reasons. |
| `registry.py` | Immutable versioned inventory; `ReproducibilityManifest` hashes training data, training config and fitted weights; `combined_hash` must match for a re-run to claim to be the same model. |
| `validation.py` | Five check families (performance, stability, data quality, fairness slices, outcome analysis) with per-tier thresholds; LOW runs the light suite (performance + data quality only). Reports carry a content hash for approval binding. |
| `challenger.py` | Paired comparison: overall delta, McNemar p-value, per-slice accuracy deltas. Promote requires overall delta ≥ +0.005, no slice regression beyond −0.02, and (for small wins) statistical significance. |
| `workflow.py` | Stage machine DEVELOPMENT → VALIDATION → APPROVED → PRODUCTION → RETIRED with an explicit transition table; HMAC-SHA-256 signatures bound to the report hash; expiry; fail-closed errors on every gate. |
| `monitoring.py` | Per-feature PSI/KS screens; accuracy-decay levels (OK / WATCH / ALERT at 0.03 / 0.06 drops); `RevalidationCalendar` with CURRENT / DUE_SOON / OVERDUE / ESCALATED (>30 days overdue) statuses. |
| `audit.py` | Append-only hash chain: `entry_hash = SHA256(seq ‖ timestamp ‖ type ‖ actor ‖ canonical(payload) ‖ prev_hash)`; `verify()` reports the first broken index. Payloads are deep-copied at append time. |
| `pipeline.py` | Orchestration of the full lifecycle; returns a `PipelineResult` with every artifact. |

## Tier thresholds

| Check | CRITICAL | HIGH | MEDIUM | LOW |
|---|---|---|---|---|
| Accuracy floor | 0.72 | 0.70 | 0.68 | 0.65 |
| AUC floor | 0.75 | 0.72 | 0.70 | 0.65 |
| Must-beat-baseline delta | 0.10 | 0.08 | 0.05 | 0.03 |
| Bootstrap CI width cap | 0.06 | 0.08 | 0.10 | 0.12 |
| Cross-fold std cap | 0.03 | 0.04 | 0.05 | 0.06 |
| Demographic parity gap cap | 0.12 | 0.15 | 0.20 | 0.25 |
| Equal-opportunity gap cap | 0.12 | 0.15 | 0.20 | 0.25 |
| ECE cap | 0.08 | 0.10 | 0.12 | 0.15 |
| Required approvers | validator + risk officer | validator + risk officer | validator | owner |
| Revalidation | 90 days | 180 days | 365 days | 730 days |

## Approval flow (sequence)

```
register(v1) ──> tier(CRITICAL) ──> validate ──> submit_for_validation
   │                                                        │
   │                     report PASSED? ── no ──> stays in VALIDATION
   │                                                        │ yes
   │                 validator signs (report hash) ──> risk officer signs
   │                                                        │
   └─ stage APPROVED ──> promote (checks expiry) ──> PRODUCTION
```

An approval that expires before promotion blocks promotion. A model in
PRODUCTION whose revalidation date passes is flagged OVERDUE, then
ESCALATED after 30 days; nothing auto-extends it.

## Audit log threat model

The chain detects tampering *after the fact*: editing, deleting or
reordering any entry breaks every subsequent hash. It does **not**
protect against an attacker who can rewrite the whole log from scratch
(no external anchoring), and the HMAC approval secret is a demo value —
production deployments need per-approver keys (ideally hardware-backed)
and the log replicated to append-only storage. The chain's job here is
to make silent edits detectable, and the test suite proves it: mutating
a payload or a hash is caught at the exact index.

## Data

All data is synthetic, generated in-process from a fixed seed. The
domain is a generic consumer-credit-style binary decision with four
financial-behaviour features and three population groups. No real
customer data, no employer data, no external files.

## Extension points

- Persistence: replace the in-memory `ModelRegistry` / `AuditLog` with
  database-backed implementations behind the same interfaces.
- Signatures: swap `sign_approval` for per-approver asymmetric keys.
- Validation: add a check by appending a `ValidationCheck` in
  `run_validation`; add a tier by extending `TIER_THRESHOLDS`.
- Monitoring: wire `screen_drift` / `check_performance_decay` to a
  scheduler and an alerting sink; the calendar already emits statuses.
