"""End-to-end governance pipeline.

Trains a champion from scratch, registers it with a reproducibility
manifest, tiers it, validates it, collects signed approvals, promotes
it to production, then evaluates a challenger (which wins overall but
regresses on group C and is rejected), and finally runs a drift screen
on shifted production data that triggers a revalidation. Every step is
written to the hash-chained audit log.

This is the single function the demo, the CLI and the API all drive, so
the numbers they report always agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from mrmp.audit import AuditLog
from mrmp.challenger import ChallengerReport, compare_challenger
from mrmp.config import Config
from mrmp.data import (
    ALL_FEATURE_NAMES,
    BASE_FEATURE_NAMES,
    Dataset,
    generate_dataset,
    split_dataset,
)
from mrmp.metrics import accuracy
from mrmp.monitoring import (
    DriftScreen,
    RevalidationCalendar,
    check_performance_decay,
    screen_drift,
)
from mrmp.registry import (
    ModelRegistry,
    ModelVersion,
    ReproducibilityManifest,
    Stage,
    hash_json,
)
from mrmp.tiering import ModelAttributes, TierAssignment, assign_tier
from mrmp.trainer import LogisticRegression
from mrmp.validation import ValidationReport, run_validation
from mrmp.workflow import Workflow


@dataclass
class PipelineResult:
    champion_accuracy: float
    champion_auc: float
    tier: TierAssignment
    validation: ValidationReport
    approvals: list[str]
    stage: str
    challenger: ChallengerReport
    drift: DriftScreen
    decay_level: str
    revalidation_due: str
    audit_ok: bool
    audit_entries: int
    champion_version: str = "v1"
    challenger_version: str = "v2-challenger"
    details: dict = field(default_factory=dict)


def build_model(
    train: Dataset,
    use_channel: bool,
    config: Config,
) -> LogisticRegression:
    X = train.with_channel() if use_channel else train.X
    model = LogisticRegression(learning_rate=0.5, epochs=400, l2=0.01)
    model.fit(X, train.y)
    return model


def _manifest(
    model: LogisticRegression,
    train: Dataset,
    train_config: dict,
    seed: int,
) -> ReproducibilityManifest:
    return ReproducibilityManifest(
        data_hash=train.content_hash(),
        config_hash=hash_json(train_config),
        weights_hash=hash_json(model.to_dict()),
        seed=seed,
    )


def run_pipeline(config: Config | None = None) -> PipelineResult:
    """Run the full governed lifecycle and return every artifact."""
    config = config or Config.from_env()
    audit = AuditLog()
    registry = ModelRegistry()
    calendar = RevalidationCalendar()
    today = date(2026, 11, 18)

    # --- 1. Data --------------------------------------------------------
    full_train = generate_dataset(n=config.n_train, seed=config.seed)
    train, holdout = split_dataset(full_train, seed=config.seed + 1)
    validation_data = generate_dataset(n=config.n_validate, seed=config.seed + 100)
    audit.append("data_generated", {"n_train": len(train), "n_validation": len(validation_data),
                                    "data_hash": train.content_hash()[:16]}, actor="pipeline")

    # --- 2. Champion: base features only --------------------------------
    train_config = {"features": BASE_FEATURE_NAMES, "learning_rate": 0.5,
                    "epochs": 400, "l2": 0.01, "seed": config.seed}
    champion = build_model(train, use_channel=False, config=config)
    champion_version = ModelVersion(
        model_id="credit-decision-logreg",
        version="v1",
        owner="model-owner@example.com",
        purpose="Generic consumer credit decision support (synthetic demo model)",
        data_lineage=["synthetic_credit_v1 (seed 42, generated in-process)"],
        training_config=train_config,
        manifest=_manifest(champion, train, train_config, config.seed),
    )
    registry.register(champion_version)
    audit.append("model_registered", champion_version.to_dict(), actor="model-owner@example.com")

    # --- 3. Risk tiering -------------------------------------------------
    attrs = ModelAttributes(
        decision_impact=3, data_sensitivity=2, autonomy=2, exposure=3,
        affects_protected_class=True, uses_external_data=False,
    )
    tier = assign_tier(attrs)
    audit.append("tier_assigned", {"model_id": champion_version.model_id, **tier.to_dict()},
                 actor="risk-officer@example.com")

    # --- 4. Validation ----------------------------------------------------
    val_report = run_validation(
        champion, validation_data, champion_version.model_id, "v1",
        tier.tier, report_id="VAL-2026-11-18-001",
    )
    audit.append("validation_completed",
                 {"report_id": val_report.report_id, "passed": val_report.passed,
                  "report_hash": val_report.report_hash()}, actor="model-validator@example.com")

    # --- 5. Approval workflow ---------------------------------------------
    workflow = Workflow(version=champion_version, tier=tier.tier, secret=config.approval_secret)
    workflow.submit_for_validation(val_report)
    approvers = [("ana.validator@example.com", "model_validator"),
                 ("omar.risk@example.com", "risk_officer")]
    approval_names: list[str] = []
    if val_report.passed:
        for name, role in approvers:
            approval = workflow.approve(name, role, val_report, signed_on=today, validity_days=180)
            approval_names.append(f"{name} ({role})")
            audit.append("approval_signed",
                         {"approver": name, "role": role, "report_id": val_report.report_id,
                          "signature": approval.signature[:16] + "..."}, actor=name)
        workflow.promote_to_production(on=today)
        audit.append("promoted_to_production",
                     {"model_id": champion_version.model_id, "version": "v1"}, actor="model-owner@example.com")
        entry = calendar.schedule(
            champion_version.model_id, "v1", tier.tier.value, today, tier.revalidation_days)
        revalidation_due = entry.due_on.isoformat()
    else:
        revalidation_due = "not scheduled — validation failed"

    # --- 6. Challenger: adds the channel score ----------------------------
    challenger = build_model(train, use_channel=True, config=config)
    champ_preds = champion.predict(validation_data.X)
    chall_preds = challenger.predict(validation_data.with_channel())
    chall_report = compare_challenger(validation_data.y, champ_preds, chall_preds, validation_data.groups)
    audit.append("challenger_evaluated",
                 {"decision": chall_report.decision, **chall_report.to_dict()},
                 actor="model-validator@example.com")

    # --- 7. Drift + decay on shifted production data ----------------------
    drifted = generate_dataset(n=1500, seed=config.seed + 200, shift=0.6)
    drift = screen_drift(
        train.X, drifted.X, BASE_FEATURE_NAMES,
        psi_threshold=config.psi_threshold, ks_threshold=config.ks_threshold,
    )
    drifted_preds = champion.predict(drifted.X)
    decay = check_performance_decay(
        accuracy(validation_data.y, champ_preds), drifted.y, drifted_preds)
    audit.append("drift_screened",
                 {"breached": drift.breached, **drift.to_dict()}, actor="monitoring")
    audit.append("performance_decay_checked", decay.to_dict(), actor="monitoring")
    if drift.breached or decay.level == "ALERT":
        audit.append("revalidation_triggered",
                     {"reason": "drift breach" if drift.breached else "performance decay",
                      "model_id": champion_version.model_id, "version": "v1"},
                     actor="monitoring")

    audit_ok, _ = audit.verify()
    champ_probs = champion.predict_proba(validation_data.X)
    from mrmp.metrics import roc_auc

    details = {
        "validation_accuracy": round(accuracy(validation_data.y, champ_preds), 4),
        "challenger_slice_accuracies": {
            g: round(accuracy(
                [validation_data.y[i] for i, x in enumerate(validation_data.groups) if x == g],
                [chall_preds[i] for i, x in enumerate(validation_data.groups) if x == g]), 4)
            for g in ("A", "B", "C")
        },
        "champion_slice_accuracies": {
            g: round(accuracy(
                [validation_data.y[i] for i, x in enumerate(validation_data.groups) if x == g],
                [champ_preds[i] for i, x in enumerate(validation_data.groups) if x == g]), 4)
            for g in ("A", "B", "C")
        },
        "drifted_accuracy": round(accuracy(drifted.y, drifted_preds), 4),
        "manifest_hash": champion_version.manifest.combined_hash()[:16],
        "audit_head": audit.head_hash[:16],
    }
    return PipelineResult(
        champion_accuracy=round(accuracy(validation_data.y, champ_preds), 4),
        champion_auc=round(roc_auc(validation_data.y, champ_probs), 4),
        tier=tier,
        validation=val_report,
        approvals=approval_names,
        stage=champion_version.stage.value,
        challenger=chall_report,
        drift=drift,
        decay_level=decay.level,
        revalidation_due=revalidation_due,
        audit_ok=audit_ok,
        audit_entries=len(audit.entries),
        details=details,
    )
