"""CLI for the model risk management platform.

Usage:
    python -m mrmp.cli demo            # full governed lifecycle, printed
    python -m mrmp.cli tier --impact 3 --sensitivity 2 --autonomy 2 --exposure 3
    python -m mrmp.cli audit-verify    # verify the demo run's audit chain
    python -m mrmp.cli serve           # start the FastAPI service
"""

from __future__ import annotations

import argparse
import sys

from mrmp.audit import AuditLog
from mrmp.config import Config
from mrmp.pipeline import run_pipeline
from mrmp.tiering import ModelAttributes, assign_tier


def cmd_demo(_args: argparse.Namespace) -> int:
    result = run_pipeline(Config.from_env())
    print(f"champion accuracy: {result.champion_accuracy:.4f}  AUC: {result.champion_auc:.4f}")
    print(f"tier: {result.tier.tier.value} (score {result.tier.score})")
    print(f"validation: {'PASSED' if result.validation.passed else 'FAILED'} "
          f"({len(result.validation.checks)} checks)")
    print(f"stage: {result.stage}  approvals: {len(result.approvals)}")
    print(f"challenger: {result.challenger.decision} "
          f"(delta {result.challenger.overall_delta:+.4f})")
    print(f"drift breached: {result.drift.breached}  decay: {result.decay_level}")
    print(f"revalidation due: {result.revalidation_due}")
    print(f"audit chain: {'OK' if result.audit_ok else 'BROKEN'} "
          f"({result.audit_entries} entries)")
    return 0


def cmd_tier(args: argparse.Namespace) -> int:
    tier = assign_tier(ModelAttributes(
        decision_impact=args.impact,
        data_sensitivity=args.sensitivity,
        autonomy=args.autonomy,
        exposure=args.exposure,
        affects_protected_class=args.protected_class,
        uses_external_data=args.external_data,
    ))
    print(f"tier: {tier.tier.value} (score {tier.score})")
    print(f"validation depth: {tier.validation_depth}")
    print(f"required approvers: {', '.join(tier.required_approvers)}")
    print(f"revalidation every {tier.revalidation_days} days")
    for reason in tier.reasons:
        print(f"  - {reason}")
    return 0


def cmd_audit_verify(_args: argparse.Namespace) -> int:
    result = run_pipeline(Config.from_env())
    print(f"audit chain: {'OK' if result.audit_ok else 'BROKEN'} "
          f"({result.audit_entries} entries)")
    return 0 if result.audit_ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("mrmp.service:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrmp", description="Model Risk Management Platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="run the full governed lifecycle")
    p_demo.set_defaults(func=cmd_demo)

    p_tier = sub.add_parser("tier", help="assign a risk tier from model attributes")
    p_tier.add_argument("--impact", type=int, default=2)
    p_tier.add_argument("--sensitivity", type=int, default=2)
    p_tier.add_argument("--autonomy", type=int, default=2)
    p_tier.add_argument("--exposure", type=int, default=2)
    p_tier.add_argument("--protected-class", action="store_true")
    p_tier.add_argument("--external-data", action="store_true")
    p_tier.set_defaults(func=cmd_tier)

    p_audit = sub.add_parser("audit-verify", help="verify the audit chain of a demo run")
    p_audit.set_defaults(func=cmd_audit_verify)

    p_serve = sub.add_parser("serve", help="start the FastAPI service")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
