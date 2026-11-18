"""Model Risk Management Platform (MRMP).

A governance platform for governed model validation and approval, in the
spirit of SR 11-7 model risk management: a versioned model inventory with
reproducibility manifests, rule-based risk tiering, tier-driven validation
suites (performance, stability, data quality, fairness, outcome analysis),
champion/challenger comparison with a per-slice regression rule, a signed
staged approval workflow, ongoing drift/performance monitoring with a
revalidation calendar, and a SHA-256 hash-chained audit log.

Everything is built from scratch on the Python standard library (no
sklearn / numpy / torch) and runs on seeded synthetic data only.
"""

from mrmp.audit import AuditLog
from mrmp.registry import ModelRegistry
from mrmp.tiering import RiskTier
from mrmp.workflow import Stage

__all__ = ["AuditLog", "ModelRegistry", "RiskTier", "Stage"]
__version__ = "1.0.0"
