"""Configuration, read from the environment with safe defaults.

No secrets are required to run the demo. ``MRMP_APPROVAL_SECRET`` is the
shared secret used to sign approvals (HMAC-style SHA-256 signatures); the
default is a clearly-marked demo value — set a real one in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 42
    n_train: int = 2000
    n_validate: int = 2000
    approval_secret: str = "mrmp-demo-secret-change-me"
    psi_threshold: float = 0.20
    ks_threshold: float = 0.15
    accuracy_floor: float = 0.70
    auc_floor: float = 0.70
    revalidation_days_critical: int = 90
    revalidation_days_high: int = 180
    revalidation_days_medium: int = 365
    revalidation_days_low: int = 730

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=os.environ.get("MRMP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MRMP_PORT", "8000")),
            seed=int(os.environ.get("MRMP_SEED", "42")),
            n_train=int(os.environ.get("MRMP_N_TRAIN", "2000")),
            n_validate=int(os.environ.get("MRMP_N_VALIDATE", "2000")),
            approval_secret=os.environ.get(
                "MRMP_APPROVAL_SECRET", "mrmp-demo-secret-change-me"
            ),
            psi_threshold=float(os.environ.get("MRMP_PSI_THRESHOLD", "0.20")),
            ks_threshold=float(os.environ.get("MRMP_KS_THRESHOLD", "0.15")),
            accuracy_floor=float(os.environ.get("MRMP_ACCURACY_FLOOR", "0.70")),
            auc_floor=float(os.environ.get("MRMP_AUC_FLOOR", "0.70")),
        )
