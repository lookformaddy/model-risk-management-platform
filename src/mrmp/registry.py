"""Versioned model inventory (registry) with reproducibility manifests.

Every model version is immutable once registered: its metadata (owner,
purpose, data lineage, training config), its stage, and a reproducibility
manifest whose SHA-256 hashes cover the training data, the training
config and the fitted weights. Two runs that claim to be the same model
must produce the same manifest hash — otherwise the model is not
reproducible and cannot be approved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Stage(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


@dataclass
class ReproducibilityManifest:
    """Content hashes that pin a model version to its inputs."""

    data_hash: str
    config_hash: str
    weights_hash: str
    code_version: str = "mrmp-1.0.0"
    seed: int = 42

    def combined_hash(self) -> str:
        material = "|".join(
            [self.data_hash, self.config_hash, self.weights_hash, self.code_version, str(self.seed)]
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "weights_hash": self.weights_hash,
            "code_version": self.code_version,
            "seed": self.seed,
            "combined_hash": self.combined_hash(),
        }


def hash_json(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class ModelVersion:
    model_id: str
    version: str
    owner: str
    purpose: str
    data_lineage: list[str]
    training_config: dict
    manifest: ReproducibilityManifest
    stage: Stage = Stage.DEVELOPMENT
    created_at: str = ""
    validation_report_id: str | None = None
    approved_by: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "owner": self.owner,
            "purpose": self.purpose,
            "data_lineage": self.data_lineage,
            "training_config": self.training_config,
            "manifest": self.manifest.to_dict(),
            "stage": self.stage.value,
            "created_at": self.created_at,
            "validation_report_id": self.validation_report_id,
            "approved_by": self.approved_by,
        }


@dataclass
class ModelRegistry:
    versions: dict[str, ModelVersion] = field(default_factory=dict)

    def register(self, version: ModelVersion) -> ModelVersion:
        key = f"{version.model_id}:{version.version}"
        if key in self.versions:
            raise ValueError(f"model version {key} is already registered (immutable)")
        self.versions[key] = version
        return version

    def get(self, model_id: str, version: str) -> ModelVersion:
        key = f"{model_id}:{version}"
        try:
            return self.versions[key]
        except KeyError:
            raise KeyError(f"no registered model version {key}") from None

    def list_models(self) -> list[ModelVersion]:
        return sorted(self.versions.values(), key=lambda v: (v.model_id, v.version))

    def production_version(self, model_id: str) -> ModelVersion | None:
        prod = [
            v for v in self.versions.values()
            if v.model_id == model_id and v.stage == Stage.PRODUCTION
        ]
        return prod[-1] if prod else None
