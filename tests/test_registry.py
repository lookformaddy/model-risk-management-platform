"""Tests for the versioned model registry and manifests."""

import pytest

from mrmp.registry import (
    ModelRegistry,
    ModelVersion,
    ReproducibilityManifest,
    hash_json,
)


def _version(model_id="m", version="v1") -> ModelVersion:
    return ModelVersion(
        model_id=model_id,
        version=version,
        owner="owner@example.com",
        purpose="test model",
        data_lineage=["synthetic_v1"],
        training_config={"lr": 0.5},
        manifest=ReproducibilityManifest("d" * 64, "c" * 64, "w" * 64),
    )


def test_register_and_get():
    reg = ModelRegistry()
    reg.register(_version())
    got = reg.get("m", "v1")
    assert got.owner == "owner@example.com"
    assert got.stage.value == "DEVELOPMENT"


def test_duplicate_registration_rejected():
    reg = ModelRegistry()
    reg.register(_version())
    with pytest.raises(ValueError):
        reg.register(_version())


def test_missing_version_raises_keyerror():
    with pytest.raises(KeyError):
        ModelRegistry().get("nope", "v9")


def test_manifest_combined_hash_changes_with_any_input():
    base = ReproducibilityManifest("a", "b", "c")
    assert base.combined_hash() != ReproducibilityManifest("a", "b", "d").combined_hash()
    assert base.combined_hash() != ReproducibilityManifest("a", "x", "c").combined_hash()
    assert base.combined_hash() == ReproducibilityManifest("a", "b", "c").combined_hash()


def test_hash_json_is_order_insensitive():
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})


def test_production_version_lookup():
    from mrmp.registry import Stage

    reg = ModelRegistry()
    v1 = _version(version="v1")
    v2 = _version(version="v2")
    v2.stage = Stage.PRODUCTION
    reg.register(v1)
    reg.register(v2)
    assert reg.production_version("m").version == "v2"
    assert reg.production_version("other") is None
