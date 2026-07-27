import json
from pathlib import Path

import pytest

from ares_engine.bundles import validate_bundle_metadata, verify_bundle
from ares_engine.config import GateConfig, load_config
from ares_engine.exceptions import BundleIntegrityError, PromotionRejected
from ares_engine.promotion import promote, resolve_champion
from ares_engine.utils import atomic_write_json, sha256_file


def make_fake_bundle(root: Path, name: str, *, score: float, passed: bool = True) -> Path:
    bundle = root / name
    bundle.mkdir(parents=True)
    payloads = {
        "model.keras": "dummy model",
        "scaler.joblib": "dummy scaler",
        "feature_spec.json": "{}\n",
        "config.json": "{}\n",
        "metrics.json": json.dumps({"score": score, "passed": passed}) + "\n",
        "provenance.json": "{}\n",
    }
    for filename, content in payloads.items():
        (bundle / filename).write_text(content, encoding="utf-8")
    manifest = {
        "format_version": 1,
        "files": {
            filename: {
                "sha256": sha256_file(bundle / filename),
                "bytes": (bundle / filename).stat().st_size,
            }
            for filename in payloads
        },
    }
    atomic_write_json(bundle / "manifest.json", manifest)
    return bundle


def test_bundle_tampering_is_detected(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "candidate", score=1.0)
    verify_bundle(bundle)
    (bundle / "metrics.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(BundleIntegrityError):
        verify_bundle(bundle)


def test_promotion_requires_improvement(tmp_path: Path) -> None:
    gates = GateConfig(min_promotion_score_improvement=0.1)
    first = make_fake_bundle(tmp_path, "first", score=1.0)
    promote(first, tmp_path, gates)
    assert resolve_champion(tmp_path) == first.resolve()

    weak = make_fake_bundle(tmp_path, "weak", score=1.05)
    with pytest.raises(PromotionRejected):
        promote(weak, tmp_path, gates)

    strong = make_fake_bundle(tmp_path, "strong", score=1.2)
    promote(strong, tmp_path, gates)
    assert resolve_champion(tmp_path) == strong.resolve()


def test_unlisted_bundle_file_is_rejected(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "candidate-extra", score=1.0)
    (bundle / "surprise.bin").write_bytes(b"not in manifest")
    with pytest.raises(BundleIntegrityError):
        verify_bundle(bundle)


def test_nested_bundle_entry_is_rejected(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "candidate-nested", score=1.0)
    (bundle / "nested").mkdir()
    with pytest.raises(BundleIntegrityError, match="nested or special"):
        verify_bundle(bundle)


def test_malformed_manifest_record_is_rejected(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "candidate-bad-manifest", score=1.0)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metrics.json"] = "not a record"
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(BundleIntegrityError, match="Invalid manifest record"):
        verify_bundle(bundle)


def test_champion_pointer_anchors_the_promoted_manifest(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "anchored", score=1.0)
    promote(bundle, tmp_path, GateConfig(min_promotion_score_improvement=0.0))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rewritten_after_promotion"] = True
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(BundleIntegrityError, match="promoted pointer"):
        resolve_champion(tmp_path)


def test_bundle_metadata_must_match_exported_config() -> None:
    config = load_config("configs/smoke.yaml")
    feature_spec = {
        "columns": ["close", "rsi_14"],
        "lookback_bars": config.model.lookback_bars,
        "timeframe": config.data.timeframe,
        "symbol": "BTC/USD",
        "feature_config": config.features.model_dump(mode="json"),
        "label_config": config.labels.model_dump(mode="json"),
        "thresholds": {
            "long": config.backtest.long_threshold,
            "short": config.backtest.short_threshold,
        },
        "costs_bps": {
            "fee": config.backtest.fee_bps,
            "slippage": config.backtest.slippage_bps,
        },
    }
    with pytest.raises(BundleIntegrityError, match="symbol"):
        validate_bundle_metadata(feature_spec, config)


def test_promotion_rejects_non_finite_scores(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "non-finite", score=float("nan"))
    with pytest.raises(BundleIntegrityError, match="finite"):
        promote(bundle, tmp_path, GateConfig(min_promotion_score_improvement=0.0))


def test_promotion_rejects_bundle_outside_artifacts_root(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    external = make_fake_bundle(tmp_path / "external", "candidate", score=1.0)
    with pytest.raises(BundleIntegrityError, match="contained"):
        promote(external, artifacts, GateConfig(min_promotion_score_improvement=0.0))


def test_champion_pointer_cannot_escape_artifacts_root(tmp_path: Path) -> None:
    external = make_fake_bundle(tmp_path / "external", "candidate", score=1.0)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    atomic_write_json(
        artifacts / "champion.json",
        {
            "bundle_path": str(external),
            "manifest_sha256": sha256_file(external / "manifest.json"),
        },
    )
    with pytest.raises(BundleIntegrityError, match="contained"):
        resolve_champion(artifacts)


def test_promotion_lock_fails_closed(tmp_path: Path) -> None:
    from filelock import FileLock

    bundle = make_fake_bundle(tmp_path, "candidate-locked", score=1.0)
    with FileLock(tmp_path / ".ares-promotion.lock"):
        with pytest.raises(PromotionRejected, match="already running"):
            promote(bundle, tmp_path, GateConfig(min_promotion_score_improvement=0.0))
