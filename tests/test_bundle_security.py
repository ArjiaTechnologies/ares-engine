"""Hostile-bundle battery against a real exported bundle. Loading must fail closed."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import ares_engine.bundles as bundles_module
from ares_engine.bundles import export_bundle, load_bundle, verify_bundle
from ares_engine.config import load_config
from ares_engine.exceptions import BundleIntegrityError
from ares_engine.models import predict_probabilities
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.training import train_candidate
from ares_engine.utils import atomic_write_json, sha256_file

pytestmark = pytest.mark.ml


def _training_config(tmp_root: Path):
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_root / "data"
    config.storage.artifacts = tmp_root / "artifacts"
    config.project.seed = 77
    config.model.lookback_bars = 12
    config.model.hidden_units = 8
    config.model.epochs = 1
    config.model.patience = 1
    config.features.ema_periods = [5, 9]
    config.features.rsi_period = 7
    config.features.bollinger_period = 10
    config.features.volatility_windows = [6]
    config.features.volume_z_window = 8
    config.labels.horizon_bars = 4
    config.validation.min_train_bars = 300
    config.validation.validation_bars = 60
    config.validation.step_bars = 60
    config.validation.max_folds = 1
    config.validation.purge_bars = 4
    config.gates.min_median_sharpe = -100.0
    config.gates.min_median_return = -10.0
    config.gates.max_worst_drawdown = 0.999999
    config.gates.max_median_turnover = 100000.0
    config.gates.min_total_trades = 0
    config.gates.require_positive_cost_stress = False
    return config


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle-fixture")
    config = _training_config(root)
    frame = make_synthetic_ohlcv(430, seed=42)
    bundle, _ = train_candidate(frame, config, bundle_name="fixture_bundle")
    probe = (
        np.random.default_rng(1).normal(size=(4, config.model.lookback_bars, 16)).astype("float32")
    )
    loaded = load_bundle(bundle)
    reference_predictions = predict_probabilities(loaded.model, probe)
    return {
        "path": bundle,
        "config": config,
        "probe": probe,
        "reference_predictions": reference_predictions,
        "scaler_mean": np.array(loaded.scaler.mean_, copy=True),
    }


def _clone(real_bundle, tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(real_bundle["path"], target)
    return target


def _rehash_manifest(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {
        item.name: {"sha256": sha256_file(item), "bytes": item.stat().st_size}
        for item in sorted(bundle.iterdir())
        if item.is_file() and item.name != "manifest.json"
    }
    atomic_write_json(manifest_path, manifest)


def test_valid_bundle_reloads_with_identical_predictions(real_bundle) -> None:
    loaded = load_bundle(real_bundle["path"])
    predictions = predict_probabilities(loaded.model, real_bundle["probe"])
    np.testing.assert_allclose(
        predictions,
        real_bundle["reference_predictions"],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(np.array(loaded.scaler.mean_), real_bundle["scaler_mean"])


@pytest.mark.parametrize(
    "filename",
    [
        "model.keras",
        "scaler.joblib",
        "feature_spec.json",
        "config.json",
        "metrics.json",
        "provenance.json",
    ],
)
def test_any_modified_file_is_detected(real_bundle, tmp_path, filename) -> None:
    bundle = _clone(real_bundle, tmp_path, f"tampered-{filename.replace('.', '-')}")
    target = bundle / filename
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    target.write_bytes(bytes(payload))
    with pytest.raises(BundleIntegrityError, match="mismatch|changed"):
        load_bundle(bundle)


def test_missing_file_fails_closed(real_bundle, tmp_path) -> None:
    bundle = _clone(real_bundle, tmp_path, "missing-file")
    (bundle / "provenance.json").unlink()
    with pytest.raises(BundleIntegrityError, match="missing"):
        load_bundle(bundle)


def test_extra_unlisted_file_fails_closed(real_bundle, tmp_path) -> None:
    bundle = _clone(real_bundle, tmp_path, "extra-file")
    (bundle / "payload.bin").write_bytes(b"surprise")
    with pytest.raises(BundleIntegrityError, match="unlisted"):
        load_bundle(bundle)


def test_nested_directory_fails_closed(real_bundle, tmp_path) -> None:
    bundle = _clone(real_bundle, tmp_path, "nested")
    (bundle / "sub").mkdir()
    with pytest.raises(BundleIntegrityError, match="nested or special"):
        load_bundle(bundle)


@pytest.mark.parametrize("name", ["/etc/passwd", "../escape.json", "a/b.json"])
def test_absolute_and_traversal_manifest_names_fail_closed(real_bundle, tmp_path, name) -> None:
    bundle = _clone(real_bundle, tmp_path, f"badname-{abs(hash(name))}")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = {"sha256": "0" * 64, "bytes": 1}
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(BundleIntegrityError, match="Invalid manifest file name|missing"):
        load_bundle(bundle)


def test_symlinked_and_broken_symlink_entries_fail_closed(real_bundle, tmp_path) -> None:
    bundle = _clone(real_bundle, tmp_path, "symlinked")
    real_metrics = bundle / "metrics.json"
    moved = tmp_path / "outside-metrics.json"
    shutil.move(real_metrics, moved)
    os.symlink(moved, real_metrics)
    with pytest.raises(BundleIntegrityError, match="symbolic"):
        load_bundle(bundle)
    moved.unlink()  # break the link
    with pytest.raises(BundleIntegrityError, match="symbolic"):
        load_bundle(bundle)


def test_symlinked_bundle_root_fails_closed(real_bundle, tmp_path) -> None:
    link = tmp_path / "root-link"
    os.symlink(real_bundle["path"], link)
    with pytest.raises(BundleIntegrityError, match="symbolic"):
        load_bundle(link)


def test_hardlink_post_verification_modification_is_caught_on_next_load(
    real_bundle, tmp_path
) -> None:
    """A hard link passes hashing while contents match, but any later edit through
    the second path changes the canonical bytes and must fail the next load."""
    bundle = _clone(real_bundle, tmp_path, "hardlinked")
    attacker_path = tmp_path / "attacker-metrics.json"
    metrics = bundle / "metrics.json"
    original = metrics.read_bytes()
    metrics.unlink()
    attacker_path.write_bytes(original)
    os.link(attacker_path, metrics)
    _rehash_manifest(bundle)
    verify_bundle(bundle)  # identical content still verifies
    attacker_path.write_bytes(original.replace(b"passed", b"PASSED"))
    with pytest.raises(BundleIntegrityError, match="mismatch|changed"):
        load_bundle(bundle)


def test_coherent_tamper_with_rehashed_manifest_is_caught_by_metadata_checks(
    real_bundle, tmp_path
) -> None:
    bundle = _clone(real_bundle, tmp_path, "coherent-tamper")
    spec_path = bundle / "feature_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["lookback_bars"] = spec["lookback_bars"] + 1
    atomic_write_json(spec_path, spec)
    _rehash_manifest(bundle)
    verify_bundle(bundle)  # self-consistent, so hashing alone cannot catch it
    with pytest.raises(BundleIntegrityError, match="disagrees with config"):
        load_bundle(bundle)


def test_reordered_feature_columns_are_refused_at_inference(real_bundle, tmp_path) -> None:
    from ares_engine.live import generate_paper_signal

    bundle = _clone(real_bundle, tmp_path, "reordered-columns")
    spec_path = bundle / "feature_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["columns"] = list(reversed(spec["columns"]))
    atomic_write_json(spec_path, spec)
    _rehash_manifest(bundle)
    frame = make_synthetic_ohlcv(430, seed=42, exchange="coinbase")
    secondary = frame.copy()
    secondary["exchange"] = "kraken"
    import pandas as pd

    as_of = pd.Timestamp(frame["timestamp"].max()) + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="Feature specification mismatch"):
        generate_paper_signal(bundle, frame, secondary_ohlcv=secondary, as_of=as_of)


def test_timeframe_and_symbol_tampering_fail_metadata_checks(real_bundle, tmp_path) -> None:
    for field, value in [("timeframe", "5m"), ("symbol", "BTC/USD")]:
        bundle = _clone(real_bundle, tmp_path, f"meta-{field}")
        spec_path = bundle / "feature_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec[field] = value
        atomic_write_json(spec_path, spec)
        _rehash_manifest(bundle)
        with pytest.raises(BundleIntegrityError, match="disagrees with config"):
            load_bundle(bundle)


def test_modification_between_verify_and_load_fails_closed(
    real_bundle, tmp_path, monkeypatch
) -> None:
    bundle = _clone(real_bundle, tmp_path, "toctou")
    real_verify = bundles_module.verify_bundle

    def tampering_verify(path):
        manifest = real_verify(path)
        scaler = Path(path) / "scaler.joblib"
        payload = bytearray(scaler.read_bytes())
        payload[0] ^= 0xFF
        scaler.write_bytes(bytes(payload))  # attacker wins the race after verification
        return manifest

    monkeypatch.setattr(bundles_module, "verify_bundle", tampering_verify)
    with pytest.raises(BundleIntegrityError, match="changed between verification and load"):
        load_bundle(bundle)


def test_existing_bundle_is_never_mutated_by_export_collision(real_bundle) -> None:
    config = real_bundle["config"]
    before = sha256_file(real_bundle["path"] / "manifest.json")
    with pytest.raises(FileExistsError):
        export_bundle(
            model=object(),
            scaler=object(),
            feature_columns=["close"],
            config=config,
            metrics={},
            provenance={},
            artifacts_root=real_bundle["path"].parent,
            name=real_bundle["path"].name,
        )
    assert sha256_file(real_bundle["path"] / "manifest.json") == before
