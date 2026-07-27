"""Atomic, hash-verified model bundles for training/inference parity."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from .config import AresConfig
from .exceptions import BundleIntegrityError
from .utils import atomic_write_json, sha256_file

BUNDLE_FILES = {
    "model.keras",
    "scaler.joblib",
    "feature_spec.json",
    "config.json",
    "metrics.json",
    "provenance.json",
    "manifest.json",
}


@dataclass(slots=True)
class LoadedBundle:
    path: Path
    model: Any
    scaler: Any
    feature_spec: dict[str, Any]
    config: AresConfig
    metrics: dict[str, Any]
    provenance: dict[str, Any]


def _manifest(directory: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "format_version": 1,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "files": files,
    }


def export_bundle(
    *,
    model: Any,
    scaler: Any,
    feature_columns: list[str],
    config: AresConfig,
    metrics: dict[str, Any],
    provenance: dict[str, Any],
    artifacts_root: Path,
    name: str | None = None,
) -> Path:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    bundle_name = name or f"ares_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    target = artifacts_root / bundle_name
    if target.exists():
        raise FileExistsError(f"Bundle already exists: {target}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{bundle_name}.", dir=artifacts_root))
    try:
        model.save(temp_dir / "model.keras")
        joblib.dump(scaler, temp_dir / "scaler.joblib")
        atomic_write_json(
            temp_dir / "feature_spec.json",
            {
                "columns": feature_columns,
                "lookback_bars": config.model.lookback_bars,
                "timeframe": config.data.timeframe,
                "symbol": config.data.symbol,
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
            },
        )
        atomic_write_json(temp_dir / "config.json", config.model_dump(mode="json"))
        atomic_write_json(temp_dir / "metrics.json", metrics)
        atomic_write_json(temp_dir / "provenance.json", provenance)
        atomic_write_json(temp_dir / "manifest.json", _manifest(temp_dir))
        os.replace(temp_dir, target)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return target


def verify_bundle(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise BundleIntegrityError("Bundle directory must not be a symbolic link")
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise BundleIntegrityError(f"Missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise BundleIntegrityError("Bundle manifest must be a JSON object")
    if manifest.get("format_version") != 1:
        raise BundleIntegrityError("Unsupported bundle manifest format version")
    manifest_files = manifest.get("files", {})
    if not isinstance(manifest_files, dict):
        raise BundleIntegrityError("Bundle manifest has an invalid files section")
    entries = list(path.iterdir())
    symlinks = sorted(item.name for item in entries if item.is_symlink())
    if symlinks:
        raise BundleIntegrityError(f"Bundle contains symbolic links: {symlinks}")
    nested_entries = sorted(item.name for item in entries if not item.is_file())
    if nested_entries:
        raise BundleIntegrityError(f"Bundle contains nested or special entries: {nested_entries}")
    actual_files = {item.name for item in entries if item.is_file()}
    listed_files = set(manifest_files)
    unlisted = actual_files.difference(listed_files | {"manifest.json"})
    if unlisted:
        raise BundleIntegrityError(f"Bundle contains unlisted files: {sorted(unlisted)}")
    missing_listed = listed_files.difference(actual_files)
    if missing_listed:
        raise BundleIntegrityError(f"Bundle manifest lists missing files: {sorted(missing_listed)}")

    for name, expected in manifest_files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise BundleIntegrityError(f"Invalid manifest file name: {name!r}")
        if not isinstance(expected, dict):
            raise BundleIntegrityError(f"Invalid manifest record for bundle file: {name}")
        expected_size = expected.get("bytes")
        expected_hash = expected.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise BundleIntegrityError(f"Invalid byte count for bundle file: {name}")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash.lower())
        ):
            raise BundleIntegrityError(f"Invalid SHA-256 digest for bundle file: {name}")
        file_path = path / name
        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            raise BundleIntegrityError(f"Size mismatch for bundle file: {name}")
        actual = sha256_file(file_path)
        if actual != expected_hash:
            raise BundleIntegrityError(f"Hash mismatch for bundle file: {name}")
    missing_required = BUNDLE_FILES.difference(actual_files)
    if missing_required:
        raise BundleIntegrityError(f"Bundle is incomplete: missing {sorted(missing_required)}")
    return manifest


def validate_bundle_metadata(feature_spec: dict[str, Any], config: AresConfig) -> None:
    columns = feature_spec.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) and column for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise BundleIntegrityError("Bundle feature columns must be a non-empty unique string list")

    expected_pairs = {
        "lookback_bars": config.model.lookback_bars,
        "timeframe": config.data.timeframe,
        "symbol": config.data.symbol,
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
    mismatches = [
        name for name, expected in expected_pairs.items() if feature_spec.get(name) != expected
    ]
    if mismatches:
        raise BundleIntegrityError(
            f"Bundle feature specification disagrees with config fields: {sorted(mismatches)}"
        )


def load_bundle(path: str | Path) -> LoadedBundle:
    bundle_path = Path(path)
    verify_bundle(bundle_path)
    from .models import keras_api

    keras = keras_api()
    with (bundle_path / "feature_spec.json").open(encoding="utf-8") as handle:
        feature_spec = json.load(handle)
    with (bundle_path / "config.json").open(encoding="utf-8") as handle:
        config = AresConfig.model_validate(json.load(handle))
    with (bundle_path / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (bundle_path / "provenance.json").open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    validate_bundle_metadata(feature_spec, config)
    return LoadedBundle(
        path=bundle_path,
        model=keras.models.load_model(bundle_path / "model.keras"),
        scaler=joblib.load(bundle_path / "scaler.joblib"),
        feature_spec=feature_spec,
        config=config,
        metrics=metrics,
        provenance=provenance,
    )
