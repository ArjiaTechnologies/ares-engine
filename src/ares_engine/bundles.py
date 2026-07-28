"""Atomic, hash-verified model bundles for training/inference parity."""

from __future__ import annotations

import hashlib
import json
import math
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

# Bounds are deliberately generous for the compact models ARES exports, while
# preventing a manifest from authorizing unbounded reads/deserialization.
BUNDLE_FILE_LIMITS = {
    "model.keras": 1_073_741_824,
    "scaler.joblib": 67_108_864,
    "feature_spec.json": 8_388_608,
    "config.json": 8_388_608,
    "metrics.json": 8_388_608,
    "provenance.json": 8_388_608,
    "manifest.json": 8_388_608,
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


def _require_finite_json(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BundleIntegrityError(f"Bundle JSON contains a non-finite value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, location=f"{location}[{index}]")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key: {key}")
            payload[key] = value
        return payload

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise BundleIntegrityError(f"Bundle {label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise BundleIntegrityError(f"Bundle {label} must be a JSON object")
    _require_finite_json(payload, location=label)
    return payload


def _single_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise BundleIntegrityError(f"Unable to inspect bundle {label}") from exc
    if not path.is_file() or path.is_symlink():
        raise BundleIntegrityError(f"Bundle {label} must be a regular, non-symbolic file")
    if stat.st_nlink != 1:
        raise BundleIntegrityError(f"Bundle {label} must not be hard-linked")
    limit = BUNDLE_FILE_LIMITS.get(path.name)
    if limit is None or stat.st_size > limit:
        raise BundleIntegrityError(f"Bundle {label} exceeds its allowed size")
    return stat


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
    _single_regular_file(manifest_path, label="manifest")
    manifest = _load_json_object(manifest_path, label="manifest")
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
    expected_payloads = BUNDLE_FILES.difference({"manifest.json"})
    if listed_files != expected_payloads:
        raise BundleIntegrityError(
            "Bundle manifest payload set is not exact: "
            f"expected {sorted(expected_payloads)}, got {sorted(listed_files)}"
        )
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
        actual_size = _single_regular_file(file_path, label=name).st_size
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


def validate_runtime_shapes(
    model: Any, scaler: Any, feature_spec: dict[str, Any], config: AresConfig
) -> None:
    feature_count = len(feature_spec["columns"])
    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features is None or int(scaler_features) != feature_count:
        raise BundleIntegrityError(
            "Bundle scaler feature count disagrees with feature specification"
        )

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise BundleIntegrityError("Bundle model must expose exactly one input")
        input_shape = input_shape[0]
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise BundleIntegrityError("Bundle model input shape is invalid")
    if input_shape[-2:] != (config.model.lookback_bars, feature_count):
        raise BundleIntegrityError("Bundle model input shape disagrees with feature specification")

    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, list):
        if len(output_shape) != 1:
            raise BundleIntegrityError("Bundle model must expose exactly one output")
        output_shape = output_shape[0]
    if not isinstance(output_shape, tuple) or len(output_shape) != 2 or output_shape[-1] != 1:
        raise BundleIntegrityError("Bundle model output shape must be a single probability")


def load_bundle(path: str | Path) -> LoadedBundle:
    """Verify and load a bundle without a verify-then-load race window.

    After manifest verification, every file is copied into a private snapshot
    directory and re-hashed against the verified manifest before anything is
    deserialized. A concurrent writer that modifies the original bundle between
    verification and deserialization can therefore only cause a
    ``BundleIntegrityError``; it can never get unverified bytes loaded.
    """
    bundle_path = Path(path)
    manifest = verify_bundle(bundle_path)
    manifest_files = manifest["files"]
    from .models import keras_api

    keras = keras_api()
    with tempfile.TemporaryDirectory(prefix=".ares-bundle-load.") as snapshot_name:
        snapshot = Path(snapshot_name)
        for name, expected in manifest_files.items():
            payload = (bundle_path / name).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != expected["bytes"] or digest != expected["sha256"]:
                raise BundleIntegrityError(
                    f"Bundle file changed between verification and load: {name}"
                )
            (snapshot / name).write_bytes(payload)
        feature_spec = _load_json_object(snapshot / "feature_spec.json", label="feature_spec")
        config_payload = _load_json_object(snapshot / "config.json", label="config")
        config = AresConfig.model_validate(config_payload)
        metrics = _load_json_object(snapshot / "metrics.json", label="metrics")
        provenance = _load_json_object(snapshot / "provenance.json", label="provenance")
        validate_bundle_metadata(feature_spec, config)
        model = keras.models.load_model(snapshot / "model.keras")
        scaler = joblib.load(snapshot / "scaler.joblib")
        validate_runtime_shapes(model, scaler, feature_spec, config)
    return LoadedBundle(
        path=bundle_path,
        model=model,
        scaler=scaler,
        feature_spec=feature_spec,
        config=config,
        metrics=metrics,
        provenance=provenance,
    )
