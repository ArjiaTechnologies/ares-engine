"""Final candidate training and complete bundle export."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bundles import export_bundle
from .config import AresConfig
from .dataset import fit_scaler, transform_sequences
from .models import backend_name, build_model, fit_model
from .utils import sha256_file, utc_now
from .validation import prepare_dataset, run_walk_forward


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def train_candidate(
    ohlcv: pd.DataFrame,
    config: AresConfig,
    *,
    source_path: Path | None = None,
    source_identity_sha256: str | None = None,
    bundle_name: str | None = None,
    repository_root: Path | None = None,
    verbose: int = 0,
) -> tuple[Path, dict[str, Any]]:
    validation = run_walk_forward(ohlcv, config, verbose=verbose)
    if not validation.passed:
        failed = [name for name, passed in validation.gates.items() if not passed]
        raise ValueError(f"Candidate failed hard validation gates: {failed}")

    _, dataset = prepare_dataset(ohlcv, config)
    train_indices = dataset.directional_mask.nonzero()[0]
    labeled_indices = np.flatnonzero(np.isfinite(dataset.labels))
    scaler = fit_scaler(dataset.X[labeled_indices])
    X_train = transform_sequences(scaler, dataset.X[train_indices])
    y_train = dataset.y_binary[train_indices].astype("float32")
    model = build_model(dataset.X.shape[1:], config.model, seed=config.project.seed)
    fit_model(model, X_train, y_train, config.model, verbose=verbose)

    provenance: dict[str, Any] = {
        "trained_at": utc_now(),
        "data_start": dataset.timestamps.min(),
        "data_end": dataset.timestamps[labeled_indices].max(),
        "latest_feature_timestamp": dataset.timestamps.max(),
        "samples": len(dataset.X),
        "labeled_samples": len(labeled_indices),
        "directional_samples": len(train_indices),
        "git_commit": _git_commit(repository_root) if repository_root else None,
        "keras_backend": backend_name(),
    }
    if source_path and source_path.exists():
        provenance["source_data"] = {
            "path": source_path,
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        }
    if source_identity_sha256 is not None:
        if len(source_identity_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_identity_sha256.lower()
        ):
            raise ValueError("source_identity_sha256 must be a SHA-256 digest")
        provenance["research_dataset_identity_sha256"] = source_identity_sha256

    bundle = export_bundle(
        model=model,
        scaler=scaler,
        feature_columns=dataset.feature_columns,
        config=config,
        metrics=validation.to_dict(),
        provenance=provenance,
        artifacts_root=config.storage.artifacts,
        name=bundle_name,
    )
    return bundle, validation.to_dict()
