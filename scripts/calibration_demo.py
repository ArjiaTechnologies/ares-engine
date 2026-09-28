"""Deterministic offline diagnostic examples; probabilities are fixtures, not a model."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ares_engine.calibration import (
    array_digest,
    probability_calibration_report,
    threshold_stability_report,
)
from ares_engine.config import BacktestConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insufficient", action="store_true")
    args = parser.parse_args()
    labels = np.tile([0, 1], 30)
    probabilities = np.tile([0.2, 0.8], 30)
    returns = np.tile([0.002, -0.001, -0.002, 0.003, 0.001, -0.002], 10)
    timestamps = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    if args.insufficient:
        timestamps, returns, probabilities = timestamps[:1], returns[:1], probabilities[:1]
    try:
        stability = threshold_stability_report(
            [(timestamps, returns, probabilities)],
            BacktestConfig(long_threshold=0.52, short_threshold=0.48),
            timeframe="1h",
            stress_cost_multiplier=2,
        )
    except ValueError as exc:
        print(json.dumps({"case": "insufficient", "status": "rejected", "reason": str(exc)}))
        raise SystemExit(2) from exc
    calibration = probability_calibration_report(
        labels, probabilities, reference_probabilities=np.full(60, 0.5)
    )
    print(
        json.dumps(
            {
                "case": "synthetic_offline_fixture",
                "claim": "arithmetic_and_boundary_demo_not_market_or_trained_model_evidence",
                "labels_sha256": array_digest(labels),
                "calibration": calibration,
                "threshold_stability": stability,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
