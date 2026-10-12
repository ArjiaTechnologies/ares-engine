"""Synthetic four-slot lifecycle; patched clock/evaluator, no model or network."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import ares_engine.paper_collection as collection
from ares_engine.exceptions import DataQualityError
from ares_engine.paper_log import record_paper_signal, utc_timestamp
from ares_engine.paper_report import paper_report


def synthetic_evaluator(bundle, frame, *, log_path, as_of, expected, **kwargs):
    hour = pd.Timestamp(expected.event.timestamp).hour
    if hour == 13:
        raise DataQualityError("constructed rejection")
    if hour == 14:
        raise RuntimeError("constructed interruption before acknowledgement")
    receipt = record_paper_signal(
        log_path,
        event=expected.event.model_dump(),
        evidence={
            "manifest_sha256": expected.manifest_sha256,
            "config_sha256": expected.config_sha256,
            "primary_sha256": "3" * 64,
            "secondary_sha256": {},
            "feature_window_sha256": "5" * 64,
            "scaled_window_sha256": "6" * 64,
            "probability": 0.7,
            "signal": 1,
            "close": 100.0,
            "data_quality_passed": True,
            "cross_venue_p95_bps": {},
            "cross_venue_latest_bps": {},
        },
        observation={
            "recorded_at": utc_timestamp(as_of),
            "as_of": utc_timestamp(as_of),
            "bundle": "constructed-demo",
            "staleness_bars": 1.0,
            "secondary_staleness_bars": {},
        },
    )
    return SimpleNamespace(recording=receipt)


def main() -> None:
    plan = {
        "schema_version": "ares-paper-plan-v1",
        "exchange": "coinbase",
        "symbol": "ETH/USD",
        "timeframe": "1h",
        "start": "2026-10-01T12:00:00+00:00",
        "end": "2026-10-01T16:00:00+00:00",
        "horizon_bars": 1,
        "manifest_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "max_recording_delay_seconds": 300,
    }
    with TemporaryDirectory(prefix="ares-collection-demo-") as directory:
        path = Path(directory) / "experiment.json"
        clock = [pd.Timestamp(plan["start"])]
        with (
            patch.object(collection, "utc_now", lambda: clock[0]),
            patch.object(collection, "read_market", lambda path: pd.DataFrame()),
            patch.object(collection, "generate_paper_signal", synthetic_evaluator),
        ):
            registration = collection.register_collection(path, plan)
            digest = registration["registration_sha256"]
            for hour in (12, 13, 14):
                clock[0] = pd.Timestamp(f"2026-10-01T{hour + 1}:00:00+00:00")
                try:
                    collection.collect_paper_slot(
                        path,
                        registration_sha256=digest,
                        candle=f"2026-10-01T{hour}:00:00+00:00",
                        bundle=Path("synthetic"),
                        primary=Path("synthetic"),
                    )
                except RuntimeError:
                    if hour != 14:
                        raise
        before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
        status = collection.collection_status(
            path, registration_sha256=digest, as_of="2026-10-01T17:00:00+00:00"
        )
        prices = [
            {
                "timestamp": "2026-10-01T13:00:00+00:00",
                "exchange": "coinbase",
                "symbol": "ETH/USD",
                "timeframe": "1h",
                "close": 110.0,
            }
        ]
        report = paper_report(
            collection.collection_signals_path(path),
            plan=status["registration"]["plan"],
            prices=prices,
            as_of=status["as_of"],
        )
        assert status["counts"] == {
            "NOT_DUE": 0,
            "NOT_STARTED": 1,
            "STARTED_UNKNOWN": 1,
            "FAILED": 1,
            "SUCCEEDED": 1,
        }
        assert report["coverage"]["timely_matching_coverage"] == 0.25
        assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
        print(
            json.dumps(
                {
                    "synthetic_demo_only": True,
                    "clock_and_evaluator_patched": True,
                    "inspection_preserved_bytes": True,
                    "attempt_report": status,
                    "price_report": report,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
