"""Hand-calculated synthetic receipt coverage; no market collection or model."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ares_engine.paper_log import record_paper_signal
from ares_engine.paper_report import paper_report


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
    with TemporaryDirectory(prefix="ares-paper-report-demo-") as directory:
        path = Path(directory) / "signals.jsonl"
        for hour, signal in ((12, 1), (14, -1)):
            record_paper_signal(
                path,
                event={
                    "exchange": "coinbase",
                    "symbol": "ETH/USD",
                    "timeframe": "1h",
                    "timestamp": f"2026-10-01T{hour}:00:00+00:00",
                },
                evidence={
                    "manifest_sha256": "1" * 64,
                    "config_sha256": "2" * 64,
                    "primary_sha256": "3" * 64,
                    "secondary_sha256": {"kraken": "4" * 64},
                    "feature_window_sha256": "5" * 64,
                    "scaled_window_sha256": "6" * 64,
                    "probability": 0.7 if signal == 1 else 0.3,
                    "signal": signal,
                    "close": 100.0,
                    "data_quality_passed": True,
                    "cross_venue_p95_bps": {"kraken": 0.0},
                    "cross_venue_latest_bps": {"kraken": 0.0},
                },
                observation={
                    "recorded_at": f"2026-10-01T{hour + 1}:00:00+00:00",
                    "as_of": f"2026-10-01T{hour + 1}:00:00+00:00",
                    "bundle": "constructed-synthetic-demo",
                    "staleness_bars": 1.0,
                    "secondary_staleness_bars": {"kraken": 1.0},
                },
            )
        before = path.read_bytes()
        prices = [
            {
                "timestamp": f"2026-10-01T{hour}:00:00+00:00",
                "exchange": "coinbase",
                "symbol": "ETH/USD",
                "timeframe": "1h",
                "close": close,
            }
            for hour, close in ((13, 110.0), (15, 90.0))
        ]
        actual = paper_report(path, plan=plan, prices=prices, as_of="2026-10-01T17:00:00+00:00")
        assert actual["coverage"]["timely_matching_coverage"] == 0.5
        assert actual["outcomes"]["available"] == 2
        assert path.read_bytes() == before
        print(
            json.dumps(
                {"synthetic_demo_only": True, "journal_unchanged": True, "report": actual}, indent=2
            )
        )


if __name__ == "__main__":
    main()
