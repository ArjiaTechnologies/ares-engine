"""Offline persistence demo with explicitly constructed synthetic evidence."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ares_engine.exceptions import AresError
from ares_engine.paper_log import evidence_digest, record_paper_signal


def main() -> None:
    event = {
        "exchange": "coinbase",
        "symbol": "ETH/USD",
        "timeframe": "1h",
        "timestamp": "2026-10-01T12:00:00+00:00",
    }
    evidence = {
        key: evidence_digest({"synthetic_demo_only": key})
        for key in (
            "manifest_sha256",
            "config_sha256",
            "primary_sha256",
            "feature_window_sha256",
            "scaled_window_sha256",
        )
    }
    evidence.update(
        secondary_sha256={"kraken": evidence_digest("synthetic secondary")},
        probability=0.7,
        signal=1,
        close=100.0,
        data_quality_passed=True,
        cross_venue_p95_bps={"kraken": 0.0},
        cross_venue_latest_bps={"kraken": 0.0},
    )
    observation = {
        "recorded_at": "2026-10-01T13:00:00+00:00",
        "as_of": "2026-10-01T13:00:00+00:00",
        "bundle": "synthetic-demo-only",
        "staleness_bars": 1.0,
        "secondary_staleness_bars": {"kraken": 1.0},
    }
    with TemporaryDirectory(prefix="ares-paper-demo-") as directory:
        path = Path(directory) / "signals.jsonl"
        first = record_paper_signal(path, event=event, evidence=evidence, observation=observation)
        before = path.read_bytes()
        replay = record_paper_signal(path, event=event, evidence=evidence, observation=observation)
        conflicting = {**evidence, "probability": 0.71}
        try:
            record_paper_signal(path, event=event, evidence=conflicting, observation=observation)
        except AresError as exc:
            conflict = str(exc)
        else:
            raise AssertionError("Expected conflicting replay to fail")
        assert path.read_bytes() == before
        print(
            json.dumps(
                {
                    "synthetic_persistence_demo_only": True,
                    "first_status": first.status,
                    "retry_status": replay.status,
                    "records": len(before.splitlines()),
                    "original_receipt_returned": replay.record == first.record,
                    "bytes_unchanged": path.read_bytes() == before,
                    "conflict": conflict,
                    "event_id": first.record["event_id"],
                    "limitation": "No forward performance or real model evidence",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
