"""Synthetic local capture failure: real Parquet/bytes, no model or network."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import ares_engine.paper_collection as collection
from ares_engine.paper_inputs import snapshot_path


def main() -> None:
    plan = dict(
        schema_version="ares-paper-plan-v1",
        exchange="coinbase",
        symbol="ETH/USD",
        timeframe="1h",
        start="2030-01-01T12:00:00+00:00",
        end="2030-01-01T14:00:00+00:00",
        horizon_bars=1,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
        max_recording_delay_seconds=300,
    )
    with TemporaryDirectory(prefix="ares-input-demo-") as directory:
        root = Path(directory)
        path, primary, secondary = (
            root / "experiment.json",
            root / "primary.parquet",
            root / "secondary.parquet",
        )
        pd.DataFrame(
            {
                "timestamp": ["2030-01-01T12:00:00Z", "2030-01-01T11:00:00Z"],
                "exchange": ["coinbase"] * 2,
                "symbol": ["ETH/USD"] * 2,
                "timeframe": ["1h"] * 2,
                "open": [100.0, 99.0],
                "high": [101.0, 100.0],
                "low": [98.0, 98.0],
                "close": [100.0, 99.0],
                "volume": [5.0, 6.0],
                "ignored_note": ["fixture"] * 2,
            }
        ).to_parquet(primary, index=False)
        secondary.write_bytes(b"constructed malformed Parquet")
        originals = [primary.read_bytes(), secondary.read_bytes()]
        with patch.object(collection, "utc_now", lambda: pd.Timestamp(plan["start"])):
            registration = collection.register_collection(path, plan)
        clock = pd.Timestamp("2030-01-01T13:00:00+00:00")

        def tick():
            nonlocal clock
            value = clock
            clock += pd.Timedelta(seconds=1)
            return value

        kwargs = dict(
            registration_sha256=registration["registration_sha256"],
            candle=plan["start"],
            bundle=root / "unused",
            primary=primary,
            secondary={"kraken": secondary},
            capture_inputs=True,
        )
        with patch.object(collection, "utc_now", tick):
            result = collection.collect_paper_slot(path, **kwargs)
        attempt = result["attempt"]
        sources = attempt["input_capture"]["sources"]
        assert [s["status"] for s in sources] == ["READY", "PARSE_REJECTED"]
        assert attempt["failure_code"] == "LOCAL_INPUT_REJECTED"
        for i, raw in enumerate(originals):
            assert snapshot_path(path, attempt["attempt_id"], i).read_bytes() == raw
            assert sources[i]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
        # Independent oracle: chronological UTC rows, numeric OHLCV, no extra column.
        expected_frame = {
            "columns": [
                "timestamp",
                "exchange",
                "symbol",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
            "rows": [
                [
                    "2030-01-01T11:00:00+00:00",
                    "coinbase",
                    "ETH/USD",
                    "1h",
                    99.0,
                    100.0,
                    98.0,
                    99.0,
                    6.0,
                ],
                [
                    "2030-01-01T12:00:00+00:00",
                    "coinbase",
                    "ETH/USD",
                    "1h",
                    100.0,
                    101.0,
                    98.0,
                    100.0,
                    5.0,
                ],
            ],
        }
        expected_digest = hashlib.sha256(
            json.dumps(expected_frame, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert sources[0]["frame_sha256"] == expected_digest
        primary.write_bytes(b"changed original")
        secondary.unlink()
        before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        retry = collection.collect_paper_slot(path, **kwargs)
        report = collection.collection_status(
            path,
            registration_sha256=registration["registration_sha256"],
            as_of="2030-01-01T15:00:00+00:00",
        )
        assert retry["dispatch"] == "REPLAY_NO_DISPATCH"
        assert before == {
            str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
        print(
            json.dumps(
                {
                    "synthetic_local_files_only": True,
                    "clock_patched": True,
                    "original_bytes_preserved": True,
                    "normalized_frame_matches_hand_oracle": True,
                    "originals_replaced_or_deleted": True,
                    "retry": retry["dispatch"],
                    "inspection_and_replay_preserved_bytes": True,
                    "attempt": attempt,
                    "counts": report["counts"],
                    "independent_forward_performance_established": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
