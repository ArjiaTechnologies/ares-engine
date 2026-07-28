"""Subprocess-crash, reader, cleanup, and process-lock tests for generations."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import ares_engine.data.ingest as ingest_module
from ares_engine.config import load_config
from ares_engine.data.ingest import ingest_market_data, recover_pending_commit
from ares_engine.data.providers import CCXTOHLCVProvider
from ares_engine.data.storage import (
    cleanup_abandoned_generations,
    create_generation,
    current_generation,
    generation_duckdb_path,
    generation_root,
    market_path,
    new_generation_id,
    publish_generation,
    read_market,
)
from ares_engine.exceptions import AresError
from ares_engine.synthetic import make_synthetic_ohlcv


class FakeProvider:
    def __init__(self, exchange_id: str, frame: pd.DataFrame) -> None:
        self.exchange_id = exchange_id
        self.frame = frame

    def fetch_range(self, symbol, timeframe, since, until, *, limit, max_pages, retries):
        del symbol, timeframe, limit, max_pages, retries
        output = self.frame[self.frame["timestamp"] >= pd.Timestamp(since)].copy()
        if until is not None:
            output = output[output["timestamp"] < pd.Timestamp(until)]
        return output


def _config(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    config.storage.artifacts = tmp_path / "artifacts"
    config.data.since = pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime()
    config.data.drop_open_candle = False
    config.data.fail_on_quality = True
    config.data.max_gap_count = 0
    config.data.max_cross_venue_p95_bps = 20.0
    config.data.min_cross_venue_overlap = 50
    return config


def _frames(rows: int, seed: int = 11) -> dict[str, pd.DataFrame]:
    primary = make_synthetic_ohlcv(rows, seed=seed, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    secondary["close"] *= 1.0001
    secondary["high"] = secondary[["high", "close"]].max(axis=1)
    return {"coinbase": primary, "kraken": secondary}


def _providers(frames: dict[str, pd.DataFrame]) -> dict[str, FakeProvider]:
    return {name: FakeProvider(name, frame) for name, frame in frames.items()}


def _activate(root: Path, frames: dict[str, pd.DataFrame]) -> str:
    generation = create_generation(
        root,
        frames=frames,
        symbol="ETH/USD",
        timeframe="1h",
        quality_documents={"latest": {"passed": True}},
        metadata={"test": True},
    )
    publish_generation(root, generation)
    return generation


@pytest.mark.parametrize(
    "crash_stage",
    [
        "generation_created",
        "venue_written:coinbase",
        "venue_written:kraken",
        "quality_written",
        "duckdb_written",
        "manifest_written",
        "before_pointer",
        "after_pointer",
    ],
)
def test_real_subprocess_crash_never_exposes_mixed_generation(
    tmp_path: Path, crash_stage: str
) -> None:
    root = tmp_path / "data"
    old_generation = _activate(root, _frames(60, seed=1))
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path
        from ares_engine.data.storage import create_generation, publish_generation
        from ares_engine.synthetic import make_synthetic_ohlcv

        root = Path(sys.argv[1])
        crash_stage = sys.argv[2]
        coinbase = make_synthetic_ohlcv(90, seed=2, exchange="coinbase")
        kraken = coinbase.copy()
        kraken["exchange"] = "kraken"
        kraken["close"] *= 1.0001
        kraken["high"] = kraken[["high", "close"]].max(axis=1)

        def crash(event):
            if event == crash_stage:
                os._exit(91)

        generation = create_generation(
            root,
            frames={"coinbase": coinbase, "kraken": kraken},
            symbol="ETH/USD",
            timeframe="1h",
            quality_documents={"latest": {"passed": True}},
            metadata={"subprocess": True},
            fault_hook=crash,
        )
        publish_generation(root, generation, fault_hook=crash)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), crash_stage],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 91

    captured = current_generation(root)
    assert captured is not None
    expected_rows = 90 if crash_stage == "after_pointer" else 60
    if crash_stage != "after_pointer":
        assert captured == old_generation
    counts = [
        len(read_market(market_path(root, venue, "ETH/USD", "1h", generation=captured)))
        for venue in ["coinbase", "kraken"]
    ]
    assert counts == [expected_rows, expected_rows]


def test_failure_before_pointer_leaves_old_generation_active(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    first_frames = _frames(120, seed=3)
    config.data.until = (
        first_frames["coinbase"]["timestamp"].max() + pd.Timedelta(hours=1)
    ).to_pydatetime()
    first = ingest_market_data(config, providers=_providers(first_frames))
    old_generation = first.generation

    expanded = _frames(150, seed=3)
    config.data.until = (
        expanded["coinbase"]["timestamp"].max() + pd.Timedelta(hours=1)
    ).to_pydatetime()
    monkeypatch.setattr(
        ingest_module,
        "publish_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crash before pointer")),
    )
    with pytest.raises(OSError, match="crash before pointer"):
        ingest_market_data(config, providers=_providers(expanded))

    assert current_generation(config.storage.root) == old_generation
    assert [
        len(
            read_market(
                market_path(config.storage.root, venue, "ETH/USD", "1h", generation=old_generation)
            )
        )
        for venue in ["coinbase", "kraken"]
    ] == [120, 120]


def test_concurrent_readers_capture_one_complete_generation(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _activate(root, _frames(60, seed=4))
    new_generation = create_generation(
        root,
        frames=_frames(90, seed=5),
        symbol="ETH/USD",
        timeframe="1h",
        quality_documents={"latest": {"passed": True}},
        metadata={"test": True},
    )
    observed: list[tuple[int, int]] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            captured = current_generation(root)
            assert captured is not None
            observed.append(
                tuple(
                    len(read_market(market_path(root, venue, "ETH/USD", "1h", generation=captured)))
                    for venue in ["coinbase", "kraken"]
                )
            )

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        publish_generation(root, new_generation)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert observed
    assert set(observed).issubset({(60, 60), (90, 90)})


def test_generation_duckdb_is_bound_to_captured_files(tmp_path: Path) -> None:
    root = tmp_path / "data"
    generation = _activate(root, _frames(75, seed=6))
    with duckdb.connect(
        str(generation_duckdb_path(root, generation)), read_only=True
    ) as connection:
        assert connection.execute("SELECT count(*) FROM ohlcv").fetchone()[0] == 150
        filenames = connection.execute("SELECT DISTINCT filename FROM ohlcv").fetchall()
    assert len(filenames) == 2


def test_cleanup_removes_only_unpublished_incomplete_generation(tmp_path: Path) -> None:
    root = tmp_path / "data"
    active = _activate(root, _frames(60, seed=7))
    abandoned = new_generation_id()
    incomplete = generation_root(root, abandoned)
    incomplete.mkdir(parents=True)
    (incomplete / "partial").write_bytes(b"partial")

    assert recover_pending_commit(_config(tmp_path)) is True
    assert not incomplete.exists()
    assert current_generation(root) == active
    assert cleanup_abandoned_generations(root) == []


def test_cross_process_ingestion_lock_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.storage.root.mkdir(parents=True)
    lock_file = config.storage.root / ".ares-ingest.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from filelock import FileLock
                lock = FileLock({str(lock_file)!r})
                lock.acquire()
                print("locked", flush=True)
                time.sleep(30)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(AresError, match="already running"):
            ingest_market_data(config, providers={})
    finally:
        holder.kill()
        holder.wait()


def test_new_rows_reports_actual_additions_not_fetch_size(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frames = _frames(120, seed=8)
    config.data.until = (
        frames["coinbase"]["timestamp"].max() + pd.Timedelta(hours=1)
    ).to_pydatetime()
    first = ingest_market_data(config, providers=_providers(frames))
    assert first.exchanges[0].new_rows == 120
    config.data.resume = False
    second = ingest_market_data(config, providers=_providers(frames))
    assert second.exchanges[0].new_rows == 0


@pytest.mark.parametrize("mode", ["same", "behind"])
def test_non_advancing_cursor_fails_closed(mode: str, monkeypatch) -> None:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    start_ms = int(start.timestamp() * 1000)

    class StallingExchange:
        has = {"fetchOHLCV": True}
        markets = {"ETH/USD": {}}

        def load_markets(self):
            return None

        def milliseconds(self):
            return start_ms + 360_000_000

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            timestamp = start_ms if mode == "same" else since - 3_600_000
            return [[timestamp, 100.0, 101.0, 99.0, 100.5, 5.0]]

    monkeypatch.setattr(CCXTOHLCVProvider, "_exchange", lambda self: StallingExchange())
    with pytest.raises(AresError, match="non-advancing"):
        CCXTOHLCVProvider("coinbase").fetch_range(
            "ETH/USD", "1h", start.to_pydatetime(), None, limit=10, max_pages=5, retries=0
        )
