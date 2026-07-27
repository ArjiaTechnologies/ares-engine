"""Bounded ingestion driven through unmodified CCXT clients against exact-shape venue simulators.

SIMULATION, NOT LIVE: the audit sandbox blocks Coinbase and Kraken. These tests
prove adapter-shape compatibility, pagination, gating, and atomicity against
wire-accurate local servers; they do not establish live endpoint compatibility.
"""

import hashlib
from datetime import UTC, datetime

import pandas as pd
import pytest
from venue_wire_sim import BASE_TS, HOUR, WireSimulator, make_real_ccxt_exchange

import ares_engine.data.ingest as ingest_module
import ares_engine.data.providers as providers_module
from ares_engine.config import AresConfig, load_config
from ares_engine.data.ingest import ingest_market_data
from ares_engine.data.providers import CCXTOHLCVProvider
from ares_engine.data.storage import market_path, read_market
from ares_engine.exceptions import AresError, DataQualityError


@pytest.fixture()
def sim():
    simulator = WireSimulator().start()
    yield simulator
    simulator.stop()


@pytest.fixture()
def wired(sim, monkeypatch):
    def patched_exchange(self):
        timeout_ms = getattr(patched_exchange, "timeout_ms", 5_000)
        return make_real_ccxt_exchange(self.exchange_id, sim.base_url, timeout_ms)

    monkeypatch.setattr(CCXTOHLCVProvider, "_exchange", patched_exchange)
    monkeypatch.setattr(providers_module.time, "sleep", lambda seconds: None)
    return patched_exchange


def _config(tmp_path, *, hours=120, until_bound=True):
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    config.storage.artifacts = tmp_path / "artifacts"
    config.storage.duckdb_path = tmp_path / "data" / "ares.duckdb"
    config.data.since = datetime.fromtimestamp(BASE_TS, tz=UTC)
    config.data.until = (
        datetime.fromtimestamp(BASE_TS + hours * HOUR, tz=UTC) if until_bound else None
    )
    config.data.page_limit = 30
    config.data.max_pages = 50
    config.data.retries = 3
    config.data.drop_open_candle = False
    config.data.resume = True
    config.data.fail_on_quality = True
    config.data.max_gap_count = 0
    config.data.max_cross_venue_p95_bps = 30.0
    config.data.max_cross_venue_latest_bps = 30.0
    config.data.min_cross_venue_overlap = 50
    return config


def _providers():
    return {
        "coinbase": CCXTOHLCVProvider("coinbase"),
        "kraken": CCXTOHLCVProvider("kraken"),
    }


def _limit_history(sim, hours=120):
    for venue in ("coinbase", "kraken"):
        sim.scenario(venue).history_end = BASE_TS + hours * HOUR


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multi_page_ingestion_covers_range_sorted_deduplicated_on_grid(sim, wired, tmp_path):
    _limit_history(sim)
    config = _config(tmp_path)
    result = ingest_market_data(config, providers=_providers())
    assert result.passed and result.committed

    coinbase_pages = [q for venue, path, q in sim.requests if venue == "coinbase"]
    kraken_pages = [q for venue, path, q in sim.requests if venue == "kraken"]
    assert len(coinbase_pages) >= 4, "120 candles at page_limit=30 must need multiple pages"
    assert len(kraken_pages) >= 4
    coinbase_starts = [int(q["start"]) for q in coinbase_pages]
    kraken_sinces = [int(q["since"]) for q in kraken_pages]
    assert coinbase_starts == sorted(coinbase_starts) and len(set(coinbase_starts)) == len(
        coinbase_starts
    )
    assert kraken_sinces == sorted(kraken_sinces) and len(set(kraken_sinces)) == len(kraken_sinces)

    for venue in ("coinbase", "kraken"):
        frame = read_market(market_path(config.storage.root, venue, "ETH/USD", "1h"))
        assert len(frame) == 120
        stamps = pd.to_datetime(frame["timestamp"], utc=True)
        assert stamps.is_monotonic_increasing
        assert not stamps.duplicated().any()
        assert (stamps.astype("int64") % (HOUR * 10**9) == 0).all()
        assert stamps.min() == pd.Timestamp(BASE_TS, unit="s", tz="UTC")
        assert stamps.max() == pd.Timestamp(BASE_TS + 119 * HOUR, unit="s", tz="UTC")
        assert set(frame["exchange"]) == {venue}
        assert set(frame["symbol"]) == {"ETH/USD"}

    cross = result.cross_venue_reports[0]
    assert cross.passed
    assert cross.stats["overlap_rows"] == 120
    assert cross.stats["p95_divergence_bps"] < 5.0


def test_second_identical_run_is_idempotent_bit_for_bit(sim, wired, tmp_path):
    _limit_history(sim)
    config = _config(tmp_path)
    first = ingest_market_data(config, providers=_providers())
    path = market_path(config.storage.root, "coinbase", "ETH/USD", "1h")
    first_hash = _file_sha(path)
    second = ingest_market_data(config, providers=_providers())
    assert first.committed and second.committed
    assert second.exchanges[0].new_rows == 0
    assert _file_sha(path) == first_hash, "identical rerun must produce identical canonical bytes"


def test_short_pages_do_not_end_ingestion_early(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    for venue in ("coinbase", "kraken"):
        sim.scenario(venue).short_page_size = 7
    config = _config(tmp_path, hours=60)
    result = ingest_market_data(config, providers=_providers())
    assert result.passed and result.committed
    frame = read_market(market_path(config.storage.root, "coinbase", "ETH/USD", "1h"))
    assert len(frame) == 60
    pages = sum(1 for venue, _, _ in sim.requests if venue == "coinbase")
    assert pages >= 9, "short 7-row pages must force many more requests, not early stop"


def test_rate_limited_pages_are_retried_until_success(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    sim.scenario("coinbase").rate_limit_first_n = 2  # HTTP 429 body
    sim.scenario("kraken").rate_limit_first_n = 2  # EAPI:Rate limit in a 200 body
    config = _config(tmp_path, hours=60)
    result = ingest_market_data(config, providers=_providers())
    assert result.passed and result.committed
    assert sim.scenario("coinbase").request_count >= 4
    assert sim.scenario("kraken").request_count >= 4


def test_transient_server_errors_are_retried(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    sim.scenario("coinbase").server_error_first_n = 2
    config = _config(tmp_path, hours=60)
    result = ingest_market_data(config, providers=_providers())
    assert result.passed and result.committed


def test_network_timeout_exhausts_retries_and_fails_closed(sim, wired, tmp_path, monkeypatch):
    _limit_history(sim, hours=60)
    wired.timeout_ms = 700
    sim.scenario("coinbase").timeout_first_n = 99
    config = _config(tmp_path, hours=60)
    config.data.retries = 1
    with pytest.raises(AresError, match="failed after 2 attempts"):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()
    assert not market_path(config.storage.root, "kraken", "ETH/USD", "1h").exists()


def test_empty_page_before_range_completion_is_rejected_as_truncation(sim, wired, tmp_path):
    _limit_history(sim, hours=120)
    sim.scenario("coinbase").empty_after_pages = 2
    config = _config(tmp_path)
    with pytest.raises(DataQualityError, match="incomplete_end|ends at|coverage"):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()
    assert not market_path(config.storage.root, "kraken", "ETH/USD", "1h").exists()


def test_truncated_history_never_commits_any_venue(sim, wired, tmp_path):
    _limit_history(sim, hours=120)
    sim.scenario("coinbase").truncate_end = BASE_TS + 90 * HOUR
    config = _config(tmp_path)
    with pytest.raises(DataQualityError):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "kraken", "ETH/USD", "1h").exists()


def test_malformed_wire_rows_fail_quality_gates_without_commit(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    sim.scenario("coinbase").malformed_rows = True
    config = _config(tmp_path, hours=60)
    with pytest.raises(DataQualityError):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()


def test_off_grid_timestamps_are_rejected(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    sim.scenario("kraken").off_grid_seconds = 300
    config = _config(tmp_path, hours=60)
    with pytest.raises(DataQualityError, match="grid|misaligned"):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()


def test_cross_venue_divergence_blocks_commit(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    sim.scenario("kraken").price_multiplier = 1.02
    config = _config(tmp_path, hours=60)
    with pytest.raises(DataQualityError, match="divergence"):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()


def test_newest_candle_mismatch_across_venues_blocks_commit(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    sim.scenario("kraken").truncate_end = BASE_TS + 58 * HOUR
    config = _config(tmp_path, hours=60)
    with pytest.raises(DataQualityError, match="not aligned|coverage|ends at"):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()


def test_depth_limited_validator_is_accepted_when_overlap_suffices(sim, wired, tmp_path):
    """Kraken-style: validator can only serve recent candles; primary has full history."""
    _limit_history(sim, hours=120)
    sim.scenario("kraken").depth_limit = 80
    config = _config(tmp_path)
    result = ingest_market_data(config, providers=_providers())
    assert result.passed and result.committed
    kraken = read_market(market_path(config.storage.root, "kraken", "ETH/USD", "1h"))
    coinbase = read_market(market_path(config.storage.root, "coinbase", "ETH/USD", "1h"))
    assert len(coinbase) == 120
    assert len(kraken) == 80
    assert result.cross_venue_reports[0].stats["overlap_rows"] == 80


def test_depth_limited_primary_still_fails_closed(sim, wired, tmp_path):
    """A primary venue that cannot serve the configured start must never commit.

    Coinbase-style venues answer pre-history windows with empty pages (the provider
    stops and the empty/coverage gates fire); Kraken-style venues ignore deep
    ``since`` and serve only the newest candles (the start-coverage gate fires).
    Both must fail closed.
    """
    _limit_history(sim, hours=120)
    sim.scenario("coinbase").depth_limit = 80
    config = _config(tmp_path)
    with pytest.raises(DataQualityError, match="No OHLCV rows|starts at"):
        ingest_market_data(config, providers=_providers())
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()

    # Kraken-style since-ignoring venue as the primary: newest-only history triggers
    # the explicit start-coverage failure.
    sim.scenario("coinbase").depth_limit = None
    sim.scenario("kraken").depth_limit = 80
    raw = _config(tmp_path).model_dump(mode="json")
    raw["data"]["primary_exchange"] = "kraken"
    raw["data"]["validation_exchanges"] = ["coinbase"]
    swapped = AresConfig.model_validate(raw)
    with pytest.raises(DataQualityError, match="starts at"):
        ingest_market_data(swapped, providers=_providers())
    assert not market_path(swapped.storage.root, "kraken", "ETH/USD", "1h").exists()


def test_incomplete_current_candle_is_dropped_before_storage(sim, wired, tmp_path, monkeypatch):
    _limit_history(sim, hours=60)
    for venue in ("coinbase", "kraken"):
        sim.scenario(venue).include_current_partial = True
    now = datetime.fromtimestamp(BASE_TS + 60 * HOUR + 1800, tz=UTC)  # half-way into candle 60
    monkeypatch.setattr(ingest_module, "utc_now", lambda: now)
    config = _config(tmp_path, until_bound=False)
    config.data.drop_open_candle = True
    config.data.max_staleness_bars = 2
    result = ingest_market_data(config, providers=_providers())
    assert result.passed and result.committed
    for venue in ("coinbase", "kraken"):
        frame = read_market(market_path(config.storage.root, venue, "ETH/USD", "1h"))
        newest = pd.Timestamp(frame["timestamp"].max())
        assert newest == pd.Timestamp(BASE_TS + 59 * HOUR, unit="s", tz="UTC"), (
            "the in-progress candle served by the venue must never reach canonical storage"
        )


def test_duplicate_and_overlapping_pages_deduplicate_deterministically(sim, wired, tmp_path):
    _limit_history(sim, hours=60)
    sim.scenario("coinbase").overlap_rows = 5
    sim.scenario("kraken").duplicate_page = True
    config = _config(tmp_path, hours=60)
    result = ingest_market_data(config, providers=_providers())
    assert result.passed and result.committed
    for venue in ("coinbase", "kraken"):
        frame = read_market(market_path(config.storage.root, venue, "ETH/USD", "1h"))
        assert len(frame) == 60
        assert not pd.to_datetime(frame["timestamp"], utc=True).duplicated().any()
