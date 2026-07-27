from datetime import UTC, datetime, timedelta

from ares_engine.data.providers import CCXTOHLCVProvider


def test_provider_keeps_paginating_when_exchange_returns_less_than_requested(
    monkeypatch,
) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    hour_ms = 3_600_000
    rows = [
        [start_ms + index * hour_ms, 100 + index, 102 + index, 99 + index, 101 + index, 10]
        for index in range(5)
    ]

    class FakeExchange:
        has = {"fetchOHLCV": True}
        markets = {"ETH-USD": {}}

        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int]] = []

        def load_markets(self) -> None:
            return None

        def milliseconds(self) -> int:
            return start_ms + 10 * hour_ms

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            assert timeframe == "1h"
            self.calls.append((symbol, since, limit))
            available = [row for row in rows if row[0] >= since]
            return available[:2]

    exchange = FakeExchange()
    monkeypatch.setattr(CCXTOHLCVProvider, "_exchange", lambda self: exchange)
    provider = CCXTOHLCVProvider("coinbase")
    frame = provider.fetch_range(
        "ETH/USD",
        "1h",
        start,
        start + timedelta(hours=5),
        limit=3,
        max_pages=10,
        retries=0,
    )

    assert len(frame) == 5
    assert len(exchange.calls) == 3
    assert {call[0] for call in exchange.calls} == {"ETH-USD"}
    assert set(frame["symbol"]) == {"ETH/USD"}
