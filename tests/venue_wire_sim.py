"""Exact-shape local simulators of the Coinbase and Kraken public OHLC endpoints.

Response structures reproduce what the real CCXT adapters (ccxt>=4.4) actually
receive on the wire, verified by driving unmodified ``ccxt.coinbase`` and
``ccxt.kraken`` clients against this server:

- Coinbase Advanced Trade "Get Market Candles":
  GET /api/v3/brokerage/market/products/{product_id}/candles
      ?granularity=ONE_HOUR&start=<unix s>&end=<unix s>
  -> {"candles": [{"start": "<unix s>", "low": "...", "high": "...",
                   "open": "...", "close": "...", "volume": "..."}, ...]}
  newest-first, decimal fields as strings.

- Kraken public OHLC:
  GET /0/public/OHLC?pair=<PAIRKEY>&interval=60&since=<unix s>
  -> {"error": [], "result": {"<PAIRKEY>": [[<unix s>, "o","h","l","c",
      "vwap","volume", count], ...], "last": <unix s>}}
  oldest-first, strictly-after ``since`` (CCXT already subtracts one interval),
  history depth capped (720 candles on the real venue), and the final row may
  be the incomplete current candle.

Scenario hooks layer adversarial behavior on top of the exact shapes without
changing the shapes themselves. Every simulated run is clearly labeled
simulation; nothing here talks to a real exchange.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HOUR = 3600
BASE_TS = 1_735_689_600  # 2025-01-01T00:00:00Z


def price_path(venue: str, index: int) -> tuple[float, float, float, float, float]:
    """Deterministic OHLCV tuple for candle *index*; venues differ by ~2 bps."""
    base = 3_000.0 + 25.0 * ((index * 37) % 40) / 40.0 + (index % 7) * 3.0
    offset = 1.0 + (0.0002 if venue == "kraken" else 0.0)
    open_price = base * offset
    close = (base + 1.5) * offset
    high = max(open_price, close) * 1.004
    low = min(open_price, close) * 0.996
    volume = 10.0 + (index % 11)
    return open_price, high, low, close, volume


@dataclass
class VenueScenario:
    """Mutable per-venue behavior switches consumed by the HTTP handler."""

    history_start: int = BASE_TS
    history_end: int = BASE_TS + 200 * HOUR  # exclusive; grid-aligned "now" boundary
    depth_limit: int | None = None  # e.g. 720 on real Kraken
    include_current_partial: bool = False  # serve the in-progress candle too
    rate_limit_first_n: int = 0  # respond as rate-limited for first N requests
    server_error_first_n: int = 0  # 5xx for first N requests
    timeout_first_n: int = 0  # hang for first N requests
    empty_after_pages: int | None = None  # serve empty payloads after N pages
    short_page_size: int | None = None  # cap rows per page below the requested limit
    duplicate_page: bool = False  # always serve the same first window
    overlap_rows: int = 0  # re-serve trailing rows of the previous page
    malformed_rows: bool = False  # inject wire-shaped but garbage rows
    off_grid_seconds: int = 0  # shift all timestamps off the UTC grid
    price_multiplier: float = 1.0  # cross-venue divergence injection
    truncate_end: int | None = None  # pretend history stops early (unix s)
    request_count: int = 0
    pages_served: int = 0
    last_served_end: int | None = None


@dataclass
class WireSimulator:
    scenarios: dict[str, VenueScenario] = field(default_factory=dict)
    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    server: ThreadingHTTPServer | None = None
    port: int = 0

    def scenario(self, venue: str) -> VenueScenario:
        return self.scenarios.setdefault(venue, VenueScenario())

    def start(self) -> WireSimulator:
        simulator = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                return

            def _send(
                self, code: int, payload: bytes, content_type: str = "application/json"
            ) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                parsed = urlparse(self.path)
                query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                venue = "coinbase" if "brokerage" in parsed.path else "kraken"
                simulator.requests.append((venue, parsed.path, query))
                scenario = simulator.scenario(venue)
                scenario.request_count += 1

                if scenario.timeout_first_n > 0:
                    scenario.timeout_first_n -= 1
                    time.sleep(30)
                    self._send(504, b"{}")
                    return
                if scenario.server_error_first_n > 0:
                    scenario.server_error_first_n -= 1
                    self._send(502, b'{"message":"upstream error"}')
                    return
                if scenario.rate_limit_first_n > 0:
                    scenario.rate_limit_first_n -= 1
                    if venue == "coinbase":
                        self._send(429, b'{"error":"rate limit exceeded"}')
                    else:
                        body = json.dumps({"error": ["EAPI:Rate limit exceeded"]}).encode()
                        self._send(200, body)
                    return

                if venue == "coinbase":
                    self._serve_coinbase(scenario, query)
                else:
                    self._serve_kraken(scenario, query)

            def _rows_for(
                self, scenario: VenueScenario, venue: str, start: int, end: int
            ) -> list[int]:
                effective_end = min(end, scenario.truncate_end or end, scenario.history_end)
                history_top = min(
                    scenario.truncate_end or scenario.history_end, scenario.history_end
                )
                available = list(range(scenario.history_start, history_top, HOUR))
                if scenario.depth_limit is not None:
                    # Real venues cap total retained history (newest N candles),
                    # not the size of one requested window.
                    available = available[-scenario.depth_limit :]
                if scenario.include_current_partial:
                    available.append(scenario.history_end)
                stamps = [
                    t
                    for t in available
                    if (start <= t < effective_end)
                    or (scenario.include_current_partial and t == scenario.history_end)
                ]
                return stamps

            def _serve_coinbase(self, scenario: VenueScenario, query: dict[str, str]) -> None:
                start = int(query.get("start", scenario.history_start))
                end = int(query.get("end", scenario.history_end))
                if scenario.duplicate_page and scenario.last_served_end is not None:
                    start = scenario.history_start
                    end = scenario.last_served_end
                if (
                    scenario.empty_after_pages is not None
                    and scenario.pages_served >= scenario.empty_after_pages
                ):
                    self._send(200, json.dumps({"candles": []}).encode())
                    return
                stamps = self._rows_for(scenario, "coinbase", start, end)
                if scenario.short_page_size is not None:
                    stamps = stamps[: scenario.short_page_size]
                if scenario.overlap_rows and scenario.last_served_end is not None:
                    overlap_start = max(
                        scenario.history_start, start - scenario.overlap_rows * HOUR
                    )
                    stamps = [t for t in range(overlap_start, start, HOUR)] + stamps
                candles = []
                for stamp in reversed(stamps):  # real endpoint answers newest-first
                    index = (stamp - BASE_TS) // HOUR
                    open_price, high, low, close, volume = price_path("coinbase", index)
                    close *= scenario.price_multiplier
                    high = max(high, close)
                    candles.append(
                        {
                            "start": str(stamp + scenario.off_grid_seconds),
                            "low": f"{low:.2f}",
                            "high": f"{high:.2f}",
                            "open": f"{open_price:.2f}",
                            "close": f"{close:.2f}",
                            "volume": f"{volume:.4f}",
                        }
                    )
                if scenario.malformed_rows and candles:
                    candles[0]["close"] = "not-a-number"
                    candles.append({"start": candles[-1]["start"]})
                scenario.pages_served += 1
                scenario.last_served_end = stamps[-1] + HOUR if stamps else start
                self._send(200, json.dumps({"candles": candles}).encode())

            def _serve_kraken(self, scenario: VenueScenario, query: dict[str, str]) -> None:
                pair = query.get("pair", "XETHZUSD")
                since = int(query.get("since", scenario.history_start - 1))
                if (
                    scenario.empty_after_pages is not None
                    and scenario.pages_served >= scenario.empty_after_pages
                ):
                    body = {"error": [], "result": {pair: [], "last": since}}
                    self._send(200, json.dumps(body).encode())
                    return
                stamps = self._rows_for(
                    scenario, "kraken", scenario.history_start, scenario.history_end
                )
                stamps = [t for t in stamps if t > since]  # real venue: strictly after since
                if scenario.short_page_size is not None:
                    stamps = stamps[: scenario.short_page_size]
                elif len(stamps) > 720:
                    stamps = stamps[:720]  # real page cap
                rows = []
                for stamp in stamps:
                    index = (stamp - BASE_TS) // HOUR
                    open_price, high, low, close, volume = price_path("kraken", index)
                    close *= scenario.price_multiplier
                    high = max(high, close)
                    rows.append(
                        [
                            stamp + scenario.off_grid_seconds,
                            f"{open_price:.2f}",
                            f"{high:.2f}",
                            f"{low:.2f}",
                            f"{close:.2f}",
                            f"{(open_price + close) / 2:.2f}",
                            f"{volume:.4f}",
                            int(volume),
                        ]
                    )
                if scenario.malformed_rows and rows:
                    rows[0][4] = "garbage"
                    rows.append([stamps[-1] if stamps else since])
                if scenario.duplicate_page and rows:
                    rows = rows + rows[: max(1, len(rows) // 10)]
                scenario.pages_served += 1
                last = rows[-1][0] if rows else since
                body = {"error": [], "result": {pair: rows, "last": last}}
                self._send(200, json.dumps(body).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def make_real_ccxt_exchange(exchange_id: str, base_url: str, timeout_ms: int = 5_000):
    """Unmodified CCXT client pointed at the simulator; markets preloaded offline."""
    import ccxt

    if exchange_id == "coinbase":
        exchange = ccxt.coinbase({"enableRateLimit": False, "timeout": timeout_ms})
        exchange.urls["api"] = {key: base_url for key in exchange.urls["api"]}
        exchange.set_markets(
            [
                {
                    "id": "ETH-USD",
                    "symbol": "ETH/USD",
                    "base": "ETH",
                    "quote": "USD",
                    "spot": True,
                    "active": True,
                    "type": "spot",
                    "precision": {"price": 0.01, "amount": 1e-8},
                    "limits": {},
                }
            ]
        )
        return exchange
    if exchange_id == "kraken":
        exchange = ccxt.kraken({"enableRateLimit": False, "timeout": timeout_ms})
        exchange.urls["api"] = {key: base_url for key in exchange.urls["api"]}
        exchange.set_markets(
            [
                {
                    "id": "XETHZUSD",
                    "symbol": "ETH/USD",
                    "base": "ETH",
                    "quote": "USD",
                    "spot": True,
                    "active": True,
                    "type": "spot",
                    "darkpool": False,
                    "precision": {"price": 0.01, "amount": 1e-8},
                    "limits": {},
                }
            ]
        )
        return exchange
    raise ValueError(f"unsupported simulated exchange: {exchange_id}")
