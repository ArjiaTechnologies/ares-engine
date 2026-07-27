"""Exchange market-data providers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd

from ..exceptions import AresError
from ..utils import timeframe_to_seconds

LOGGER = logging.getLogger(__name__)
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class OHLCVProvider(Protocol):
    exchange_id: str

    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None,
        *,
        limit: int,
        max_pages: int,
        retries: int,
    ) -> pd.DataFrame: ...


@dataclass(slots=True)
class CCXTOHLCVProvider:
    """Fetch paginated OHLCV through CCXT with explicit cursor advancement."""

    exchange_id: str
    options: dict[str, Any] | None = None

    def _exchange(self) -> Any:
        try:
            import ccxt
        except ImportError as exc:
            raise AresError("CCXT is not installed. Install ARES base dependencies first.") from exc

        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            raise AresError(f"Unknown CCXT exchange id: {self.exchange_id}")
        params: dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 30_000,
        }
        if self.options:
            params.update(self.options)
        return exchange_class(params)

    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None,
        *,
        limit: int,
        max_pages: int,
        retries: int,
    ) -> pd.DataFrame:
        exchange = self._exchange()
        exchange.load_markets()
        if not exchange.has.get("fetchOHLCV"):
            raise AresError(f"{self.exchange_id} does not expose fetchOHLCV through CCXT")
        requested_symbol = symbol
        fetch_symbol = symbol
        if fetch_symbol not in exchange.markets:
            close_matches = [
                market for market in exchange.markets if market.replace("-", "/") == fetch_symbol
            ]
            if close_matches:
                fetch_symbol = close_matches[0]
            else:
                raise AresError(f"{requested_symbol} is not listed by {self.exchange_id}")

        step_ms = timeframe_to_seconds(timeframe) * 1000
        cursor_ms = int(since.astimezone(UTC).timestamp() * 1000)
        until_ms = (
            int(until.astimezone(UTC).timestamp() * 1000) if until else exchange.milliseconds()
        )
        rows: list[list[float]] = []

        for page in range(max_pages):
            if cursor_ms >= until_ms:
                break
            batch: list[list[float]] | None = None
            for attempt in range(retries + 1):
                try:
                    batch = exchange.fetch_ohlcv(
                        fetch_symbol,
                        timeframe=timeframe,
                        since=cursor_ms,
                        limit=limit,
                    )
                    break
                except Exception as exc:  # CCXT uses exchange-specific exception subclasses.
                    if attempt >= retries:
                        raise AresError(
                            f"{self.exchange_id} OHLCV fetch failed after {retries + 1} attempts"
                        ) from exc
                    sleep_seconds = min(2**attempt, 16)
                    LOGGER.warning(
                        "Fetch failed for %s page %s; retrying in %ss: %s",
                        self.exchange_id,
                        page,
                        sleep_seconds,
                        exc,
                    )
                    time.sleep(sleep_seconds)
            if not batch:
                break

            ordered = sorted(batch, key=lambda row: row[0])
            usable = [row[:6] for row in ordered if cursor_ms <= int(row[0]) < until_ms]
            rows.extend(usable)
            latest_ms = int(ordered[-1][0])
            next_cursor = latest_ms + step_ms
            if next_cursor <= cursor_ms:
                raise AresError(f"{self.exchange_id} returned a non-advancing OHLCV cursor")
            cursor_ms = next_cursor
            if cursor_ms >= until_ms:
                break
        else:
            raise AresError(
                f"Reached max_pages={max_pages} while fetching {self.exchange_id}; "
                "increase the limit or narrow the date range"
            )

        frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        if frame.empty:
            return pd.DataFrame(columns=[*OHLCV_COLUMNS, "exchange", "symbol", "timeframe"]).astype(
                {
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                }
            )

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        frame["exchange"] = self.exchange_id
        frame["symbol"] = requested_symbol
        frame["timeframe"] = timeframe
        return (
            frame.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
