"""Exchange market-data providers."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from ..exceptions import AresError
from ..utils import timeframe_to_seconds

LOGGER = logging.getLogger(__name__)
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(slots=True)
class FetchTelemetry:
    """Measured transport and pagination evidence for exactly one fetch_range call."""

    exchange: str
    symbol: str = ""
    timeframe: str = ""
    http_request_count: int = 0
    fetch_ohlcv_call_count: int = 0
    page_count: int = 0
    raw_rows_received: int = 0
    normalized_rows: int = 0
    duplicate_rows_removed: int = 0
    overlapping_rows_removed: int = 0
    out_of_range_rows_removed: int = 0
    empty_pages: int = 0
    short_pages: int = 0
    first_request_cursor_ms: int | None = None
    last_request_cursor_ms: int | None = None
    cursor_progression_ms: list[int] = field(default_factory=list)
    retry_count: int = 0
    rate_limit_responses: int = 0
    response_statuses: list[int | None] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    last_trace: FetchTelemetry = field(init=False)

    def __post_init__(self) -> None:
        self.last_trace = FetchTelemetry(exchange=self.exchange_id)

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
        self.last_trace = FetchTelemetry(
            exchange=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
        )
        trace = self.last_trace
        for credential_name in [
            "apiKey",
            "secret",
            "password",
            "uid",
            "privateKey",
            "walletAddress",
        ]:
            if getattr(exchange, credential_name, None):
                raise AresError(
                    f"{self.exchange_id} public OHLCV provider refuses configured credentials"
                )

        original_fetch = getattr(exchange, "fetch", None)
        original_on_rest_response = getattr(exchange, "on_rest_response", None)
        fetch_callable: Any = original_fetch
        current_cursor_ms: int | None = None

        def measured_fetch(*args: Any, **kwargs: Any) -> Any:
            trace.http_request_count += 1
            raw_url = str(args[0] if args else kwargs.get("url", ""))
            parts = urlsplit(raw_url)
            safe_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            method = str(args[1] if len(args) > 1 else kwargs.get("method", "GET"))
            event: dict[str, Any] = {
                "sequence": trace.http_request_count,
                "method": method,
                "url": safe_url,
                "cursor_ms": current_cursor_ms,
                "status": None,
                "error": None,
            }
            try:
                response = fetch_callable(*args, **kwargs)
                status = getattr(exchange, "last_response_status", None)
                event["status"] = int(status) if isinstance(status, int) else None
                trace.response_statuses.append(event["status"])
                return response
            except Exception as exc:
                status = (
                    getattr(exc, "status", None)
                    or getattr(exc, "http_status", None)
                    or getattr(exchange, "last_response_status", None)
                )
                event["status"] = int(status) if isinstance(status, int) else None
                event["error"] = type(exc).__name__
                trace.response_statuses.append(event["status"])
                if event["status"] == 429 or "ratelimit" in type(exc).__name__.lower():
                    trace.rate_limit_responses += 1
                raise
            finally:
                trace.requests.append(event)

        if callable(original_fetch):
            if callable(original_on_rest_response):

                def measured_on_rest_response(status: int, *args: Any, **kwargs: Any) -> Any:
                    exchange.last_response_status = status
                    return original_on_rest_response(status, *args, **kwargs)

                exchange.on_rest_response = measured_on_rest_response
            exchange.fetch = measured_fetch
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
                rate_limits_before = trace.rate_limit_responses
                try:
                    current_cursor_ms = cursor_ms
                    trace.fetch_ohlcv_call_count += 1
                    if trace.first_request_cursor_ms is None:
                        trace.first_request_cursor_ms = cursor_ms
                    trace.last_request_cursor_ms = cursor_ms
                    trace.cursor_progression_ms.append(cursor_ms)
                    if attempt > 0:
                        trace.retry_count += 1
                    batch = exchange.fetch_ohlcv(
                        fetch_symbol,
                        timeframe=timeframe,
                        since=cursor_ms,
                        limit=limit,
                    )
                    break
                except Exception as exc:  # CCXT uses exchange-specific exception subclasses.
                    lowered = f"{type(exc).__name__}: {exc}".lower()
                    if trace.rate_limit_responses == rate_limits_before and (
                        "429" in lowered or "rate limit" in lowered or "ratelimit" in lowered
                    ):
                        trace.rate_limit_responses += 1
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
                trace.page_count += 1
                trace.empty_pages += 1
                break

            trace.page_count += 1
            trace.raw_rows_received += len(batch)
            if len(batch) < limit:
                trace.short_pages += 1
            ordered = sorted(batch, key=lambda row: row[0])
            trace.overlapping_rows_removed += sum(int(row[0]) < cursor_ms for row in ordered)
            trace.out_of_range_rows_removed += sum(int(row[0]) >= until_ms for row in ordered)
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
        duplicate_count = int(frame["timestamp"].duplicated().sum())
        normalized = (
            frame.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
        trace.duplicate_rows_removed = duplicate_count
        trace.normalized_rows = len(normalized)
        return normalized
