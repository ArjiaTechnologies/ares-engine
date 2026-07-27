"""Parquet-first storage with a DuckDB query layer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from ..exceptions import AresError
from ..utils import slugify_symbol


def market_path(root: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return root / "raw" / exchange / slugify_symbol(symbol) / f"{timeframe}.parquet"


def quality_path(root: Path, name: str) -> Path:
    safe_name = name.replace("/", "-").replace("~", "__").lower()
    return root / "quality" / f"{safe_name}.json"


def read_market(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def merge_market(current: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge a candidate batch in memory without mutating canonical storage."""
    if current.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = current
    else:
        merged = pd.concat([current, incoming], ignore_index=True)
    if merged.empty:
        return merged
    if "timestamp" in merged:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        merged = (
            merged.sort_values("timestamp", na_position="last")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
    return merged


def write_market(path: Path, frame: pd.DataFrame) -> None:
    """Atomically replace one canonical Parquet file after validation has passed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".parquet",
    )
    os.close(descriptor)
    temp_path = Path(raw_temp_path)
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def upsert_market(path: Path, incoming: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for callers that already validated their candidate data."""
    merged = merge_market(read_market(path), incoming)
    if not merged.empty:
        write_market(path, merged)
    return merged


def sync_duckdb(database_path: Path, parquet_root: Path) -> None:
    """Create a DuckDB view over all raw Parquet files without duplicating data."""
    try:
        import duckdb
    except ImportError as exc:
        raise AresError("DuckDB is not installed. Install ARES base dependencies first.") from exc

    database_path.parent.mkdir(parents=True, exist_ok=True)
    raw_root = parquet_root / "raw"
    parquet_files = list(raw_root.rglob("*.parquet")) if raw_root.exists() else []
    connection = duckdb.connect(str(database_path))
    try:
        if not parquet_files:
            connection.execute(
                """
                CREATE OR REPLACE VIEW ohlcv AS
                SELECT
                    CAST(NULL AS TIMESTAMPTZ) AS timestamp,
                    CAST(NULL AS DOUBLE) AS open,
                    CAST(NULL AS DOUBLE) AS high,
                    CAST(NULL AS DOUBLE) AS low,
                    CAST(NULL AS DOUBLE) AS close,
                    CAST(NULL AS DOUBLE) AS volume,
                    CAST(NULL AS VARCHAR) AS exchange,
                    CAST(NULL AS VARCHAR) AS symbol,
                    CAST(NULL AS VARCHAR) AS timeframe,
                    CAST(NULL AS VARCHAR) AS filename
                WHERE FALSE
                """
            )
            return
        glob = (raw_root / "**" / "*.parquet").resolve().as_posix().replace("'", "''")
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW ohlcv AS
            SELECT *
            FROM read_parquet('{glob}', union_by_name = true, filename = true)
            """
        )
    finally:
        connection.close()
