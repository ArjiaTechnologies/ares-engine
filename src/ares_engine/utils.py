"""Small utilities shared across ARES Engine."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_TIMEFRAME_RE = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>[mhdw])$")
_TIMEFRAME_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert CCXT-style minute/hour/day/week timeframes to seconds."""
    match = _TIMEFRAME_RE.fullmatch(timeframe.strip().lower())
    if match is None:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected forms like 5m, 1h, 1d")
    return int(match.group("count")) * _TIMEFRAME_SECONDS[match.group("unit")]


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so interrupted jobs cannot leave half-written metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        tf = None
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            return
        return

    try:
        import keras  # type: ignore
    except ImportError:
        return
    keras.utils.set_random_seed(seed)


def slugify_symbol(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-").lower()
