#!/usr/bin/env python3
"""Standalone public-ingestion audit runner for a clean installed wheel.

Calls the real ARES production ingestion/validation code (no parallel
pipeline). Usage:

    python scripts/verify_public_ingestion.py \
        --symbol ETH/USD --timeframe 1h \
        --start 2026-06-01T00:00:00Z --end 2026-06-15T00:00:00Z \
        --output artifacts/public-ingestion-audit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default="coinbase")
    parser.add_argument("--validation", default="kraken")
    parser.add_argument("--symbol", default="ETH/USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", required=True, type=_parse_utc)
    parser.add_argument("--end", required=True, type=_parse_utc)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--page-limit", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    from ares_engine.public_audit import (
        run_public_ingestion_audit,
        validate_public_ingestion_report,
    )

    report = run_public_ingestion_audit(
        primary=args.primary,
        validation=args.validation,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        output_dir=args.output,
        page_limit=args.page_limit,
        retries=args.retries,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    problems = validate_public_ingestion_report(args.output)
    if problems:
        print("Report validation problems: " + "; ".join(problems), file=sys.stderr)
        return 4
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
