# External live-ingestion verification handoff (macOS)

Purpose: execute the one audit requirement this sandbox could not — bounded,
credential-free ingestion against the real public Coinbase and Kraken
endpoints — and package the evidence. Total time: ~10 minutes.

Prerequisites: macOS with Python 3.11–3.13 (`python3 --version`), network
access, and the audited deliverables (`ARES-Engine-Fable5-Audited-source.tar.gz`
plus `ARES-Engine-Fable5-Audit-SHA256SUMS.txt`).

```bash
# 0) Verify and unpack the audited source
cd ~/Downloads
shasum -a 256 -c ARES-Engine-Fable5-Audit-SHA256SUMS.txt --ignore-missing
tar -xzf ARES-Engine-Fable5-Audited-source.tar.gz
cd ARES-Engine-Fable5-Audited

# 1) Clean environment
python3 -m venv .venv-ingest-audit
source .venv-ingest-audit/bin/activate
python -m pip install --quiet --upgrade pip

# 2) Install the audited wheel (build it from this exact tree)
python -m pip install --quiet build
python -m build --wheel --outdir dist-audit
python -m pip install --quiet dist-audit/ares_eth_engine-0.1.0-py3-none-any.whl

# 3) Run the real Coinbase/Kraken ingestion audit (public endpoints, no keys).
#    Two weeks of 1h ETH/USD = 336 candles per venue -> multiple pages at
#    --page-limit 100. Pick any recent UTC window with both ends in the past.
ares verify-public-ingestion \
  --primary coinbase \
  --validation kraken \
  --symbol ETH/USD \
  --timeframe 1h \
  --start 2026-07-01T00:00:00Z \
  --end 2026-07-15T00:00:00Z \
  --page-limit 100 \
  --output artifacts/public-ingestion-audit
echo "exit code: $?"        # 0 = all gates passed; 1 = a gate failed; 4 = report invalid

# 4) Validate the generated report independently
ares validate-ingestion-report artifacts/public-ingestion-audit

# 5) Package the evidence
tar -czf ARES-Public-Ingestion-Evidence.tar.gz -C artifacts public-ingestion-audit

# 6) Checksums for the record
shasum -a 256 ARES-Public-Ingestion-Evidence.tar.gz artifacts/public-ingestion-audit/*
```

Wheel-only alternative for step 3 (no checkout needed once the wheel is
installed): `python scripts/verify_public_ingestion.py --symbol ETH/USD
--timeframe 1h --start 2026-07-01T00:00:00Z --end 2026-07-15T00:00:00Z
--output artifacts/public-ingestion-audit`.

Review checklist for the produced `public_ingestion_report.json`:

- `live_public_endpoints_reached: true`, `fixture_mode: false`
- `credentials_used: false`, `orders_possible: false`
- `primary_pages` and `validation_pages` ≥ 4 (pagination actually exercised)
- `primary_rows` = `validation_rows` = expected candle count for the window
- all five gates true and `overall_passed: true`
- `canonical_hashes_run1` == `canonical_hashes_run2` (idempotency)
- `SHA256SUMS.txt` matches every artifact (the validator re-checks this)

If `overall_passed` is true and validation is clean, the CONDITIONAL verdict's
outstanding requirement is satisfied; attach the evidence tarball to the audit
record. If it fails, the JSON carries the failing gate and a failure
classification distinguishing endpoint/environment blocks from ingestion logic.
