# Fable 5 audit — command log and environment record

Working directory for all repository commands: the restored tree (baseline
commit `b7736ac`, branch `audit/fable-5-hardening`). Exit code 0 unless
stated. Long raw logs are summarized; no secrets exist or appear anywhere.

## 0. Environment

- Host: sandboxed Ubuntu 22.04, linux-aarch64, 4 CPU, ~3 GB RAM; all egress
  through an allowlist proxy at 127.0.0.1:3128 (PyPI and github.com git
  protocol allowed; exchange APIs, GitHub release assets, Ubuntu pool blocked).
- Interpreters (built from source in-session because the sandbox ships only
  3.10 and binary distribution channels were blocked): CPython 3.11.15+
  (branch 3.11 @ d2b2f5e), 3.12.13+, 3.13.14+, each linked against locally
  built OpenSSL 3.0-dev, SQLite (trunk), bzip2, xz; `ssl/sqlite3/ctypes/bz2/
  lzma` verified importable in each.
- Key packages (3.11 audit venv): tensorflow 2.21.0 | numpy 2.4.6 | pandas 2.3.3 | scikit-learn 1.9.0 | ccxt 4.5.68 | duckdb 1.5.5 | pyarrow 23.0.1 | optuna 4.9.0 | pydantic 2.13.4
- Tools: ruff 0.16.0 | mypy 1.20.2 | pytest 9.1.1 | pip 26.1.2
- Docker daemon: absent. GitHub access: git protocol only; nothing pushed.

## 1. Identity verification (Phase 0)

```
sha256sum ARES-Engine-v0.1.0-restored.tar.gz
  = cc86e2b38cc1136421c574a32dc37c0bb45f8d6c4bfedee4d1579f01c7d47a13
tar -tzvf …   # 16 entries; no traversal/absolute/symlink/non-regular entries
cat payload.part.00..06 | base64 -d > ares-source.tar.gz   # decoded manually; bundled script NOT executed
sha256sum ares-source.tar.gz
  = 0a18e017996746764325b18185f4de320898e41c84bbc91ab695b1979b52a7b0   (== original release)
tar entries: 82 (70 files); top-level ares-engine-0.1.0/ only
git init && git add -A && git write-tree
  = e48e6ad8e45643ca06e6fff21cb27d5044c04a87   (exact match);  git ls-files | wc -l = 70
git commit "Import verified ARES Engine v0.1.0 source" = b7736ac; branch audit/fable-5-hardening
```

## 2. Baseline (unmodified source)

```
python -m pip install -e ".[dev,ml]"      -> FAILED (F-006): No matching distribution for tensorflow-cpu (linux-aarch64)
python -m pip install -e ".[dev]" && pip install "tensorflow>=2.20,<2.22"   -> OK (TF 2.21.0 aarch64)
python -m compileall src tests            -> OK
ruff check .                              -> 4 errors (I001, 3×UP037)
ruff format --check .                     -> 19 files would be reformatted
mypy src                                  -> 31 errors in 17 files
pytest -q -m "not ml"                     -> 49 passed (1.49 s)
pytest -q --cov=ares_engine --cov-branch  -> 50 passed incl. ML smoke; TOTAL branch coverage 76%
```

## 3. Final state (audit branch, every check green)

```
ruff check . ; ruff format --check . ; mypy src   -> all clean (0 findings, 24 source files)
pytest -q --cov=ares_engine --cov-branch          -> 186 passed (145 non-ML + 41 ML); branch coverage ≈85%
python -m build && twine check dist/*             -> sdist+wheel built; both PASSED
```

## 4. Python matrix (wheel installs, tests run against the installed package)

| Interpreter | non-ML | ML (TF 2.21.0) |
|---|---|---|
| 3.11.15+ (editable audit env) | 145 passed | 41 passed |
| 3.12.13+ (wheel-only venv) | 145 passed | 41 passed |
| 3.13.14+ (wheel-only venv) | 145 passed | 41 passed |

Wheel-only CLI exercised from outside the checkout: import + version + all 14
subcommand helps + `ares demo --bars 600` (packaged-config fallback) — green.
One environment note: the source-built 3.12's pip needed
`SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` (system CA path for a
custom OpenSSL prefix); unrelated to ARES.

## 5. Offline lifecycle

```
ares verify-offline --config configs/smoke.yaml --bars 2000
  -> exit 0 in 3.0 s: quality PASS, walk-forward passed (score 0.0351 under
     permissive smoke gates), promotion approved, 7-file bundle verified,
     reloaded, paper signal 0 @ p=0.4385, backend tensorflow,
     profitability_claim=false
ares verify-offline --bars 1600  -> refused up front (F-011 fix): needs ≥1805
```

## 6. Simulated venue ingestion (NOT live; exact-shape wire simulators)

- Venues: local HTTP simulators of Coinbase Advanced Trade candles and Kraken
  public OHLC (shapes derived from the installed ccxt 4.5.68 adapters and both
  venues' public API docs; see tests/venue_wire_sim.py docstring).
- Clients: unmodified ccxt.coinbase / ccxt.kraken through the production
  CCXTOHLCVProvider and ingest_market_data.
- Symbol/timeframe: ETH/USD @ 1h. Window: 2025-01-01T00:00Z + 120 h
  (120 candles/venue) at page_limit 30 -> ≥4 pages/venue, strictly advancing
  cursors (coinbase start params, kraken since params, both strictly
  increasing and unique), full coverage, sorted, deduplicated, on-grid.
- Reruns bit-identical (canonical SHA-256 equality). 16 scenario tests +
  6 atomicity/concurrency tests + 10 harness tests all green; failure
  scenarios (rate limit both dialects, 5xx, timeout, truncation, malformed,
  off-grid, divergence, misalignment, depth caps, open candle, torn commit,
  lock contention) preserve canonical state.

## 7. Concurrency evidence

- Ingestion + promotion lock contention from REAL subprocesses -> fail-closed.
- Two-process promotion race -> single coherent manifest-anchored champion.
- Crash injection: between staging and journal, between renames, between
  decision and pointer, during pointer write -> previous state preserved or
  rolled forward; nothing torn after recovery.

## 8. Generated-artifact hashes

Recorded in the deliverable manifest `ARES-Engine-Fable5-Audit-SHA256SUMS.txt`
(covers archives, bundle, patch, reports, and FABLE_5_TEST_RESULTS.json).

## 9. Network-restriction evidence (live endpoints blocked)

```
# Environment network-restriction evidence  (2026-07-27T22:41:05Z)

## Proxy environment variables
ALL_PROXY=socks5h://localhost:1080
CLAUDE_CODE_HOST_HTTP_PROXY_PORT=46471
CLAUDE_CODE_HOST_SOCKS_PROXY_PORT=44865
CLOUDSDK_PROXY_ADDRESS=localhost
CLOUDSDK_PROXY_PORT=3128
CLOUDSDK_PROXY_TYPE=https
DOCKER_HTTPS_PROXY=http://localhost:3128
DOCKER_HTTP_PROXY=http://localhost:3128
FTP_PROXY=socks5h://localhost:1080
GIT_SSH_COMMAND=ssh -o ProxyCommand='socat - PROXY:localhost:%h:%p,proxyport=3128'
GRPC_PROXY=socks5h://localhost:1080
HTTPS_PROXY=http://localhost:3128
HTTP_PROXY=http://localhost:3128
NO_PROXY=localhost,127.0.0.1,::1,*.local,.local,169.254.0.0/16,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
RSYNC_PROXY=localhost:1080
all_proxy=socks5h://localhost:1080
ftp_proxy=socks5h://localhost:1080
grpc_proxy=socks5h://localhost:1080
http_proxy=http://localhost:3128
https_proxy=http://localhost:3128
no_proxy=localhost,127.0.0.1,::1,*.local,.local,169.254.0.0/16,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

## DNS resolution
api.exchange.coinbase.com -> DNS FAIL: [Errno -3] Temporary failure in name resolution
api.coinbase.com -> DNS FAIL: [Errno -3] Temporary failure in name resolution
api.kraken.com -> DNS FAIL: [Errno -3] Temporary failure in name resolution

## curl: Coinbase Exchange public candles (no credentials)
* Uses proxy env variable no_proxy == 'localhost,127.0.0.1,::1,*.local,.local,169.254.0.0/16,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16'
* Uses proxy env variable https_proxy == 'http://localhost:3128'
*   Trying 127.0.0.1:3128...
* Connected to (nil) (127.0.0.1) port 3128 (#0)
* allocate connect buffer!
* Establish HTTP proxy tunnel to api.exchange.coinbase.com:443
> CONNECT api.exchange.coinbase.com:443 HTTP/1.1
> Host: api.exchange.coinbase.com:443
> User-Agent: curl/7.81.0
> Proxy-Connection: Keep-Alive
>
< HTTP/1.1 403 Forbidden
< Content-Type: text/plain
< X-Proxy-Error: blocked-by-allowlist
<
* Received HTTP code 403 from proxy after CONNECT
* CONNECT phase completed!
* Closing connection 0
--- body ---
head: cannot open '/tmp/cb_body.txt' for reading: No such file or directory


## curl: Kraken public OHLC (no credentials)
* Uses proxy env variable no_proxy == 'localhost,127.0.0.1,::1,*.local,.local,169.254.0.0/16,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16'
* Uses proxy env variable https_proxy == 'http://localhost:3128'
*   Trying 127.0.0.1:3128...
* Connected to (nil) (127.0.0.1) port 3128 (#0)
* allocate connect buffer!
* Establish HTTP proxy tunnel to api.kraken.com:443
> CONNECT api.kraken.com:443 HTTP/1.1
> Host: api.kraken.com:443
> User-Agent: curl/7.81.0
> Proxy-Connection: Keep-Alive
>
< HTTP/1.1 403 Forbidden
< Content-Type: text/plain
< X-Proxy-Error: blocked-by-allowlist
<
* Received HTTP code 403 from proxy after CONNECT
* CONNECT phase completed!
* Closing connection 0
--- body ---
head: cannot open '/tmp/kr_body.txt' for reading: No such file or directory


## control: pypi.org via same proxy (allowed host)
pypi.org HTTP 200 in 4.960243s

## Python requests library: minimal standalone calls (no credentials, Python 3.11 venv)
coinbase: URLError: <urlopen error Tunnel connection failed: 403 Forbidden>
kraken: URLError: <urlopen error Tunnel connection failed: 403 Forbidden>

## CCXT 4.5.68: minimal standalone public fetch_ohlcv (no credentials)
ccxt 4.5.68
coinbase: NetworkError: coinbase GET https://api.coinbase.com/v2/currencies
kraken: NetworkError: kraken GET https://api.kraken.com/0/public/Assets

## Classification
All failures originate at the sandbox proxy (127.0.0.1:3128) during the CONNECT handshake with header 'X-Proxy-Error: blocked-by-allowlist'; direct DNS resolution is also unavailable. No TLS session with either exchange is ever established, so no exchange-side response exists. Control request to pypi.org through the identical proxy returns HTTP 200. Conclusion: environment restriction, not an ARES defect and not an exchange rejection.
```
