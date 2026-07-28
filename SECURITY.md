# Security policy

## Supported code

The current `0.1.x` release line and the current `main` branch receive security fixes.

## Reporting

Use the repository's private vulnerability-reporting or security-advisory flow. Do not put credentials, exploit details, sensitive data, or issues that could endanger systems or funds in a public issue.

## Trust boundaries

- ARES contains no order-routing code and needs no exchange credentials for its public-data workflow. Do not add or commit exchange keys, private keys, signing keys, account exports, or personal data.
- Exchange responses are untrusted until schema, coverage, continuity, UTC-grid, OHLCV, freshness, alignment, and divergence checks pass.
- Canonical data is an immutable generation selected by an atomic pointer. Manifests detect corruption; filesystem permissions and host integrity remain external controls.
- `scaler.joblib` and Keras model deserialization can execute or instantiate trusted code. Load bundles only from an authenticated publisher and protected storage. The bundled SHA-256 manifest detects changed bytes but is not a digital signature and does not establish authorship.
- Champion promotion trusts the local artifacts root and its process-level file lock. Cryptographic signing, a transparency log, and hardware-backed publisher identity are not implemented.
- Dependencies and container bases remain supply-chain inputs. Review Dependabot, dependency-audit, CodeQL, and pinned-action changes before merging.
- Public-ingestion request traces intentionally omit credentials and query strings; evidence establishes read-only endpoint contact, not exchange authenticity beyond TLS/CCXT and the host trust store.

Never disable integrity, data-quality, freshness, insolvency, execution-delay, or promotion gates to force a strategy through.
