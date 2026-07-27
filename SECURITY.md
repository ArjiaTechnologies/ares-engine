# Security policy

## Supported versions

Only the latest release on the default branch is supported during the alpha phase.

## Reporting a vulnerability

Do not open a public issue for leaked credentials or a vulnerability that can put funds or systems at risk. Contact the maintainers privately through GitHub's security-advisory flow.

## Trust boundaries

ARES uses public market-data endpoints by default and contains no order-routing code. Never commit API keys, private keys, exchange secrets, model-signing keys, raw account exports, or personally identifying data. Use read-only credentials when authenticated market data is unavoidable.

Model bundles are executable trust objects, not harmless documents. `scaler.joblib` uses Python serialization and can execute code when loaded; Keras model files may also contain custom objects in future versions. Load bundles only from a trusted source whose manifest and release provenance you have verified. The SHA-256 manifest detects accidental or unauthorized changes after creation, but it does not prove who created the bundle. A future signed-release design is tracked in the roadmap.

Run dependency updates through review and CI. Do not disable data gates, freshness checks, execution delay, or bundle verification to “fix” a failing strategy. That converts an honest failure into a security and financial-risk problem.
