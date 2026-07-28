# Sol GitHub Actions evidence

Initial status: **PENDING**.

The private workflows are implemented with read-only default permissions and action commits pinned:

- Python 3.11, 3.12, and 3.13 non-ML compile, Ruff, strict mypy, pytest/coverage.
- TensorFlow CPU ML tests and 2,000-bar offline lifecycle.
- Source/wheel build, Twine checks, clean installed-wheel CLI smoke outside checkout.
- Dependency audit.
- Non-root Docker build, CLI/demo/offline lifecycle, and graceful shutdown.
- CodeQL with `security-events: write` only in its analysis job.
- Dependency review with `contents: read` only, where GitHub supports it.
- Manual `workflow_dispatch` public Coinbase/Kraken ingestion only; never automatic on pull requests.
- Release workflow intentionally fails with an explanatory message and has no tag, release, package, or write permission.

This document will record the private PR URL, run URLs, job conclusions, merge commit, and final repository privacy/no-release proof after execution. Until then the Sol audit remains `CONDITIONAL`.
