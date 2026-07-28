# Releasing ARES Engine

Release automation is intentionally disabled. The current repository is a private, release-ready research codebase, not an authorized public release.

Before a future release, a separately authorized task must:

1. Re-run the complete local, live-ingestion, Docker, and GitHub Actions evidence matrix.
2. Review unresolved risks, dependencies, licensing, provenance, security reporting, and public documentation.
3. Update the version, changelog, and citation metadata intentionally.
4. Enable a least-privilege tag workflow only after reviewing the exact tag and destination.
5. Create the tag and GitHub release explicitly. PyPI publication requires separate authorization and trusted publishing configuration.

Do not publish model bundles or market datasets without separate license, provenance, hash, privacy, and risk review. No current workflow creates releases or publishes to PyPI.
