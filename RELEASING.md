# Releasing ARES Engine

Releases are deliberate maintainer operations. The workflow never publishes to PyPI, contacts an exchange, uses exchange credentials, or executes orders.

## Required gate

1. Start from a reviewed release branch and confirm the package, CLI, changelog, citation, and release-note versions agree.
2. Run the complete Python matrix, ML lifecycle, dependency audit, package/wheel smoke, Docker job, CodeQL, Dependency Review, and full-history secret scan.
3. Confirm no unresolved Critical or High security finding exists.
4. Merge only through the protected `main` branch after required checks pass.
5. Verify the exact `main` commit, then create and push an annotated `v*` tag pointing to it.

## Tag workflow

The pinned, tag-only release workflow checks out the exact annotated tag, installs `uv.lock` with `--frozen`, repeats static/non-ML/ML/audit/package checks, and tests the installed wheel outside the checkout. It builds with `SOURCE_DATE_EPOCH` set from the tag commit, generates a reproducible CycloneDX JSON SBOM and SHA-256 manifest, and stages a draft GitHub release.

Review the draft asset names, sizes, hashes, SBOM validation, workflow logs, release notes, and tag target before publishing the draft. Delete a malformed draft and investigate rather than replacing assets silently.

Do not attach market data, model bundles, scalers, local databases, credentials, audit recovery bundles, environments, or caches. PyPI publication requires a separate authorization and trusted-publishing design.
