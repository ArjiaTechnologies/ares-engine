# Public release security results

## Current status

The repository is public and the privacy-clean replacement object is in use.
The initial public analysis runs are pending.

## Completed controls

- Dependency graph and Dependabot alerts enabled.
- Dependabot security updates enabled.
- Weekly pip and GitHub Actions version-update configuration present.
- GitHub secret scanning enabled.
- GitHub push protection enabled.
- Private vulnerability reporting enabled.
- Gitleaks 8.30.1 full-history result: clean.
- TruffleHog 3.96.0 full-history result: clean.
- Manual author, committer, path, artifact, and credential review: clean.

GitHub reports non-provider-pattern scanning and validity checks as unavailable
or disabled for this repository tier. Core secret scanning and push protection
are enabled.

## Pending results

- Real public CodeQL run ID and alert review.
- Real public Dependency Review run ID and result.
- Dependabot, code-scanning, and secret-scanning alert enumeration.
- Protected `main` rules with public required checks.

No unresolved Critical or High finding is known. The release remains blocked
until the pending public analyses execute and their results are reviewed.

