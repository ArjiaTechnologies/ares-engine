# Public release security results

## Current status

The repository is public and the privacy-clean replacement object is in use.
All pre-merge public analyses have completed successfully.

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
- Public CodeQL run `30398479199`: real analysis completed successfully;
  analysis `1540257885` reported zero results across 43 rules.
- Public Dependency Review run `30398479138`: completed successfully.
- Public CI run `30398479089`: seven required jobs passed.
- Code-scanning alerts: zero.
- Dependabot alerts: zero.
- Secret-scanning alerts: zero.
- GitHub dependency-graph SBOM inventory: 38 packages.
- Private vulnerability reporting enabled.
- Branch protection requires pull requests and all applicable public checks;
  force pushes and deletion of `main` are disabled.

GitHub reports non-provider-pattern scanning and validity checks as unavailable
or disabled for this repository tier. Core secret scanning and push protection
are enabled.

Dependabot created ten expected version-update branches when public version
updates activated. Their GitHub-generated metadata and reachable commits were
included in repeated full-history scans: Gitleaks examined 40 commits with no
leaks and TruffleHog examined 560 chunks with zero verified or unverified
secrets. Two standalone CodeQL action-update PRs have internally mismatched
action versions and therefore failing analysis jobs; they are routine version
updates against the pre-release `main`, not findings in the release PR. The
release PR's CodeQL analysis is green and its alert count is zero.

No unresolved Critical or High finding exists. Post-merge and tag-build
verification remain before final `PASS`.
