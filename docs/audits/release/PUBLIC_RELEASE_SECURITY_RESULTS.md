# Public release security results

## Final status

PASS

The public release uses replacement repository object `R_kgDOTmei-A`. The old
repository object `R_kgDOTlVbxQ` was deleted after GitHub Support had not yet
purged its read-only pull-request ref. The offending commit is absent from the
replacement mirror, returns HTTP 422 through the repository commit API, and
returns HTTP 404 anonymously.

## Scanner results

- Gitleaks 8.30.1, `gitleaks git . --log-opts="--all"`: no leaks.
- TruffleHog 3.96.0, `trufflehog git file://$(pwd) --only-verified`: zero
  verified and zero unverified secrets.
- The pre-public scan covered 29 clean-history commits and 524 TruffleHog
  chunks. A repeat after Dependabot activation covered 40 reachable commits
  and 560 chunks with the same clean result.
- Manual complete-history author, committer, path, artifact, encoding, and
  credential review: clean.
- Clean replacement mirror `git fsck --full`: passed.

No `.local` identity, machine hostname, private email, unintended username,
workspace path, credential, authorization header, private key, certificate,
cookie, token, generated dataset, model, scaler, database, recovery archive,
or unrelated reconstructable blob was found.

## GitHub security analyses

- Final release-PR CodeQL run `30399029899`, analysis `1540290727`: real
  `analyze` job passed, 43 rules, zero results.
- Final release-PR Dependency Review run `30399030808`: real review action
  passed.
- Post-merge CodeQL run `30399219968`, analysis `1540303603`: passed on release
  commit `35475b63accd64fe484f04a5f1cec4dc14090186`, 43 rules, zero results.
- Code-scanning alerts: zero.
- Dependabot alerts: zero.
- Secret-scanning alerts: zero.
- Dependency audit: passed with no known vulnerabilities.
- No unresolved Critical or High finding exists.

## Enabled controls

- Dependency graph and Dependabot alerts.
- Dependabot security updates.
- Weekly pip and GitHub Actions version updates.
- GitHub secret scanning and push protection.
- CodeQL code scanning.
- Dependency Review on pull requests.
- Private vulnerability reporting.
- Protected `main` requiring pull requests and strict passing checks for the
  Python matrix, TensorFlow lifecycle, package/installed-wheel smoke,
  dependency audit, Docker, CodeQL, and Dependency Review.
- Force pushes and deletion of `main` disabled; administrator enforcement and
  conversation resolution enabled.

GitHub reports non-provider-pattern scanning and validity checks as unavailable
or disabled for this repository tier. Core secret scanning and push protection
are enabled.

Dependabot created ten expected version-update branches when public version
updates activated. Their GitHub-generated metadata and reachable commits were
included in the repeat scans. Two independent CodeQL-action update PRs used
temporarily mismatched init/analyze action versions and therefore failed their
own CodeQL jobs; this is a dependency-update PR coordination issue, not a
finding in the v0.1.0 release. The release PR and post-merge main analyses are
green and report zero alerts.

## Publication exclusions

The release contains only a wheel, source distribution, checksum manifest, and
CycloneDX 1.6 SBOM. It contains no credentials, generated market data, model
bundle, scaler, database, private audit or recovery bundle, environment, or
cache. No PyPI publication occurred, and no live-order capability was added.
