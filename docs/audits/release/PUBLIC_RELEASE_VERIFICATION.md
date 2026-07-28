# ARES Engine v0.1.0 public release verification

## Verdict

PASS

The controlled public-release gates completed on 2026-07-28. The released
software is commit `35475b63accd64fe484f04a5f1cec4dc14090186`, annotated tag
object `c0b9612258b1cfe8c8ac30c0beea298982b3a614` (`v0.1.0`). No unresolved
Critical or High security finding remains.

## Provenance and privacy gate

- Original pre-rewrite main: `90c6f3646ca32a4a0ca0e68a887e0853418cc5cd`
- Original pre-rewrite tree: `181a15cae0c44fbaacd6101e6e83ef17a43d2156`
- Rewritten clean main: `3de1360cbe95d004c04a05345a89ba3eab29f240`
- Release-preparation commit: `870f09555e6a01d1914f76c57893b9151509053c`
- Final release commit: `35475b63accd64fe484f04a5f1cec4dc14090186`
- Final release tree: `40ed026ffc0032518a10765051a1790efd17b9f7`
- Annotated tag object: `c0b9612258b1cfe8c8ac30c0beea298982b3a614`
- Old repository ID: `R_kgDOTlVbxQ`
- Replacement repository ID: `R_kgDOTmei-A`
- Original recovery bundle SHA-256: `b8a5601ca07ffb475f1a60ddee09c7d8d86799fc4de1362549e013888c2380b1`
- Clean pre-replacement bundle SHA-256: `e6e39e57909b1bc116b96eb324b755def649a9b89a44ddd0b695a98788f2fea0`
- GitHub Support ticket: `4610730` (still pending when replacement began)
- Clean release PR: <https://github.com/ArjiaTechnologies/ares-engine/pull/1>

GitHub Support had not purged the managed ref in time, so the private
repository object was replaced. The old object was deleted only after both
recovery bundles, the clean replacement history, its managed PR ref, and its
private CI were verified. In the replacement repository, the offending commit
`226c17e528e8881d72f7c79fcd8e902715708afd` returns HTTP 422 through the
authenticated commit API and HTTP 404 anonymously. It is absent from the
verified clean mirror and cannot be resolved through the replacement object.

## Secret and metadata review

- Gitleaks `8.30.1`: `gitleaks git . --log-opts="--all"`; 29 clean-history
  commits before the public transition; no leaks.
- TruffleHog `3.96.0`: `trufflehog git file://$(pwd) --only-verified`; 524
  chunks before the public transition; zero verified and zero unverified
  secrets.
- After Dependabot activation, Gitleaks examined all 40 reachable commits with
  no leak and TruffleHog examined 560 chunks with zero verified or unverified
  secrets.
- Complete author and committer review found no `.local` identity, hostname,
  private email, ChatGPT/Codex workspace path, or unintended local username.
- `git fsck --full` completed cleanly in the replacement mirror.

## Visibility and public presentation

- Public visibility timestamp: `2026-07-28T20:52:58Z`
- Repository: <https://github.com/ArjiaTechnologies/ares-engine>
- Anonymous repository and release requests: HTTP 200
- Anonymous old-commit request: HTTP 404
- Default branch: `main`
- GitHub detects the MIT license and renders the public README.
- The repository description and topics accurately identify a machine-learning
  trading research and paper-inference project without profit or live-capital
  claims.

## CI, analysis, and test results

- Clean replacement private CI: run `30397558077`; seven required jobs passed.
- Final release-PR CI: run `30399030376`; seven required jobs passed.
- Final release-PR CodeQL: run `30399029899`, analysis `1540290727`; 43 rules,
  zero results.
- Final release-PR Dependency Review: run `30399030808`; passed.
- Post-merge CI: run `30399219862`; seven required jobs passed.
- Post-merge CodeQL: run `30399219968`, analysis `1540303603`; 43 rules, zero
  results on the exact release commit.
- Tag-driven release build: run `30399424098`; passed every required step.
- Python 3.11, 3.12, and 3.13: 195 non-ML tests passed on each interpreter;
  50 were deselected; branch-aware non-ML coverage was 82%.
- TensorFlow CPU lifecycle: 50 ML tests passed; 195 were deselected; bounded
  offline verification passed.
- Complete local suite: 245 tests passed with 90% branch-aware coverage.
- Dependency audit: no known vulnerabilities.
- Package build, Twine check, isolated installed-wheel smoke, Docker non-root
  smoke, and graceful shutdown: passed.
- Fresh anonymous clone on Python 3.12.13: 195 non-ML tests passed; `ares doctor`
  and `ares demo --bars 500` passed. An initial run using the host's unsupported
  Python 3.9.6 was rejected by package metadata as designed.

## Release artifacts

Release: <https://github.com/ArjiaTechnologies/ares-engine/releases/tag/v0.1.0>

- `ares_eth_engine-0.1.0-py3-none-any.whl`:
  `0baff3ef3260003ef6560d32ddc95908424f3ed25e41a4b8b001fba3b82f3398`
- `ares_eth_engine-0.1.0.tar.gz`:
  `0ac9059fcf9a886fb4f6e9b22f63761a5a180f842e3d9fdf563bf55f13325ca1`
- `SHA256SUMS.txt`:
  `1265394974c330ff88c717539dbfb51324fca6ed7d2df621a64342cc7d7001a1`
- `ares-engine-v0.1.0.cdx.json` (CycloneDX 1.6, 61 components):
  `20f4433b1256aaafb486e3680249a75f0d6dc10ca8850db6d44f6c65b51dca04`

All three payload hashes match `SHA256SUMS.txt`. Archive inspection found no
market data, generated model or scaler, database, credential, private recovery
bundle, virtual environment, or cache. PyPI returned HTTP 404 for
`ares-eth-engine`; no PyPI publication occurred.

## Security controls

Dependency graph, Dependabot alerts, Dependabot security updates, weekly
version updates, secret scanning, push protection, CodeQL, Dependency Review,
and private vulnerability reporting are enabled. Code-scanning, Dependabot,
and secret-scanning alert counts were all zero at release. `main` requires pull
requests and strict passing checks for Python 3.11, 3.12, and 3.13; TensorFlow;
package/installed-wheel smoke; dependency audit; Docker; CodeQL; and Dependency
Review. Force pushes and branch deletion are disabled, enforcement includes
administrators, and conversation resolution is required.

## Remaining risks

- ARES does not establish profitability, provide investment advice, or place
  live orders. Historical performance is not evidence of future results.
- The system does not implement a locked nested post-search final holdout.
- OHLC-based simulation cannot establish intrabar ordering or actual fills;
  fees, slippage, latency, spread, depth, liquidity, and market impact remain
  assumptions.
- Public exchange endpoints, schemas, rate limits, retention, and CCXT mappings
  may change.
- `joblib` and Keras deserialization remain trust boundaries; model bundles
  must be treated as authenticated, trusted local artifacts.
- Host integrity, dependency supply chains, TLS trust, filesystem durability,
  and market-regime change remain external risks.

No credentials, generated market/model artifacts, or live-order path were
published or added.
