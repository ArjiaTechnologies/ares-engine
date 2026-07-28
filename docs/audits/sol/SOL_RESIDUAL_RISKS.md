# Sol residual risks

No unresolved Critical or High software defect was found in the reviewed scope. The following risks remain and prevent stronger claims than software release-readiness.

| Severity | Risk | Current control / required future work |
|---|---|---|
| Medium | `joblib` and Keras deserialization are code/object trust boundaries; hashes do not authenticate a publisher. | Load only protected, trusted local bundles. Add signed manifests, authenticated publisher identity, and transparency/provenance verification before external bundle distribution. |
| Medium | Search and early stopping do not provide an automated nested untouched final holdout. | Treat walk-forward scores as selection estimates. Add a locked post-search dataset and one-time evaluation before performance claims. |
| Medium | OHLC bars cannot determine intrabar ordering or true fills, latency, spread depth, and market impact. | Same-bar dual barrier hits are neutral; use tick/order-book replay and conservative execution modeling for future studies. |
| Medium | A single host/filesystem compromise can modify code, dependencies, evidence, data pointers, and trusted bundles. | Private repo, CI, manifests, locks, and least privilege reduce accidents, not a hostile host. Add signed builds, isolated runners, protected environments, and backup/restore controls. |
| Low | External exchange APIs, retention, CCXT mappings, TLS trust, rate limits, and schema behavior can change. | Bounded retries, fail-closed data checks, two venues, manual live workflow. Re-run live evidence before release and monitor provider changes. |
| Low | `fsync` durability depends on OS/filesystem semantics; atomic replace is guaranteed only within the same filesystem. | Generation and pointer temporaries are created under the same root. Document supported filesystems and add power-loss/platform testing for production operations. |
| Low | Dependency audit and CodeQL cannot prove dependency or container-base absence of all vulnerabilities. | Pin action commits, use version bounds, run Dependabot/audits/CodeQL, review and rebuild regularly; consider image/SBOM scanning and digest-pinned base images. |
| Low | Process locks serialize cooperating ARES processes but cannot stop privileged or non-cooperating writers. | Protect artifact/data directories with OS ownership and deploy ARES under a dedicated unprivileged account. |
| Informational | Research results can fail under regime change even when software is correct. | No profitability claim; require forward paper testing, drift monitoring, and independent quantitative review. |
| Informational | Live trading controls do not exist. | Intentional scope boundary. Any order service requires a separate security/risk architecture and audit. |

The repository must remain private in this task. No tag, GitHub release, PyPI publication, credentials, or live-order feature is authorized.
