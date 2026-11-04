# CI cache and smoke change

Scope: setup-uv policy across workflows and a separate Linux smoke job, following decisions 9, 11 and 12 in internal/notes/ci_and_native_builds_2026_11_02.md. No issue requested.

- [x] Inspect pinned action cache semantics and implement policy and smoke split.
- [ ] Run local checks and open draft PR with temporary smoke failure.
- [ ] Verify failure propagation and no uv saves in PR logs, then remove perturbation.
- [ ] Cross-review, final CI timing comparison against main full run 41453451599 (newer main runs skip after queue validation), update description and mark ready.

All phases are sequential. Preserve toolchain installs, build hooks and Windows warming.

Local pre-merge hook passed with SCALY_BUILD_SOLVERS=skip: 2810 tests passed, 50 skipped; type, lint, format, benchmark smoke, docs and private-name checks passed. Review found stale contributor prose, now fixed. Kept gbench saves because restore-only request applies to uv, and documented immutable uv cache tradeoff. Probe run 41809991799 remains running. Review delta check, probe removal, final run and timing evidence remain.
