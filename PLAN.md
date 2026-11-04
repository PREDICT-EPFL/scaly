# CI cache and smoke change

Scope: setup-uv policy across workflows and a separate Linux smoke job, following decisions 9, 11 and 12 in internal/notes/ci_and_native_builds_2026_11_02.md. No issue requested.

- [x] Inspect pinned action cache semantics and implement policy and smoke split.
- [ ] Run local checks and open draft PR with temporary smoke failure.
- [ ] Verify failure propagation and no uv saves in PR logs, then remove perturbation.
- [ ] Cross-review, final CI timing comparison against main full run 41453451599 (newer main runs skip after queue validation), update description and mark ready.

All phases are sequential. Preserve toolchain installs, build hooks and Windows warming.

Local pre-merge hook passed with SCALY_BUILD_SOLVERS=skip: 2810 tests passed, 50 skipped; type, lint, format, benchmark smoke, docs and private-name checks passed. Review found stale contributor prose, now fixed. Kept gbench saves because restore-only request applies to uv, and documented immutable uv cache tradeoff. Probe run 41809991799 remains running. Review delta check, probe removal, final run and timing evidence remain.

Delta review approved. Verdict posted on PR. Contributor prose wraps corrected locally; do not push until probe completes because pushing cancels its CI run. Latest main runs skip all checks after merge-queue validation; latest full main push run is 41453451599, 27m06s wall time, 102.07 runner minutes across 25 jobs (older layout, so disclose this limit). Data: /tmp/scaly-main-baseline-jobs.json. A more comparable queue run on current main would be useful alongside that required main baseline. Full probe logs are unavailable through gh until run completion.

Maintainer correction: use uv.lock keys with one shared matrix.platform suffix, retaining pruning and main/baseline-nightly save guards. Stable-key choice is superseded. Current probe still runs the earlier code; preserve it until propagation evidence is captured, then push corrected key policy with probe removal. Do not claim ten fixed entries. Expected entries: five per lock generation accessed within seven days, plus runner-image generations. Need pruned-byte measurement from first main/nightly save; no new review round.

Probe smoke failed as intended. Linux test log126537053187 and Windows build log126506770881 explicitly report save-cache:false and skipping save. Aggregate ci waits on Windows test; do not push until it fails. Probe removed locally. Windows build rebuilt native libraries for ~80min despite restoring a fallback key with different recipe hash, so final timing comparison must disclose cold native confound. Logs in /tmp/scaly-{smoke-probe,test-probe-linux,build-probe-windows}.log.
