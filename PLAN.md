# Vendor Fatrop, issue #237

Implement on `ted/issue-237`, based on main. The package supplies native libraries and headers,
not the Scaly solver adapter tracked by #239. Prerequisite #250 is merged. Parent #236 supplies
the context. No open PR changes the native hooks or regenerates snapshots for this work.

Sources: issue #237 and parent #236, `plugins/scaly-piqp/{hatch_build,native_build}.py`,
`plugins/solver_build.py`, `internal/notes/ci_and_native_builds_2026_11_02.md`,
`internal/notes/benchmark-buildout.md` section 3.1, `internal/notes/vendored_solvers.md`, and the
project architecture, codebase, conventions, contributing and issue-tracking guides.

1. [x] Inspect prerequisites, parent, checkout and open PRs. Record work in the tracker.
2. [ ] Inspect Fatrop v1.2.0, then write and run failing native C and packaging tests.
3. [ ] Add the package, static hidden BLASFEO build, cache/release reuse and Linux/macOS CI.
4. [ ] Prove the C solve and coexistence with PIQP, wheel licenses and cache reuse. Perturb the
   implementation to verify the new tests fail. Record build and license constraints.
5. [ ] Run all merge checks, commit, push and open a linked draft PR.
6. [ ] Run cross-review, address findings and confirm fixes with a delta check. Remove this
   temporary plan, mark the PR ready, update the tracker and watch CI.

Steps are sequential. Cross-platform CI provides Linux arm64 and macOS evidence unavailable
on the local Linux x86 machine. Keep generated C snapshots byte-identical.
