# Issue #249

Implement decisions 13 to 15 in `internal/notes/ci_and_native_builds_2026_11_02.md`.
Prerequisite #248 is merged. The starting point is `nightly.yml`, `ci-build.yml`,
`ci-test.yml`, the retired nightly-only workflows, and `release.yml`. The release
record and contributor guide must describe the finished behavior.

- [x] Inspect the issue, prerequisite, open PRs, worktree, and Project metadata.
- [ ] Add executable checks for classifier-derived Python entries and failure labels.
- [ ] Reuse CI builds and tests for three independent nightly axes. Start compile and
  Google Benchmark caches cold, save baseline platform caches for CI, and expose a
  Monday source-rebuild override. Preserve native cache keys and Windows exclusions.
- [ ] Make releases call the full nightly and collect only that run's tested files.
- [ ] Update the contributor guide and release record through `write-docs`.
- [ ] Run targeted checks, perturb the new checks, then run `wt hook pre-merge --yes`.
- [ ] Push a draft PR, run manual nightly validation, and obtain one independent
  cross-review followed by a check limited to any fixes.
- [ ] Remove this plan, mark the PR ready, update Project status, and watch CI.

Implementation and validation are sequential in this worktree. The independent
review follows passing local checks. No compiler changes or snapshot regeneration
are in scope, and devrush has no applicable nightly workflow tests to port.

The main-only compile-cache handoff and TestPyPI release require post-merge evidence.
The implement skill excludes deployment workflow execution, so a TestPyPI dispatch
needs the maintainer's authorization once the change is concrete and reviewable.
