# Windows pre-merge checks, issue #198

Implementation profile. Base and PR target: main. Branch: t3code/issue-198-windows-checks.

Scope follows #198, #156's scope comment, parent #84, the decisions in
internal/notes/windows_toolchain_options_2026_10_09.md, the compiler roadmap's order and
rules, and internal/notes/windows_native_2026_10_22/. Architecture, codebase, conventions
and contributing instructions were read. Native measurements remain on Linux la015.
No core changes, snapshots, packaging, workflows or public installation documentation.

- [x] Inspect the issue, prerequisites, checkout and overlapping PRs. Project access is
  blocked by missing read:project scope; issue and PR work can proceed.
- [x] Add focused failing tests for timeout cleanup, deadlines and smoke selection.
- [x] Guard Windows paths while preserving POSIX behavior and supported numerical checks.
- [x] Update the full collection baseline; prove the new checks fail under perturbation.
- [x] Run Windows types, smoke and the full wt pre-merge hook with the toolchain extra.
- [ ] Obtain one cross-provider review and a fixes-only check if needed.
- [ ] Commit and push, open/link a draft PR against main, report verdict to the coordinator.
- [ ] Record criterion evidence and CI state, register the PR watch.

Implementation and validation are sequential. Independent review follows passing checks.
#156 starts from this committed branch after the review verdict. PR #166 overlaps the smoke
and regression files and owns snapshots; all snapshots must stay byte-identical.

Validation: toolchain extra synced; focused tests 41 passed, 6 skipped. Removing each
Windows guard made its regression fail, then all guards were restored. Native Windows
ty, smoke and wt hook pre-merge --yes passed. Full suite: 2474 passed, 147 skipped,
2 xfailed. Full collection: 2623. C snapshots unchanged. Linux/macOS remain for PR CI.
