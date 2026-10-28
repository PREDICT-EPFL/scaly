# Ipopt native producer, issue 229

Base: `t3code/issue-228-native-bundle-builder`, PR #231. This branch owns producer workflows,
validation scripts and maintainer instructions. Builder/schema changes belong to #228 and
consumer pins and ordinary wheel setup belong to #230.

Inputs read: issue #229, issue #228's fixed design comment, PR #231 description,
`internal/notes/ipopt_native_bundle_contract.md`, `internal/notes/ipopt_native_controls.md`,
the architecture/codebase/conventions/contribution guides, existing wheel/test/release workflows,
`windows_install.py`, `solve_smoke.py`, the builder and solver tests.

- [x] Inspect scope, stack and existing interfaces. Project access lacks `read:project`.
- [x] Add failing checks for trigger selection, dependency/floor audits and publication gates.
- [x] Implement five-target candidate generation with manylinux image identities and retained diagnostics.
- [x] Implement isolated validation using existing solver-required tests and smoke commands, native C/C++
  linkage, load identities, runtime-removal control and retained-source numerical comparison.
- [x] Implement guarded publication of exact validated bytes, with a testable read-only preflight.
- [x] Document candidate/health/publication/rollback procedures and expected pins.
- [ ] Run focused checks and `wt hook pre-merge --yes`; open a draft PR against the builder branch.
- [ ] Run all five candidates in this PR's CI. Iterate single targets when needed, especially Windows.
- [ ] Run one cross-family review and one delta check. Record evidence and maintainer handoff.
- [ ] Remove this plan when complete, mark ready and watch the PR.

Dependencies: validation follows candidate generation; publication consumes only a completed full
main-branch run. Target builds are independent. Monthly health never publishes. No publication,
tag, merge, default-branch push or repository setting change is authorized here.

Local evidence: 14 admission/rejection tests pass. A mutation disabling the glibc floor rejection
makes its test fail. Both workflows pass actionlint. All new Python scripts pass pinned Ruff and ty.
The complete Windows validator passes with a diagnostic metadata repack of the retained #228
archive. It includes 125 existing solver tests and both closed-loop probes. This is validation
script evidence, not a successful new candidate build. The native bytes remain those of #228.

The real audit found #228's `objdump` import inventory is empty for the repaired Windows Ipopt
DLL. A minimal builder fix reads ordinary/delay imports through its already-required `pefile`.
The coordinator was told before committing, then asked to keep the fix out of #229's history.
The fix and regression will land in #228. Locally it remains only as an uncommitted diagnostic
patch. Await the coordinator's new #231 head, then rebase onto
`origin/t3code/issue-228-native-bundle-builder` and push with `--force-with-lease`.
No schema or fixture changed.

Draft PR: https://github.com/PREDICT-EPFL/scaly/pull/233. Initial producer run 40605435231 uses
the incomplete Windows inventory and was cancelled. The next run is on the pushed draft.
Full pre-merge type, format and lint passed. The suite has 2743 passed, 107 skipped and the known
workstation `test_native_recipe_render_policies_separate_cache_entries` failure from missing `cc`.
The benchmark smoke is running separately. The docs build reports no issues. The cold Windows
source build still takes about an hour.
