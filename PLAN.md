# Ipopt native producer, issue 229

Base: `t3code/issue-228-native-bundle-builder`, PR #231. This branch owns producer workflows,
validation scripts and maintainer instructions. Builder/schema changes belong to #228 and
consumer pins and ordinary wheel setup belong to #230.

Inputs read: issue #229, issue #228's fixed design comment, PR #231 description,
`internal/notes/ipopt_native_bundle_contract.md`, `internal/notes/ipopt_native_controls.md`,
the architecture/codebase/conventions/contribution guides, existing wheel/test/release workflows,
`windows_install.py`, `solve_smoke.py`, the builder and solver tests.

- [x] Inspect scope, stack and existing interfaces. Project access lacks `read:project`.
- [ ] Add failing checks for trigger selection, dependency/floor audits and publication gates.
- [ ] Implement five-target candidate generation with manylinux image identities and retained diagnostics.
- [ ] Implement isolated validation using existing solver-required tests and smoke commands, native C/C++
  linkage, load identities, runtime-removal control and retained-source numerical comparison.
- [ ] Implement guarded publication of exact validated bytes, with a testable read-only preflight.
- [ ] Document candidate/health/publication/rollback procedures and expected pins.
- [ ] Run focused checks and `wt hook pre-merge --yes`; open a draft PR against the builder branch.
- [ ] Run all five candidates in this PR's CI. Iterate single targets when needed, especially Windows.
- [ ] Run one cross-family review and one delta check. Record evidence and maintainer handoff.
- [ ] Remove this plan when complete, mark ready and watch the PR.

Dependencies: validation follows candidate generation; publication consumes only a completed full
main-branch run. Target builds are independent. Monthly health never publishes. No publication,
tag, merge, default-branch push or repository setting change is authorized here.
