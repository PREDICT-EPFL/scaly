# Remove repeated expression verification in lowering

The constructor verifies every concrete function's inputs and results, including custom-rule
residuals. Lowering currently repeats a subset of that verification in `lower_function` and
`_lower_to_proc`. Normalization returns a new function through `_replace`, which verifies the
transformed graph. The investigation and independent Opus check found no supported bypass.

Scope is limited to those two calls, their import, a regression test, the test collection
baseline, and the verification descriptions in the architecture and IR pages and verifier docstring.
Constructor verification and program verification stay. Generated C snapshots must not change.
This follows roadmap item #38's requirement that graphs are verified before lowering.

- [x] Add a regression for root and callee graphs and observe the redundant checks fail it.
- [x] Remove the repeated checks, update the descriptions, and prove the regression can fail
  when a repeated check is restored. Refresh the full test collection baseline.
- [x] Run `wt hook pre-merge --yes`, including the full suite, C snapshots and benchmark smoke.
- [ ] Commit and open a draft pull request on this branch. Apply `cross-review`, handle its
  findings, remove this temporary plan, mark the pull request ready and watch its checks.

The test must precede the fix. Validation follows the implementation. Independent reviews can
run together after validation, and their fixes precede any required delta check.

The regression failed in both cases before the fix, passed after it, and failed again when a
process-local wrapper restored the root's redundant verification. The focused verifier,
lowering and C snapshot run passed all 74 tests. Full collection passes with 2,502 tests.
The merge hook passed type checking, formatting, lint, 2,413 tests with 89 skips, benchmark smoke,
the documentation build and the private-name check. Solver-dependent checks skip where the
plugin is unavailable. Generated C snapshots are unchanged.

Opus and sol approved with no actionable findings. Opus noted that the test pinned traversal order
and could obstruct skipping unchanged normalization copies, and that the IR guide's wording still
named lowering as a separate verification boundary. The test now compares node counts independent
of order and deliberately changes both fixture graphs during normalization. The IR guide is
aligned. Focused validation and a short Opus delta check remain before removing this plan.

Focused validation after the review notes passed all 74 verifier/lowering/snapshot tests, Ruff,
types and the documentation build. Opus's unchanged-copy normalization probe now passes both
cases, while restoring the root verification still fails both cases as expected. Only the short
Opus delta check remains before removing this plan and preparing the final pull request.
