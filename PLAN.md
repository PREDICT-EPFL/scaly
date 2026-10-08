# Issue #38

Run expression verification at Function construction and before lowering, complete the SLICE
and SOLVER_CALL contracts, test operation coverage, and replace three recursive graph walks.

Read the compiler roadmap Foundations and rules, compiler maintenance context, architecture,
codebase, conventions, contributing and issue tracking. The dispatched brief limits overlap with
#26, #61 and #16. Base and PR base are main. Shared vocabulary merges sequentially.

- [x] Inspect issue metadata, open PR overlaps and current implementation.
- [x] Add failing verifier, coverage and deep Euler graph regressions. Source of the depth case:
  origin/devrush internal/notes/code_review_2026_09_27.html, finding 8.
- [x] Hook verifier into current construction and lowering, add rules, make walks iterative.
- [x] Maintainer chose to retain and pin existing AD formulas and arithmetic identities.
  Generic pass code may not add individual elementwise references. #20 owns shared partials.
- [x] Show coverage fails on a missing classification and an individual elementwise dispatch.
- [x] Updated collection baseline; all seven pre-merge checks passed. C snapshots unchanged.
- [x] #138 remained open and main unchanged at review launch. Draft PR #141 pushed and linked.
- [ ] Opus 5.5 found one blocker: operand CONST guards could mask a missing dispatch.
  Fixed the extractor to read only expr.op comparisons; delta check pending.
- [ ] Address findings, delta check, final evidence, remove plan, ready PR and watch.

Implementation and tests are sequential. Independent checks can run together. Support-rule coverage
joins the invariant in #40, and graph-digest coverage in #64. No change to their current scope.

Validation: 1370 tests passed, 3 skipped; Ruff format/check and strict ty passed.
C snapshots are unchanged. Benchmark smoke, documentation build and the private-name check also passed.
