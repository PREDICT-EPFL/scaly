# #118: lean lane helpers

Branch `t3code/issue-118-lean-lanes`, stacked on `t3code/issue-10-name-authority` (#136).
Design: the issue's *Proposed fix*. Scope: `src/scaly/passes/program/widen_ranges.py`, the lane
defines and vector typedefs in `src/scaly/codegen/c.py`, the `name_scope` keys, tests.

Maintainer decisions after the first review: share the type and macro under `lanes="auto"` too and
regenerate `tests/baseline/c/` (the diff must be only that sharing); land as "Part of #118" with a
follow-up issue under #71 for the source-size criterion; measure kernel timings on this host, which
is la015. Wait for the coordinator before restacking on the rebased #136.

Reproduction: `/tmp/scaly118/measure.py` (copied into the pull request description). Baseline on
this host (AVX-512, native lanes 8): CT lanes=1 5.25 s, 0.29 MB; lanes=8 10.43 s, 3.64 MB, 732 helpers.

- [x] Failing test in `tests/codegen/test_vectors.py`: scalar below twice the lanes, one shared
      typedef and width macro, bytes equal to the unwidened program
- [x] Scalar rule
- [x] Shared width macro and vector type, allocated through `NameScope`
- [x] Single walk of the loop body in `transform`
- [x] Measure CT at lanes 1, 4, 8; perturb the rule and see the test fail
- [x] `wt hook pre-merge`, snapshots byte-identical
- [x] Draft PR #142 on #136, linked as stack #143 with `gh stack link 136 142`
- [x] Cross-review (gpt-6.1-sol, high) and delta check (gpt-6-astra): resolved, posted on the PR
- [x] Share under `auto` too; snapshots regenerated, diff is only the sharing
- [ ] Follow-up issue for the source-size criterion, under #71; reference it in the PR
- [ ] Kernel timings (unbumpercars, race-car, npmpc) on la015 under the benchmark protocol
- [ ] Restack on the rebased #136 when the coordinator says so; rerun the checks; mark ready
