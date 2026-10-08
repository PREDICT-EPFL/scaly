# #118: lean lane helpers

Branch `t3code/issue-118-lean-lanes`, stacked on `t3code/issue-10-name-authority` (#136).
Design: the issue's *Proposed fix*. Scope: `src/scaly/passes/program/widen_ranges.py`, the lane
defines and vector typedefs in `src/scaly/codegen/c.py`, the `name_scope` keys, tests.

The C snapshots render at `lanes="auto"` and several of their helpers share caps, so sharing the
type and macro applies only when the recipe fixes the lane count. That keeps `tests/baseline/c/`
byte-identical (brief) and is listed as an open point on the pull request.

Reproduction: `/tmp/scaly118/measure.py` (copied into the pull request description). Baseline on
this host (AVX-512, native lanes 8): CT lanes=1 5.25 s, 0.29 MB; lanes=8 10.43 s, 3.64 MB, 732 helpers.

- [x] Failing test in `tests/codegen/test_vectors.py`: scalar below twice the lanes, one shared
      typedef and width macro, bytes equal to the unwidened program
- [x] Scalar rule
- [x] Shared width macro and vector type (fixed lanes only), allocated through `NameScope`
- [x] Single walk of the loop body in `transform`
- [x] Measure CT at lanes 1, 4, 8; perturb the rule and see the test fail
- [x] `wt hook pre-merge`, snapshots byte-identical
- [x] Draft PR #142 on #136, linked as stack #143 with `gh stack link 136 142`
- [ ] Cross-review (gpt-6.1-sol, high) and delta check; post verdict on the PR
- [ ] Kernel timings on la015 (unreachable from this session: open point, PR stays draft)
- [ ] Source-size criterion: not met by the decided design (CT 1.50 MB at 4 lanes vs 0.58 MB
      bound). Evidence and options posted on #118; waiting for the maintainer's decision
