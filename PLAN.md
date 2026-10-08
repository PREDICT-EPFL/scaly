# #118: lean lane helpers

Branch `t3code/issue-118-lean-lanes`, stacked on `t3code/issue-10-name-authority` (#136).
Design: the issue's *Proposed fix*. Scope: `src/scaly/passes/program/widen_ranges.py`, the lane
defines and vector typedefs in `src/scaly/codegen/c.py`, the `name_scope` keys, tests.

The C snapshots render at `lanes="auto"` and several of their helpers share caps, so sharing the
type and macro applies only when the recipe fixes the lane count. That keeps `tests/baseline/c/`
byte-identical (brief) and is listed as an open point on the pull request.

Reproduction: `/tmp/scaly118/measure.py` (copied into the pull request description). Baseline on
this host (AVX-512, native lanes 8): CT lanes=1 5.25 s, 0.29 MB; lanes=8 10.43 s, 3.64 MB, 732 helpers.

- [ ] Failing test in `tests/codegen/test_vectors.py`: scalar below twice the lanes, one shared
      typedef and width macro, bytes equal to the unwidened program
- [ ] Scalar rule
- [ ] Shared width macro and vector type (fixed lanes only), allocated through `NameScope`
- [ ] Single walk of the loop body in `transform`
- [ ] Measure CT at lanes 1, 4, 8; perturb the rule and see the test fail
- [ ] `wt hook pre-merge`, snapshots byte-identical
- [ ] Draft PR on #136 via `gh stack`, cross-review (Codex), kernel timings on la015 or open point
