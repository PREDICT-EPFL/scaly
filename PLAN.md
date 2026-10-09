# #20: one elementwise table

Branch `t3code/issue-20-elementwise-table`, stacked on `t3code/issue-19-one-forward-traversal` (#166).
Design: `internal/notes/core_compiler_roadmap.md`, *One AD engine*, [#20], and decision 2.

- [x] Tests first: `tests/ad/test_rules.py` (table rows, float32 partials, lazy partials); #161's xfails removed
- [x] `src/scaly/ad/rules.py` (import layer 2): rows with partials, prescribed `jvp`, NumPy fold, program op, C spelling, `expensive`; `tangent` and `cotangent` helpers
- [x] Forward and reverse read the table; per-mode elementwise formulas, `_seed_axis`, `_broadcast_tangent`, `_unbroadcast` move or go
- [x] Fold (`OpInfo.numpy`), lowering maps, `_UNARY_C`/`_BINARY_C`, `_EXPENSIVE_OPS` derive from the table
- [x] `IMPORT_LAYERS`, op-coverage pins, package map, *Where to add things*
- [x] Checks: `wt hook pre-merge --yes`, snapshots byte-identical, perturbed entry fails the #15 harness
- [ ] Draft PR, `gh stack link`, cross-review, description, ready
- Follow-up filed as #168: float64 `zeros_like`/`_ones_like` in reverse structural rules and `gradient` break float32 reverse mode
