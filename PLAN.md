# Issue #26: accumulating scatter

Base and PR base: main. Independent branch. Design: `internal/notes/core_compiler_roadmap.md`, Foundations and Rules for every item.

Scope: accept duplicate scatter indices, preserve unique stores, accumulate repeats in index order, add segment_sum builder, replace gather/unbroadcast adjoint scans, extend scatter-sum combination. Preserve C snapshots. Avoid #38/#61 ownership and minimize overlap with #136 in lowering/gather.py.

- [x] Inspect issue, prerequisite #9, open PR overlap, checkout, project instructions and devrush witnesses.
- [x] Add differential tests and graph-size evidence; reproduce failures.
- [x] Implement focused builder, lowering, adjoint and optimization changes.
- [x] Document builder through write-docs; check examples.
- [ ] Prove tests detect perturbations; run pre-merge checks and unchanged snapshots.
- [ ] Commit, push draft PR, link in T3, obtain cross-review and confirm fixes.
- [ ] Remove this plan, mark PR ready, update issue to In review, watch CI.

Relevant implementation: src/scaly/ir/expr.py, ad/reverse.py, passes/lowering/gather.py, passes/program/combine_scatter_sums.py. Tests: tests/ad/test_expr.py, tests/passes/test_program.py, tests/test_c_snapshot.py. Port source: origin/devrush tests/core/codegen/test_control.py, scatter cases; tests/core/ad/test_expr.py, repeated scatter. Segment extrema and runtime indexing are outside scope.

Validation so far: 12 expected failures before the fix; 28 focused tests pass; four targeted perturbations fail as intended. Full suite: 1370 passed, 3 skipped, including byte-identical C snapshots. Ruff format/check and strict ty pass. Pre-merge benchmark smoke and docs build still running. Guide snippet and examples/function_composition.py pass.
