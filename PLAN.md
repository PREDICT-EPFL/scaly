# Support on every expression node

Implement PREDICT-EPFL/scaly#40 on `t3code/39-sparse-pattern`, second in the
#39 through #43 stack. The sparse-tensors section and Rules for every item in
`internal/notes/core_compiler_roadmap.md`, issue #40's discussion, PR #274,
the compiler maintenance notes and project contribution rules govern the work.

The stored pattern remains a type contract even when support is smaller.
Support is lazy, iterative, cached outside interning, and defaults to unknown.
CALL uses only the callee's own support. Argument specialization belongs to #221.

## Boundaries

- Leave `src/scaly/passes/lowering/` untouched. File the zero-skipping follow-up
  after #27. Leave full-size allocation and row-pointer work to #41.
- Keep C snapshots byte-identical. Stop and report if zero folding moves one.
- Keep sparse names out of the guide and filter support from the API page.
- SELECT, general REDUCE and LOOP do not exist on this base. Cover SUM and VMAP
  now and identify the later op rules in the handoff.
- Keep edits in `ir/expr.py` focused because #25 edits it in parallel.
- Do not run `gh stack`, merge, tag or push to main.

## Work

- [x] Read the task, parent, prerequisite, design, existing rules and checks.
- [x] Add failing rule, cache, simplification and derivative tests against NumPy.
- [x] Implement support with the shared pattern algebra and connect simplification.
- [x] Restrict derivative relations using support, without the #41 rewrite.
- [x] Run randomized soundness and mutation checks, the suite, snapshots and hooks.
- [x] Add the guide sentence, API filter and named follow-ups.
- [x] Commit, push and open the stacked draft PR, #277.
- [ ] Finish the cross-review fixes and delta checks.
- [ ] Record criterion evidence, mark ready, update tracker and watch the PR.

Phases depend on the preceding phase. Independent validation commands may run together.

## Evidence so far

- The initial support suite failed all 39 cases before implementation.
- The focused IR/AD suite passed 1,265 cases. After the metadata and coverage
  integration, the focused support, sparsity, coverage, lowering and snapshot
  run passed 113 cases.
- Twelve in-memory perturbations were detected, including missing support,
  missing coefficient/row filters, disabled disjoint folding, missing metadata
  classification and recomputation after the first support read.
- Benchmark smoke, docs build, two linked examples, Ruff and strict ty pass.
- The full collection has 2,654 tests. The first full run found two structural
  expectations to update, the empty-product reduction loop and the op coverage
  allowlist. The final full run passes 2,602 tests with 52 skips, and the full
  `wt hook pre-merge --yes` passes, including benchmark smoke, docs and private names.
- C snapshots and lowering source have no diff against the stack base.
- Scalar zero skipping is tracked by #276, after #27 and #40. SELECT belongs
  to #30, axis REDUCE to #66 and LOOP to #31. Their ops are absent here.

## Review fixes

- Sol reproduced loss of custom derivative rows outside primal support. Removed
  the blanket result-support filter, retaining sparse input bounds and the local
  coefficient filters. Ordinary operations propagate supported relations without
  that filter. Custom declared and undeclared derivatives survive calls, maps,
  nested calls, unary operations, multiplication, division and matrix products.
- All eight new custom-derivative cases failed before the fix. The focused run
  now passes 116 tests. A new check verifies ordinary relations remain within
  every intermediate node's support. Both new regression gates detect in-memory
  perturbations, bringing the mutation checks to fourteen.
- Sol confirmed that fix at low effort. The full merge hook on it passes 2,611
  tests with 52 skips, plus all remaining checks.
- Opus reproduced expensive sorting, validation and digesting in flat unions and
  intersections. They now use direct set algebra under the shared `OP_INFO`
  rules, retaining `CanonicalPattern.matmul` for Boolean matrix products. The
  same `chain_jac:65` generation reproduction takes 6.6 seconds here against
  5.0 seconds on the base, rather than the review's 79 to 88 seconds before the
  fix. Remaining support-cache memory is recorded on #41.
- Replaced the guide's abstract sentence with an executed example returning
  `[0, 0]` while NumPy returns `[0, nan]`. Updated the simplify docstring,
  contributor support-classification instructions, descriptor typing and cache
  wording. Print elimination follows existing semantics. The non-snapshot
  `npmpc` kernel's removal of a zero accumulation term is expected.
- Fifteen perturbations are detected, including changing flat intersection to
  union. The focused run passes 117 cases. The full merge hook is rerunning on
  these fixes. Opus's short delta check remains before readiness.
