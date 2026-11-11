# Index operands for #35

Branch: `t3code/35-index-operands`. Pull request base: main. Started from the
provided main checkout. Do not rebase merely because origin/main moved.

The user brief, `internal/notes/compiler_run_2026_11_10/plan.md` and
`~/.scaly-suite/RULES.md` govern this task. Full suites run detached only through
the suite runner, with exit-file polling in this turn. GitHub uses REST.

- [x] Read the required project context, implement, tracker, write-plan,
  port-from-devrush, write-docs and cross-review procedures.
- [x] Verify #26/#30 closed, inspect #35/#23 and overlapping PRs, record work on #35.
- [x] Migrate index operands and consumers, retaining constant affine recovery and C snapshots.
- [x] Implement runtime addresses, axes, fill, scratch writes and ordered combinations.
- [x] Implement forward/reverse derivatives, tie splitting, sparsity and support.
- [x] Implement all requested builders and public exports.
- [x] Add guide section/API entries and execute the changed guide examples.
- [x] Test NumPy references, finite differences, ties, NaN, duplicates, empty sources,
  extreme indices, scalar/block hints, and exact set sparsity.
- [x] Prove the tests fail before implementation; perturb tie counting and confirm failure.
- [x] Focused suites, snapshots, format, lint and types pass. Regenerate collection baseline.
- [x] Detached full pre-merge through ~/.scaly-suite/run exited 0: 3796 passed, 55 skipped; smoke, docs, lint, format, types and private-name checks passed.
- [ ] Commit and push review-ready state, open draft PR and immediately link in T3.
- [ ] One Opus cross-review, fix findings, one short delta check. Settle delegates.
- [ ] Remove PLAN.md in the final readiness commit; record criterion evidence/review verdict,
  push all review fixes together and queue own PR with squash auto merge after green checks.

## Evidence and remaining constraints

Initial constant-operand focused run: 1743 passed, including unchanged C snapshots.
After runtime/axis/builder work: 1766 passed in the IR, AD, lowering, snapshot,
operation-coverage and public-boundary selection. A broader passes selection also
completed; retain its log at /tmp/scaly35-current.log until final validation.
Final narrow selection: 75 passed, including source-level memset and snapshots.
Ruff formatting/lint and ty passed. Guide snippets printed [7, -1, 3] and [4, 0, 6, 0].
Tie-count perturbation failed as intended and was restored. No PR or delegate exists yet.

Ports use devrush's tests/core/integration/test_indexing_ops.py (ordered updates,
base derivatives and segment reductions), test_index_edge_cases.py (padded lanes,
empty sources, leading axes, cast indices, scalar/block hints and extreme indices),
and tests/core/ad/test_reduction_derivative_edge_cases.py (extrema ties). In-place
loop tests and configurable nonsmooth conventions are outside #35.

Roadmap departure applied: constant-index zero-base scatters retain their existing
zero-fill loops to preserve byte-identical C; runtime zero bases emit memset.

Existing gap to mention in PR: Python numerical input leaves currently support
float64 only (#55/#56 own typed boundaries). Tests compute int64 index operands by
casting float64 inputs in the graph. Do not expand into typed-boundary work.
No follow-up issue has been filed. Do not touch LOOP behavior or sparse kernel
position-table designs beyond necessary operand migration.
