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
- [x] Opened and linked PR #409. Initial Opus review completed and its thread was settled.
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
Tie-count perturbation failed as intended and was restored. PR #409 is draft; one Opus review completed, with fixes awaiting the single delta check.

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


## Review fixes

All seven act-on findings are addressed locally. New tests reproduced all five
code findings before fixing them, then passed. Sparse values with nonzero bases
are densified. Public scatter has only the base/index/values signature; internal
zero-base calls use _scatter_zero, and old compiler tests now call that helper.
Axis updates require the correct shape. Constant invalid int64 entries normalize
to -1 before affine recovery. Forward rules use one batched op for all seeds.
Published IR assembly was regenerated, the cast limitation and guide wording
were corrected, and the roadmap departure now names its precise scope.

Cheap review notes addressed: zero_fill asserts a single zero store, carry-mask
comments allow nonnegative weights, fill values normalize to their payload dtype,
and scatter constant folding uses ordered NumPy ufunc.at instead of a Python loop.
segment_sum/segment_max bounds defaults and per-position dynamic_slice fill are
explicit in the guide. Whole-row runtime axis sparsity and fill scatter work-buffer
copies remain documented PR gaps, alongside typed numerical boundaries (#55/#56).

Focused selection after fixes: 2237 passed, including unchanged C snapshots.
Additional new regression selection: 83 passed. Format, lint and types passed.
Collection baseline updated to 3860 tests. The detached full suite on fixes exited 0: 3805 passed, 55 skipped, all hooks passed.
The reviewer numerical scripts also passed 48 Jacobian and 16 Hessian cases.
Need the one short delta review, removal of this plan, a single combined fixes push,
ready status, CI and the merge queue. Do not launch another full review.
