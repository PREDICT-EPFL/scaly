# float32 lowering and exact integer constants

Implement #55 under `internal/notes/compiler_run_2026_11_10/plan.md` on main.
Stack base: `t3code/30-predicates-select-cast`, PR #320. Section coordinator: #24.
The #55 paragraph of `internal/notes/core_compiler_roadmap.md` decides the scope.
Run-wide rules: `~/.scaly-suite/RULES.md`, read before testing.

- [x] Read the issue, parent, run plan, compiler design and required repository guides.
- [x] Add failing NumPy differential tests for float32 operations, folding, store rounding,
  lanes, integer constants and sums, verifier errors and workspace/coalescing limits.
- [x] Implement dtype-faithful constants, folds, C spellings, lowering and program passes.
- [x] Remove ProgramNode's default dtype and make all callers explicit.
- [x] Run focused tests, perturb the new checks, regenerate the collection baseline,
  and run all pre-merge checks through the shared detached runner.
- [ ] Open the stacked draft PR, obtain Opus cross-review and sol correctness review,
  address findings together, and obtain a delta check where needed.
- [ ] Remove this temporary plan, report criteria evidence and verdict, mark ready.
- [ ] Once #320 merges, rebase onto main, retarget the PR, verify and queue it.
  Handle CI and queue ejections until merged.

## Boundaries and reversible choices

- Entry signatures, headers, JIT arrays and the public float32 parity gate belong to #56.
  Test float32 internals through float64 leaves and direct typed raw procedures.
- Keep non-float64 buffers on the stack until #56 supplies typed workspace rules.
- Leave float32 coalescing untouched and pin it with a test, as the roadmap permits.
- #340 owns signed wrapping arithmetic. This task owns literals, integer abs/min/max
  and integer accumulator initialization. #339 owns masks inside called Functions.
- Watch overlaps with #315, #349, #331, #314, #327/#346, #330 and #340 when rebasing.
- Float64 C snapshots must stay byte-identical. Any required change needs reviewed
  source/operation counts and generated-C compile time under the run plan.

## Evidence and remaining work

- Initial tests reproduced unrounded float32 arithmetic and literals, exact integer abs
  losing low bits, and the integer sum builder refusing integer inputs.
- 84 new focused cases pass, including special-value and libm cases adapted from
  `origin/devrush:tests/core/codegen/test_elementwise_edge_cases.py`. The broader pass, program-IR and vector selection passed
  754 tests, with 47 compiler-dependent skips. Eight existing C snapshot checks passed
  without changing a snapshot. Lint, format and strict type checks pass.
- Mutations restoring double folding and float32 workspace spills both fail their tests.
- After moving onto the updated #320 base, the complete collection contains 3264 tests.
- Other devrush float32 cases cover derivatives and typed entry points owned by #168/#56,
  and dynamic indexing owned by #35.
- Roadmap departure recorded beside #55: defer non-float64 spills to #56 so this layer
  does not implement its typed workspace. Report this in the PR and to the coordinator.
- Rebased onto the updated #320 branch. Resolved call lowering by retaining main's
  sibling-result grouping and adding explicit dtypes. A new workspace regression on
  that base also needed an explicit CALL dtype. Strict types pass after that correction.
- Full pre-merge checks are detached under `/tmp/scaly-55-suite-2.*`, with the
  original shared-suite queue position retained. The first runner was cancelled while
  still waiting for its slot, before any checks ran.
- Full suite passed 3208 tests with 52 skips. Four special-value tests added afterwards
  passed in the 84-case focused selection. The shared pre-merge hook exited 0, including
  benchmark smoke, documentation and private-name checks. The documentation build
  has existing link warnings. The final 3264-test collection, lint, format and types pass.
- Draft PR: https://github.com/PREDICT-EPFL/scaly/pull/360, linked to this thread.
- Sol correctness review found two midpoint rounding defects: Python integer conversion
  could round through double, and nine-digit table formatting ran before float32 rounding.
  Both have failing reproductions and fixes using direct NumPy array conversions.
  The focused arithmetic, dtype and snapshot selection passes 259 tests. Eight added
  midpoint cases bring the dtype selection to 92 cases and the full collection to 3272.
  The full hook rerun under `/tmp/scaly-55-suite-3.*` exited 0: 3220 passed, 52 skipped,
  benchmark smoke, docs and private names passed. Lint, format and strict types passed.
  Obtain the short sol fix check and the main Opus review before the final push.
- Sol's short fix check approved both midpoint fixes, with all 92 dtype cases passing.
- Opus's main review requested expression-boundary midpoint conversion, the stale
  lowering policy paragraph, and deletion of the unreachable float32 CAST fallback.
  Those are fixed. Integer abs is now admitted by the expression API and covered in
  both integer widths; lane-mask names/comments and the workspace roadmap wording
  are corrected. The const_float convenience default stays for hand-built programs.
- Floating expression conversion uses copy=False to preserve compact broadcast
  storage; the existing huge-uniform test now covers both float widths. The focused
  dtype/compact selection passes 105 tests, with 103 dtype cases. Full collection
  is 3284 tests. Run the shared hook again, then request one Opus delta check only.
- Non-actionable review notes: NumPy and libm float32 transcendentals can differ
  within the ulp policy; folded and runtime sums use different orders already on
  main; native C contraction differences also predate this task. No new issue needed.
