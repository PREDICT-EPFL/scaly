# Unbumpercars contraction regression

Apply the validated prototype from `/tmp/unb-fix.patch`, preserved locally in
`internal/notes/perf_2026_10_29/unb-fix.patch`. The investigation is in the related
issue #85 comment and the main checkout's `internal/notes/perf_2026_10_29/unb_findings.md`.
The relevant ownership rules are in `docs/dev/codebase.md`, with ordered reductions
in `src/scaly/passes/lowering/contraction.py` and their differential test in
`tests/passes/lowering/test_lowering.py`.

Completion requires four-row narrow panels, eight-column wider panels, exact ordered
results including tails, unchanged C snapshots, passing required checks and one cross-review.
Chain M=5 and M=9 are outside scope. Do not measure benchmarks on la015, merge or queue the PR.

All phases are sequential because validation and review depend on the final implementation.

- [x] Preserve the prototype and read its investigation and compiler context.
- [x] Apply the test and prove it rejects the old schedule, then apply the lowering fix.
- [x] Perturb the accumulator and confirm numerical failure, then restore.
- [x] Run the required checks after removing stale perturbation bytecode.
- [ ] Commit and obtain one cross-review, addressing findings and checking only any fixes.
- [ ] Remove this temporary plan, push and open the PR with validation and prior measurements.

The schedule assertion rejected the old lowering in seven cases. The initializer perturbation
failed the five-row/two-column numerical test. After restoring the source and removing stale
Python bytecode, all 89 lowering tests pass. The first whole-suite attempt loaded stale
perturbation bytecode and must be replaced by a clean run. Benchmark smoke is excluded because
it records timings on la015, forbidden by this task.

Clean non-benchmark pre-merge hooks passed: strict ty, formatting, lint, the full suite
with 2,759 passed and 53 skipped, documentation build, and private-name check. Explicit
`uv run ruff format` and `uv run ty check` also passed. C snapshots remain unchanged.
