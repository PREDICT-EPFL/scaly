# Signed integer wrapping, #340

Base: `t3code/30-predicates-select-cast`, pull request #320. The compiler run plan on
main and `~/.scaly-suite/RULES.md` override the implement workflow where they differ.
Design: `internal/notes/core_compiler_roadmap.md`, Dtypes and the scalar vocabulary.

- [x] Read the task, parent #24, prerequisite #30, run plan, rules and compiler context.
- [x] Inspect C rendering, shared constant evaluation and expression folding. Check the
  frozen devrush edge tests, already represented by main's select/cast tests.
- [x] Reproduce signed overflow with NumPy differential tests for int32/int64 in block
  and scalar lowering, plus a standalone C executable with the signed overflow sanitizer.
- [x] Render signed data arithmetic through unsigned values and wrap scalar constant
  evaluation. Keep bounded compiler index expressions and float64 C snapshots unchanged.
- [x] Prove the tests detect regressions, refresh the collected test baseline, run focused
  checks, then detached full pre-merge checks through the suite queue.
- [ ] Commit, push once for review and open a draft PR. Cross-review with Opus and a
  correctness review with sol, address findings and confirm any fixes with a delta check.
- [ ] Delete this temporary plan before ready. After #320 merges, rebase onto main,
  retarget the PR, verify, queue after the verdict and green checks, and watch until merged.

The change excludes literals, libm, abs/min/max, accumulators, floor division, remainder,
and reverse-mode masks. Open overlapping work includes #315, #331 and #55.

Focused proof: 102 tests passed, including the C snapshots. Removing the C wrapping made
the sanitizer report signed overflow. Removing program folding's modulo conversion made
both dtype folding cases fail. The test baseline now contains 3082 IDs. Ruff and ty pass.
Expression folding already wraps correctly for scalar and varying constants, so it needs
no production change. Int32 scalar expansion is gated off until #55, so the renderer test
locally permits int32 in the exact-store set without changing the production gate.

Rebased onto #320's replacement head, verified against the force-pushed remote before
the first push. Full pre-merge validation passed through the detached suite runner:
3137 passed, 52 skipped, benchmark smoke, docs and all other hooks passed. The first
queued process received SIGTERM before testing; the retry retained its queue time.
