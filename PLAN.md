# COND implementation, issue #53

Base: `t3code/31-loop`, PR #411. COND follows decision 7 and the conditional section of
`internal/notes/core_compiler_roadmap.md`. The compiler run plan and `~/.scaly-suite/RULES.md`
control validation and GitHub access. Keep snapshots unchanged and exclude loop derivatives,
VMAP folding and unrolling. Export the builder with a docstring, keep it out of the API guide.

- [x] Read project instructions, roadmap, compiler maintenance context and run rules.
- [x] Add differential failing tests for branch selection, clamping, effects, nesting and AD.
- [x] Add expression COND, verifier, builder, union support and sparsity.
- [x] Add program IF, lowering, C emission and pass coverage; reproduce failures before guards.
- [x] Add branch tangent and adjoint helpers with primal recomputation.
- [x] Run focused checks, perturb a new gate, refresh node baseline and detached full checks.
- [x] Commit, check PR #411 status, push and open a draft PR on the current base; link immediately.
- [ ] One Opus 5.5 high cross-review and one delta check, fixing findings locally.
- [ ] After #411 merges, rebase onto main, retarget, rerun detached suite and push once.
- [ ] Delete this plan, mark ready, await green CI and report verdict to the coordinator.

No departures yet. Do not queue without the coordinator's explicit instruction.

Focused tests cover every program pass, union analyses, sparse layouts, custom derivatives,
selected effects and unused loop results. Scalar preparation needed IF recursion; all other
passes already preserved the reproduced conditional program. LE-to-LT mutation failed the
selector-zero test. Devrush has no COND implementation or tests to port. The base ABI takes
float64 leaves, so tests cast runtime float selectors inside the graph.

Full pre-merge via `/tmp/scaly-cond-full-v1.log` passed: 3,890 tests, 55 skips,
benchmark smoke, docs, strict types, Ruff format/lint and private-name check.
QP branch traversal reproduced five failures before the fix. No snapshots changed.

PR #417 is open on the LOOP base. Opus 5.5 high review found CSC operand canonicalization
captured only in branch zero and temporary CALL IDs reused in sparsity analysis. Both new
regressions failed first. Fixes move canonicalization before branch capture and retain CALLs
in each analysis context. Notes addressed: early branch result-type errors, four text renderers,
and derivative semantics in the builder docstring. The reciprocal-hoisting gap for hand-built
IF assignments is unreachable from current lowering and stays in the PR description.
