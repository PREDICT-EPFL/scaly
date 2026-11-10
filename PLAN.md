# Sparse arithmetic simplification, issue #303

The stored pattern is fixed by the operation's pattern rule. Simplification may remove
arithmetic but must preserve that pattern, including explicit zero slots. Scope is
`passes/expr.py`, `passes/arith.py` and regression tests. Support analysis and sparse
kernels remain outside this task.

Read the issue, roadmap decisions 4 and #39/#40, compiler maintenance context, project
conventions and the compiler run plan at
`internal/notes/compiler_run_2026_11_10/plan.md` in the coordinator's worktree.
The run plan authorizes queueing after review and green checks. No runtime timings.

- [x] Inspect the clean linked worktree, task metadata and surrounding implementation.
- [x] Add failing tests for folded values, stored patterns and lowering hints, including
      dense/sparse mixtures, broadcasting and zero matrix products.
- [x] Remove sparse bypasses, preserve patterns when replacing operands, and build stored
      zeros with PACK over a zero values tensor. Run focused checks and perturb a rule.
- [ ] Refresh the node-ID baseline and run all merge checks. Keep C snapshots byte-identical.
- [ ] After PR #289 merges, fetch and rebase the task commits onto origin/main from the
      supplied sparse-stack tip, then rerun checks.
- [ ] Commit, push, open and link the PR. Complete one correctness cross-review and any
      required delta check. Remove this temporary plan before making the PR ready.
- [ ] With green CI and the review verdict, queue with `gh pr merge --squash`, update
      tracker status, and report criteria evidence, decisions, limitations and follow-ups.

PR #289 was open at the initial inspection. The requested prerequisite stack is already
present locally. No overlapping PR currently changes the two pass modules.

The session was killed during the mutation check. Restored the temporary sparse bypass
and repeated the check with two workers: both sparse subtraction cases fail as expected
when the bypass is reintroduced. The implementation is restored. Focused validation
before the interruption passed 230 tests, Ruff formatting/lint and strict ty.

Run-wide memory rule from the coordinator: full checks run detached under
`flock ~/.scaly-suite.lock env PYTEST_XDIST_AUTO_NUM_WORKERS=8 wt hook pre-merge --yes`,
with a temporary log and exit file. Focused pytest uses at most two workers. Run full
pre-merge only for a state intended for push, and retry a killed worker or timeout once.
