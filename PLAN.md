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
- [x] After PR #289 merges, fetch and rebase the task commits onto origin/main from the
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

Read `~/.scaly-suite/RULES.md` on the coordinator's instruction. Full checks run detached
through `~/.scaly-suite/run`, with two slots, eight workers, and temporary log and exit files.
Focused pytest uses at most two workers. GitHub reads use REST; check PR state at most
once per five minutes. Queue once after review with `gh pr merge N --squash --auto`.

REST confirmed #289 merged. The required rebase onto origin/main succeeded and dropped
the inherited header-client change already upstream. All 166 arithmetic and C snapshot
tests passed again after rebasing. The first detached suite waiter was terminated before
running; its retry was voluntarily cancelled while waiting to switch to the two-slot runner.
