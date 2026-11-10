# Constant keys under digest collisions

Implement PREDICT-EPFL/scaly#301 in `t3code/301-constant-key-compare` under the
compiler-run plan at `internal/notes/compiler_run_2026_11_10/plan.md`.

The issue and the roadmap's #41 paragraph fix the design: compare compacted bytes
when digests match. Preserve logical-array compaction, the sparse scale gate and
byte-identical C snapshots. Constant handling outside keys is out of scope.

- [x] Read the issue, parent, run plan, repository rules and key consumers.
- [x] Add forced-collision and compact-serialization regression tests and observe failure.
- [x] Retain compacted bytes in the cached key and verify all key consumers.
- [ ] Run the full merge checks. The 51 focused expression tests pass. Removing the
      byte comparison makes four regressions fail; doubling the serialized payload
      makes the compaction check fail. Both mutations were restored.
- [x] Wait for #289 to merge, fetch and rebase onto main using the supplied stack base.
      The inherited header-test change was dropped as already present upstream.
      Rerun checks after rebasing. Do not open a PR before this point.
- [ ] Push, open and link a draft PR, obtain one correctness cross-review and a delta
      check if needed, and remove this temporary plan before marking ready.
- [ ] Record completion evidence, wait for green CI, queue with `gh pr merge --squash`,
      and report the PR, evidence, departures, decisions and remaining work.

Read: `ir/expr.py`, `passes/expr.py`, `ir/types.py`, their expression tests,
`tests/integration/test_sparse_scale.py`, the architecture, codebase, conventions,
contributing and issue-tracking guides, and compiler maintenance context. Frozen
`origin/devrush:tests/core/ir/test_expr.py` has the predecessor interning tests but
no digest-collision regression to port.

The supplied checkout includes one later stack-owned header-test commit. The
required rebase must leave no header changes in this task's PR.

PR #329 is open against main. Opus xhigh and Sol xhigh both approve. Opus noted
that rebuilding attribute keys duplicates retained compact bytes, so the node
now caches its construction-time attribute key for structural equality and CSE.
The compaction regression failed with four serializations instead of two before
this correction. Its delta check is pending. No roadmap departure or public
interface change was needed. The suite runner was still waiting for a slot and
was stopped before changing the reviewed state; only the final state's run counts.

The initial full run was interrupted by la015's memory exhaustion and proves
nothing. No test processes survived it. The first detached attempt exited 143
before producing check results. Follow the current `~/.scaly-suite/RULES.md`:
full checks run through `~/.scaly-suite/run`, launched detached with `nohup setsid`
and a log and exit file under a temporary directory. Focused tests use at most
two workers. Retry killed-worker or timeout failures once. Run the full checks
only on a state intended for pushing. GitHub reads use REST.
