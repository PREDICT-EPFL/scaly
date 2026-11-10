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
- [ ] Wait for #289 to merge, fetch and rebase onto main using the supplied stack base.
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

The initial full run was interrupted by la015's memory exhaustion and proves
nothing. No test processes survived it. The run now serializes full suites with
`flock ~/.scaly-suite.lock env PYTEST_XDIST_AUTO_NUM_WORKERS=8 wt hook pre-merge --yes`,
launched detached with `nohup setsid`, a log and an exit file under a temporary
directory. Focused tests use at most two workers. Retry killed-worker or timeout
failures once. Run the full checks only on a state intended for pushing.
