# Accumulators and contraction scheduling, #45

Run under `internal/notes/compiler_run_2026_11_10/plan.md`, read first from the
coordinator's checkout until #291 merges. Base `origin/main`, stack layer 1 of
`#45 -> #46 -> #47`, branch `t3code/45-accumulators`.

Design is settled in `internal/notes/core_compiler_roadmap.md`, Tensor core and the
loop compiler, #45 and decisions 5 and 13. Read the compiler maintenance context,
project architecture, codebase, conventions and contributing guide, #45's two
comments and parent #292. Runtime measurements wait for Phase 4. Snapshot
regenerations are concurrent; rebase and regenerate when overlapping work lands.
Do not edit `passes/lowering/plans.py`, which belongs to #46.

- [x] Read the run plan and implement, inspect-issues, update-issues, write-plan,
      cross-review and unslop skills. Confirm linked worktree on la015, inspect
      issue dependencies and open PRs, mark #45 In progress, initialize gh stack.
- [x] Add failing differential and structural tests for SUM and all current
      MATMUL shape classes, empty reductions, both schedules, mutable scalar
      expansion and ordered vector lanes. Inspect devrush's reduction tests.
- [x] Add a single lowering scheduler and migrate reductions and contractions.
      Keep each output's reference summation order; identity initialization must
      survive empty reduction ranges. Delete layout-specific helpers.
- [x] Show new checks fail under an implementation perturbation. Run narrow
      checks, regenerate node IDs and C snapshots, record operation counts,
      generated source size and C compile time against main.
- [ ] Rebase over overlapping main changes, regenerate snapshots and rerun the
      full suite and `wt hook pre-merge --yes`.
- [ ] Commit and push only task changes, open draft PR targeting main, link it
      in T3 and register its gh stack. Run one cross-review and any delta check.
- [ ] Put the verdict in the PR description immediately, address findings, file
      any uncovered follow-ups with `source:agent`, mark Ready and In review.
- [ ] Remove this temporary plan before final readiness. Queue the PR with
      `gh pr merge --squash` once review and checks pass, watch pending CI/merge,
      and report criterion evidence, verdict, open points and queue/merge state.

Current findings: main already supports ASSIGN in scalarize for matrix panels.
The new regression must exercise mutable accumulation, empty loops and calls.
The sparse stack, #30 and #28 may require rebases before final evidence.

Validation so far: the three structural checks fail against main. With the new
scheduler the targeted run passed 387 tests with 50 compiler-extra skips. GCC and
Clang both passed ordered reductions in GNU and scalar C for widths 2, 4 and 8.
Changing the identity to 11, suppressing the strided schedule and using two
output lanes each failed its new check. Five source snapshots changed, all
headers and the sparse-Jacobian and mapped sources stayed identical. Ruff and
ty pass. The full pre-merge hook is running, including the whole test suite.

The first full run exposed a required hoisting fix: ASSIGN initialization,
updates and uses cannot split between the invariant prologue and mapped body.
`hoist_invariant._split` now keeps each scalar's uses together. A new mapped-sum
NumPy test reproduced an undeclared C accumulator before the fix. The 54 custom
derivative/JIT regression tests pass after it.

Compile evidence is in `/tmp/scaly-45-evidence-{main,branch}/evidence.json` and the
draft PR description in `/tmp/scaly-45-pr.md`. Counts match for every snapshot.
Clang 20.1.8 compiles the generated C with gather/scatter disabled. Compile CPU
seconds are the median of three compilations, not runtime benchmark evidence.

The program tests also caught greedy workspace reuse letting a scalar sum pin
a freed 1600-element matrix slot, doubling workspace. Packing now keeps scalar
results out of spill-sized slots. The existing packing witness fails
before the fix and passes after it; 15 packing/hoisting tests pass. Regenerating
the snapshots again changes no files.

Current validation job: detached `nohup setsid` shell, log
`/tmp/scaly-45-rebased.log`, exit code `/tmp/scaly-45-rebased.exit`. It runs
`flock ~/.scaly-suite.lock env PYTEST_XDIST_AUTO_NUM_WORKERS=8 wt hook pre-merge --yes`
on the state rebased over main's sparse stack at 8d8c7870.
The coordinator's latest rule supersedes earlier caps: every full suite uses
this lock; focused tests run unlocked with `-n 2`, never `-n=auto` outside the lock.
The earlier unlocked runs were stopped. Poll the exit file; do not start another
full run. Retry a killed-worker or timeout failure once before diagnosing it.
The coordinator now requires polling the local exit file within this turn, with
sleep between polls. Do not end the turn while the local run is pending. The
earlier scheduled continuation is disabled. Do not schedule GitHub polling.
At the 19:09 continuation, the locked run had reached 95% and had two failure
markers. Wait for the final failure summaries and exit code before classifying
them, then apply the one-rerun rule to any killed-worker or timeout failure.

At 19:12 the locked run reported 2666 passed, 55 skipped and two assertions.
The metric expected an accumulator buffer that no longer exists, so its buffer
count changes from six to five. The affine Hessian source-size gate exposed the
packing rule applying to all small buffers; it now applies only to scalar
results. Both failures and the spill-reuse witness pass in a focused 3-test run.
Ruff, ty and snapshot regeneration pass, with no further snapshot changes.
The old hook's exit was 143 and included "Terminated"; the corrected pushed
state gets one complete locked rerun, also covering that resource interruption.

Rebased onto origin/main after the sparse stack merged. Resolved contraction
and reduction conflicts by keeping the existing sparse product bridge and
reducing SUM over storage_size. Scalarize retains main's compact-storage work.
All 204 focused lowering/scalarize/widening and previous-failure cases pass.
Ruff, ty and fresh snapshot regeneration pass, with no further snapshot diff.
The old pre-rebase job was cancelled. Refresh the evidence against this main,
push the rebased branch with force-with-lease and run the new locked hook.

GitHub reads now use REST only. Check PR checks at most every five minutes;
use watch_pull_request for wakes. Patch its body with gh api -X PATCH and
queue once after the verdict with gh pr merge 315 --squash --auto. A quota
error waits until the reset from gh api rate_limit, rather than failing work.

Draft PR: https://github.com/PREDICT-EPFL/scaly/pull/315, registered with gh stack.
No reviewer has been launched yet; the cross-review skill requires green local
checks first. Once green, run Opus xhigh and a separate sol xhigh correctness
review, then only delta checks for fixes. Put the verdict line in the PR body
immediately so the coordinator can start #46. Remove this plan before readiness.
