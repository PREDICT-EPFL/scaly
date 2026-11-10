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
      generated source size and C compile time against main. Measurements remain
      pending; the test and snapshot work is complete.
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
derivative/JIT regression tests pass after it. Full checks run in tmux session
`scaly45-premerge`, logging to `/tmp/scaly-45-pre-merge-final.log`, with exit status
in `/tmp/scaly-45-pre-merge-final.status`, so thread interruptions do not kill them.

Compile evidence is in `/tmp/scaly-45-evidence-{main,branch}/evidence.json` and the
draft PR description in `/tmp/scaly-45-pr.md`. Counts match for every snapshot.
Clang 20.1.8 compiles the generated C with gather/scatter disabled. Compile CPU
seconds are the median of three compilations, not runtime benchmark evidence.

The program tests also caught greedy workspace reuse letting a scalar sum pin
a freed 1600-element matrix slot, doubling workspace. Packing now reserves
spill-sized slots for spill-sized values. The existing packing witness fails
before the fix and passes after it; 15 packing/hoisting tests pass. Regenerating
the snapshots again changes no files. Restart the whole hook with this fix.
