# Sparse mapped window layout, issue #300

The issue and the roadmap's #42 paragraph require raw mapped window buffers to use each
sparse leaf's declared layout, just like direct numerical arguments. Internal VMAP windows
remain canonical. This is a public convention change before the sparse API is published.

The compiler run plan at
`internal/notes/compiler_run_2026_11_10/plan.md` overrides the implement workflow. The branch
starts on the sparse interface stack. Do not open a pull request until #289 merges, then rebase
with `git rebase --onto origin/main 248b2159` and rerun checks. Queue the reviewed, green pull
request with `gh pr merge --squash`. No generated-code timing measurements.

- [x] Read issue, parent, run plan, roadmap, repository conventions and relevant code.
- [ ] Make declared-order direct/window numerical regressions fail before changing code.
- [ ] Keep numerical mapped arguments in declared order; convert symbolic raw windows to
      canonical compact values at the frontend. Preserve ordinary symbolic sparse arguments.
- [ ] Cover CSC, CSR, coordinate order, a non-involutive permutation, batched leaves, empty
      patterns and map lengths, overlapping and broadcast windows, and derivative routing.
- [ ] Run focused tests and show the regressions fail when the fix is perturbed. Refresh the
      complete pytest collection baseline and run the pre-merge hook, preserving C snapshots.
- [ ] After #289 merges, fetch and rebase onto main as directed, resolve only in-scope conflicts,
      rerun checks, push and open a draft pull request against main.
- [ ] Obtain cross-review and any required delta check, remove this temporary plan, mark ready,
      and queue after green checks. Record completion evidence and the public convention decision.

No roadmap departure is expected. The rejected public convention is canonical row-major order
for raw windows while direct numerical values use a different declared order.
