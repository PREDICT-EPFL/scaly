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
- [x] Make declared-order direct/window numerical regressions fail before changing code.
- [x] Keep numerical mapped arguments in declared order; convert symbolic raw windows to
      canonical compact values at the frontend. Preserve ordinary symbolic sparse arguments.
- [x] Cover CSC, CSR, coordinate order, a non-involutive permutation, batched leaves, empty
      patterns and map lengths, overlapping and broadcast windows, and derivative routing.
- [x] Run focused tests and show the regressions fail when the fix is perturbed. Refresh the
      complete pytest collection baseline and run the pre-merge hook, preserving C snapshots.
- [ ] After #289 merges, fetch and rebase onto main as directed, resolve only in-scope conflicts,
      rerun checks, push and open a draft pull request against main.
- [ ] Obtain cross-review and any required delta check, remove this temporary plan, mark ready,
      and queue after green checks. Record completion evidence and the public convention decision.

No roadmap departure is expected. The rejected public convention is canonical row-major order
for raw windows while direct numerical values use a different declared order.

Recovery after the machine ran out of memory: the 120 focused tests passed before the interruption.
Replacing the symbolic conversion's inverse permutation with the forward permutation made the
non-involutive CSC regression fail, and the correct source was restored. Ruff and ty passed.
The collection refresh did not complete. No full pre-merge check has run yet.

All full-suite runs now use the shared `~/.scaly-suite.lock` and eight workers, detached with
`nohup setsid`. Run `flock ~/.scaly-suite.lock env PYTEST_XDIST_AUTO_NUM_WORKERS=8 wt hook
pre-merge --yes`, capture a temporary log and exit file, and poll the exit file. Run the full
hook only on a state intended for push. Focused tests use at most two workers.

The detached retry passed the complete pre-merge hook, including 2998 passed / 52 skipped,
benchmark smoke, documentation and private-name checks. Log and exit file:
`/tmp/scaly-300-retry.Uoz8GT/premerge.{log,exit}`. The first detached lock wait was terminated
with exit 143 before any checks ran. Collection has 3050 tests; the focused run has 120 passed.
C snapshots are unchanged. No dedicated timing measurements were taken.

The coordinator now requires GitHub reads through REST because the shared GraphQL quota ran
out. Check `pulls/289` at most once per five minutes; the 18:16 UTC status still has #286,
#287 and #289 in the queue. Check the task PR's checks at most once per five minutes, use T3
watches for wakes, and queue it once with `gh pr merge --squash --auto` after verdict and green
checks. A rate-limit error means waiting for the reset rather than declaring failure.
