# Declared-layout window sparse Jacobians, #332

The compiler run plan on main and `~/.scaly-suite/RULES.md` govern this task.
Base is origin/main, with no stack. Only `src/scaly/ad/sparse.py` and its tests
are in scope. Preserve #300's declared layouts and internal canonical order.
Design comes from the core compiler roadmap's Sparse tensors section and the
compiler maintenance context's Mapped derivative construction section.

- [x] Read the issue, prerequisite #300, section #293, project instructions and design.
      REST Project discovery returned HTTP 404; progress is recorded on the issue.
- [x] Add a regression that rejects global coloring for CSC and coordinate windows.
      Compare values and sparsity against an independent reference for overlapping
      and zero-stride windows, non-involutive orders, batched leaves, empty patterns
      and zero-length maps. See it fail before the fix.
- [x] Recognize constant gathers of wrt and route stage-local columns through their
      slot mapping. Keep computed actuals on the existing fallback.
- [x] Run focused tests with at most two workers, demonstrate the regression gate
      fails under perturbation, and measure the issue reproduction's source size,
      Program IR arithmetic and generated-C compile time. No run-time timings.
      112 tests passed, including all C snapshots. Removing gather recognition made
      both 20-stage layout cases fail as intended. The issue reproduction is 17,847
      bytes and 70 arithmetic operations per stage before, 6,839 bytes and 9 after.
      Generated-C compilation with cc -O2 took 0.1295 s before and 0.0657 s after;
      these are single observations during concurrent compiler work.
- [x] Commit the initial candidate and run the detached full pre-merge hook through
      `~/.scaly-suite/run`, polling its exit file within this turn. Preserve snapshots.
      The complete hook passed: 3,528 tests passed, 52 skipped; all other checks passed.
- [ ] Open and link one draft PR after local checks pass. Obtain the cross-review
      verdict from Opus and the required independent correctness review. Fix findings
      together, check any fixes, and settle the completed delegate threads.
      PR #390 is linked. Sol approved without findings. Opus requested a regression
      for repeated gather contributions whose count equals nnz without full coverage,
      and one unified column-mapping path. Both are applied; the regression fails
      when the old coverage condition is restored. All 216 focused tests pass,
      snapshots and the reproduction's generated C are byte-identical. The public
      docstring now describes constant gathers; the optional maintenance-note edit
      stays outside this task's source-and-tests scope. The initial Opus thread is
      settled. Next: full hook on the revised candidate, then one Opus delta check.
- [ ] Remove this temporary plan, record criterion evidence and review in the PR,
      mark ready, and queue once with `gh pr merge --squash --auto`.

Only push the reviewed candidate and any combined review fixes or forced rebases.
Do not rebase merely because main moved. Report any roadmap departure, open points
and follow-up issues. The run authorizes merging through the queue.
