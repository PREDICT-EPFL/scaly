# Declared-layout window sparse Jacobians, #332

The compiler run plan on main and `~/.scaly-suite/RULES.md` govern this task.
Base is origin/main, with no stack. Only `src/scaly/ad/sparse.py` and its tests
are in scope. Preserve #300's declared layouts and internal canonical order.
Design comes from the core compiler roadmap's Sparse tensors section and the
compiler maintenance context's Mapped derivative construction section.

- [x] Read the issue, prerequisite #300, section #293, project instructions and design.
      REST Project discovery returned HTTP 404; progress is recorded on the issue.
- [ ] Add a regression that rejects global coloring for CSC and coordinate windows.
      Compare values and sparsity against an independent reference for overlapping
      and zero-stride windows, non-involutive orders, batched leaves, empty patterns
      and zero-length maps. See it fail before the fix.
- [ ] Recognize constant gathers of wrt and route stage-local columns through their
      slot mapping. Keep computed actuals on the existing fallback.
- [ ] Run focused tests with at most two workers, demonstrate the regression gate
      fails under perturbation, and measure the issue reproduction's source size,
      Program IR arithmetic and generated-C compile time. No run-time timings.
- [ ] Commit the final candidate and run the detached full pre-merge hook through
      `~/.scaly-suite/run`, polling its exit file within this turn. Preserve snapshots.
- [ ] Open and link one draft PR after local checks pass. Obtain the cross-review
      verdict from Opus and the required independent correctness review. Fix findings
      together, check any fixes, and settle the completed delegate threads.
- [ ] Remove this temporary plan, record criterion evidence and review in the PR,
      mark ready, and queue once with `gh pr merge --squash --auto`.

Only push the reviewed candidate and any combined review fixes or forced rebases.
Do not rebase merely because main moved. Report any roadmap departure, open points
and follow-up issues. The run authorizes merging through the queue.
