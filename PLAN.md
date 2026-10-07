# PIQP option validation, issue #80

Base: `t3code/issue-79-piqp-0.6.4`, PR #122. Stack layer 2 of 3.
Design: issue #80 and `internal/notes/documentation_api_review.md#piqp-option-errors-reach-c-compilation`.
Reference: vendored PIQP 0.6.4 `piqp_typedef.h`, QP construction in `src/scaly/solvers/qp.py`, plugin wrapper and tests, solver-backend guide.

- [x] Inspect issue, dependencies, checkout, and stack. #79 remains open and is the authorized base.
- [x] Reproduce unknown-option acceptance with dense and sparse construction tests.
- [x] Validate names in Python before any matrix probe or C rendering. Pin accepted names to 0.6.4 and preserve valid-option rendering.
- [x] Update the internal note and guide sentence using write-docs.
- [ ] Run focused tests, perturb the check, and run `wt hook pre-merge`.
- [ ] Commit, push a draft stacked PR, and get one cross-review. Apply findings and check fixes once.
- [ ] Remove this plan, mark ready, update Project, and watch the PR.

Implementation is sequential. Independent verification commands may run in parallel. #112 owns moving options to runtime; #79 owns vendoring, build hooks and notices. No edits to those files.

Focused plugin tests: 19 passed. Before the fix and with validation disabled, all six unknown-name cases fail. Full pre-merge hook is running.
