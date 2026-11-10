# Sparse gate child peak memory

Scope is issue #362 and only `tests/integration/test_sparse_scale.py`. The issue body decides the design. Read the compiler run plan on `origin/main`, the la015 suite rules, the repository architecture and contribution rules, and the implement, inspect-issues, update-issues, write-plan and cross-review skills.

- [x] Verify base, scope, blockers and the run rules.
- [ ] Add the freed-parent-buffer regression inside the existing gate test and see it fail.
- [ ] Read Linux `VmHWM`; preserve other platforms unless evidence shows inheritance. Confirm the regression and perturb the measurement.
- [ ] Run the complete pre-merge checks through the detached suite runner. Check snapshots are unchanged.
- [ ] Open and link the pull request, get one sol correctness review, and check any fixes once.
- [ ] Remove this temporary plan, make the PR ready, and queue once after green checks and its verdict. Report criterion evidence and any open points.

The phases are sequential. Independent reads and checks may run together. No compiler, limits, snapshots, or unrelated files change.
