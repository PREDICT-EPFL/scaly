# Sparse gate child peak memory

Scope is issue #362 and only `tests/integration/test_sparse_scale.py`. The issue body decides the design. Read the compiler run plan on `origin/main`, the la015 suite rules, the repository architecture and contribution rules, and the implement, inspect-issues, update-issues, write-plan and cross-review skills.

- [x] Verify base, scope, blockers and the run rules.
- [x] Add the freed-parent-buffer regression inside the existing gate test and see it fail. Old measurement reported 283,107,328 bytes.
- [x] Read Linux `VmHWM`; preserve other platforms unless evidence shows inheritance. Three focused tests pass. Disabling the Linux branch makes the final regression fail in 0.68 seconds. Apple XNU assigns ru_maxrss from the task's resident_size_max; Windows uses the process's PeakWorkingSetSize. CI runs the regression on both unchanged paths.
- [ ] Run the complete pre-merge checks through the detached suite runner. Check snapshots are unchanged.
- [ ] Open and link the pull request, get one sol correctness review, and check any fixes once.
- [ ] Remove this temporary plan, make the PR ready, and queue once after green checks and its verdict. Report criterion evidence and any open points.

The phases are sequential. Independent reads and checks may run together. No compiler, limits, snapshots, or unrelated files change.

Ruff format/lint, ty, diff checks and the snapshot-file comparison pass. Full pre-merge checks run detached through the suite runner; logs are recorded in `/tmp/scaly-362-validation`.
