# Sparse SELECT, #359

Implement the decided sparse SELECT extension on the approved #46 branch. The compiler run plan
and `~/.scaly-suite/RULES.md` govern validation and merging. #369 owns sparse-sparse products;
#382 owns published documentation. Keep lowering edits in the SELECT region.

- [x] Read the run plan, task, prepared tests, roadmap and surrounding implementation.
- [x] Run the prepared cases before implementation; add constant-condition and dense-branch gates.
- [x] Add SELECT storage/support regions and preserve #30 masks on stored-coordinate derivatives.
- [x] Prove region and mask gates fail under perturbation. Keep dense C snapshots byte-identical.
- [x] Run pre-merge through the detached suite runner and commit the candidate.
- [ ] Push once and open a draft on #367.
- [ ] Obtain cross-review, address findings and confirm any fixes.
- [ ] After #367 merges, rebase only this task's commits onto main, retarget, rerun the full suite,
      push once and queue once when green. Report the guide correction to #382's owner.

Sources: issue #359 and prepared comment 6779722333; roadmap sections Sparse tensors and Dtypes
and the scalar vocabulary; `passes/lowering/plans.py`; `ad/rules.py`; `ad/reverse.py`.

Focused validation: 199 sparse, support, SELECT, coverage and C snapshot checks passed, plus the
three added sparse-leaf and sparse-constant-condition cases. Both deliberate perturbations failed
their intended gates and were restored. Ruff and strict types pass. Prepared boolean boundary
inputs were adapted to boolean expressions inside the graph because typed boundaries belong to
#56 and are absent from the requested base. The condition's own broadcast storage is included in
its buffer bound. No public docs were edited.

The detached pre-merge run uses `/tmp/scaly-359-premerge.log` and `.exit`. Its full suite passed
with 3,894 tests and 55 skips. Benchmark smoke, docs build and all remaining hooks passed. The runner exited zero.
#367 has rebased upstream since this branch started. Keep the original approved ancestor in the
user's `rebase --onto` command, so only this task's commits move when #367 lands.
