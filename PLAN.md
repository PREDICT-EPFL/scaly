# Sparse SELECT, #359

Implement the decided sparse SELECT extension on the approved #46 branch. The compiler run plan
and `~/.scaly-suite/RULES.md` govern validation and merging. #369 owns sparse-sparse products;
#382 owns published documentation. Keep lowering edits in the SELECT region.

- [x] Read the run plan, task, prepared tests, roadmap and surrounding implementation.
- [ ] Run the prepared cases before implementation; add constant-condition and dense-branch gates.
- [ ] Add SELECT storage/support regions and preserve #30 masks on stored-coordinate derivatives.
- [ ] Prove region and mask gates fail under perturbation. Keep dense C snapshots byte-identical.
- [ ] Run pre-merge through the detached suite runner, commit, push once and open a draft on #367.
- [ ] Obtain cross-review, address findings and confirm any fixes.
- [ ] After #367 merges, rebase only this task's commits onto main, retarget, rerun the full suite,
      push once and queue once when green. Report the guide correction to #382's owner.

Sources: issue #359 and prepared comment 6779722333; roadmap sections Sparse tensors and Dtypes
and the scalar vocabulary; `passes/lowering/plans.py`; `ad/rules.py`; `ad/reverse.py`.
