# Sparse SELECT, #359

Implement the decided sparse SELECT extension on the approved #46 branch. The compiler run plan
and `~/.scaly-suite/RULES.md` govern validation and merging. #369 owns sparse-sparse products;
#382 owns published documentation. Keep lowering edits in the SELECT region.

- [x] Read the run plan, task, prepared tests, roadmap and surrounding implementation.
- [x] Run the prepared cases before implementation; add constant-condition and dense-branch gates.
- [x] Add SELECT storage/support regions and preserve #30 masks on stored-coordinate derivatives.
- [x] Prove region and mask gates fail under perturbation. Keep dense C snapshots byte-identical.
- [x] Run pre-merge through the detached suite runner and commit the candidate.
- [x] Push once and open a draft on #367: PR #406.
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
#367 rebased upstream since this branch started. This branch now includes its notes-only update,
and source and tests are byte-identical to the reviewed candidate. Use the updated ancestor in
the coordinator's latest `rebase --onto` instruction when #367 lands. Hold the next push until
the review fixes are together. Sol found two
mask-loss paths: nested masks lost their inner condition during the compact transpose, and pattern
normalization hid a mask in PACK/GATHER. Both are fixed locally through the existing mask helper,
including the corresponding dense conversion. Three regression cases failed first and now pass
with gradient and Hessian checks. Focused SELECT and C snapshot validation passed 51 tests, and
strict types and Ruff passed. Opus found an out-of-bounds sparse condition read with dense branches,
missing dense-product and PACK mask gates, and needless enumeration of unknown dense support.
These are fixed and tested. Constant conditions on sparse results are also projected to stored
coordinates at construction, so even a full-shape uniform condition has no logical matrix buffer.
The repeated zero construction is now one helper, and sparse mask unions use bool stored vectors.
All five dense-product/PACK mask perturbations fail their new gates. Final focused validation
passed 195 tests. The final full pre-merge rerun passed, with 3,908 tests, 55 skips, benchmark
smoke and docs build clean. Log: `/tmp/scaly-359-review-premerge.log`, exit zero. A separate
100,000-square constant-condition case with six stored entries lowered values, gradient,
Jacobian and sparse Hessian under a 4 GB memory limit, with at most 36 private entries.
Ask both reviewers for short delta checks on fixes only.
Do not push the rebase separately. The guide examples in #382 must account for a dense zero branch
giving a full pattern, and sparse predicates still requiring an explicit packed values predicate.
