# NumPy names, issue #479

This is the second of three stacked changes, based on `t3code/430-integer-mean`, PR #482.
The decided design is in `internal/notes/core_compiler_roadmap.md`, "Tensor core and the loop compiler".
The issue and maintainer brief fix the naming decision and scope.

- [x] Read the issue, parent, project status, repository instructions and implementation.
- [ ] Add one NumPy comparison for max/min functions and methods, tuple axes, keepdims and ravel, and observe its failure.
- [ ] Rename the public builders and every caller in src, tests, docs and examples. Keep Python's built-in shape calculations intact.
- [ ] Update public exports, API listings, norm docstrings, changelog and test-ID baseline.
- [ ] Perturb the implementation to prove the regression detects wrong results, then restore it.
- [ ] Run focused checks and detached pre-merge checks through ~/.scaly-suite/run. Every generated-C snapshot must remain byte-identical.
- [ ] Open a draft PR based on the mean branch, obtain one claude-opus-5-5 high review and a delta check if fixes are needed.
- [ ] Remove this temporary plan, mark ready, update the tracker and report evidence and CI state.

Do not change mean, shape helpers, cast folding, leaf-dtype prose, extern documentation or generated-C baselines.
Do not merge, queue, push to main or create tags. If #482 squash-merges, rebase onto main using the old base tip,
push with --force-with-lease and retarget the PR. Keep implementation steps here, out of tracker comments and the PR description.
