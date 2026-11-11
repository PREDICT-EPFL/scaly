# Axis reductions, #66

Branch `t3code/66-axis-reductions`, based on `origin/main`, first layer of #66, #67,
#68. The compiler run plan and `~/.scaly-suite/RULES.md` override the ordinary
implementation procedure. Runtime measurements wait for phase 4. Do not edit
`ir/einsum.py` or generalize matmul.

Read: issue #66 and its comment, parent #296, roadmap decision 4 and tensor item
#66, architecture, codebase, conventions, contributing, compiler maintenance
context, local instructions and the implement/tracker/plan skills.

- [x] Confirm prerequisites, worktree, scope and concurrent changes.
- [x] Add failing NumPy axis, internal extremum, derivative and folding tests.
- [x] Replace SUM with REDUCE across vocabulary, verification, support, sparsity,
      rewrites, AD, lowering and visualization. Add sum/mean/diag builders.
- [x] Exercise sparse stored groups and loaded-bound scheduling, identities,
      negative axes, empty and singleton axes. Remove unbroadcast scatter.
- [x] Update guide/API with write-docs, collection baseline and C snapshots if
      needed. Perturb new checks. Record snapshot counts, size and compile time.
- [x] Run focused checks with at most two workers. Run the complete pre-merge
      hook detached through the suite queue on the intended review state.
- [ ] Commit and submit the stack PR. Link it to this thread immediately.
- [ ] Get one cross-review and one delta check. Fix findings and put the verdict
      in the PR description immediately so #67 can start.
- [ ] Remove this temporary plan, run required checks, push all fixes together,
      mark ready, update tracker and queue squash auto-merge once. Hand CI and
      merge watching to the parent. Rebase only on conflicts or queue ejection.

Overlapping PRs at start: #320, #355, #314, #367, #369, #392, #382.
Implemented and focused checks pass. The AD differential suite passes 1,301
cases. The changed snapshot is only `jac.c`, with temporary-name changes.
Compared with origin/main it has 5,452 versus 5,456 bytes, 134 lines each,
static operator token counts +50, -5, *40, /0 each, and byte-identical compiled
`.text` sections of 454 bytes. Five cold object compilations with `/usr/bin/cc
-O3 -fno-math-errno -c` had medians 49.61 ms here and 50.47 ms on main. The
machine is busy, so these are compile evidence, not a speed claim. Runtime
measurements remain deferred. Eleven deliberate perturbations were detected,
including the loaded-bound scheduling and no-scatter checks.

Project status is unavailable through the REST-only read policy; branch and
issue comments record active ownership. The guide's new example and the
architecture's expression example were executed. Full validation includes the
existing README and example checks.

The detached pre-merge hook passed at `/tmp/scaly66-suite-vxKqGW/premerge.log`.
Full suite: 4,346 passed, 55 skipped. Benchmark smoke, docs build, ty, Ruff
and private-name checks all passed. The worktree is ready for review.
