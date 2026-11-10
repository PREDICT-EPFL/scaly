# Avoid workspace parameter shadowing

Issue #311, based on main. Scope is the typed C wrapper's workspace parameter and its compile regression. The exported pointer entry keeps its name.

Read the compiler-run plan from the coordinator worktree while #291 remains open, and follow `~/.scaly-suite/RULES.md`. The run permits concurrent snapshot regeneration and authorizes queuing this pull request after review and green checks.

- [x] Read the issue, naming implementation, project conventions and compiler-run rules. No blockers or parent issue.
- [x] Add a generated-header compile-and-call regression and reproduce the failure. Both input-name cases fail with GCC's reported shadowing error.
- [x] Allocate the workspace parameter through the existing naming scope and regenerate snapshots and test node IDs. Both regression cases detect a reverted fix. The focused naming, wrapper and snapshot checks pass, 43 tests.
- [ ] Run all pre-merge checks through the detached suite runner. Started detached on la015, with evidence under `/tmp/scaly-311.nSgvDF/`. Source and compile measurements are in `measurements.json` there. Only two lines of `workspace.h` change, its header grows by 6 bytes, and every generated C source stays identical. The workspace module has 4,137 source bytes and 12 unique scalar operation nodes. GCC compiles it in 0.050435 seconds with `-O3 -fno-math-errno -c`; the original header fails compilation.
- [ ] Commit, push, open the pull request and obtain one correctness review with `cross-review`. Address findings and check any fixes.
- [ ] Remove this temporary plan, mark ready, update the tracker and queue after green checks. Rebase and regenerate if needed.
