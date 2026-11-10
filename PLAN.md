# Hash-seed-independent generated C

Implement #163 under section 6 (#295) of the compiler run plan. Base and PR base are main.
The compiler run plan overrides the implement skill, including authorization to queue the PR
after cross-review and green checks. No JIT key, index, or invalidation changes.

Read: compiler run plan, architecture, codebase, conventions, contributing, issue tracking,
compiler maintenance context, core compiler roadmap Foundations #64 and Rules for every item.

- [x] Inspect issue, dependencies, checkout, and overlapping PRs. No blockers; sparse stack
      overlap and snapshot regeneration are allowed by the run plan.
- [x] Add a subprocess corpus test covering mapped stage, nested call/map derivatives, and
      other representative Functions. Confirm failure on the original implementation.
- [x] Locate the first unstable compiler pass and fix iteration using graph or statement order.
      Audit other seed-dependent iteration in lowering, program passes, and rendering.
- [ ] Verify differential behavior, mutation detection, snapshots, full suite, and merge hooks.
      If snapshots move, review why and collect the run plan's required evidence.
- [ ] Commit, push, open and link draft PR. Obtain one correctness cross-review and a delta
      check if needed. Remove this temporary plan before marking ready.
- [ ] Update issue status, handle CI/rebase, queue squash merge with green checks, and report
      evidence, remaining points, and any source:agent follow-ups to the coordinator.

Each phase depends on the previous one, except the iteration audit can accompany diagnosis.

Evidence so far:

- The new subprocess test failed before the fix on map_of_call_1 at seed 1. An aggregate probe
  also showed the mapped stage differing at seeds 4, 5, and 42. Both map_of_call seed counts
  differed at seeds 1, 4, 5, 7, and 42 relative to seed 0.
- Fusion now follows the declared output order. Eleven Functions and their C headers match
  across seeds 0, 1, 4, 5, 7, and 42. Reverting just this change made the regression fail again.
- The audit found workspace packing updating dependencies while iterating a set of writes.
  A direct Program regression, swapping two buffer names, failed before its fix. Dependency
  closures now use values from before the statement. An in-memory mutation of this fix failed
  the regression again. The corresponding frozen-devrush test covered scan, which main lacks;
  the small Program reproduction covers the same packing hazard on main's IR.
- The first narrow run passed 56 tests, including the existing NumPy references for both
  reported examples and all C snapshots. No snapshot regeneration so far.
- Full merge hooks are running; final code needs a complete run after the workspace change.
