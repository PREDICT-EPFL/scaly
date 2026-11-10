# Hash-seed-independent generated C

Implement #163 under section 6 (#295) of the compiler run plan. Base and PR base are main.
The compiler run plan overrides the implement skill, including authorization to queue the PR
after cross-review and green checks. No JIT key, index, or invalidation changes.

Read: compiler run plan, architecture, codebase, conventions, contributing, issue tracking,
compiler maintenance context, core compiler roadmap Foundations #64 and Rules for every item.

- [x] Inspect issue, dependencies, checkout, and overlapping PRs. No blockers; sparse stack
      overlap and snapshot regeneration are allowed by the run plan.
- [ ] Add a subprocess corpus test covering mapped stage, nested call/map derivatives, and
      other representative Functions. Confirm failure on the original implementation.
- [ ] Locate the first unstable compiler pass and fix iteration using graph or statement order.
      Audit other seed-dependent iteration in lowering, program passes, and rendering.
- [ ] Verify differential behavior, mutation detection, snapshots, full suite, and merge hooks.
      If snapshots move, review why and collect the run plan's required evidence.
- [ ] Commit, push, open and link draft PR. Obtain one correctness cross-review and a delta
      check if needed. Remove this temporary plan before marking ready.
- [ ] Update issue status, handle CI/rebase, queue squash merge with green checks, and report
      evidence, remaining points, and any source:agent follow-ups to the coordinator.

Each phase depends on the previous one, except the iteration audit can accompany diagnosis.
