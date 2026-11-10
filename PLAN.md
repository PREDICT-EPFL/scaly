# Type structural derivative constants

Implements #168 under section #24 of the compiler run. The owning plan is
`internal/notes/compiler_run_2026_11_10/plan.md`, temporarily read from the coordinator's
`t3-e7c13084-2480097323e0b716` worktree until #291 lands. Base and pull request target are main.

The decided rule in `internal/notes/core_compiler_roadmap.md`, One AD engine, gives zeros,
ones and seeds the primal's dtype. Read the architecture, codebase, conventions, contributing
guide and compiler maintenance context. #20 is closed. The sparse stack and #30 overlap these
files, so keep the diff focused and rebase if needed.

- [x] Inspect #168, parent #24, prerequisites, project status, open pull requests and checkout.
- [x] Read the structural derivative rules and existing float32 regressions. The relevant
      devrush reduction test is already present on main. Python scalar coercion already types
      the repeated-dot factor correctly. Public float32 printing remains outside this task.
- [ ] Add float32 graph and numerical regressions for reductions, matrix products, repeated
      operands, PRINT and unreached inputs across the requested derivative builders. See them fail.
- [ ] Type the structural zero, one and repeated-dot constants. Run the regressions and prove
      the new assertions reject a deliberate perturbation.
- [ ] Refresh the collection baseline and run all pre-merge checks. Keep C snapshots unchanged.
- [ ] Open and link a draft pull request. Use cross-review with Opus and the required independent
      sol correctness review, fix findings, and request a delta check only on fixes.
- [ ] Delete this temporary plan, record evidence and the review verdict in the pull request,
      mark ready, queue when checks pass, and resolve any queue ejection.

No lowering, typed entry signatures, new operations or snapshot regeneration is in scope.
