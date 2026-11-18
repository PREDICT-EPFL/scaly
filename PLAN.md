# NumPy shape helpers, #475

Base and PR base: main. Branch: `t3code/475-shape-helpers`. Independent work.

The issue and maintainer brief fix the scope. Follow `internal/notes/core_compiler_roadmap.md`,
"Tensor core and the loop compiler", and `~/.scaly-suite/RULES.md`. Add builders from existing
operations only. Do not change generated-C snapshots, reduction/vec names, mean, cast folding,
or the leaf-dtype prose. A snapshot difference stops this task before any further push.

- [x] Read the issue, parent, dependencies, project rules and surrounding implementation.
- [x] Write lean NumPy comparisons, invalid-input checks, a gradient check per helper and one
  reshape-only workspace check. Run them before implementation and observe the missing helpers.
- [x] Implement methods where NumPy has them and sc functions, plus None indexing through reshape
  and existing slices. Add docstrings, API entries, the functions guide and export assertions.
- [x] Run focused checks, demonstrate the tests fail under a perturbation, update the test-node
  baseline, and run the detached full pre-merge checks through the suite runner. Snapshots must
  remain byte-identical.
- [ ] Commit and push a reviewable state, open a draft PR against main and link it to T3. Get one
  Opus 5.5 high review, fix findings, and confirm fixes with a short delta check if needed.
- [ ] Remove this temporary plan, update the PR and tracker, mark ready, and report evidence,
  review, CI status, open points and follow-ups. Do not merge, queue, push main or tag.

Only the cross-review delegates work. Checks and implementation remain in this worktree. The full
suite depends on final implementation and docs; review depends on all local checks passing.

Focused validation: 104 tests passed, including the snapshot, import and expression tests.
All nine gradient checks rejected a deliberately zeroed VJP. The workspace check rejects an
inserted shared intermediate, with frame_budget=0 so stack allocation cannot conceal it.
The test-node baseline has 5471 entries, including 11 new tests. Ruff and ty pass, and the docs
build passes. The README, guide example and representative examples pass. The detached full
pre-merge run exited 0: 5143 tests passed, 16 skipped, benchmark smoke passed, docs built without
issues, and the private-name check passed. Generated-C snapshots remain unchanged.

The Opus review found two small fixes: correct the indexing docstring and reject symbolic pad
constants with a helper-naming error. Both are fixed. One added invalid-input case failed before
the guard and will verify the fix. The optional export assertion and guide wording are simplified.
The signature's subok argument and direct flatten reshape are retained to follow NumPy and avoid
coupling to the parallel vec rename. The existing identity slice is retained, with no workspace.
Remaining: checks on these fixes, delta check, remove plan, final push, ready PR and tracker report.
