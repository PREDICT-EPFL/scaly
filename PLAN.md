# Procedure frame budget, #76

Implement the decided frame budget under the compiler run plan at
`internal/notes/compiler_run_2026_11_10/plan.md`, currently read from the coordinator's
worktree until #291 merges. Base and PR target are `origin/main` and `main`.

Scope: packed local buffers, inlined callee locals and a fixed scalar allowance;
`frame_budget` in bytes, default 65536, with zero spilling eligible buffers;
largest buffers first; header estimate; GCC and Clang frame-warning gate.
Workspace dtype sizes and alignment belong to #56, bare-metal builds to #207.

Read: #76 and parent #24, architecture, codebase, conventions, contributing,
issue tracking, compiler maintenance context, roadmap decisions, foundations,
order and #76. Frozen devrush inspected for relevant workspace tests.

- [x] Inspect task, prerequisites, worktree and overlap; mark In progress.
- [x] Add failing tests for packing, call frames, zero budget and compiled module.
- [x] Implement final-program estimate and option through existing render options.
- [x] Update guide/API and regenerate reviewed C snapshots.
- [ ] Perturb the gate; run narrow checks, full pre-merge checks and examples.
- [x] Record operation counts, source sizes and generated-C compile time, no run-time timings.
  Data: `/tmp/scaly-76-measurements.json`. Other compiler-run tasks were active, so timings
  are diagnostics, not performance evidence. All seven before/after C cases build with GCC and Clang.
- [ ] Commit, push, draft PR, Opus cross-review plus required correctness review; fix and delta check.
- [ ] Remove temporary plan, make PR ready, queue after green checks and report merge.

No roadmap departures so far. Snapshots may overlap during the run; rebase and
regenerate if needed, and give the regenerated diff to the reviewer.
