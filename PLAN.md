# Reject declarations that collide in C

Issue #388 under `internal/notes/compiler_run_2026_11_10/plan.md` and
`~/.scaly-suite/RULES.md`. Branch `t3code/388-c-name-collisions` is stacked on
`t3code/374-exit-var-clash`; its PR starts with that base. #387 already has
review and delta approval and entered the merge queue once.

The verifier uses `utils/names.py::c_ident`, as the renderer does, and rejects
same-scope collisions among exit variables, local buffers, scalar declarations
and buffer parameters. Preserve nested C shadowing, flattened BLOCK semantics,
scalar updates and the renderer's deduplication of repeated local buffers.
Buffers render before the procedure's other statements. Header macros are #326.
If a newly rejected case is reachable through today's lowering, stop and report
instead of renaming or changing lowering.

- [x] Inspect the issue, declaration emission and neighboring verification tests.
- [ ] Add regressions compared with C99 syntax checks and observe failure before the fix.
- [ ] Extend declaration scopes, prove failure when the new check is removed, and run focused checks with at most two workers.
- [ ] Refresh test collection; run the full pre-merge hook detached through the shared runner; preserve all C snapshots.
- [ ] Push once, open and link a stacked draft PR.
- [ ] Obtain one Opus high correctness review and one short delta check; settle both delegates.
- [ ] After #387 squash-merges, perform the one forced rebase onto origin/main and retarget the PR to main through REST.
- [ ] Remove this plan, publish completion evidence, and queue the PR once with squash auto-merge.

Keep `passes/program/hoist_invariant.py` untouched. Existing task #380 owns it.
