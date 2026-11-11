# Reject duplicate exit-loop declarations

Issue #374, under `internal/notes/compiler_run_2026_11_10/plan.md` and
`~/.scaly-suite/RULES.md`. Base: origin/main after #379. The decided contract is
the #29 paragraph in `internal/notes/core_compiler_roadmap.md`.

The verifier must reject repeated exit-variable declarations in one C scope.
Blocks render without braces; procedure and loop bodies have separate scopes.
Nested shadowing remains legal. LOOP lowering belongs to #31; do not edit
`passes/program/hoist_invariant.py`, which #380 owns.

- [x] Inspect the issue discussion, prerequisites, verifier, renderer, and existing tests.
- [x] Add the issue reproduction and scope regressions; observe the rejection test fail.
- [x] Extend the existing context walk with exit-declaration scopes; prove the check fails when removed.
- [x] Refresh the collection baseline and run focused checks with at most two workers.
- [ ] Run the full pre-merge hook detached through the shared suite runner; keep C snapshots byte-identical.
- [ ] Push once, open a draft PR against main, and link it in T3.
- [ ] Obtain one correctness review from claude-opus-5-5 at high effort, fix findings, and obtain one short delta check.
- [ ] Settle used review threads, remove this plan, publish the evidence, and queue once with squash auto-merge.

The issue currently has one discussion comment. It explicitly moves the two
`while_loop` criterion to #31 and permits closing #374 with the verifier fix.
Devrush's carried-loop tests cover recurrence and break placement, already covered
on main; they contain no duplicate-declaration regression to port.

Evidence so far: three rejection cases fail before the fix and with the check
temporarily removed. The focused verifier, carried-range C, import-boundary,
import-layer, and C snapshot run passes 276 tests with two workers. Both plain
and carried outer loops permit a same-name inner exit variable and match a
Python loop after compiling the C. All 3510 test IDs collect successfully.
Strict ty and Ruff pass. Logs are under `/tmp/scaly-374/`.
