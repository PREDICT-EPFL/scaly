# Float32 printing, #373

Section #24 of `internal/notes/compiler_run_2026_11_10/plan.md`. Base and PR base are
`origin/main` and `main`. The issue and roadmap fix the design: admit float32 in
`ir/expr.py::PRINTABLE` and render it with `%.9g` in `codegen/c.py::_PRINTF`.
C promotes a float argument to double for variadic printf. Float64 output and C snapshots
must stay byte-identical. Embedded build #207 and other printing behavior are out of scope.

- [x] Read the run plan, suite rules, issue, section contract and compiler conventions.
- [ ] Add exactly three tests for JIT evaluation and round-trip output, rendered format,
  and removal with `-DSCALY_PRINTF(...)=`. Observe the existing rejection.
- [ ] Add the two float32 table entries and update the guide and API dtype descriptions.
- [ ] Run focused tests and perturb their gates. Refresh the full collection baseline.
- [ ] Run the detached, queued pre-merge hook, including the full suite and C snapshots.
- [ ] Commit and push the reviewed candidate, open a draft PR, and link it in T3.
- [ ] Obtain one cross-review from claude-fable-5-2 at high effort. Fix findings together
  and confirm only the fixes if needed. Settle finished review threads.
- [ ] Delete this temporary plan, record the verdict and evidence, mark ready, and queue
  with `gh pr merge --squash --auto`. Watch until merged and report any ejection.

The phases run in this order. Focused tests use no more than two workers. Full suites
run only through `~/.scaly-suite/run`; Codex polls the detached run's exit file in its turn.
Push once for review and once for all review fixes if needed. Never push to main.
