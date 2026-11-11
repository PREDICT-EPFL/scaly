# Float32 printing, #373

Section #24 of `internal/notes/compiler_run_2026_11_10/plan.md`. Base and PR base are
`origin/main` and `main`. The issue and roadmap fix the design: admit float32 in
`ir/expr.py::PRINTABLE` and render it with `%.9g` in `codegen/c.py::_PRINTF`.
C promotes a float argument to double for variadic printf. Float64 output and C snapshots
must stay byte-identical. Embedded build #207 and other printing behavior are out of scope.

- [x] Read the run plan, suite rules, issue, section contract and compiler conventions.
- [x] Add exactly three tests for JIT evaluation and round-trip output, rendered format,
  and removal with `-DSCALY_PRINTF(...)=`. Observe the existing rejection.
- [x] Add the two float32 table entries and update the guide and API dtype descriptions.
- [x] Run focused tests and perturb their gates. Refresh the full collection baseline.
  All 29 print and snapshot checks pass. Eight-digit formatting fails both precision tests;
  overriding print removal fails the removal test. The README solve/export and print example run.
- [x] Rebase onto current origin/main for the coordinator's flush.
- [x] Run the detached, queued pre-merge hook, including the full suite and C snapshots.
  The first hook found an obsolete IR test expecting float32 prints to raise. That case is
  removed; the refreshed baseline collects 6407 tests. Focused print, IR and snapshot checks
  pass, 34 tests total. No C snapshot changed.
  The rebased full hook passed: 6351 tests passed, 56 skipped. Types, formatting, lint,
  benchmark smoke, the documentation build and private-name checks also passed.
- [x] Report the branch, head, ahead/behind main, hook counts and snapshot changes. Stop.

The resumed brief and `~/.scaly-suite/super/flush-common.md` override the original push and
review steps. Push nothing, open no PR and start no Claude review. The coordinator schedules
the PR and review after #448 lands and explicitly releases this branch to push.

The phases run in this order. Focused tests use no more than two workers. Full suites
run only through `~/.scaly-suite/run`; Codex polls the detached run's exit file in its turn.
Never push to main.
