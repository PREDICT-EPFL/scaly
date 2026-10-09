# Differential AD tests for #15

The decided design is the One AD engine section of internal/notes/core_compiler_roadmap.md. This is the first layer of the AD stack, based on main. Production code, C snapshots, and other layers' files are outside scope.

Read docs/how_it_works/architecture.md, docs/dev/codebase.md, docs/dev/conventions.md, docs/dev/contributing.md, internal/notes/compiler_maintenance_context.md, current forward/reverse rules and tests, and the frozen devrush AD fixtures.

- [x] Build explicit per-ExprOp cases and refusal inventory. Keep migration comparisons separate from permanent finite differences and duality. Include seed counts, broadcasts, empty tensors, repeated operands, all matmul forms and structural axes.
- [x] Extend joint-call, map and constant-seed tests with roadmap fixtures. Record current wrong answers as strict expected failures and file follow-up issues.
- [x] Prove sensitivity to a wrong partial and seed permutation without modifying production files. Measure suite wall time against the base and regenerate collection baseline.
- [x] Run pre-merge checks and confirm C snapshots remain unchanged.
- [ ] Commit and open a draft PR against main, get one independent cross-provider review, apply fixes and check the delta, then mark ready and update tracking.

Each phase depends on the prior phase. Validation commands can run independently; the independent review follows all checks.

The differential harness and fixtures pass with no wrong answers exposed. Temporary module-scoped monkeypatches made both selected sine checks fail, then the checks passed without corruption. Full-suite attempts reproduced the documented isolated solver-loader crashes on the base and changed checkout. Collection stderr must stay separate when regenerating node IDs; a stale-baseline error can otherwise join one printed node ID.

All pre-merge hooks passed. Full suite: 2260 passed, 3 skipped in 40.35 s. Base: 1483 passed, 3 skipped in 38.47 s. Observed added wall time: 1.88 s, 4.9%, with 16 workers and warm caches. PR https://github.com/PREDICT-EPFL/scaly/pull/160 is linked in T3. Claude Opus 5.5 at xhigh is reviewing the first differential-test revision, request scaly-15-review-1. Await its result, apply findings, run the delta check if needed, remove this plan, mark ready and update tracking.
