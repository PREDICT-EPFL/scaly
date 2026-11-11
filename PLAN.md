# Reuse compatible external library namespaces

Task #357, based on origin/main, under the unattended compiler run plan in
`internal/notes/compiler_run_2026_11_10/plan.md` and `~/.scaly-suite/RULES.md`.
Read the #355 diff, its `calls_and_carried_loops/extern.md` note, #159, the compiler
roadmap's external-code decision and the project contribution rules.

- [x] Inspect ownership, dependencies, loader, artifact/index records and existing tests.
- [x] Add Linux subprocess regressions for more than 16 compatible dependency sets,
  conflicting versions, callable old handles and namespace exhaustion. Show failures first.
- [x] Reuse namespaces by dependency compatibility, preserving the solver namespace,
  cached artifacts and process-lifetime handles. Record the rule for #342.
- [x] Run focused extern/JIT/solver checks with at most two workers, prove the gates
  reject broken reuse/isolation, and keep C snapshots byte-identical.
- [ ] Run all merge checks detached through `~/.scaly-suite/run`, then commit, push
  once and open/link a draft PR against main.
- [ ] Obtain one Opus 5.5 high correctness review, fix findings and obtain one short
  delta check. Settle both delegate threads after using their results.
- [ ] Remove this temporary plan, finish checks and the PR evidence, mark ready and
  queue once with `gh pr merge N --squash --auto`. Watch until landed or blocked.

Dependent phases run in order. No solver hook changes, cache-key redesign, macOS or
Windows loader changes, or broader environment-lifetime work. Namespace and handle
retention stays process-wide; reclamation is unnecessary for the completion criterion.

Evidence so far: all five new cases failed on the old loader. The fixed tests pass;
disabling compatibility checks fails the path, SONAME and transitive cases, and
disabling deep binding fails the independent case. Focused extern/JIT/solver checks
passed with 368 passed and 2 skipped. Collection has 4,758 tests. Ruff and ty pass.
The detached full merge check passed through the shared runner: 4,702 passed and
56 skipped, then benchmark smoke, documentation build and private-name checks passed.
Logs and exit status are at `/tmp/scaly357-full.log` and `/tmp/scaly357-full.exit`.
