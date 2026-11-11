# Keep scalar assignments with their readers

Implement #380 on origin/main under
`internal/notes/compiler_run_2026_11_10/plan.md` and `~/.scaly-suite/RULES.md`.
The decided design is the roadmap's "Loops, conditionals and printing", #29.
Only `passes/program/hoist_invariant.py`, its tests and the collection baseline change.
Preserve the carried-range guard from #379. Do not edit `ir/program_spec.py`.

- [x] Read the issue, parent, run rules, design and existing pass helpers.
- [x] Add the issue's compiling, numerical reproduction and scalar dependency cases.
- [x] Observe the reproduction fail, add the smallest placement rule, and prove reverting it fails.
- [x] Check focused tests and byte-identical C snapshots, then run all pre-merge checks detached through the shared runner.
- [ ] Push once for a draft PR and request one Opus 5.5 high correctness review through cross-review.
- [ ] Fix findings together and request one short delta check on those fixes alone.
- [ ] Settle review delegates, remove this plan, mark ready, update the issue and queue once with squash auto-merge.

Focused tests use at most two workers. GitHub reads use REST. No main push, tags,
releases or deployment workflows. Review results and criterion evidence go in the PR.

The six compiled cases cover the reported body reader, hoisting both statements, a
scalar chain, reassignment, a nested-loop reader and a variant assignment. All six
pass with the fix. Restoring the pass from origin/main makes five fail with
undeclared scalar errors, including the exact reproduction. Restored the fix afterwards.
Ruff formatting, lint and strict ty pass. The collection baseline adds six node IDs,
and the complete collection passes with 3508 tests.
All 97 focused pass, carried-range and snapshot tests pass. The detached shared
runner exits zero: 3456 passed, 52 skipped, benchmark smoke passed, documentation
built without issues, and the private-name check passed.
