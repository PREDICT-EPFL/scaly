# Keep scalar assignments with their readers

Implement #380 on origin/main under
`internal/notes/compiler_run_2026_11_10/plan.md` and `~/.scaly-suite/RULES.md`.
The decided design is the roadmap's "Loops, conditionals and printing", #29.
Only `passes/program/hoist_invariant.py`, its tests and the collection baseline change.
Preserve the carried-range guard from #379. Do not edit `ir/program_spec.py`.

- [x] Read the issue, parent, run rules, design and existing pass helpers.
- [ ] Add the issue's compiling, numerical reproduction and scalar dependency cases.
- [ ] Observe the reproduction fail, add the smallest placement rule, and prove reverting it fails.
- [ ] Check focused tests and byte-identical C snapshots, then run all pre-merge checks detached through the shared runner.
- [ ] Push once for a draft PR and request one Opus 5.5 high correctness review through cross-review.
- [ ] Fix findings together and request one short delta check on those fixes alone.
- [ ] Settle review delegates, remove this plan, mark ready, update the issue and queue once with squash auto-merge.

Focused tests use at most two workers. GitHub reads use REST. No main push, tags,
releases or deployment workflows. Review results and criterion evidence go in the PR.
