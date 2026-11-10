# Allow empty dense leaves through CasADi

Issue: PREDICT-EPFL/scaly#290. The decided contract is in the issue,
`docs/how_it_works/generated_interface.md` and `docs/dev/versioning.md`.
The unattended compiler-run plan overrides the implement skill for merging and
snapshot measurements. Work starts above the sparse stack at `248b2159`.

- [x] Read the issue, parent, run rules, project conventions and interface implementation.
- [x] Add compiled CasADi regressions for empty dense inputs and outputs, dimensions,
      and preservation of nonempty null-pointer errors. Observe failures before fixing.
- [x] Emit leaf null checks only when storage is nonempty. Update the cache version
      and the generated-interface explanation. Verify regressions and perturb the fix.
      All 16 regressions failed before the fix and under a restored old predicate.
      The focused run passed all 48 CasADi, pointer-entry and C snapshot tests.
- [ ] Refresh the test collection baseline, then run all merge checks.
- [ ] After #289 merges, fetch and rebase onto main with
      `git rebase --onto origin/main 248b2159`, then rerun checks.
- [ ] Push, open the pull request against main, link it, and obtain cross-review.
      Apply findings and run one delta check if needed.
- [ ] Remove this temporary plan, record completion evidence, and queue with
      `gh pr merge --squash` once review and CI pass.

No dense matrix layout change, sparse interface change, or run-time measurements.
Sibling #103 also owns `codegen/casadi.py`; avoid editing it unless needed.
