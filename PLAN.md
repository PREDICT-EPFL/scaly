# Scaly-only model setup for the acados example

The maintainer requested updating #265 so its Scaly solver needs no handwritten
CasADi dynamics or cost. The pure-CasADi model remains the independent reference.
A local probe already confirmed that shaped zero placeholders work with acados
5033648, including exact Hessians, when all generated dynamics and cost modules
and their metadata queries are replaced before compilation.

Keep all integration helpers in the example. Do not change the library,
benchmarks, dependencies, acados, or wheel. Preserve the approved integer-flag
wording and the byte-identical compiler snapshots.

- [x] Extend the small-instance test to reject CasADi model construction on the
  Scaly path, and observe its failure.
- [x] Add a small example helper that generates zero placeholders, exports Scaly
  dynamics and all three stage costs with derivatives, and builds the solver.
- [x] Update the example docstring and code-generation guide using write-docs.
- [x] Run the wheel-backed tests and standalone example, verify skips without
  acados and dynamics/cost perturbation failures, and run the pre-merge checks.
- [ ] Commit and push, update #265, obtain one review of this revision, and
  confirm any fixes with the required short check.
- [ ] Remove this plan, record final evidence and review, and watch CI. Do not
  merge or close issues.

Steps run sequentially except independent validation commands. Existing evidence
and the wheel are in /tmp/scaly-acados-243/ on la015. The sparse-output public API
gap is already recorded on follow-up #48.

The strengthened test failed on the old Scaly path because it called the CasADi
model builders. Both targeted wheel tests now pass, and both skip without acados.
The helper takes expression builders, reads dimensions from the OCP SX vectors,
and replaces all 12 dynamics/cost files including sparse metadata queries.

All pre-merge checks pass: 2,411 tests passed, 91 skipped, smoke and docs.
The 20-interval example converges in five SQP iterations at cost 65.80579739.
Both mutations fail state iterate 1, with maximum errors 0.00798451 and
0.00568691. Restored wheel tests pass. This revision changes only the example,
its test and guide prose, plus this temporary plan.
