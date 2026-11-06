# acados race-car example

Implements #243, parent #241, on main. #242's findings comment completes the
prerequisite while the issue remains open. The route decision on #243 requires
ordinary CasADi solver generation followed by replacing dynamics and external-cost
C sources with Scaly CasADi-compatible exports before building. No C glue or
changes to Scaly's exporter, benchmarks, acados, wheel, or project dependencies.

Read the findings and route comments, the two probes on
`origin/t3code/242-acados-generic-research`, and the race-car benchmark model and
cost. The private 0.6.0 wheel downloaded on la015 with the recorded SHA-256.

- [x] Inspect scope, prerequisites, checkout, tracker, conventions and examples.
- [x] Add and run a failing small-instance iterate comparison test.
- [x] Implement a standalone example with independent Scaly and CasADi models,
  matching stage semantics, sparse derivatives and source replacement.
- [x] Verify the example and test with the wheel, demonstrate dynamics and cost
  perturbation failures, and verify skips without acados.
- [x] Document setup and link from the code generation guide using write-docs.
- [x] Refresh the node-ID baseline and run all pre-merge checks.
- [ ] Commit, push, open the sole draft PR against main, link it, obtain one
  cross-review, address findings and confirm fixes with a delta check.
- [ ] Remove this plan, complete the PR evidence and tracker status, mark ready
  and watch CI. Never merge or close issues.

Phases run sequentially. Only independent checks may run concurrently. A source
swap failure permits #242's native-adapter/custom-update fallback, with evidence
in the PR. If both routes fail, report on #243 and stop.

The source swap works without C glue. Wheel tests pass, and both tests skip in
the ordinary environment. Negative checks temporarily changed only Scaly's RK4
step by 1% or its throttle penalty by 10%. Both failed the state comparison at
iterate 1, stage 1, with differences 0.00126089 and 0.002697. Changes were restored.
The standalone 20-interval solve converges in five iterations at cost 65.80579739.
Logs and downloaded wheel are in `/tmp/scaly-acados-243/` on la015.

Pre-merge checks passed, including 2,411 tests with 91 skipped, smoke and docs.
The six-interval case converges in five SQP steps with maximum iterate differences
below 2e-11, including state, control, dynamics and inequality multipliers.
