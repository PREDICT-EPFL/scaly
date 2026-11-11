# Seed-pair lowering for #225

Implement the Sparse tensors roadmap under the compiler run plan and ~/.scaly-suite/RULES.md.
Branch t3code/225-seed-pair-lowering, PR #419, base main. No budget or support-rule changes.

- [x] Read the run plan, rules, #225/#214/#220 and overlapping pull requests through REST.
- [x] Capture the launch-base chain M=9, 40-stage Hessian baseline with default budgets.
- [x] Implement per-entry seed lists, argument-support variants and mapped-call workspace reuse.
- [x] Differential tests against NumPy, including actual mapped JVPs and a deliberately broken list.
- [x] Regenerate snapshots; all existing snapshots and the new seed-pair snapshot are byte-identical.
- [x] Independent Sol review and delta: preserve transitive call-input producer lifetimes.
- [x] Opus primary review: fix int/bool typed zeros and contraction tables proportional to operands.
- [x] Remove sparse-vector matmul/gather/scatter per-seed AD stacks, with size and loop-growth gates.
- [x] Forced rebase onto merged #406, preserving its sparse SELECT lowering.
- [ ] Complete detached full pre-merge through ~/.scaly-suite/run and poll its exit inside this turn.
- [ ] One Opus low-effort delta on the fixes and regenerated snapshot diff; settle after using result.
- [ ] Remove this plan, push all review fixes together and link PR immediately after the push.
- [ ] Update PR and #214 with evidence, mark ready, queue once with squash auto-merge, report.

The section coordinator re-cut #225 under run-plan rule 5: #427 owns the operation target and
straight-line floor investigation; it is blocked by #225 and blocks #214. Do not park #419 for
its 67,308 count. #225 still owns the sparse-data Jacobian source-growth criterion, now covered.
No roadmap departure. Runtime waits for Phase 4; #216 does not exist and #214 stays open.

Evidence lives under /tmp/scaly-225/ on la015. Both builds use the exact descriptor Hessian from
benchmarks.harness.sweep._descriptor_kernel(chain_nlp(9,40), "hess"), render_c_module defaults,
and gcc -O2 -fPIC -shared ... -lm. Source size is the C translation unit excluding its header.
- Launch main: 253,786 ops/stage, 632,057 B, 5.1856 s generation, 9.0952 s compile.
- Corrected: 67,308 ops/stage, 1,121,802 B, 5.6640 s generation, 6.6690 s compile.
- Lifted-budget diagnostic on unchanged launch main: 58,465 ops/stage (not production).
- Mapped arithmetic gate: launch base 196 > 60; candidate 52. Seed-table perturbation changes values.
- Sparse-data Jacobian loops at 8/16/32 seeds: matmul 7/7/7, gather 1/1/1, scatter 1/1/2;
  source bytes respectively 2,225/3,300/7,635; 1,317/1,895/4,204; 1,514/4,204/10,299.
- Contraction review probe, n=64, 90%/50%/30% seeds: 293,035/249,965/228,328 B, previously
  4,325,579/about 2.49 MB/about 1.57 MB. Numeric max error below 2.2e-13.
- New typed-zero and source-size tests failed on reviewed code before their fixes (five failures).
  All three sparse-data growth tests also failed with the per-seed stacks before the AD change.
- Fourth pre-merge (before Opus fixes) passed: 4,662 tests, 56 skipped. Final suite pending.

Opus's cheap notes are addressed by sharing mapped support windows with analysis through a minimal
ir/support.py hook and removing a dead patterned-node check. No support rules are changed.
The primary review thread and cancelled quota-failed request are settled; the delta is still owed.
