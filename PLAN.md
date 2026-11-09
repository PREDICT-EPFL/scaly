# Fatrop stage Functions, issue #240

Third of the #282 / #280 / #240 stack. This branch and the PR base are
`t3code/239-fatrop-backend`. Do not rewrite either lower branch.

The contract is `StageFunctions` and the recorded layout in `solvers/ocp.py`,
`docs/dev/solver_plugins.md`, and "Block-structured solvers" in the compiler
roadmap. Stage forms select the mapped path automatically. General problems
keep the whole-problem path. List this selection rule as an open point.

- [x] Inspect #240, parent #236, prerequisites, stack and project conventions.
- [x] Add failing differential tests for dense stage derivatives and solver
  agreement, including per-stage parameters and carried global variables.
- [x] Assemble mapped dense stage derivatives in core and consume them in the
  Fatrop plugin, sharing wrapper setup, bounds, multipliers and statistics.
  Keep the builder API unchanged and whole-problem changes to small hooks.
- [x] Solve a free-final-time problem and compare both paths' iterations on
  builder problems. Perturb new correctness checks to prove they fail.
- [x] Extend the three Fatrop documentation sections using write-docs. Check
  parent #236's documentation requirements and run the examples.
- [x] Run all pre-merge checks and preserve the C snapshots byte for byte.
- [x] Commit and push, post the concrete benchmark-study plan, obtain the
  required launch choice, then record chain function-evaluation time for both
  paths from native timers on la015. Store the evidence under internal/notes.
- [x] Open a draft PR against #239's branch, link in T3 and `gh stack link 284`.
- [ ] One cross-review with the required reviewers, fix findings and confirm
  fixes with a delta check. Delete this plan before marking ready, update #240
  to In review, and watch the PR.

Global-variable bounds (#283) and #215 are outside scope. Stop if the stage
contract or whole-problem implementation needs more than a small change.

Progress: core and all Fatrop tests pass. The derivative test also checks an
unrolled NumPy formulation. Perturbing dense Hessians or reversing the dynamics
Jacobian makes the derivative and workload gates fail. Chain M=3/N=10 and race
car N=10 match iterations and solutions. Free final time matches IPOPT.
Both solver-guide fences and both paper-chain scripts run. Benchmark launch
authorized by Ted; CPU-policy choice pending because la015 is powersave/boost on.

Pre-merge hooks passed: 2548 passed, 52 skipped, smoke, ty, Ruff, docs and
private names. Clean strict docs build passed. README code and export ran. Ted
set the CPU policy; performance and disabled boost verified.

Measurement completed: eight cells, 240 successful timed solves, no iteration
mismatches. Stage FE medians are 0.643/1.546/4.002/12.617 ms versus
0.818/1.948/4.803/15.143 ms. Evidence and raw-summary copy are in the study note.
PR #285 is draft, based on #239 and linked into stack #284. Cross-review remains.
