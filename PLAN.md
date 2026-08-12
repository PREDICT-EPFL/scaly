# B3 closeout and B4 readiness

## Goal

Close the remaining benchmark-harness gaps from B3, then start `alloy-sqp` on a
plugin contract that can support the comparison and instrumentation promised by
B4. B3 is complete only when each canonical closed-loop artifact can drive its
canonical function-evaluation (FE) benchmark and the solver-backed gates run in
continuous integration (CI).

## Decisions

- Use exact Lagrangian Hessians by default wherever the backend supports them:
  checks, closed-loop runs, and FE sweeps. Limited-memory Hessians remain an
  explicit diagnostic opt-out, never the default.
- Make the unbumpercars exact Lagrangian Hessian the primary isolated FE
  scalability kernel. Keep the Jacobian as a secondary kernel only if the
  harness can express both without duplicating its interfaces; otherwise replace
  it. The existing measurements show little Jacobian difference and a much
  larger Hessian difference.
- Do not add a Gauss-Newton residual contract for B4. `alloy-sqp` will initially
  support the exact Lagrangian Hessian and the objective Hessian (the existing
  Hessian oracle with constraint multipliers set to zero). Decide during Phase 4
  whether to add solver-local BFGS; it requires no NLP contract change.

## Context and affected files

- The canonical unbumpercars run now uses the discrete-time model and harvests
  35,079 weights, while `benchmarks/harness/sweep.py` still builds the
  continuous-time oracle expecting 4,807 weights. The handoff currently fails.
- Solver-backed problem gates live in `benchmarks/problems/*/checks.py`, but
  `.github/workflows/ci.yml` runs them without solver libraries and does not
  rerun them in the solver job.
- `src/alloy/solvers/stats.py` cannot represent separate QP and globalization
  timings. `src/alloy/solvers/registry.py`, `docs/solver_plugins.md`, and the
  structural registry tests define the plugin protocol that B4 must use.
- `ROADMAP.md` is the source for B3/B4 exit criteria; update it only after the
  corresponding acceptance checks below pass.

## Phase 1 — coherent canonical FE harvesting

- Make the canonical unbumpercars sweep model match the canonical closed loop:
  discrete plant/filter, C=8. Preserve continuous-time results as historical
  documentation or an explicit non-canonical variant, not as an implicit
  consumer of the discrete artifact.
- Harvest every input required by an exact Lagrangian Hessian, including the
  solved primal, objective factor, constraint multipliers, state/desired input,
  model weights, physics, and time step.
- Add the exact-Hessian Alloy and CasADi MX kernel builders and a dense
  correctness comparison before timing. Prefer a small kernel selector if it
  cleanly supports both Hessian and Jacobian; otherwise make the Hessian the sole
  unbumpercars sweep kernel.
- Validate all harvested field shapes at the handoff, not only `z`. Add a test
  that feeds a canonical-shaped C=8 artifact through both backend builders.
- Re-run the three canonical handoffs. Chain M=5 and race-cars N=40 must remain
  unchanged; unbumpercars C=8 must now pass instead of failing on `pw`.

## Phase 2 — exact-Hessian gates enforced in CI

- Remove the opt-in environment gate around the unbumpercars exact-Hessian
  check. Keep the test small enough to run in the solver CI job.
- Assert the default exact-Hessian configuration in all three problem checks;
  any limited-memory run must be requested explicitly.
- In the solver CI job, run `benchmarks/run.py smoke --select problems` after the
  solver libraries are restored. Missing IPOPT or an unexpected skipped
  solver-backed gate must fail this job. The core job may continue running the
  dependency-free subset.
- Run Ruff, ty, pytest, full benchmark smoke, and all public closed-loop smoke
  commands on both supported CI platforms.

## Phase 3 — freeze the B4 plugin contract

- Extend the solver stats ABI with separate QP and globalization/line-search
  timings. Define an additive timing invariant, update PIQP/IPOPT to fill zero
  for inapplicable fields, and bump both the stats ABI and solver-plugin protocol.
- Update `docs/solver_plugins.md`, `docs/solvers.md`, and the fake-backend
  structural tests. After this protocol bump, B4 itself must require no edits to
  `src/alloy/`.
- Prove with a minimal plugin-level fixture how the same SQP implementation will
  call either Alloy-generated or CasADi-generated C-ABI oracles. If external
  oracle declarations cannot be represented by the current descriptor, include
  the smallest explicit support in this protocol bump rather than special-case
  it in `alloy-sqp`.

## Phase 4 — implement and accept `alloy-sqp`

- Create `plugins/alloy-sqp` against the frozen protocol, link PIQP through
  `piqp_c`, and add plugin-local correctness, nested-solve, status, and timing
  tests.
- Implement exact Lagrangian Hessian and objective-Hessian modes first. Evaluate
  BFGS here and add it only if it materially improves robustness or cost; do not
  introduce Gauss-Newton residuals.
- Add the one-solver/two-oracle columns for chain, race cars, and unbumpercars,
  with identical SQP settings and FE/QP/globalization timing reports.
- Run smoke episodes before canonical episodes. Accept B4 only when both oracle
  providers agree to declared tolerances and no core special case was added.

## Ordering

Phases 1 and 2 can be developed in parallel but both must pass before B3 is
marked complete. Phase 3 starts after their artifact and timing requirements are
stable. Phase 4 is strictly after Phase 3. Delete this file once all four phases
are complete and the roadmap has been updated.
