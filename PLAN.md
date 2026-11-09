# #88 discrete-time OCP builder

Branch `t3code/88-ocp-builder`, stack 1 of 3 (#88, #239, #240). Deleted before the PR is ready.

## Design choices (open points for the PR)

- `sc.DiscreteOCP(N=..., dynamics=F, stage_cost=l, terminal_cost=Vf, path=..., terminal=...,
  x_bounds=..., u_bounds=..., stage_params=(...), global_vars=(...), name=...)` in
  `src/scaly/solvers/ocp.py`; `sc.StageConstraint(fn, lo, hi)` for path and terminal constraints.
- Function inputs after `x` (and `u`) are matched by name: a global variable, a per-stage
  parameter, or else a global parameter. Parameters keep the order the Functions first name them,
  after `initial_state`.
- Per-stage parameters take N + 1 rows (the terminal stage reads the last), not N.
- `ocp.problem()` is the general lowering; `ocp.stage_problem()` the stage lowering, a `Problem`
  with `stage_layout` and `stage_functions`.
- Variables `z` (N, nx + nu), `x_N`, then the global variables. Stage form: `z` (N, nx + ng + nu),
  `x_N` (nx + ng), globals after the state. `ocp.pack` / `ocp.unpack` convert trajectories.
- Path constraints and costs apply at stages 0..N-1; state bounds skip x_0.

## Steps

- [ ] tests/solvers/test_ocp.py (ported devrush DiscreteOCP tests + lowering/stage-form tests)
- [ ] src/scaly/solvers/ocp.py, Problem.stage_functions, exports, import layers, codebase map
- [ ] chain and race-car builder formulations + gates in checks.py
- [ ] docs/guide/ocp.md, nav, API page
- [ ] checks (`wt hook pre-merge --yes`), cross-review, PR
