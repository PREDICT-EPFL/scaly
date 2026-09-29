# TinyMPC in Scaly

[TinyMPC](https://github.com/TinyMPC/TinyMPC) solves linear MPC problems with box and second-order-cone
constraints by ADMM, with the Riccati gain cached so that each iteration is a backward pass for the
affine terms, a rollout, projections and dual updates. Here the algorithm is written once in Python
and Scaly generates C specialized to each problem: dimensions, horizon, dynamics and cache become
constants of the code, and the loops over the horizon stay loops.

| File | Content |
| --- | --- |
| `problem.py` | `Problem` (with `Settings` and `Cone` from `scaly.ocp.tinyadmm`); the Riccati cache as the library computes it (`tinympc_cache`); `reference_solve`, a line-by-line NumPy port of the library's `solve` (the test oracle) |
| `solver.py` | `build_solver(problem)`: the generated `tiny_solve`, one `Function` `(state, x0, xref, uref, bounds...) -> (state_next, iterations, solved, u0)`, which is `scaly.ocp.tinyadmm.admm_solver` with the library's linear cost; `Solver`, a warm-started Python wrapper. The same ADMM with the problem's own references and terminal cost is the OCP method `sc.ocp.TinyADMM` |
| `problems.py` | the three problem families of [mcu-solver-benchmarks](https://github.com/RoboticExplorationLab/mcu-solver-benchmarks) as closed-loop scenarios, with their sweeps |
| `random_mpc.py` | random QP-MPC (ICRA 2024): tracking with `|u| <= 3` |
| `safety_filter.py` | predictive safety filter on a double integrator (CDC 2024 benchmarks) |
| `rocket_landing.py` | rocket soft landing with a thrust cone (Conic-TinyMPC) |
| `benchmark/` | the generated solver against the TinyMPC library: solve time, code size, compile time ([report](../../internal/notes/tinympc_benchmark_report.html)) |

```bash
uv run examples/tinympc/random_mpc.py          # nx nu N, default 10 4 10
uv run examples/tinympc/safety_filter.py       # nx N, default 4 20
uv run examples/tinympc/rocket_landing.py      # N, default 32
uv run examples/tinympc/benchmark/run_benchmark.py --workdir /tmp/tinympc-bench
uv run examples/tinympc/benchmark/report.py
```

## How the solver is built

One call of the generated function is one `tiny_solve`:

- the solver state (trajectories, box slacks and their previous values, box and cone duals) comes
  in and goes out, so the caller warm-starts the next solve the way the library keeps its
  workspace; `pack_state`/`unpack_state` give the layout;
- `sc.while_loop` runs the ADMM iterations until the residuals meet the tolerances or `max_iter`;
- inside it, `sc.scan` runs the backward pass for `d_k`, `p_k` over the horizon in reverse (negative
  strides), and a second `scan` rolls the trajectory out with the cached gain;
- clipping, cone projections (`where`, `sqrt`), dual updates, `norm_inf` residuals and the stopping
  test are plain expressions over the whole horizon.

Disabled bounds are not inputs, and cone code is generated only for the cones a problem declares.
`build_solver(..., fixed_bounds=bounds)` makes the bounds constants; `lowering="block"` keeps the
stage steps as loops over constant tables instead of straight-line code.

## Fidelity to the library

The generated solver takes the same iterations as the TinyMPC library (commit `023f36b`) on every
step of every benchmark instance, and returns the same inputs to 1e-13 on the QPs. On the rocket
the library computes part of its cone projection in `float`, which leaves 1e-5; with that made
`double` the two agree to 1e-13. To compare iterates, the port keeps the library's conventions
where they differ from textbook ADMM (listed in `problem.py`): the linear cost uses `Q + rho I`,
one `rho` is folded into the cache however many constraint sets act on a variable, the cone
projection scales the axis by `mu`, and termination tests the box slacks only.

Iteration counts are today's library's, which reorders the ADMM steps relative to the version
behind the published result sheets.
