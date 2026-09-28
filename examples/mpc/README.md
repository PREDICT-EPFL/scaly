# Model predictive control

Notebooks for `scaly.mpc` (`from scaly import mpc`): optimal control problems over a horizon
(`mpc.OCP`), their solvers and control laws (`mpc.MPC`), closed-loop simulation, and the offline
design of terminal ingredients. The models come from `scaly.integrators` (`../integrators/`). Every
notebook needs a vendored solver, PIQP for the linear problems and IPOPT or scaly's SQP for the
nonlinear ones. Each ends with a cell of assertions against its references, and
`tests/integration/test_notebooks.py` runs every notebook top to bottom. They are saved with their
outputs, so they read on GitHub without running.

| Notebook | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `nmpc_transcriptions.ipynb` | M/L | A cart-pole swing-up to a terminal equality, transcribed five ways, *IPOPT* | one `mpc.OCP` per transcription (`si.MultipleShooting` with `si.rk4` and with Radau IIA `si.implicit`, `si.Collocation` at Radau and Legendre points, `si.Pseudospectral`): sizes, IPOPT iterations and times; each solution checked interval by interval against `solve_ivp`; refining the grid takes the spread between them from 1e-4 to 1e-7; `cost="points"` against `cost="integral"`; the `law` written as C |
| `nmpc_closed_loop.ipynb` | M | The cart-pole swing-up in closed loop against a pole 20% heavier than modelled, *IPOPT*, *SQP* | `mpc.MPC` with IPOPT and with scaly's SQP; `mpc.simulate` on an `si.adaptive` plant and its `mpc.ClosedLoop`; a soft `mpc.Path` track limit that gives way only where a hard one is infeasible; `ctrl.solve` returning an `mpc.Solution`, `reset` and `initial_guess`; `ctrl.law` run by hand with one guess buffer, as a C caller would |
| `reference_tracking.ipynb` | S/M | A cart tracking a stepped and sinusoidal reference, its mass a parameter, *PIQP* | parameters by name (`mpc.Quadratic(x_ref="r")`, `mpc.TerminalEquality(x_ref="r")`, a step Function taking `mass`, `ocp.params`); a `varying=("r",)` preview against a held reference; `mpc.simulate` with a parameter that is a callable of the step; one solve checked against NumPy's KKT solve |
| `linear_mpc.ipynb` | M | Two masses joined by a spring, and a double integrator, under linear MPC, *PIQP* | `si.zoh` into `mpc.linear` with `mpc.Quadratic` costs; `mpc.lqr` as the unconstrained limit (`u = Kx` and cost `x'Px` for N = 1, 5, 20); a `mpc.max_invariant_set` terminal set; the sparse QP against `condensed=True` (sizes, PIQP times); recursive feasibility and cost decrease in an `mpc.simulate` closed loop; `ctrl.law` written as C |
| `terminal_sets.ipynb` | M | Terminal sets for a double integrator's MPC and the regions of attraction they give, *PIQP*, *IPOPT* | `mpc.Polytope` (`box`, `intersect`, `preimage`, `support`, `chebyshev_center`, `remove_redundancy`, `vertices`); `mpc.max_invariant_set` against a brute-force closed loop; `mpc.largest_ellipsoid`; `TerminalEquality`, the ellipsoid (IPOPT), the polytope (PIQP) and no terminal set on one grid of initial states, checked against HiGHS LPs; `NotQuadratic` for the ellipsoid on PIQP |

```bash
uv run --with jupyterlab jupyter lab examples/mpc
```

Matplotlib is in the dev group; Jupyter is not, and `--with jupyterlab` adds it without changing the
lock file. The plots use `../notebooks/plotstyle.py`. Each notebook ends by writing the C of a
controller's `law` to `examples/generated/mpc/<name>/` (git-ignored). The law is one Function,
`law(x0, *params, guess) -> (u, guess_next)`, with the solver and the warm start's shift in the
generated code, so a C caller keeps one buffer of `guess_size` doubles between calls.

## Where each piece of the package is used

| Names | Notebooks |
| --- | --- |
| `OCP(ode=..., transcription=...)`, `cost=` | `nmpc_transcriptions.ipynb`, `nmpc_closed_loop.ipynb` |
| `OCP(step=...)`, `linear`, `Quadratic` | `linear_mpc.ipynb`, `terminal_sets.ipynb`, `reference_tracking.ipynb` |
| `Path` (hard and `soft=`) | `nmpc_transcriptions.ipynb`, `nmpc_closed_loop.ipynb` |
| `TerminalEquality` | `nmpc_transcriptions.ipynb`, `reference_tracking.ipynb`, `terminal_sets.ipynb` |
| parameters by name, `varying=` | `reference_tracking.ipynb` |
| `condensed=True` | `linear_mpc.ipynb` |
| `MPC`, `solve`, `Solution`, `law` | all five; the law called by hand in `nmpc_closed_loop.ipynb` |
| `initial_guess`, `reset` | `nmpc_closed_loop.ipynb`, `linear_mpc.ipynb`, `terminal_sets.ipynb`, `reference_tracking.ipynb` |
| `simulate`, `ClosedLoop` | `nmpc_closed_loop.ipynb`, `reference_tracking.ipynb`, `linear_mpc.ipynb` |
| `lqr`, `Polytope`, `max_invariant_set` | `linear_mpc.ipynb`, `terminal_sets.ipynb` |
| `Ellipsoid`, `largest_ellipsoid` | `terminal_sets.ipynb` |

## What the examples ran into

- The swing-up is multimodal. Without a terminal equality, IPOPT stops at local minima with the pole
  short of upright, and different transcriptions find different ones, so `nmpc_transcriptions.ipynb`
  ends the horizon with `TerminalEquality()`.
- `x_bounds` constrain a collocation's internal states as well as the grid points. Refining the
  degree then tightens the problem instead of refining it, and the transcriptions stop converging to
  one another. `nmpc_transcriptions.ipynb` puts the track limit on the grid points with a `Path`.
- Scaly's SQP needs a good start on the swing-up. From the default guess it reaches a worse local
  minimum; seeded with IPOPT's first solution (`reset(solution.guess)`), it follows IPOPT's closed
  loop to 5e-6 in fewer iterations per step.
- PIQP 0.6.2 takes no warm start, so its law's guess buffer only carries the layout. On an
  infeasible start it runs to `MAX_ITER` in the sparse form rather than reporting infeasibility,
  which the condensed form does.
