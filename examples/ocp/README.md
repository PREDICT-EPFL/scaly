# Optimal control

Notebooks for `scaly.ocp` (`from scaly import ocp`): optimal control problems over a horizon
(`ocp.ContinuousOCP`, transcribed into an `ocp.DiscreteOCP`), their solvers (`ocp.solver` with the
direct method `ocp.Direct`), receding-horizon control written as a loop over the solver and
`ocp.shift`, and the offline design of terminal ingredients. The models come from
`scaly.integrators` (`../integrators/`). Every notebook needs a vendored solver, PIQP for the linear
problems and IPOPT or scaly's SQP for the nonlinear ones. Each ends with a cell of assertions against
its references, and `tests/integration/test_notebooks.py` runs every notebook top to bottom. They are
saved with their outputs, so they read on GitHub without running.

| Notebook | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `nmpc_transcriptions.ipynb` | M/L | A cart-pole swing-up to a terminal equality, transcribed five ways, *IPOPT* | one `ContinuousOCP` transcribed five ways (`ocp.MultipleShooting` with `si.RK4` and with `si.RadauIIA`, `ocp.Collocation` at Radau and Legendre points, `ocp.Pseudospectral`): sizes, IPOPT iterations and times; each solution checked interval by interval against `solve_ivp`; refining the grid takes the spread between them from 1e-4 to 1e-7; `cost="points"` against `cost="integral"`; a control law written as C |
| `nmpc_closed_loop.ipynb` | M | The cart-pole swing-up in closed loop against a pole 20% heavier than modelled, *IPOPT*, *SQP* | `ocp.solver` with `ocp.Direct` on IPOPT and on scaly's SQP; a closed loop over the solver and `ocp.shift` on an `si.adaptive` plant; a soft `ocp.Path` track limit that gives way only where a hard one is infeasible; whole plans from the solver, `ocp.initial_guess` and a pumping warm start; the control law as one `@sc.function` of the solver and the shift, run by hand as a C caller would |
| `reference_tracking.ipynb` | S/M | A cart tracking a stepped and sinusoidal reference, its mass a parameter, *PIQP* | parameters by name (`ocp.Quadratic(x_ref="r")`, `ocp.TerminalEquality(x_ref="r")`, a step Function taking `mass`, `problem.params`, the solver's inputs); a `varying=("r",)` preview against a held reference; a closed loop that passes each step's reference; one solve checked against NumPy's KKT solve |
| `linear_mpc.ipynb` | M | Two masses joined by a spring, and a double integrator, under linear MPC, *PIQP* | `si.zoh` into `si.affine` with `ocp.Quadratic` costs; `ocp.lqr` as the unconstrained limit (`u = Kx` and cost `x'Px` for N = 1, 5, 20); an `ocp.max_invariant_set` terminal set; the sparse QP against `ocp.Direct(..., form="condensed")` (sizes, PIQP times); recursive feasibility and cost decrease in closed loop; the control law written as C |
| `terminal_sets.ipynb` | M | Terminal sets for a double integrator's MPC and the regions of attraction they give, *PIQP*, *IPOPT* | `sc.sets.Polytope` (`box`, `intersect`, `preimage`, `support`, `chebyshev_center`, `remove_redundancy`, `vertices`); `ocp.max_invariant_set` against a brute-force closed loop; `ocp.largest_ellipsoid`; `TerminalEquality`, the ellipsoid (IPOPT), the polytope (PIQP) and no terminal set on one grid of initial states, checked against HiGHS LPs; `NotQuadratic` for the ellipsoid on PIQP |

```bash
uv run --with jupyterlab jupyter lab examples/ocp
```

Matplotlib is in the dev group; Jupyter is not, and `--with jupyterlab` adds it without changing the
lock file. The plots use `../notebooks/plotstyle.py`. Each notebook ends by writing the C of a
control law to `examples/generated/ocp/<name>/` (git-ignored). The law is one Function the notebook
composes from the solver and the shift, `law(x0, *params, warm) -> (u, warm_next)`:

```python
solve, shift = ocp.solver(problem, method), ocp.shift(problem, method)

@sc.function(sc.L("x0", nx), sc.L("warm", method.warm_size(problem)), output=sc.G("u", "warm_next"))
def law(x0, warm):
  _, us, point, _ = solve(x0, warm)
  return us[0], shift(point)
```

so the generated code holds the solver and the warm start's shift, and a C caller keeps one buffer of
`warm_size` doubles between calls.

## Where each piece of the package is used

| Names | Notebooks |
| --- | --- |
| `ContinuousOCP`, `transcribe`, `cost=` | `nmpc_transcriptions.ipynb`, `nmpc_closed_loop.ipynb` |
| `DiscreteOCP`, `si.affine`, `Quadratic` | `linear_mpc.ipynb`, `terminal_sets.ipynb`, `reference_tracking.ipynb` |
| `Path` (hard and `soft=`) | `nmpc_transcriptions.ipynb`, `nmpc_closed_loop.ipynb` |
| `TerminalEquality` | `nmpc_transcriptions.ipynb`, `reference_tracking.ipynb`, `terminal_sets.ipynb` |
| parameters by name, `varying=` | `reference_tracking.ipynb` |
| `Direct(..., form="condensed")`, `to_problem` | `linear_mpc.ipynb` |
| `solver`, `Direct`, a law of the solver and `shift` | all five; the law called by hand in `nmpc_closed_loop.ipynb` |
| `initial_guess`, `Direct.layout` | all five; a warm start built from the layout in `nmpc_closed_loop.ipynb` |
| a closed loop over `solver` and `shift` | `nmpc_closed_loop.ipynb`, `reference_tracking.ipynb`, `linear_mpc.ipynb` |
| `lqr`, `sc.sets.Polytope`, `max_invariant_set` | `linear_mpc.ipynb`, `terminal_sets.ipynb` |
| `sc.sets.Ellipsoid`, `largest_ellipsoid` | `terminal_sets.ipynb` |

## What the examples ran into

- The swing-up is multimodal. Without a terminal equality, IPOPT stops at local minima with the pole
  short of upright, and different transcriptions find different ones, so `nmpc_transcriptions.ipynb`
  ends the horizon with `TerminalEquality()`.
- `x_bounds` constrain a collocation's internal states as well as the grid points. Refining the
  degree then tightens the problem instead of refining it, and the transcriptions stop converging to
  one another. `nmpc_transcriptions.ipynb` puts the track limit on the grid points with a `Path`.
- Scaly's SQP needs a good start on the swing-up. From the default guess it reaches a worse local
  minimum; started from IPOPT's first solution, the point that solve returned, it follows IPOPT's
  closed loop to 5e-6 in fewer iterations per step.
- PIQP 0.6.2 takes no warm start, so its law's warm buffer only carries the layout. On an
  infeasible start it runs to `MAX_ITER` in the sparse form rather than reporting infeasibility,
  which the condensed form does.
