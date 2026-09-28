# Integrators

Notebooks for `scaly.integrators` (`from scaly import integrators as si`), which turns a
continuous-time model `f(x, u, ...) -> xdot` into discrete-time maps and transcription constraints.
Each notebook covers one part of the package: the maths, the Scaly code in small cells, plots of the
results, and a check against a reference (`expm`, `solve_ivp` at tight tolerance, NumPy's solves,
published coefficients, closed forms or central differences). A last cell of assertions holds those
comparisons to tolerances, and `tests/integration/test_notebooks.py` runs every notebook top to
bottom. None of them needs a solver. They are saved with their outputs, so they read on GitHub
without running.

| Notebook | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `explicit_methods.ipynb` | S/M | The ten named explicit Runge-Kutta methods and one written by hand, on a linear system and Van der Pol | `si.explicit(model, method, dt=None)` with the step as an input, so one compiled map per method serves a whole step-size sweep; observed orders against `expm` and `solve_ivp`; `si.Tableau`, `si.tableau`, `si.order_conditions` (embedded rows included); a `Tableau` claiming too high an order, refused |
| `discrete_maps.ipynb` | S/M | A torque-driven pendulum with friction, and a chain of masses whose size comes from the state | the map contract: the model's own signature and the `{model}_rk4` name, the zero-order hold against `solve_ivp`, `dt=None`, `steps=` unrolled up to `si.UNROLL_STEPS` and then a `scan` whose C is as long at 500 substeps as at 5, two maps in one Function as a local error estimate, `sc.jacobian` in `x` and `dt`, `si.linearize` against `si.zoh`, a template model giving a template map and its `instances` |
| `stiff_implicit.ipynb` | M | Van der Pol, Prothero-Robinson at `h lambda = -1e4`, and Robertson's kinetics | `si.implicit` for the four collocation families and the five named methods at their orders; L-stable against A-stable decay of a stiff transient next to each tableau's `R(h lambda)`; Radau IIA(3) through Robertson at steps up to 3.7e4 s; `newton_iters=k` against `tol=` for simplified and full Newton; step Jacobians by the implicit function theorem against central differences |
| `frequency_response.ipynb` | M | Den Hartog's tuned mass damper: frequency response and H-infinity tuning | `sc.linalg.lu` with partial pivoting (the first pivot is zero), `lu_solve` for two inputs from one factor and with `trans=True` for the adjoint, `solve(assume="gen")` in a `vmap` over 400 frequencies against NumPy's complex solve, `sc.gradient` through the solves driving L-BFGS-B to two equal peaks, next to Den Hartog's formula |
| `linearize_and_discretize.ipynb` | S/M | A cart-pole at its upright equilibrium: continuous and discrete linear models | `si.linearize` evaluated as generated code against the closed form; `si.zoh` and `si.foh` against `solve_ivp` with a held and a ramping input; linearize-then-`zoh` against `si.linearize` of an `si.rk4` map, which differ by RK4's local error, 32 times smaller per halving of `dt` |
| `adaptive_plant.ipynb` | M | The Arenstorf orbit of the restricted three-body problem as a plant model | `si.adaptive` with `dopri5` and `bs32` from 1e-4 to 1e-10 against DOP853 and `solve_ivp`'s RK45 (errors, and time per call); NaN when `max_steps` runs out; `dt` as an input or folded in; `sc.jacobian` through the `while_loop` against the variational equations; fixed-step `si.explicit` needing about 100 times the steps |
| `symplectic_orbits.ipynb` | S/M | A thousand orbits of the Kepler problem at eccentricity 0.6 | `si.symplectic` (Störmer-Verlet, symplectic Euler) with `split=2` and 100 steps per call as a `scan`: the energy error stays bounded and the angular momentum holds to rounding, while `si.rk4`'s energy error grows tenfold at the same step; orders 2, 1 and 4 against `solve_ivp` |
| `polynomials.ipynb` | S | Gauss, Radau and Lobatto nodes and the collocation tableaus built on them | `scaly.integrators.polynomial`: quadrature exact to degree 2n-1, 2n-2 and 2n-3 and no further; differentiation and interpolation matrices converging spectrally on `exp(sin 3t)`; `si.gauss_legendre`, `si.radau_iia`, `si.lobatto_iiia` against published coefficients, with `si.order_conditions`; a generated spectral derivative and integral with the matrices as constants |
| `transcriptions.ipynb` | M | One interval of a driven pendulum under five transcriptions, each solved by Newton | the `si.Interval` contract that `scaly.mpc` consumes (`n_internal`, `n_residual`, `state_times`, `control_times`, `guess`) for `MultipleShooting` with RK4 and with Radau IIA, `Collocation` at Radau and Gauss points, and `Pseudospectral`; collocation equal to the `si.implicit` Radau IIA and Gauss-Legendre steps to rounding; orders 5 and 4, and spectral convergence in the node count |

```bash
uv run --with jupyterlab jupyter lab examples/integrators
```

Matplotlib is in the dev group; Jupyter is not, and `--with jupyterlab` adds it without changing the
lock file. The plots use `../notebooks/plotstyle.py`. Each notebook ends by writing the C of its main
Function to `examples/generated/integrators/<name>/` (git-ignored).

Two habits keep them quick. A sweep over step sizes uses one map with `dt=None`, the step as an
input, rather than one map per step size, since every new Function compiles on its first call. And
where several maps are compared at once (`stiff_implicit.ipynb` runs fifteen implicit methods side by
side), one `sc.function` calls them all and stacks the results, so the sweep compiles once.

## Where each piece of the package is used

| Module | Names | Notebooks |
| --- | --- | --- |
| `explicit.ipynb` | `explicit`, `rk4` | `explicit_methods.ipynb`, `discrete_maps.ipynb`, `symplectic_orbits.ipynb` |
| | `adaptive` | `adaptive_plant.ipynb` |
| | `symplectic` | `symplectic_orbits.ipynb` |
| `tableau.ipynb` | `Tableau`, `TABLEAUS`, `tableau`, `order_conditions` | `explicit_methods.ipynb`, `polynomials.ipynb`, `stiff_implicit.ipynb` |
| | `gauss_legendre`, `radau_iia`, `lobatto_iiia`, `lobatto_iiic` | `polynomials.ipynb`, `stiff_implicit.ipynb` |
| `implicit.ipynb` | `implicit` | `stiff_implicit.ipynb`, `transcriptions.ipynb` |
| `linear.ipynb` | `zoh`, `foh`, `linearize` | `linearize_and_discretize.ipynb`, `discrete_maps.ipynb` |
| `model.ipynb` | `UNROLL_STEPS`, the map contract | `discrete_maps.ipynb` |
| `polynomial.ipynb` | nodes, `lagrange_integrals`, `differentiation_matrix`, `interpolation_matrix` | `polynomials.ipynb` |
| `transcription.ipynb` | `MultipleShooting`, `Collocation`, `Pseudospectral`, `Interval` | `transcriptions.ipynb` |
| `scaly.linalg` | `lu`, `lu_solve`, `solve(assume="gen")` | `frequency_response.ipynb` |

The MPC examples in `../mpc/` use these maps and transcriptions inside optimal control problems.
