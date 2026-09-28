# Examples

Each script runs on its own (`uv run examples/<name>.py`), prints what it found, and writes the C it
generated to `examples/generated/<name>/` (git-ignored), so you can read the code behind every
result. `tests/integration/test_examples.py`, `tests/integration/test_examples_gallery.py` and
`tests/linalg/test_examples.py` check each one against a NumPy or SciPy reference, central
differences or a published value. The examples marked *solver* need the vendored PIQP or IPOPT
libraries. The folders `integrators/` and `mpc/` hold the examples of `scaly.integrators` and
`scaly.mpc` as notebooks, with their C in `examples/generated/integrators/` and
`examples/generated/mpc/`.

The tables group the examples by the Scaly feature they are mainly about; most use several. Size is
a rough guide: **S** is a first read, **L** a complete application.

## Functions, trees and code generation

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `simple.py`, `multiple_shooting.py` | S | Small introductions to `Function`s, derivatives and a solver | |
| `deploy_in_c.py` | S | An attitude propagator with its Jacobians for an EKF, shipped to C | `Function.factory` bundling a value and two Jacobians; `write_module` in C and C++; a C `main` compiled with `cc` against the typed-buffer header (`_call`) and checked against the JIT; the CasADi layer (`casadi=True`) loaded by `casadi.external`, with compact sparse Jacobians (`SpJac`) |

## Derivatives

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `derivatives_tour.py` | S | A Lennard-Jones cluster of seven atoms: forces, normal modes, rigidity, the global minimum | `gradient`, `hessian`, `forward` (Hessian-vector products), `jacobian` and `sparse_jacobian` (the rigidity matrix), `adjoint`, a `factory` of E, grad and Hess driving SciPy's `trust-exact`; static `gather` tables |
| `lagrangian_mechanics.py` | M | An n-link pendulum's equations of motion from its Lagrangian alone | derivatives with respect to expressions (`M(q)` as a Hessian in `qd`, the Coriolis terms as a Jacobian of a gradient), forward dynamics by dense `cholesky`, an RK4 `scan` recording the energy, the finite-time Lyapunov exponent from `jacobian` through the scan |
| `option_greeks.py` | S/M | Black-Scholes prices, Greeks and implied volatilities for a book of options | `factory` with `Grad` and `Hess` per option, `vmap` over the book, a safeguarded Newton `while_loop` with `sc.gradient` inside, vmapped; one reverse sweep for all the book's sensitivities; `erf` |
| `hanging_chain.py` | M | Calibrating a hanging chain's stiffness and mass from two photographs | Newton in a `while_loop` with AD inside the body, a step safeguarded by `isfinite`, `sc.custom_derivative` with the implicit-function rule |
| `gaussian_process.py` | M | Gaussian-process regression: hyperparameters by maximum marginal likelihood | gradient and Hessian *through* a dense `cholesky` and `solve_triangular` (loops at order 120), checked against the trace formula; the Laplace posterior; a generated predictive mean and variance |

## Sparsity

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `bratu_newton.py` | M | The Bratu PDE (solid-fuel ignition) by Newton with continuation in lambda | `jacobian_sparsity` and `column_coloring` finding the five-point stencil (colors constant in the grid size), `sparse_jacobian` into `SparseMatrix.from_sparse_jacobian`, `SparseLDL` and `inertia()` inside a `while_loop` |
| `sqp_newton_sparse.py` | M | Pendulum swing-up: SQP Newton steps on a sparse KKT system | `sparse_hessian`, `sparse_jacobian`, `SparseLDL` with `inertia()` and `health()` |
| `kalman_update.py` | M | A Kalman update of a spatial field with a sparse information prior | a quasi-definite `SparseLDL` solve and its Jacobians (the Kalman gain) |
| `heat_control.py` | M | Optimal heating of a plate made of two materials | `SparseMatrix` assembly, `SparseLDL` implicit time stepping, `sc.S` passing the sparse system matrix between `Function`s, adjoint gradients from the solves' implicit rules |

## Loops: `scan`, `while_loop`, `vmap`

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `robot_arm_ik.py` | S/M | Forward and inverse kinematics of a 7-joint arm, for 500 targets at once | forward kinematics as a `scan` over a Denavit-Hartenberg table, `jacobian` through it, damped least squares in a `while_loop` with `params`, the IK solver `vmap`ped over a batch |
| `periodic_orbit.py` | M | The Van der Pol limit cycle by single shooting, and its Floquet multipliers | an RK4 `scan` with a broadcast (stride-0) step size, `jacobian` through it (the monodromy matrix), Newton on `(amplitude, period)` in a `while_loop`, Liouville's formula as a check |
| `ekf_identification.py` | M/L | Maximum-likelihood identification of a forced Duffing oscillator | a whole extended Kalman filter as one `scan` (step Jacobians from `jacobian`, Joseph-form update), reverse mode through all 600 steps, L-BFGS |
| `lqr_tuning.py` | M | Tuning LQR weights for a quadrotor in a gust, on a heavier vehicle than modelled | two `scan`s (the Riccati recursion backwards, the closed loop forwards with a negative stride), the step number (`index=True`), reverse mode through both scans and the small solves in them |
| `ilqr.py` | L | Parking a car-like robot past two obstacles by iterative LQR | a complete trajectory optimizer as one C function: rollout `scan`, backward-pass `scan` with negative strides and AD-built local models, a line search `while_loop` (`index=True`) inside an outer `while_loop` adapting the regularization, NaN from a failed `cholesky` as the rejection signal |
| `tinympc/` | L | TinyMPC's ADMM for linear MPC with box and cone constraints, on the three problem families of the TinyMPC microcontroller benchmarks, and a benchmark against the TinyMPC library | `while_loop` around two `scan`s (backward pass with negative strides, rollout), cone projections with `where`, warm-start state in and out; see `tinympc/README.md` |

## Indexing and graph-structured data

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `mdp_value_iteration.py` | S/M | Planning for a robot on a slippery floor: value iteration for a stochastic shortest path | `segment_sum`, `segment_min`, `where` and comparisons, `while_loop` stopped on `norm_inf`, `cast` to an `int64` policy |
| `sinkhorn_transport.py` | M | Entropic optimal transport between two histograms, and its gradient | log-domain Sinkhorn with `segment_max`/`segment_sum` log-sum-exps over static segment ids, reverse mode through the `while_loop` matching the envelope theorem |
| `truss_sizing.py` | M | Minimum-compliance truss design by optimality criteria | `take` and `put_add` with run-time `int64` indices (the bar table is an input), dense `cholesky`, gradients through the solve and the scatter |
| `optimal_power_flow.py` | L | AC optimal power flow on the WSCC 9-bus system (MATPOWER `case9`), *solver* | the network as index tables: line flows from `gather`, bus balances by `segment_sum` and `scatter`, line ratings as a `bounded` group; IPOPT reaches the published 5296.69 $/h |

## Iterative numerical methods written in Scaly

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `lasso_admm.py` | S/M | Compressed sensing: a sparse signal from few measurements, the LASSO by ADMM | `while_loop`, soft thresholding with `copysign` and `maximum`, `logical_or` of residual tests, dense `cholesky` and `cho_solve`, a `bool` output |

`bratu_newton.py`, `ilqr.py`, `periodic_orbit.py`, `robot_arm_ik.py`, `sinkhorn_transport.py` and
`tinympc/` above are also solvers written in Scaly and generated as C.

## Solvers: PIQP, IPOPT and scaly-sqp

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `tiny_qp.py` | S | A two-variable parametric QP with a closed-form solution, *solver* | the smallest `sc.problem` for PIQP: one variable leaf, a scalar parameter, box bounds and one equality; the whole generated C (oracle and PIQP wrapper) short enough to read |
| `cbf_safety_filter.py` | M | A control-barrier-function filter keeping a unicycle clear of three obstacles, *solver* | a QP whose constraint data are Lie derivatives from `jacobian`, proven quadratic for PIQP; the solver *nested* in a controller `Function`, so the nominal law, the oracles and PIQP compile to one library |
| `portfolio_qp.py` | S/M | A Markowitz efficient frontier with a factor risk model, *solver* | a sparse QP (`options={"sparse": True}`) with matrix parameters, a parameter sweep over one generated solver, `qp_problem` for the dense formulation |
| `nmpc_cartpole.py` | L | Nonlinear MPC swinging up a cart-pole on a bounded track, in closed loop, *solver* | multiple shooting with `vmap`ped RK4 defects, a variable tree with per-leaf bounds and slacks, `bounded` groups, warm starts, `solver_stats()` |
| `mhe.py` | M/L | Moving-horizon estimation of a pendulum's state and friction, *solver* | `sc.solver(problem, "sqp")` warm-started in a receding horizon, a stride-0 parameter in the `vmap`ped defects; the same `Problem` handed to IPOPT gives the same estimate |
| `qp_solvers/` | L | Four QP families (linear MPC, a factor-model portfolio, an SVM, a dense random QP) written once as `sc.problem`s, *solver* | the generated PIQP (`scaly.solvers.ipm`) reached from a problem, next to the PIQP library and IPOPT; `compare.ipynb` compares answers, iterations, code size, build, generation and compile time and solve time from C; see `qp_solvers/README.md` |
| `casadi/` | L | CasADi's own Python examples (17 of them: NLPs, QPs, shooting, collocation, pseudospectral, MHE, system identification, code generation), each written in CasADi and in Scaly, *solver* | `compare.py` checks the two give the same answers and compares code lines, setup time and run time; see `casadi/README.md` |

## Integrators and model predictive control

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `integrators/` | L | Nine notebooks on `scaly.integrators`: explicit and implicit Runge-Kutta maps, stiff problems, adaptive and symplectic stepping, exact discretization, collocation building blocks and transcriptions, and dense LU on a tuned mass damper | every public name of the package, with the maths, plots, and a last cell asserting agreement with `expm`, `solve_ivp`, published coefficients or central differences; see `integrators/README.md` |
| `mpc/` | L | Five notebooks on `scaly.mpc`: a cart-pole swing-up under five transcriptions and in closed loop, reference tracking, linear MPC and terminal sets, *solver* | `OCP`, `MPC`, the control law exported to C, `simulate`, soft path constraints, parameters and references, `linear`, `lqr`, `Polytope`, invariant sets and ellipsoids, the condensed form, with plots of each and a last cell of assertions; see `mpc/README.md` |

## Notebooks

`examples/notebooks/` has seven of the examples as documented, executed notebooks, two notebooks
of their own on sparse matrices and seven on problems outside control: the derivation step by step with the maths, the Scaly code in small
cells, plots of the results and of the algorithms at work, and the generated C at the end. They are saved with their outputs, so they read
on GitHub without running. They write their generated C to the same `examples/generated/<name>/`
folders as the scripts, and `tests/integration/test_notebooks.py` runs each one top to bottom.
The fourteen notebooks in `integrators/` and `mpc/` are listed in the section above.

| Notebook | What it adds to the script |
| --- | --- |
| `lagrangian_mechanics.ipynb` | the mass matrix against the textbook, the triple pendulum's path and energy error, two runs 1e-9 apart diverging at the rate the Jacobian through the scan predicts |
| `bratu_newton.ipynb` | the Jacobian pattern and its colouring on the grid, colours constant from 100 to 25 600 unknowns, quadratic Newton convergence, the solution branch up to the turning point |
| `ekf_identification.ipynb` | the record, the gradient against differences, the L-BFGS path, a 1600-point likelihood landscape, innovation whiteness and the filter's error band |
| `gaussian_process.ipynb` | the gradient through the Cholesky against the trace formula, the likelihood landscape with the Laplace ellipses, the posterior band |
| `ilqr.ipynb` | the three loops explained one by one, the iterates in the plane, cost and regularization per iteration (the loop body driven from Python), a first-order optimality check |
| `cbf_safety_filter.ipynb` | the filtered and unfiltered paths, the barriers over time, the QP's feasible set in the input plane at an active step; needs PIQP |
| `nmpc_cartpole.ipynb` | states, force and bounds in closed loop, a filmstrip of the swing-up, IPOPT iterations and times per sample, the C size against the horizon; needs IPOPT |

The two sparse-matrix notebooks have no script version:

| Notebook | Problem | What it shows |
| --- | --- | --- |
| `sparse_fem_topology.ipynb` | A conduction tree by topology optimization: P1 finite elements on a plate, a heat sink on one edge, SIMP with a density filter, optimality criteria | the `SparseMatrix` algebra on two triangles (`from_coo` summing duplicates, `+` as pattern union, `*` as intersection, sparse–sparse `@`, `tril`, `with_pattern`, `position`); assembly of 16 200 triplets and the restriction `R @ K @ R.T`; `sc.S` between two `Function`s and a call refused for the wrong pattern; fill-in and elimination trees of natural, RCM and minimum-degree orderings from `linalg.analyze`; the adjoint gradient through the solve, the assembly and a constant sparse filter, against the hand-derived formula |
| `sparse_kkt_mpc.ipynb` | The KKT system of unconstrained linear MPC for a chain of oscillating masses, checked against the Riccati recursion | `SparseMatrix.block` over stages with `from_dense`, `zeros` and `add_diagonal`; `K + K.tril(-1).T` for the symmetric product; a hand-made stage ordering through `analyze(perm=...)` and `SparseLDL(symbolic=...)` against automatic, dual-first and random ones; regularization with refinement against the unregularized matrix; `inertia()` as a convexity certificate and `health(signs=...)`; the LQR gain and `dV/dA` by `sc.jacobian` and `sc.gradient` through the solve; Ruiz equilibration with `segment_max`, `scale_rows` and `scale_cols` |

Seven notebooks go beyond control, to problem classes with the same traits: a structure fixed when
the graph is built, data that change between calls, and a need for derivatives or an embedded solve.
They have no script version either:

| Notebook | Problem | What it shows |
| --- | --- | --- |
| `pose_graph_slam.ipynb` | 2-D pose-graph SLAM: four laps of odometry and loop closures, by Levenberg–Marquardt | one factor `vmap`ped over the edges after a static `gather`; the pattern of `J.T @ J` from the graph and the fill of its factor under three orderings; the whole LM loop (`while_loop`, accept/reject by `where`) in C, against SciPy's MINPACK; one solver re-used for a Monte Carlo checked against the Laplace covariance |
| `kinetics_estimation.ipynb` | Arrhenius parameters of A → B → C from four batch runs, by multiple shooting, *IPOPT* | nested `vmap` (intervals inside runs) with a stride-0 parameter; the arrowhead Lagrangian Hessian and its colouring; data as the problem's parameter; single shooting (SciPy) for comparison; the Gauss–Newton covariance from `jacobian` through a `scan`, against a Monte Carlo |
| `circuit_transient.ipynb` | A Cockcroft–Walton voltage multiplier by nodal analysis, trapezoidal steps and Newton | the netlist as `gather`/`segment_sum` tables; SPICE-style junction limiting with `where`; a Newton `while_loop` inside the time `scan`, with `SparseLDL` on the netlist-shaped Jacobian; against a hand-stamped SciPy implementation; the source waveform as an input |
| `opf_day_ahead.ipynb` | AC and DC optimal power flow on `case9` over a 24-hour load profile, *IPOPT*, *PIQP* | loads as problem parameters, warm-started re-solves; the Jacobian's pattern equal to the bus adjacency; the DC-OPF as a sparse QP from the same tables; losses and marginal prices, AC against DC |
| `conductivity_inversion.ipynb` | Imaging conductivity from boundary potentials: a PDE-constrained inverse problem | assembly with `from_coo` on a fixed pattern; one `SparseLDL` factor shared by thirteen sources; the adjoint gradient and Hessian–vector products (`jvp` of the gradient) against differences; L-BFGS against Newton–Krylov (`trust-krylov`) |
| `spike_deconvolution.ipynb` | Streaming spike inference from a calcium trace: a nonnegative deconvolution QP per window, *PIQP* | a parameter inside the constraint matrix on a fixed pattern; one generated solver called per window; the bidiagonal formulation against the dense one, solve time against window length; L-BFGS-B as reference |
| `surrogate_optimization.ipynb` | A reactor's operating point optimized on a neural-network surrogate, *IPOPT* | the training loss as a `vmap` over samples with `gradient` for L-BFGS; the trained network as a constant inside an NLP with exact Hessians; the surrogate optimum against the true one; `hessian` against differences; batched C inference |

They use Matplotlib (in the dev group) and a Jupyter kernel, which the dev group does not include:
`uv run --with jupyterlab jupyter lab examples/notebooks` works without changing the lock file.
`plotstyle.py` holds the shared colours and Matplotlib settings.

## Interpolation and lookup tables

| Example | Size | Problem | What it shows |
| --- | --- | --- | --- |
| `interp/` | L | Six notebooks on `scaly.interp`: interpolation kinds, n-D lookup tables, tables learned from data, shape-constrained fits, contouring control around a race track, and spline trajectories, *solver* | every kind against SciPy; search strategies, table layouts and batches timed; a table passed to C at run time; calibration and identification with `Expr` data and coefficients, and the Jacobian's pattern at known against symbolic points; monotone, convex and bounded fits by PIQP; a closed-loop MPCC lap on one periodic spline; a last cell of assertions in each; see `interp/README.md` |
| `interp/pairs/` | L | Five problems written in CasADi and in Scaly: batched lookups, table calibration, Hammerstein identification, contouring MPC, heat-pump MPC, *solver* | `uv run examples/casadi/compare.py --dir examples/interp/pairs` checks they agree and compares code lines, setup and run time |

## Ideas not written yet

- `nonsmooth_conventions.py` (S): how `sc.options(nonsmooth=...)` changes the generated derivative of
  `maximum`, `abs` and a hinge loss at ties.
- `sensor_localization.py` (M): range-only network localization from an edge list, by
  Levenberg-Marquardt on a sparse Jacobian.
- `goddard_rocket.py` (L): the free-final-time Goddard rocket, with the final time as a variable and
  a scaled time grid.
- `minimal_surface.py` (L): a COPS-style minimal surface on a grid, a large sparse NLP for IPOPT.
