# CasADi's examples in Scaly

The Python examples that ship with [CasADi](https://github.com/casadi/casadi/tree/main/docs/examples/python),
each written twice: `<name>_casadi.py` follows the original, `<name>_scaly.py` solves the same
problem with Scaly. `compare.py` runs both, checks that they return the same numbers and measures
code size, setup time and run time.

Every CasADi example in the repository (commit `e1bac07`, 2026-09-26) was run with the CasADi 3.8.0
wheel. The 17 below run there and have a Scaly counterpart; the others are listed at the end with
the reason they are not here.

```bash
uv run examples/casadi/rocket_casadi.py        # either half on its own prints its results
uv run examples/casadi/rocket_scaly.py
uv run examples/casadi/compare.py              # every pair: agreement, code lines, setup, run
uv run examples/casadi/compare.py rocket race_car --processes 5
```

## The pairs

| Pair | Problem | CasADi | Scaly |
| --- | --- | --- | --- |
| `rosenbrock` | Rosenbrock's problem as an equality-constrained NLP | `nlpsol` IPOPT | `sc.opt.problem`, IPOPT |
| `simple_nlp` | closest point to the origin on a half-plane | `nlpsol` IPOPT | `sc.opt.bounded` inequality, IPOPT |
| `hs015` | Hock-Schittkowski 15 | `nlpsol` Uno (IPOPT preset) | IPOPT; Scaly has no Uno backend |
| `nlp_codegen` | NLP whose oracles are generated as C, compiled and loaded | `generate_dependencies`, gcc, `nlpsol` from the `.so` | `sc.opt.solver` (which always does this), `write_module` |
| `c_code_generation` | gradient of det of a 7x7 matrix, compiled at `-O0`, `-O3`, `-Os` | SX `det`, `generate`, gcc, `external` | expansion by minors with each minor computed once, `SCALY_CC_OPT` |
| `parallel_map` | sin applied 100 000 times, at 300 points | SX chain, `map` | `scan` for the chain, `vmap` for the points |
| `simple_lp` | two-variable LP | `conic` qpOASES | `sc.opt.QP`, PIQP |
| `chain_qp` | hanging chain on a sloped floor, 80 variables | `qpsol` qpOASES, sparse | QP proof, PIQP with `sparse=True` |
| `rocket` | minimum-effort rocket, 50 intervals of 20 Euler steps | MX, `expand` | `scan` over the intervals |
| `race_car` | minimum-time race with a speed limit | `Opti`, RK4 | RK4 defect under `vmap`, the final time broadcast |
| `direct_single_shooting` | Van der Pol OCP, 20 intervals of 4 RK4 steps | MX, loop of calls | `scan` stacking end states and costs |
| `direct_multiple_shooting` | the same, multiple shooting | MX | one `vmap`, stride 3 |
| `direct_collocation` | the same, Legendre collocation of degree 3 | MX | one `vmap`, stride 9 |
| `biegler_10_1` | Biegler's example 10.1, Radau collocation on 1 to 10 elements | 10 `nlpsol`s | 10 solvers |
| `pseudospectral_collocation` | LGL pseudospectral OCP with an analytical solution, degrees 5 to 45 | `Opti`, `map` | `vmap`, dense differentiation matrix, costates from `lam_eq` |
| `mhe_spring_damper` | moving-horizon estimation over 990 windows, EKF arrival cost | `struct_symSX`, `factory` Jacobians | `vmap` of the RK4 step, `sc.jacobian`, parameters for the data |
| `sysid` | four-parameter fit to 2000 samples, single then multiple shooting | `mapaccum`, `map`, Gauss-Newton Hessian, JIT | nested `scan`s, `vmap`, exact Hessian |

`_common.py` holds the result printer and the IPOPT option helpers both halves use, and `_lgl.py`
the NumPy Legendre-Gauss-Lobatto routines both pseudospectral halves import. Plots are left out.

## Results

<!-- compare.py output starts -->
Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 logical CPUs, Linux 6.18.44-fc-v42; Python 3.14.3, CasADi 3.8.0, Scaly from this checkout, NumPy 2.4.6; cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

| Example | Agree (max rel. diff) | Iterations CasADi / Scaly | Code lines CasADi / Scaly | Setup CasADi · JIT · Scaly | Run CasADi · JIT · Scaly |
| --- | --- | --- | ---: | ---: | ---: |
| `biegler_10_1` | yes (1e-15, z_N9) | n/a | 49 / 45 | 187.6 ms · 3.00 s · 3.51 s | 14.4 ms · 13.4 ms · 17.1 ms |
| `c_code_generation` | yes (4e-15, gd) | n/a | 35 / 49 | 14.06 s · n/a · 1.38 s | 42 µs · n/a · 9 µs |
| `chain_qp` | yes (8e-12, z) | n/a | 40 / 27 | 142.1 ms · n/a · 302.8 ms | 433 µs · n/a · 163 µs |
| `direct_collocation` | yes (7e-14, x1) | 16 / 16 | 80 / 61 | 198.7 ms · 2.87 s · 2.75 s | 12.1 ms · 7.1 ms · 5.9 ms |
| `direct_multiple_shooting` | yes (2e-16, u) | 9 / 9 | 56 / 48 | 156.0 ms · 13.71 s · 2.64 s | 45.7 ms · 3.1 ms · 2.5 ms |
| `direct_single_shooting` | yes (1e-13, x1) | 37 / 37 | 60 / 48 | 165.0 ms · 144.89 s · 3.60 s | 232.8 ms · 16.1 ms · 11.0 ms |
| `hs015` | yes (4e-10, f) | n/a | 17 / 18 | 139.9 ms · 335.8 ms · 179.9 ms | 1.2 ms · 994 µs · 2.8 ms |
| `mhe_spring_damper` | yes (9e-16, dx_est) | 991 / 991 | 100 / 89 | 223.4 ms · 612.2 ms · 786.5 ms | 1.85 s · 1.88 s · 1.38 s |
| `nlp_codegen` | yes (0e+00) | n/a | 21 / 19 | 331.2 ms · n/a · 174.6 ms | 605 µs · n/a · 991 µs |
| `parallel_map` | yes (0e+00) | n/a | 26 / 30 | 966.1 ms · failed · 447.0 ms | 367.9 ms · failed · 365.0 ms |
| `pseudospectral_collocation` | yes (9e-10, error_N40) | 62 / 62 | 44 / 44 | 259.9 ms · 36.32 s · 11.17 s | 92.5 ms · 53.0 ms · 49.6 ms |
| `race_car` | yes (6e-07, u) | 35 / 38 | 35 / 39 | 296.9 ms · 9.52 s · 3.63 s | 56.5 ms · 31.6 ms · 19.8 ms |
| `rocket` | yes (4e-14, lam_u) | 14 / 14 | 25 / 29 | 1.10 s · failed · 3.34 s | 44.0 ms · failed · 5.6 ms |
| `rosenbrock` | yes (9e-31, lam_g) | 10 / 10 | 12 / 16 | 135.4 ms · 335.3 ms · 191.0 ms | 1.8 ms · 2.0 ms · 2.2 ms |
| `simple_lp` | yes (6e-11, lam_a) | n/a | 15 / 16 | 133.5 ms · n/a · 142.8 ms | 41 µs · n/a · 75 µs |
| `simple_nlp` | yes (0e+00) | 5 / 5 | 12 / 15 | 129.2 ms · 298.8 ms · 171.8 ms | 1.1 ms · 1.2 ms · 1.5 ms |
| `sysid` | yes (7e-10, params_single) | 5 / 6; 7 / 10 | 67 / 74 | 6.31 s · n/a · 27.02 s | 453.4 ms · n/a · 591.8 ms |
<!-- compare.py output ends -->

Setup is modelling, derivative construction, solver creation and any C generation and compilation,
up to the first result; run is the median time of one more `run()` (the whole numerical part of the
example, Python calls included). CasADi · JIT is the CasADi file with `CASADI_JIT=1`: `expand` and
`jit` with the flags Scaly's JIT uses (`-O2 -ftree-vectorize -march=native -fno-math-errno`), for
the examples that evaluate their oracles in CasADi's virtual machine. Code lines exclude comments,
docstrings and blank lines. Each number is the median of three fresh processes on a small cloud
machine (the first line of the output); they are not comparable with those on
[the results pages](../../docs/results/index.md), which use the reference machine and protocol in
[fairness](../../docs/results/fairness.md). Rerun `compare.py` to get them for yours.

What the numbers say:

- **Same answers.** Every pair agrees to 1e-6 or better, most to 1e-10, and where the NLP is the same
  IPOPT takes the same number of iterations.
- **Code size is a wash.** The Scaly versions are between 0.7 and 1.4 times as long. They are shorter
  where CasADi builds variable and bound lists entry by entry (`chain_qp`, `direct_collocation`) and
  longer where Scaly needs a typed `sc.function` for what CasADi writes as an expression.
- **Scaly always compiles.** Its setup is 0.2 s for the small NLPs and 2.6 to 3.6 s for the
  optimal-control problems, most of it Python-side lowering rather than the C compiler (about 1 s of
  the 3.6 s in `direct_single_shooting`). CasADi's interpreted setup is 0.1 to 0.3 s. `sysid` is the
  outlier, see below.
- **Compiling CasADi's oracles costs more than compiling Scaly's.** With `expand` and `jit`, CasADi's
  setup on the optimal-control problems is 2.9 s to 145 s where Scaly's is 2.6 to 3.6 s: about equal
  for `direct_collocation`, 5 times longer for multiple shooting and 40 times for single shooting,
  whose expanded Hessian is one long expression. On two examples gcc does not finish: it runs out of
  memory on the 100 000-operation chain of `parallel_map` and crashes on the 1000 expanded Euler
  steps of `rocket`. Scaly keeps both as loops (`scan`).
- **Compiled oracles win the runs.** Against interpreted CasADi, Scaly is 2 to 21 times faster where
  function evaluation dominates (the shooting and collocation problems, `rocket`, `race_car`).
  Against CasADi with JIT the gap is 1.1 to 1.6 times.
- **Small problems measure IPOPT.** On the problems solved in a millisecond Scaly is 1.2 to 1.6 times
  slower. For `rosenbrock`, 1.7 of Scaly's 2.1 ms are inside IPOPT and 0.35 ms in the wrapper, which
  creates the IPOPT problem on every call; evaluating the oracles takes 2.5 µs.

## Differences that affect the comparison

- **Formulation.** `race_car` puts the control and final-time bounds into IPOPT's variable bounds;
  `Opti` makes them general constraints, so IPOPT takes 38 iterations instead of 35 and the
  solutions agree to 6e-7 (IPOPT's tolerance). `sysid` uses the exact Hessian where the original
  gives IPOPT the Gauss-Newton one, which Scaly's IPOPT backend cannot take; at the zero-residual
  optimum both reach the true parameters. `c_code_generation` differs in the determinant: CasADi's
  expansion gives 45 000 operations and a 1 MB C file, the Scaly version computes each of the 128
  minors once. That, not the tool, is why its compile is ten times faster.
- **Solvers.** `hs015` runs Uno in CasADi and IPOPT in Scaly. The two QPs run qpOASES (active set) in
  CasADi and PIQP (proximal interior point) in Scaly, so their run times compare solvers, not tools.
  Both IPOPTs are 3.14.19 with MUMPS, but CasADi's is the build in its wheel and Scaly's the vendored
  static one; [fairness](../../docs/results/fairness.md) shows how much that alone can move a time.
- **What is timed.** Run times include Python on both sides: argument conversion, the call, and for
  `mhe_spring_damper` the EKF update in NumPy between the 990 solves.

## A slow spot found on the way

`sysid`'s multiple-shooting problem has an arrowhead Hessian: the four parameters couple to all 4000
states. Of the 13.5 s `sc.opt.solver` takes to build that solver, 8.9 s are in `star_coloring`
(`src/scaly/ad/sparsity.py`) and 3.7 s in `_star_recovery_indices` (`src/scaly/ad/sparse.py`),
whose Python loops grow with the number of variables times the degree of the dense rows. That, and
not code generation or compilation, is why Scaly's setup is 27 s in that row.

## Not ported

| Example | Why |
| --- | --- |
| `vdp_collocation`, `vdp_collocation2`, `dae_collocation` | ask IPOPT for HSL MA27, which the CasADi wheel does not ship: `Library loading failure` |
| `dae_single_shooting`, `dae_multiple_shooting` | the same, and IDAS integrators |
| `simulation`, `multipoint_simulation`, `sensitivity_analysis`, `bouncing_ball`, `dae_reduced_index`, `lotka_volterra_minlp` (also Bonmin) | CVODES/IDAS adaptive integrators, with events for the bouncing ball; Scaly has fixed-step integration only, written as `scan` |
| `implicit_runge-kutta`, `vdp_indirect_single_shooting`, `vdp_indirect_multiple_shooting` | `rootfinder` on an unsymmetric system; Scaly's dense solves are Cholesky and LDL, and the last two also integrate with CVODES |
| `nlp_sensitivities` | parametric sensitivities through `nlpsol`; Scaly does not differentiate through a solver call |
| `callback` | Python callbacks inside the graph; Scaly generates C for every node |
| `accessing_mx_algorithm`, `accessing_sx_algorithm` | walk CasADi's own instruction list |
| `daebuilder`, `bouncing_ball_daebuilder`, `breaking_spring`, `fmu_collocation`, `fmu_export`, `modelica_fmu_import` | DaeBuilder, FMUs and Modelica; four of them also fail in the wheel |
| `ipopt_nl` | reads an AMPL `.nl` file; fails in the wheel |
| `lqr_control` | the LQR design and simulation are commented out upstream; what runs is an eigenvalue and rank check |
| `vdp_dynamic_programming` | NumPy only, no CasADi |

The CasADi examples are MIT No Attribution licensed; the `_casadi.py` files and `_lgl.py` are
adapted from them.
