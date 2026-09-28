# Case study E8: 6-DoF powered descent by successive convexification (OpenSCvx)

OpenSCvx's 6-DoF rocket landing (`examples/rocket/6DoF_pdg.py`), solved by its penalized trust region
(PTR) method in JAX with CVXPY and QOCO, against the same PTR written in Scaly as one generated C
function: the discretization, the convex subproblem solved by Scaly's generated PIQP, and the loop.

```bash
examples/case_studies/scvx/baseline/setup.sh     # OpenSCvx pinned, and a venv with its pinned stack
uv run examples/case_studies/scvx/compare.py --out examples/case_studies/scvx/results/compare.json
uv run --with jupyterlab jupyter lab examples/case_studies/scvx/scvx.ipynb
```

`scvx.ipynb` explains the problem and the method, checks Scaly's discretization against OpenSCvx's
matrices, builds and runs the whole PTR without OpenSCvx, and plots the trajectory and the recorded
comparison. Its `RUN_BASELINES` flag reruns `compare.py`.

## The problem and the two sides

OpenSCvx (OpenSCvx/OpenSCvx at `9f23a62`, 2026-09-21; openscvx 0.5.3.dev57, JAX 0.11.2, CVXPY 1.9.3,
QOCO 0.3.2) states the landing as its example does:

- 14 states (mass, position, velocity, attitude quaternion, angular rate), body-frame thrust, free final
  time; five nodes;
- the time `t` and a penalty state `y` appended, `y`'s rate the squared violation of every
  continuous-time constraint (state boxes, glide slope, tilt, gimbal, thrust bounds, ...), everything
  in normalized time with a time dilation `s` as a control;
- each PTR iteration: discretize about the reference (two fixed Tsit5 steps per interval, the thrust a
  first-order hold, `A_d`, `B_d`, `C_d` by forward-mode JVPs through diffrax); solve the convex
  subproblem (the linearized dynamics with virtual control, the boundary conditions, the boxes, a
  quadratic trust region, an l1 penalty on the virtual control, maximal final mass); take the solution
  as the next reference; stop when `J_tr < 5e-3` and `J_vc < 1e-6`.

Every nonconvex and conic constraint sits in the penalty state, so the subproblem is a QP. OpenSCvx
hands it to CVXPY, which canonicalizes it for QOCO (`abstol` 1e-6, `reltol` 1e-9).

**Scaly** (`model.py`, `scaly_impl.py`): the rates exactly as the example writes them; the two Tsit5
steps written out with diffrax's coefficients, the tangent `d x / d(x_k, u_k, u_{k+1})` carried through
every stage by the variational equation (for an explicit Runge-Kutta step, the step's forward-mode
derivative, stage for stage), one call to a Function returning the rates and their Jacobians per stage,
the intervals in a `vmap`; the subproblem as an `sc.opt.problem`, the l1 term split into nonnegative parts,
solved by Scaly's generated PIQP (`examples/qp_solvers/generated_piqp.py`, `eps_abs` 1e-9) as a
Function called inside the loop body; the PTR a `while_loop`. `ptr(X0, U0)` returns the trajectory, the
iteration count and the `J_tr` and `J_vc` traces.

## Results

Apple M3 Max, macOS 26, Python 3.12 on the OpenSCvx side, Apple clang 21 at
`-O2 -mcpu=native -fno-math-errno`. Not the reference machine of
[fairness](../../../docs/results/fairness.md); none of this belongs on the results pages. Five rounds of
fresh processes behind the quiet-machine gate (one-minute load 2.6 to 3.6 at each start), the fastest
round per row:

| Stack | Start | Imports, s | Build, s | Compile and first solve, s | End to end, s | Solve, ms | PTR iterations | Final mass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenSCvx | cold | 1.08 | 0.37 | 3.12 | 4.61 | 43.8 | 5 | 1.4171783 |
| OpenSCvx | warm | 1.07 | 0.37 | 3.16 | 4.64 | 45.3 | 5 | 1.4171783 |
| Scaly | cold | 0.12 | 0.81 | 4.13 | 5.06 | **2.13** | 5 | 1.4171783 |
| Scaly | warm | 0.13 | 0.32 | 1.38 | **1.83** | **2.13** | 5 | 1.4171783 |

- *Build* is OpenSCvx's example module (the `Problem`) or Scaly's `ptr_function`, which builds the
  generated QP solver too. *Compile and first solve* is OpenSCvx's `initialize()` (tracing, XLA
  compilation, CVXPY canonicalization and one warm-up iteration) or Scaly's first call (C generation,
  compile or cache load, and the first solve). *Solve* is OpenSCvx's `solve()`, and Scaly's whole PTR
  from C (`fatrop_chain/time_kernel.c`, the best 1 ms batch). OpenSCvx's `post_process()`, 0.83 s, is
  not counted.
- *Cold* starts with empty caches; *warm* reuses the cold run's: OpenSCvx's compilation cache
  (`OPENSCVX_CACHE_DIR`), Scaly's JIT cache.

Per PTR iteration (`compare.py`'s `parts`; OpenSCvx's split from its instrumented run in
`results/reference_6dof.json`, iterations 2 to 5):

| Milliseconds | OpenSCvx | Scaly |
|---|---:|---:|
| whole iteration | 5.11 | **0.425** |
| QP solve | 0.558 (QOCO, inside CVXPY's 2.19) | **0.346** |
| discretization and the rest | 2.92 (two discretizations) | **0.046** (one) |

- **The same iterates.** Both take 5 PTR iterations to final mass 1.4171783 and `t_f` = 8.485601 s, with
  the same `J_tr` at every iteration (to 7 to 9 digits, 5 at the last, where it is 2e-4). The trajectories agree to 1.2e-7 (states) and
  4.5e-7 (controls), the tolerance of OpenSCvx's QOCO settings; the last `J_vc` is 8e-11 there and 4e-15
  in Scaly's tighter PIQP. The discretization alone matches OpenSCvx's `A_d`, `B_d`, `C_d` and `x_prop`
  to 5e-13 at every iteration.
- **The solve: 21x.** 2.13 ms against 43.8 ms. The QP solvers are within 1.6x of each other (PIQP at a
  tighter tolerance than QOCO); the difference is everything around them. OpenSCvx discretizes twice
  per iteration, the iterate and the candidate (`openscvx/algorithms/scvx/iteration.py`: "roughly 2x
  discretization work"), in JAX dispatched from Python, and passes through CVXPY's parameter update and
  QOCO's interface each time; Scaly's iteration is one discretization (46 us) and one QP solve (346 us)
  in the same C function. OpenSCvx's first iteration after `initialize()` is slower (27 ms of the instrumented run's 48 ms solve).
- **End to end, cold: level.** 5.06 s against 4.61 s. Scaly's first call is C generation (1.4 s) and the
  C compiler on 702 kB of C (about 2.7 s); OpenSCvx's `initialize()` is XLA and CVXPY.
- **End to end, warm: 2.5x.** OpenSCvx's compilation cache gives it nothing (4.64 s against 4.61 s cold).
  Scaly's warm start loads the compiled library, but 1.4 s of its 1.83 s is still C generation: the JIT
  cache is keyed on the rendered C, so a cache hit renders it anyway (CS-16 in `internal/todo.md`).
  Keyed on the Function instead, the warm start would be about 0.5 s.
- **Setup took work.** The first version inlined the rates into every Runge-Kutta stage and took
  `sc.jacobian` through the steps: about 44 s from build to first solution. The rates as a call node
  (derivatives of a call call its derivative) brought it to about 11.5 s, and carrying the tangent by the
  variational equation (one `rates_jac` call per stage) to the table's 4.9 s. `interval_ad` keeps the
  AD-through-the-steps form, to check the other.

**Against the plan.** The plan's Phase A (the PTR with IPOPT, each cone a smooth constraint) is not
needed for this example: its subproblem has no cones. So the study is Phase C directly, the whole PTR
as one C function with a generated QP solver. "Phase A at least 10x faster than OpenSCvx end to end,
setup included" is not met: 0.9x cold, 2.5x warm (the solve alone is 21x). "Phase C inside Reynolds et
al.'s ~100 ms class for the same node count on one core" is met at 2.1 ms, but on a smaller problem than
theirs: five nodes, and a QP rather than their SOCP. The comparison the plan meant, with second-order
cones in the subproblem, needs cones in the generated IPM first (CS-17).

## Files

| File | What it is |
| --- | --- |
| `baseline/setup.sh` | OpenSCvx pinned, and a venv with the stack it ran |
| `baseline/run_openscvx.py` | the example, run once in a fresh process with its phases timed |
| `model.py` | the example's rates and constraints, and the Tsit5 tableau |
| `scaly_impl.py` | the discretization, the subproblem, and the whole PTR as one Function |
| `run_scaly.py`, `compare.py` | Scaly's side in a fresh process, timed from C; both sides, cold and warm |
| `results/reference_6dof.json` | OpenSCvx's run: every iteration's reference, matrices, subproblem solution and timings |
| `results/compare.json` | the five rounds |
| `scvx.ipynb` | the study as a notebook |

`tests/integration/test_case_study_scvx.py` checks, on a pendulum with RK4, the variational tangent
through hand-written stages against `sc.jacobian` through the same steps and against central
differences, and a generated QP solver called inside a `while_loop` against the same solver called from
a Python loop.
