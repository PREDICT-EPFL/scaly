# Integrators and MPC: implementation plan (2026-09-27, v1)

Status: **signed off 2026-09-27** (answers in §11), in progress on `claude/integrators-mpc`. Two new in-tree sub-packages, `scaly.integrators` and
`scaly.mpc`. The first turns a continuous-time model into discrete-time maps and transcription
constraints. The second builds, solves, analyses and deploys model predictive control laws. The
phase list is §7. The questions still open for sign-off are §11.

Decisions settled with Colin on 2026-09-27:

1. **Methods mean transcriptions.** Direct methods are shooting (explicit or implicit Runge-Kutta
   integrators) and local collocation. Pseudospectral means global orthogonal collocation (LG, LGR,
   LGL) with hp segments. Indirect methods (Pontryagin: costates and stationarity by AD, a two-point
   boundary value problem by indirect multiple shooting) are the final, separate phase.
2. **In-tree packages.** `src/scaly/integrators/` and `src/scaly/mpc/`, reached as
   `sc.integrators` and `sc.mpc` the way `sc.linalg` is. Offline design (DARE, invariant sets,
   LPs) uses SciPy. SciPy is already a runtime dependency (`pyproject.toml`), whatever `AGENTS.md`
   says (§11, question 3).
3. **Registry backends first.** Linear MPC goes through `piqp`, sparse or condensed. Nonlinear MPC
   goes through `ipopt` for a full solve or `sqp`, including real-time iteration. The generated IPM
   plugs in when Tier 4's `backend="scaly"` lands (piqp_plan.md, Tier 4 #30), not in this work.
4. **The standard set of terminal ingredients.** These are:
   - a terminal equality;
   - an LQR terminal cost;
   - ellipsoidal invariant sets for linear systems, and for nonlinear ones by the quasi-infinite
     horizon method;
   - the polytopic maximal positively invariant set for linear systems;
   - a certification check.

   Control-invariant and robust (tube) sets are deferred.

## 1. What exists today

`src/scaly` has no integrator, OCP or MPC module. The repository writes both by hand, repeatedly:

| Pattern | Copies | Where |
| --- | --- | --- |
| RK4 step (`k1 = f(x); ...`), substeps as `rk4(rk4(...))` or a Python loop | 20+ | `examples/{nmpc_cartpole,mhe,ekf_identification,periodic_orbit,lagrangian_mechanics}.py`, `examples/casadi/*_scaly.py`, all four benchmark problems, `tests/integration/test_{scan,stage_transcription}.py` |
| NumPy plant duplicating the symbolic model | 9 | chain (2), race_cars, unbumpercars (2), npmpc (2), cartpole, mhe |
| Local collocation (`np.poly1d` coefficient build) | 2 | `examples/casadi/direct_collocation_scaly.py` (Legendre), `biegler_10_1_scaly.py` (Radau) |
| Pseudospectral (LGL nodes, `D @ X`) | 1 | `examples/casadi/pseudospectral_collocation_scaly.py` + `_lgl.py` |
| Implicit steps | 2 | `examples/heat_control.py` (implicit Euler), `circuit_transient.ipynb` (trapezoid, Newton `while_loop`) |
| ZOH by `expm` | 3 | `examples/qp_solvers/problems.py`, `lqr_tuning.py`, `sparse_kkt_mpc.ipynb` |
| Riccati | 4 | TinyMPC cache, `sparse_kkt_mpc.ipynb`, `lqr_tuning.py` (differentiable scan), npmpc (`scipy.linalg.solve_discrete_are`) |
| Horizon stacking, `vmap` stage tuples, `x0` equality, bound tiling, warm-start shift | 5 variants | cartpole, chain, npmpc, race_cars (shifts multipliers too), mhe |
| Invariant sets, Lyapunov equations | 0 | — |

What the compiler already provides: `sc.vmap` (one loop per stage Function, C size constant in N),
VMAP-aware sparse Jacobians and Hessians, `sc.scan` and `sc.while_loop` (with `params=`),
`sc.custom_derivative`, `sc.jacobian` of an `Expr` with respect to a slice inside a body, dense
`cholesky`/`ldl`/`solve_triangular`, and `sc.solver` nested inside a larger Function
(`examples/cbf_safety_filter.py`). What it lacks is **a general dense solve**: `linalg.solve` is
symmetric only, with no LU and no pivoting. Newton on implicit Runge-Kutta stage equations needs one
(PR I2).

## 2. Packages and import layers

```
src/scaly/integrators/
  __init__.py       the public integrator surface
  tableau.py        Butcher tableaus: the named families, construction from collocation nodes, order conditions
  explicit.py       explicit Runge-Kutta steps, embedded pairs, adaptive simulation
  implicit.py       implicit Runge-Kutta steps: Newton on the stage equations, derivatives by the implicit function theorem
  linear.py         exact discretization of linear time-invariant systems, and linearization
  polynomial.py     Gauss, Radau, Lobatto and Chebyshev nodes; quadrature weights; differentiation and interpolation matrices
  transcription.py  MultipleShooting, Collocation, Pseudospectral: how one interval or segment becomes variables and constraints

src/scaly/mpc/
  __init__.py       the public MPC surface
  ocp.py            OCP: the declaration (dynamics, costs, constraints, horizon, transcription) and the sc.problem it builds
  controller.py     MPC: the solver, the control law Function with its warm start, closed-loop simulation
  polytope.py       Polytope in halfspace form and the LPs on it (SciPy HiGHS)
  terminal.py       terminal ingredients: LQR, Lyapunov, ellipsoids, maximal invariant sets, certification
  indirect.py       Pontryagin's boundary value problem and indirect shooting (phase X)
```

Both packages sit at **import layer 5**, the user-facing request layer. They need `function/api`
(5), `function/sugar` (4), `linalg` (5) and `solvers` (5). `mpc` imports `integrators` within the
same layer, and nothing imports `mpc`. Neither imports `codegen`: the user calls
`scaly.codegen.write_module(ctrl.law, ...)`, and calling a Function compiles it through the
existing seam. `scaly/__init__.py` gains `from . import integrators, mpc`, with no flat
re-exports. `polytope.py` imports `scipy.optimize` inside the functions that solve LPs, so that
`import scaly` does not pay for it. Each module gets an `IMPORT_LAYERS` entry, a one-line docstring
and a mirrored test file (`tests/integrators/`, `tests/mpc/`).

## 3. `scaly.integrators`

### 3.1 The model and the discrete map

A model is any `sc.Function` whose first input is the state and whose single output is the state
derivative: `f(x, u, p) -> xdot`, `f(x, u)` or `f(x)`. The discrete map returned by every
integrator has **the model's own signature**, with `xnext` as its output. Every input after `x` is
held constant over the step, which gives a zero-order hold on `u`. `dt` is either a float folded
into the code or `None`, which appends a `dt` input (free final time, runtime sampling). The name
defaults to `{f.name}_{method}`, and `name=` overrides it. Two differently configured integrators
of one model in one graph need distinct names, the rule the lowering already enforces.

```python
import scaly as sc
from scaly import integrators as si

@sc.function(4, 1, output="xdot")
def cartpole(x, u): ...

F = si.rk4(cartpole, dt=0.025)                                   # F(x, u) -> xnext
F = si.explicit(cartpole, "ssprk3", dt=0.05, steps=4)            # substeps; a scan above a threshold
F = si.explicit(cartpole, si.Tableau(A, b, c), dt=None)          # F(x, u, dt)
F = si.implicit(cartpole, "radau_iia", stages=3, dt=0.05, newton_iters=4)   # derivatives by the implicit function theorem
F = si.implicit(cartpole, "sdirk3", dt=0.05, tol=1e-12, max_iter=20)       # a while_loop Newton
S = si.adaptive(cartpole, "dopri5", rtol=1e-8, atol=1e-10, max_steps=10_000)  # S(x, u, t) -> x(t), for plants
Ad, Bd = si.zoh(A, B, dt); Ad, Bd, Bd1 = si.foh(A, B, dt)           # exact, NumPy/SciPy
A, B = si.linearize(F, x_eq, u_eq)                               # numeric Jacobians of any discrete map
```

### 3.2 The methods

| Family | Methods | Order | Stability | Notes |
| --- | --- | --- | --- | --- |
| Explicit RK | `euler`, `heun`, `midpoint`, `ralston`, `rk3`, `ssprk3`, `rk4`, `rk38`, any explicit `Tableau` | 1-4 | — | Zero tableau entries skipped; each stage is one `CALL` of `f` |
| Embedded pairs | `bs32` (Bogacki-Shampine), `dopri5` (Dormand-Prince 5(4)) | 3, 5 | — | Fixed step uses the high-order row; `si.adaptive` controls the error in a `while_loop`, with the step sequence frozen for derivatives |
| Collocation IRK | `gauss_legendre(s)`, `radau_iia(s)`, `lobatto_iiia(s)`, `lobatto_iiic(s)` | 2s, 2s-1, 2s-2, 2s-2 | A-stable; Radau IIA and Lobatto IIIC L-stable | Tableaus built from the nodes (§3.4), any s |
| Named implicit | `backward_euler`, `implicit_midpoint`, `trapezoidal` | 1, 2, 2 | L, A, A | Aliases of the families above where they coincide |
| DIRK | `sdirk2`, `sdirk3` (Alexander) | 2, 3 | L-stable | Stage by stage: s solves of order nx, not one of order s·nx |
| Symplectic | `symplectic_euler`, `stormer_verlet` | 1, 2 | — | For `q' = v, v' = a(q, u, p)`, a model with `split=nq` |
| Exact LTI | `zoh`, `foh` | exact | — | Numeric matrices; `sc.const` them to use in a graph |

### 3.3 Efficiency

- The model is one `Function` and every stage evaluation is a `CALL` of it, not an inlined copy.
  Forward mode reuses the cached derivative Function, and lowering scalarizes small callees.
- Substeps are unrolled up to a threshold and form a `sc.scan` above it. Loops stay loops (the
  decision taken for the IPM), so the C size is constant in the step count. PR I1 sets the
  threshold by measurement.
- Implicit steps run Newton on the stage residual `G(K; x, u, p) = 0`, starting from the explicit
  Euler predictor:
  - `newton_iters=k` runs a fixed number of iterations, the deterministic-time choice for control.
  - `tol=` runs a `while_loop`.
  - The whole step is a `sc.custom_derivative`: `dK = -G_K^{-1} (G_x dx + G_u du + G_p dp)`, one
    factorization of `G_K` per derivative evaluation. AD therefore never differentiates the Newton
    loop, the pattern `SparseLDL.solve` set.
  - `G_K = I - dt (A ⊗ J)` is nonsymmetric, which is why PR I2 comes first.
  - DIRK methods factor `I - dt a_ii J`, of order nx, once per stage.
- Simplified Newton (the Jacobian at the predictor, factored once per step) is an option, and the
  default for `newton_iters`. Full Newton refactors every iteration.

### 3.4 Collocation and pseudospectral building blocks (`polynomial.py`, `transcription.py`)

`polynomial.py` is exact, NumPy-only construction:

- Nodes: Legendre-Gauss (LG), Legendre-Gauss-Radau (LGR, right or left), Legendre-Gauss-Lobatto
  (LGL) and Chebyshev-Gauss-Lobatto, all by Golub-Welsch or Newton on Legendre polynomials.
- The rest: barycentric weights, quadrature weights, the differentiation matrix, interpolation
  matrices and the collocation tableau `(A, b, c)` for any node set.

`transcription.py` is the contract `mpc` consumes. A transcription says how one interval (or
segment) of the horizon becomes decision variables and constraints:

```python
si.MultipleShooting(si.rk4, steps=2)         # no extra variables; defect F(x, u, p) - xnext
si.MultipleShooting(si.implicit, method="radau_iia", stages=2, newton_iters=3)
si.Collocation("radau", degree=3)            # d·nx collocation states per interval; d·nx + nx equations
si.Pseudospectral("lgr", nodes=20, segments=1)   # global; controls at the nodes; hp by segments > 1
```

Each transcription provides four things:

- the per-interval variable count;
- the defect `Function`;
- the quadrature of an integral cost (the RK weights, or Gauss weights on collocation states);
- the time grid and the node times at which path constraints are imposed.

Two equivalences serve as the differential tests. Local collocation at Radau points gives the Radau
IIA step exactly, and at Gauss points the Gauss-Legendre step.

## 4. `scaly.mpc`

### 4.1 Declaring and solving

```python
from scaly import mpc

ocp = mpc.OCP(
  ode=cartpole, dt=0.05, horizon=40,                  # or step=F for a discrete-time model
  transcription=si.MultipleShooting(si.rk4, steps=2),   # or si.Collocation(...), si.Pseudospectral(...)
  stage_cost=l, terminal_cost=Vf,                     # Functions of (x, u[, p]) and (x[, p]); mpc.quadratic(Q, R, ...) builds them
  x_bounds=(x_lo, x_hi), u_bounds=(-15.0, 15.0),
  constraints=[mpc.path(g, hi=1.0, soft=1e3)],        # soft: slacks with an exact l1 (or l2) penalty
  terminal=mpc.terminal.Ellipsoid(P, alpha),          # or TerminalEquality(), Polytope(H, h)
  params=sc.L("p", 3), references=sc.L("r", NX),      # shared parameters; per-stage references sliced by stage
)
ctrl = mpc.MPC(ocp, "ipopt", options={"tol": 1e-8})   # or "sqp", "piqp" (the QP is proved and extracted as today)
u = ctrl(x)                                           # Python: warm-started from the last call
sol = ctrl.solve(x)                                   # xs, us, t, costs, multipliers, SolverStatus
law = ctrl.law                                        # sc.Function law(x, guess[, p, r]) -> (u, guess_next)
scaly.codegen.write_module(law, "gen/")               # one C artifact; the shift runs in generated code
hist = mpc.simulate(ctrl, plant=si.adaptive(cartpole, "dopri5"), x0=x0, steps=120)
```

- Layout is decided in one place (`ocp.py`) and hidden behind `Solution`. It is stage-major, so
  every stage is one `vmap` and the KKT ordering sees the banded structure.
- `law` takes the whole primal-dual guess and returns the next one, already shifted. The shift
  moves primal blocks by one stage and repeats the last one, and moves multipliers by stage blocks
  (the race_cars variant). For pseudospectral it applies the constant interpolation matrix from
  the old nodes to the shifted ones. A C caller keeps one buffer. Status reaches C through the
  existing `<symbol>_stats()`.
- A linear model is `step=mpc.linear(A, B)` with `mpc.quadratic` costs. That problem is a QP, so
  `"piqp"` accepts it through the existing proof. `transcription=mpc.SingleShooting()` eliminates
  the states with a `scan` rollout, giving the condensed dense QP. The sparse form is the default.
- Real-time iteration is `mpc.MPC(ocp, "sqp", rti=True)`: one warm-started SQP step per sample.
  PR M4 decides whether `max_iter=1` of the existing plugin is enough or the plugin needs a
  no-globalization mode.

### 4.2 Terminal ingredients (`terminal.py`, `polytope.py`)

| Tool | Systems | How |
| --- | --- | --- |
| `TerminalEquality(x_s)` | any | Equality rows |
| `lqr(A, B, Q, R) -> (K, P)` | linear, or linearized at `(x_s, u_s)` | `scipy.linalg.solve_discrete_are`, residual checked |
| `Polytope` | — | Halfspace form. `box`, `intersect`, `preimage`, `remove_redundancy`, `contains`, `chebyshev_center`, `support`, `vertices` (2-D and 3-D plotting), all LPs by HiGHS |
| `max_invariant_set(A_K, X)` | linear, `u = Kx` | Gilbert-Tan: intersect preimages until the new rows are redundant (one LP per row). Raises when not finitely determined within `max_iter` |
| `Ellipsoid(P, alpha)`, `largest_ellipsoid(P, polytope)` | linear | `alpha = min_i h_i^2 / (H_i P^{-1} H_i^T)` over the state and `Kx` input rows |
| `quasi_infinite_horizon(F, x_s, u_s, Q, R, X, U)` | nonlinear | Chen-Allgöwer. P from a discrete Lyapunov equation of the closed-loop linearization with a margin; `alpha` by bisection until the decrease condition holds on the level set, checked by sampling or by a multi-start NLP built with scaly and IPOPT |
| `certify(ocp, terminal, samples=...)` | any | Samples the set and checks invariance and `Vf(F(x, κ(x))) - Vf(x) + l(x, κ(x)) <= 0`; returns the worst point |

A polytopic terminal set is `sc.bounded` rows. An ellipsoid is one nonlinear inequality, so the QP
backends refuse it with `NotQuadratic`, and the error message points to the polytope. Steady-state
targets for tracking come from `mpc.steady_state(F, y_ref, ...)`: a linear solve for linear models,
an NLP otherwise.

### 4.3 Indirect methods (phase X)

`mpc.indirect` builds `H = l + λᵀ f`, the costate dynamics `λ' = -H_x` and the stationarity
condition `H_u = 0` by AD. `u*(x, λ)` comes in closed form when the user gives it, and otherwise
from a Newton solve on `H_u`. Input boxes are handled by projection for control-affine problems
with a cost quadratic in `u`. The two-point boundary value problem is solved by indirect multiple
shooting, with the integrators of §3 and Newton on the LU of PR I2. Pseudospectral solutions report
costate estimates from their multipliers, which gives the cross-check between the two.

## 5. Solvers

| Problem | Backend | Warm start | Status |
| --- | --- | --- | --- |
| Linear dynamics, quadratic cost, polytopic constraints | `piqp` (sparse or condensed) | none (PIQP 0.6.2) | `<symbol>_stats()` |
| Nonlinear, full solve | `ipopt` | primal; duals with `warm_start_init_point` | same |
| Nonlinear, SQP or RTI | `sqp` (PIQP subsolver) | primal and duals | same |
| Any QP, library-free C | generated IPM | — | **after Tier 4 #30**, a todo item |

## 6. What changes elsewhere

- **New examples**, each run by `tests/integration/test_examples_gallery.py`:
  - `integrators_tour.py`: work-precision curves and stiff versus non-stiff behaviour;
  - `linear_mpc.py`: double integrator, maximal invariant terminal set, closed loop, C export;
  - `nmpc_transcriptions.py`: one swing-up by shooting, collocation and pseudospectral.
- **Existing examples stay as they are** (sign-off, §11). `tests/mpc/` carries its own
  self-contained copy of the hand-written cartpole formulation as the differential reference.
- **Benchmarks and the CasADi mirrors stay as they are.** Their formulations are pinned by
  `checks.py` gates and CasADi parity, so migrating them is a separate todo item.
- **Docs:**
  - guide pages `docs/guide/integrators.md` and `docs/guide/mpc.md`;
  - API pages `docs/api/integrators.md` and `docs/api/mpc.md`;
  - their `zensical.toml` nav entries and rows in `docs/api/index.md`.
- **`internal/todo.md`:** one item per PR. The "Next id" counter reads 135, but C-141 is taken, so
  it moves to 142 before any new id is assigned.

## 7. Phases

One commit per PR on `claude/integrators-mpc`, cut from `claude/function-templates`. Each PR follows
the handoff procedure:

- a todo id;
- tests, then mutation checks: break by hand, see red, restore;
- a benchmark script in `internal/notes/perf_2026_09_2x_integrators/` or `..._mpc/`;
- an HTML report;
- the node-ID baseline regenerated.

| PR | Todo | Content | Gate |
| --- | --- | --- | --- |
| I1 | API-142 | Package skeleton, `tableau.py` (explicit families, order-condition checks to order 5), `explicit.py` (steps, substeps as unroll/scan, `dt` input, extra inputs), `rk4` | Empirical order within 0.1 of theoretical on a linear system (expm reference) and Van der Pol (`solve_ivp`, rtol 1e-13), each method. Library RK4 generates the same C as `nmpc_cartpole`'s hand-written RK4, or times within 3% A/B. C size constant in `steps` above the threshold |
| I2 | C-143 | Dense LU with partial pivoting: `ExprOp.LU` (packed factor and row permutation) with loop and unrolled lowering, verify rule and sparsity; `linalg.lu`, `lu_solve`, `solve(a, b, assume="gen")` with the implicit derivative by `custom_derivative`. The factor op itself has no derivative, as for `SPARSE_LDL` | Matches `numpy.linalg.solve` to 1e-12 relative on random, permutation-requiring (zero leading pivot) and ill-conditioned (cond 1e10) matrices, orders 1 to 40. Derivatives match finite differences. Full suite. Timed against LAPACK `dgesv` through the C-side harness |
| I3 | API-144 | `implicit.py`: the collocation IRK families from nodes, DIRK, Newton (fixed or `tol`, simplified or full), implicit-function-theorem derivatives in both modes | Orders 2s, 2s-1 and 2s-2 observed. A stiff test (Robertson; `y' = -1e6 (y - cos t)`) stays stable at `dt·λ = 1e4` for L-stable methods. Derivatives agree with AD through the unrolled Newton to 1e-10 and with finite differences to 1e-6. Radau IIA(3) against `solve_ivp(method="Radau")` |
| I4 | API-145 | `linear.py` (`zoh`, `foh`, `linearize`), embedded pairs and `si.adaptive`, symplectic methods | ZOH against a fine RK reference and the `expm` block formula. Adaptive global error tracks `rtol` over 1e-4 to 1e-10. Störmer-Verlet energy drift bounded over 1e5 steps where RK4's grows |
| I5 | API-146 | `polynomial.py`, `transcription.py` (`MultipleShooting`, `Collocation`, `Pseudospectral`) | Quadrature exact to degree 2n-1 (LG), 2n-2 (LGR) and 2n-3 (LGL). Differentiation matrices exact on polynomials. Collocation(Radau, d) equals the Radau IIA(d) step to 1e-12. Spectral convergence on a smooth problem |
| M1 | API-147 | `ocp.py`, `controller.py`: OCP to `sc.problem`, every transcription, bounds, path and soft constraints, terminal equality, params and references, `Solution`, `law` with the in-graph shift, `simulate` | The library-built cartpole matches the hand-written formulation (copied into `tests/mpc/`) to 1e-8 and solves within 3% A/B. Shooting, collocation and pseudospectral agree as the grid refines. `law` through the JIT equals the Python loop step for step |
| M2 | API-148 | `mpc.linear`, `mpc.quadratic`, the condensed form, `polytope.py`, `lqr`, `max_invariant_set`, ellipsoids, polytope and ellipsoid terminal constraints | Unconstrained MPC with the LQR terminal cost returns `u = Kx` to 1e-9 for every N. O∞ of textbook examples matches published facet counts and is invariant (LP-checked). A randomized closed loop is recursively feasible, and the optimal cost decreases by at least the stage cost |
| M3 | API-149 | `quasi_infinite_horizon`, `certify`, `steady_state` | The Chen-Allgöwer example (discretized) certifies. Closed-loop cost decreases from sampled points of the terminal set. `certify` fails on a deliberately inflated `alpha` |
| M4 | API-150 | RTI, the C deployment example (a C `main` running the closed loop on the generated `law`), the three new examples, guide and API docs | Deployment example compiled and run by the gallery test. RTI tracks the full solve's closed loop on cartpole within a stated gap. Docs build clean |
| R | API-151 | Review round with five agents (IR/AD for LU and implicit derivatives, integrator numerics, MPC theory, performance, docs/tests), fixes, a summary report, tag `integrators-mpc-complete` | — |
| X | API-152 | `mpc.indirect` (§4.3) | Indirect and pseudospectral solutions of a textbook problem (the Bryson-Ho brachistochrone, or an LQ problem with a known costate) agree, costates included |

Deferred todo items opened with the plan:

- API-153: DAEs (semi-explicit index 1) in implicit RK and collocation.
- API-154: control-invariant and robust (tube) terminal sets.
- API-155: the generated IPM as an MPC backend, once Tier 4 #30 lands.
- API-156: migrate the benchmark problems' hand-written integrators and NumPy plants.

Kill criteria:

- If I1's library RK4 cannot match the hand-written C within 3%, stop and find out why before
  building on it.
- If I2's LU is more than 3x LAPACK at n ≤ 12, the implicit integrators still proceed (correctness
  first), and the gap becomes a compiler item.

## 8. Tests

- `tests/integrators/` and `tests/mpc/` mirror the packages. Every numeric claim is differential,
  checked against one of: `scipy.integrate.solve_ivp` at tight tolerance, `scipy.linalg.expm`,
  `numpy.linalg.solve`, `scipy.linalg.solve_discrete_are`, a hand-unrolled formulation, or a
  closed form (LQR).
- Derivatives are checked against finite differences and against AD through the unrolled
  alternative.
- Tests needing a solver carry `@pytest.mark.solver("ipopt"|"piqp")`. Tests of offline design
  (polytopes, DARE) need none.
- Mutation checks per PR, reported. Planned mutants include:
  - a tableau coefficient;
  - the sign in the implicit derivative;
  - the pivot choice;
  - the shift direction;
  - one Gilbert-Tan termination row;
  - the LQR `P` perturbed by 1e-6;
  - `alpha` inflated.

## 9. Benchmarks

Every comparison follows the protocol in `docs/results/fairness.md` and the handoff's traps: an A/B
against a worktree, both sides warmed, minima over equal samples, timed in C where the Python call
dominates.

| Suite | Scaly side | Baseline |
| --- | --- | --- |
| Integrator step and step Jacobian | library `rk4`, `radau_iia(3)`, `sdirk3` | hand-written scaly RK4; CasADi SX of the same step with `jacobian()`, both compiled at -O2 |
| Work-precision | every method | `solve_ivp` for the reference solution only, not for timing |
| LU | `linalg.lu_solve` at n ∈ {4, 8, 12, 24, 40} | LAPACK `dgesv` from C |
| NMPC closed loop | cartpole via `scaly.mpc` (ipopt, sqp, RTI) | the hand-written example; CasADi Opti with IPOPT on the same NLP |
| Linear MPC | sparse and condensed via piqp at nx ∈ {4, 12}, N ∈ {10, 50} | PIQP called directly on the same QP data |

Numbers stay in the per-PR reports under `internal/notes/`. None reach `docs/` unless they go
through the results pages.

## 10. Risks

- **LU pivoting in the program dialect.** Row swaps need loads at computed indices and an argmax
  kept in scalars. The unrolled small-order form can use selects instead. If the looped form
  fights the passes (fusion, `pack_workspace`), I2 ships the unrolled form plus a scan-based loop,
  and the op follows.
- **Name collisions.** A library that wraps user models must derive every Function name from the
  model's name and configuration, or two instances collide at lowering. Tests build two integrators
  of one model into one graph.
- **The implicit derivative recomputes `G_K`.** The Newton loop's final factor is not reused by the
  derivative rule in v1. If the benchmark shows that dominates, the step returns the factor as a
  hidden output.
- **Pseudospectral warm starts.** The shift interpolates on non-uniform nodes. Accuracy is tested,
  not assumed.
- **Plugin reach.** RTI may need a small `sqp` plugin option. That is a plugin change with its own
  cold rebuild of 5 to 8 minutes.

## 11. Questions for sign-off

Answered 2026-09-27: approved as written, no scope trims; the full Tier process per PR; existing
examples untouched, only the three new ones added; `AGENTS.md` corrected to name SciPy as a runtime
dependency. The branch is `claude/integrators-mpc` from `claude/function-templates`.

1. **Branch:** `claude/integrators-mpc` from `claude/function-templates`, which carries Tier 3, the
   speed pass and templates. Fine, or a different base?
2. **Examples:** port `nmpc_cartpole.py` onto `scaly.mpc` (the hand-written form becomes a test
   reference), and leave the benchmarks and CasADi mirrors untouched for now (API-156)?
3. **SciPy:** `AGENTS.md` says NumPy is the only runtime dependency, but `pyproject.toml` lists
   SciPy and `src/scaly` imports it. This plan uses SciPy for offline design. Should I also
   correct that line in `AGENTS.md`?
4. **Process:** the Tier procedure per PR (todo id, mutation checks, benchmark script, HTML report),
   roughly ten reports. Keep it, or lighten it for this work (for example, one report per package)?
5. **Scope trims, if you want them:** symplectic methods (I4), `si.adaptive` (I4) and
   pseudospectral warm-start interpolation are the items easiest to defer.
