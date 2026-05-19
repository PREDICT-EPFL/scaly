# QP and NLP solvers

Phase 5 of [`roadmap.md`](roadmap.md). This document describes the user-facing
`al.qp(...)` / `al.nlp(...)` builders, their internal oracle assembly, the
ctypes plumbing to vendored PIQP and IPOPT, and the current limitations.

The shared libraries themselves (`src/alloy/lib/libpiqpc.{dylib,so}`,
`src/alloy/lib/libipopt.{dylib,so}`) and their headers are built by the hatch
hook; see [`vendored_solvers.md`](vendored_solvers.md) for build-side issues.

## Problem shapes

Both shapes follow the explicit-categories convention rather than CasADi's
bilateral `lba ≤ Ax ≤ uba`.

**QP**

```text
min   0.5 xᵀ P x + cᵀ x
s.t.  A_eq x = b_eq
      l_ineq ≤ G_ineq x ≤ u_ineq      (two-sided general inequalities)
      x_lb  ≤ x ≤ x_ub                (box)
```

**NLP**

```text
min   f(x, p)
s.t.  h_eq(x, p) = 0
      l_ineq ≤ g_ineq(x, p) ≤ u_ineq  (two-sided general inequalities)
      x_lb  ≤ x ≤ x_ub                (box)
```

Inequalities are kept two-sided rather than reduced to one-sided form because
PIQP supports them natively and the IPOPT path stacks `[h_eq; g_ineq]` into
`g(x)` with bounds `[0; l_ineq] ≤ g ≤ [0; u_ineq]` — a one-shot conversion at
solver-construction time, not per call.

## Public API

```python
import alloy as al
import numpy as np

# QP — PIQP backend
qp = al.qp(
    P=P_expr,         c=c_expr,
    A_eq=A_eq_expr,   b_eq=b_eq_expr,
    G_ineq=G_ineq_expr, l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr,     x_ub=ub_expr,
    solver="piqp",
)

# NLP — IPOPT backend
nlp = al.nlp(
    x=x_sym, p=p_sym,
    f=f_expr,
    h_eq=h_eq_expr,
    g_ineq=g_ineq_expr, l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr,       x_ub=ub_expr,
    solver="ipopt",
)
```

Every symbolic input may be:

- an Alloy `Expr` over a set of free parameters `p` (the canonical case),
- a NumPy array / Python scalar (becomes a constant),
- or `None` for the optional pieces (`A_eq`/`b_eq`, `G_ineq`/`l_ineq`/`u_ineq`,
  `x_lb`/`x_ub`).

Both builders return an `al.SolverFunction`, an opaque callable that mimics
`Function`'s call surface (`input_names`, `output_names`, `__call__` accepting
positional or keyword args). Calling it runs the bound backend.

### Call-time inputs

- `x0` (size `n`): initial primal iterate.
- `lam_eq0` (size `p`): initial equality multipliers.
- `lam_ineq0` (size `m`): initial inequality multipliers.
- followed by every free parameter detected in the symbolic inputs, in
  deterministic name/id order.

Initial dual values are accepted in the signature for API symmetry. Warm-start
plumbing into PIQP and IPOPT is not implemented yet.

### Call-time outputs

QP:

- `x` (size `n`), `cost` (scalar),
- `lam_eq` (size `p`), `lam_ineq` (size `m`, signed: `z_u − z_l`),
- `lam_box` (size `n`, signed: `z_bu − z_bl`).

NLP:

- `x`, `f`,
- `h_eq`, `g_ineq` (constraint values at the optimum),
- `lam_eq`, `lam_ineq`, `lam_box` (signed; `lam_box = mult_x_U − mult_x_L`).

`SolverFunction.last_status` exposes the most recent `SolverStatus(code, name,
iter)`. `status.ok` is `True` for PIQP `solved` and IPOPT
`solve_succeeded`/`solved_to_acceptable_level`/`feasible_point_found`.

### Symbolic-parameter example

```python
mu = al.sym("mu", 2)
qp = al.qp(P=np.eye(2), c=-mu)
for mu_val in [np.array([0.0, 0.0]), np.array([1.5, -0.3])]:
    out = qp(x0=np.zeros(2), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0), mu=mu_val)
    # out["x"] approaches mu_val each time
```

## Internals

### Oracle assembly

Both builders construct ordinary Alloy `Function`s for the data the backend
needs at call time. Those Functions go through the existing JIT path (Phase
4), so caching, compilation, and dispatch happen the same way as for any user
function. Only the final solver call is opaque.

**QP oracle.** One Function whose inputs are the discovered free parameters
and whose outputs are the flattened QP data (`P`, `c`, `A_eq`, `b_eq`,
`G_ineq`, `l_ineq`, `u_ineq`, `x_lb`, `x_ub`). At call time this Function is
evaluated once with the user-supplied parameter values; the resulting numeric
arrays are handed to PIQP via `piqp_update_dense` + `piqp_solve`.

**NLP oracles.** Three Functions, all built through `Function.factory(...)`:

- `nlp_base` — `(x, *params) → (f, g_all)` where `g_all = concat([h_eq, g_ineq])`.
- `nlp_grad` — `factory(..., ["grad:f:x"])`: dense gradient of the objective.
- `nlp_jac` — `factory(..., ["spjac:g:x"])`: compact sparse Jacobian of the
  stacked constraint vector. The `(rows, cols)` pattern is attached to the
  Function via `output_sparsities` and used directly by IPOPT.
- `nlp_hess` — `factory(..., ["sphess:gamma:x:x"], aux={"gamma": ["f", "g"]})`:
  compact sparse Lagrangian Hessian. At call time `lam:f = obj_factor` and
  `lam:g = lam` (concatenated equality + inequality multipliers from IPOPT).
  IPOPT expects only the lower triangle; the symmetric COO pattern returned by
  the factory is filtered to `i ≥ j` at build time and the matching values are
  gathered at call time.

A small `nlp_bounds` Function evaluates the (param-dependent) `x_lb`, `x_ub`,
`l_ineq`, `u_ineq` exprs and is queried once per solve.

### Backend wiring

`src/alloy/solvers/` contains:

- `_lib.py` — locates `libpiqpc` / `libipopt` next to the package and dlopens
  them via `ctypes.CDLL`.
- `_piqp.py` — `ctypes.Structure` mirrors of `piqp_data_dense`, `piqp_settings`,
  `piqp_info`, `piqp_result`, `piqp_workspace`. The `PIQPDenseSolver` handle
  owns the column-major numpy buffers that PIQP reads through pointers, and
  defers `piqp_setup_dense` until the first `update(...)` so PIQP sees the
  real problem instead of `±PIQP_INF` placeholders (which otherwise produce
  spurious "free constraint" warnings at setup time). PIQP's signed duals
  `z_l, z_u, z_bl, z_bu` are combined into a single signed `lam_ineq` /
  `lam_box` (upper − lower) on the way out.
- `_ipopt.py` — `CFUNCTYPE`s for the five `Eval_*_CB` callbacks plus bindings
  for `CreateIpoptProblem`, `IpoptSolve`, `AddIpopt{Str,Num,Int}Option`, and
  `FreeIpoptProblem`. `solve_ipopt(...)` wraps Python evaluators in the
  callback ABI: when IPOPT passes `iRow`/`jCol` with a NULL `values` it
  receives the precomputed sparse pattern; with a non-NULL `values` the Alloy
  oracle is invoked and the compact buffer is memcpy'd into IPOPT's array.
  Exceptions raised inside callbacks are captured and re-raised after IPOPT
  returns, so a buggy oracle does not silently swallow the diagnostic.
- `_oracle.py` — `collect_free_inputs(exprs)` walks the expression graph and
  returns the unique `Ops.INPUT` exprs in deterministic name/id order.
- `solver_function.py` — `SolverFunction` opaque wrapper. Mimics
  `Function`'s `(input_names, output_names, __call__)` surface; the backend is
  a Python callable rather than a JIT path.
- `qp.py` / `nlp.py` — the user-facing builders.

### Sign and ordering conventions

- `lam_ineq`, `lam_box` are signed: positive ⇒ upper bound active, negative ⇒
  lower bound active. Both backends report nonneg `z_l`/`z_u` separately
  internally; the conversion happens in the SolverFunction backend.
- For NLP, `lam = [lam_h; lam_g]` is the IPOPT-side stacked multiplier vector.
  The Lagrangian aux `gamma` is built off `["f", "g"]` in that order, so
  `lam:g` corresponds to the same stacked vector — this is what makes
  `sphess:gamma:x:x` produce the correct Lagrangian Hessian.
- Sparse Jacobian and Hessian patterns are COO `(rows, cols)` in the order
  produced by `al.sparse_jacobian` / `al.sparse_hessian`.

## Status

Implemented:

- `al.qp(...)` with the PIQP dense backend.
- `al.nlp(...)` with the IPOPT backend (sparse Jacobian, sparse Lagrangian
  Hessian filtered to lower triangle).
- Two-sided general inequalities on both backends.
- Symbolic parameters in the oracle; QP data and NLP bounds can be Alloy
  `Expr`s of free `p`.
- ctypes bindings to vendored `libpiqpc`/`libipopt`.
- Reference tests against CasADi+IPOPT, analytic KKT solutions, and box-only
  optima (`tests/alloy/test_solvers.py`).

Deferred (tracked in [`roadmap.md`](roadmap.md)):

- C codegen of the solver wrapper itself. `SolverFunction.__call__` currently
  runs the backend in Python; the oracle Functions go through the normal JIT
  path. Generating C that drives PIQP/IPOPT directly is what enables the C++
  harness use case.
- A C++ harness driving the safety filter end-to-end through the universal
  ABI.
- Sparse PIQP backend (current implementation uses the dense interface —
  enough for the input-affine CBF QP).
- Warm-start handover for `lam_eq0`/`lam_ineq0` into PIQP and IPOPT.
- Inlining a solver as a node inside a larger `Function` expression graph
  (the roadmap calls these "opaque calls"); for now `SolverFunction`s are
  terminal callables.

## Limitations and gotchas

- `libpiqpc`/`libipopt` must exist under `src/alloy/lib/`. The first `uv sync`
  triggers the build hook (~5–8 min cold). Missing libs raise
  `SolverLibraryError` at import time of `alloy.solvers`.
- `±PIQP_INF` (`1e30`) and `±IPOPT_INF` (`2e19`) are treated as "no bound" by
  the respective backends. Pass `±np.inf` and the SolverFunction backend will
  substitute the right sentinel.
- IPOPT options default to `print_level=0`, `sb="yes"` (no banner). Pass
  `options={...}` to override or extend.
- PIQP options pass through to `piqp_settings` field names. Unknown names
  raise `ValueError`.
