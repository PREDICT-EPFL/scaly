# QP and NLP solvers

Phase 5 of [`roadmap.md`](roadmap.md). This document describes the user-facing
`al.qp(...)` / `al.nlp(...)` builders, their internal oracle assembly, the
ctypes plumbing to vendored PIQP and IPOPT, and the current limitations.

The shared libraries and headers live in the `alloy-piqp` and `alloy-ipopt`
plugin packages and are built by their `plugins/*/hatch_build.py` hooks; see
[`vendored_solvers.md`](vendored_solvers.md) for build-side issues.

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

Initial dual values are accepted in the signature for API symmetry. IPOPT seeds
its equality and inequality multipliers from them; PIQP warm-start plumbing is
not implemented yet.

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
iter, stats)`. IPOPT stats contain iteration and value-callback counts; PIQP
leaves stats as `None`. `status.ok` is `True` for PIQP `solved` and IPOPT
`solve_succeeded`/`solved_to_acceptable_level`/`feasible_point_found`.
During the generated-path soak period, `SolverStatus.code` and `.name` use the
Alloy enum on the generated path but solver-native values on the Python path;
`ok` agrees across both, and the distinction disappears when the Python path is deleted.

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

- `_lib.py` — delegates native solver discovery to `alloy.toolchain`, then
  dlopens `libpiqpc` / `libipopt` via `ctypes.CDLL`.
- `_piqp.py` — `ctypes.Structure` mirrors of `piqp_data_dense`, `piqp_settings`,
  `piqp_info`, `piqp_result`, `piqp_workspace`. The `PIQPDenseSolver` handle
  owns the column-major numpy buffers that PIQP reads through pointers, and
  defers `piqp_setup_dense` until the first `update(...)` so PIQP sees the
  real problem instead of `±PIQP_INF` placeholders (which otherwise produce
  spurious "free constraint" warnings at setup time). PIQP's signed duals
  `z_l, z_u, z_bl, z_bu` are combined into a single signed `lam_ineq` /
  `lam_box` (upper − lower) on the way out.
- `_ipopt.py` — `CFUNCTYPE`s for the five `Eval_*_CB` callbacks and the
  intermediate callback plus bindings
  for `CreateIpoptProblem`, `IpoptSolve`, `AddIpopt{Str,Num,Int}Option`, and
  `FreeIpoptProblem`. `solve_ipopt(...)` wraps Python evaluators in the
  callback ABI: when IPOPT passes `iRow`/`jCol` with a NULL `values` it
  receives the precomputed sparse pattern; with a non-NULL `values` the Alloy
  oracle is invoked and the compact buffer is memcpy'd into IPOPT's array.
  Initial multiplier buffers support warm starts, while the intermediate and
  evaluation callbacks record iteration and invocation counts. Exceptions
  raised inside callbacks are captured and re-raised after IPOPT returns, so a
  buggy oracle does not silently swallow the diagnostic.
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

## Nesting: solver as a graph node

A `SolverFunction` is a real `alloy.Function` whose body is one
`Ops.SOLVER_CALL` per solver output. Two consequences:

1. You can use `solver.call([x0_expr, lam_eq0_expr, lam_ineq0_expr, *param_exprs])`
   the same way you would call any other `Function`. The result is a tuple of
   `Expr`s — one per solver output — usable in further symbolic computation.
2. The outer Function's semantic graph contains an `Ops.CALL` node whose callee is the
   `SolverFunction`; the solver's own outputs are `Ops.SOLVER_CALL` nodes
   whose attrs carry the `SolverDescriptor`. From the outer Function's
   point of view, the solver behaves like any other named callee.

This is the safety-filter assembly pattern from the roadmap:

```python
@al.function("safety_filter", {"x": (NX,), "u_ref": (NU,)})
def safety_filter(x, u_ref):
    f_x = dynamics.f.call([x])[0]
    g_x = dynamics.g.call([x])[0]
    h, h_grad = cbf(x)
    P = al.const(np.eye(NU))
    c = -u_ref
    G_ineq = h_grad @ g_x
    l_ineq = -(h_grad @ f_x + alpha * h)
    u_ineq = al.const(np.full(NG, np.inf))
    qp = al.qp(P=P, c=c, G_ineq=G_ineq, l_ineq=l_ineq, u_ineq=u_ineq)
    out = qp.call([
        al.const(np.zeros(NU)),   # x0
        al.const(np.zeros(0)),    # lam_eq0
        al.const(np.zeros(NG)),   # lam_ineq0
        x, u_ref,                 # params (auto-detected by al.qp)
    ])
    return {"u": out[0]}
```

The `SOLVER_CALL` op is marked non-differentiable; `al.jacobian` /
`al.gradient` / sparse-pattern queries through it return zero. Implicit
function theorem AD (KKT-residual adjoints) is future work.

### AOT: integrating a generated solver into a C++ application

The same `render_c_module(fun)` path used by the existing benchmarks works
for solver-containing Functions; the only extra thing the consumer needs is
to know which solver libs to link against. ``alloy.codegen.solver_c`` exposes
that:

```python
from alloy.codegen.c import render_c_module
from alloy.codegen.solver_c import solver_compile_flags

module = render_c_module(safety_filter, typed_buffers=False)
(out_dir / module.header_name).write_text(module.header)
(out_dir / module.source_name).write_text(module.source)
flags = solver_compile_flags(safety_filter)
# -> ['-I/.../alloy/include', '-L/.../alloy/lib',
#     '-Wl,-rpath,/.../alloy/lib', '-lpiqpc']  (and/or '-lipopt')
```

`solver_compile_flags` returns the include/lib/rpath/library flags as a
list, ready to splice into a `subprocess.run(...)` invocation of `cc` /
`c++`. Pass `rpath=False` if you'll bundle the libs into your own
install tree and set the rpath yourself.

A minimal C++ driver against the universal ABI looks like:

```cpp
#include "safety_filter.h"
#include <cstdio>

int main() {
    double x[NX] = {/* state */};
    double u_ref[NU] = {/* reference */};
    double u[NU];
    const double* arg[] = {x, u_ref};
    double* res[] = {u};
    double w[(safety_filter_SZ_W > 0 ? safety_filter_SZ_W : 1)];
    int rc = safety_filter(arg, res, nullptr, w, nullptr);
    return rc;
}
```

A small Google-Benchmark example (PIQP variant + IPOPT variant) exercising
this AOT path lives at `benchmarks/alloy_solver_aot_demo.py`:

```bash
uv run python benchmarks/alloy_solver_aot_demo.py \
    --variant both -- --benchmark_min_time=0.05s
```

That script generates two toy safety filters (a CBF-style QP and the same
shape routed through IPOPT), writes the rendered headers/sources to
`benchmarks/gen/alloy_solver_aot_demo/`, links them against the vendored
solver libs via `solver_compile_flags`, and runs Google Benchmark on the
AOT-compiled binaries. On a recent macOS arm64 dev box the QP path lands
around 4 µs per solve, the NLP path around 450 µs.

The realistic safety-filter workload (7-state bicycle + 256/128 MLP,
HOCBF pair + wall constraints — see [`safety_filter.md`](safety_filter.md))
lives separately at `benchmarks/alloy_safety_filter_benchmark.py` and times
the constraint/cost forward + derivative pieces that an external QP/NLP
solver would call. Wiring it onto a solver via this AOT path is the next
step.

### Status of the C codegen path

**Both backends supported.** A SolverFunction renders to a self-contained C
function that drives PIQP (dense QP) or IPOPT (NLP).

For PIQP, the generated wrapper calls the rendered oracle to fill QP data,
transposes `P`/`A`/`G` to PIQP's column-major layout, lazily sets up a
static `piqp_workspace`, and dispatches `piqp_update_dense` + `piqp_solve`
on every call.

For IPOPT, the generated source emits a static context struct (parameter
pointers), static `const int` arrays for the sparse Jacobian and
lower-triangular Hessian patterns, and five static `eval_*` callbacks that
bridge IPOPT into the rendered `base`/`grad`/`jac`/`hess` Functions. The
wrapper body computes bounds via the rendered `bounds_raw`, calls
`CreateIpoptProblem`, applies the descriptor's options, runs `IpoptSolve`,
and writes the seven NLP outputs (`x`, `f`, `h_eq`, `g_ineq`, `lam_eq`,
`lam_ineq`, `lam_box`). The Hessian callback gathers the lower-triangle via
a precomputed static index table — same mask the Python backend uses.

The JIT auto-adds `-I<alloy/include>`, `-L<alloy/lib>`, `-lpiqpc`/`-lipopt`,
and `-Wl,-rpath,<alloy/lib>` to the compile command when a SolverFunction is
reachable from the function being compiled. A nested safety-filter Function
(QP or NLP variant) compiles to one `.so` that links directly against the
vendored solver libs and is callable from C++ through the universal ABI.

## Sign and ordering conventions

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
- `Ops.SOLVER_CALL` IR op + `SolverFunction` subclassing `Function`, so
  solvers compose with the rest of the IR.
- **C codegen for `SOLVER_CALL` (PIQP + IPOPT)**: nested QP/NLP solvers
  render to a single `.so` that links against the vendored solver libs and
  runs without any Python in the hot path.
- Reference tests against analytic KKT solutions and box-only optima, plus
  JIT-compiled nesting tests, live in `plugins/alloy-{piqp,ipopt}/tests/`;
  structural solver tests remain in `tests/alloy/`.

Deferred (tracked in [`roadmap.md`](roadmap.md)):
- A C++ harness driving the safety filter end-to-end through the universal
  ABI.
- Sparse PIQP backend (current implementation uses the dense interface —
  enough for the input-affine CBF QP).
- Warm-start handover for `lam_eq0`/`lam_ineq0` into PIQP and IPOPT.
- Implicit-function-theorem AD through `SOLVER_CALL` (today: zero gradients).
- Cross-call deduplication across distinct solver invocations remains a future optimization.
  Multiple outputs selected from the same `solver.call(...)` site lower to one Program IR
  `CALL`, but two separate call sites with identical arguments are not globally CSE'd yet.

## Limitations and gotchas

- `libpiqpc`/`libipopt` must exist in their plugin package `lib/` directories.
  The first `uv sync` triggers the plugin build hooks (~5–8 min cold). Missing
  libs raise `SolverLibraryError` when a solve or solver-bearing JIT needs them.
- `±PIQP_INF` (`1e30`) and `±IPOPT_INF` (`2e19`) are treated as "no bound" by
  the respective backends. Pass `±np.inf` and the SolverFunction backend will
  substitute the right sentinel.
- IPOPT options default to `print_level=0`, `sb="yes"` (no banner). Pass
  `options={...}` to override or extend.
- PIQP options pass through to `piqp_settings` field names. Unknown names
  raise `ValueError`.
