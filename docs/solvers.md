# QP and NLP solvers

Phase 5 of [`roadmap.md`](roadmap.md). This document describes the user-facing
`al.qp(...)` / `al.nlp(...)` builders, their internal oracle assembly, the
generated C solve path driving vendored PIQP and IPOPT, and the current
limitations.

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

# NLP — IPOPT or SQP backend
nlp = al.nlp(
    x=x_sym, p=p_sym,
    f=f_expr,
    h_eq=h_eq_expr,
    g_ineq=g_ineq_expr, l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr,       x_ub=ub_expr,
    solver="sqp",       # or "ipopt"
)
```

`alloy-sqp` uses a bounded objective/violation filter by default. Its solver
options are `globalization="filter"|"l1"`, `watchdog=<non-negative int>` (only
with `l1`), primal `tol` (default `1e-6`), `dual_tol` for stationarity and
complementarity (default `1e-4`), `max_iter`, `hessian="exact"|"objective"`,
`regularization`, `line_search_beta` (default `0.7`), `merit` (the l1 penalty
offset, default `10`), `qp="sparse"|"dense"` (default `"sparse"`), `qp_tol`
(default `1e-6`), `qp_max_iter` (default `50`), and the debug-only `trace` flag.
Invalid combinations are rejected when the wrapper is generated.

Both globalization paths include equality, inequality, and variable-bound
violations. Accepted primal and signed dual iterates use the same step length.
Success requires primal feasibility plus KKT stationarity and conventional
signed-multiplier complementarity; a merely feasible `MAX_ITER` result is not
upgraded to success. Following LAOPT, a QP that stops at its iteration limit or
returns an infeasibility certificate does not abort the solve: the SQP continues
with PIQP's best iterate and the globalization plus the KKT test decide whether
the step is accepted. Only PIQP's numerics/unsolved/invalid-settings statuses
fail the solve. The l1 path uses an Armijo test and can take bounded
non-monotone watchdog steps before restoring its last complete primal/dual/QP
checkpoint for a normal line search.

The QP subproblem is assembled through PIQP's sparse interface. The Jacobian
and Hessian patterns are fixed across SQP iterations, so the wrapper bakes the
CSC index tables of `P` (the Hessian's upper triangle, unioned with the full
diagonal and with the constraint-normal term's pattern), `A` and `G` as static
arrays at codegen time and refills only the value arrays per iteration, through
a permutation table mapping each oracle value buffer's own order onto CSC
order. `qp="dense"` assembles the same QP for PIQP's dense interface instead;
it is slower at every benchmark size measured, so it exists as a differential
check on the sparse assembly rather than as a configuration to reach for.

Equality-constrained QPs regularize the Hessian first in the constraint-normal
space, `H + rho * A.T @ A + regularization * I`, accumulated pair by pair over
each equality row's own Jacobian entries. The added quadratic is constant over
the linearized equality manifold, so it does not damp the feasible SQP
direction. `rho` escalates by decades until the model is positive definite; if
an equality-constrained exact Lagrangian Hessian still has negative reduced
curvature, the SQP model falls back to the objective Hessian, which leaves
exact Hessians as the default without letting a large diagonal shift create a
false stationarity floor. Inequality-only models retain their exact
active-constraint curvature.

Positive definiteness is both tested and repaired by one modified LDL^T pass
over the assembled `P`, using an elimination tree and factor column counts
computed at codegen time. Whenever a pivot is not at least `regularization` it
is raised to its own magnitude; since only diagonal entries are touched,
`L*D*L^T` equals the assembled `P` plus that diagonal, so the repair is an
exact diagonal shift and it is zero whenever the model is already positive
definite. The pass costs `O(nnz(L))`, which is what makes escalating `rho`
cheap enough to do inside every SQP iteration.

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
- `lam_box0` (size `n`, NLP only): initial signed box multipliers — the
  wrapper sign-splits into IPOPT's `z_L = max(-lam_box0, 0)` /
  `z_U = max(lam_box0, 0)`, the inverse of the `lam_box` output convention.
- followed by every free parameter detected in the symbolic inputs, in
  deterministic name/id order.

IPOPT seeds its multipliers from the initial duals (pass
`options={"warm_start_init_point": "yes"}` to make IPOPT use them). For QP the
dual inputs are accepted for API symmetry only: PIQP's C interface has no
warm-start entry point — the cross-solve reuse it does offer (persistent
workspace, `piqp_update_*` + re-solve) is already how the generated wrapper
drives it.

### Call-time outputs

QP:

- `x` (size `n`), `cost` (scalar),
- `lam_eq` (size `p`), `lam_ineq` (size `m`, signed: `z_u − z_l`),
- `lam_box` (size `n`, signed: `z_bu − z_bl`).

NLP:

- `x`, `f`,
- `h_eq`, `g_ineq` (constraint values at the optimum),
- `lam_eq`, `lam_ineq`, `lam_box` (signed; `lam_box = mult_x_U − mult_x_L`).

`SolverFunction.last_stats` carries the full `SolverStats` after each solve
(alloy + native status, iterations, objective, the `t_fe`/`t_solver`/`t_qp`/
`t_globalization`/`t_glue` timing split, and the five oracle evaluation
counters; PIQP fills
`n_eval_f = 1` for its single oracle evaluation). `SolverFunction.last_status`
is the derived `SolverStatus(code, name, iter, stats)` view; its `code`/`name`
are the alloy status enum (`stats.py`), and `ok` is `True` for `OK` and
`ACCEPTABLE` (IPOPT `Feasible_Point_Found` maps to `ACCEPTABLE`).

Stats version 3 added six per-solve diagnostics (zero when the backend has no
such concept):

| field | meaning | SQP | PIQP | IPOPT |
|---|---|---|---|---|
| `primal_viol` | constraint violation (inf norm) at the returned `x` | recomputed at the returned iterate (equalities, inequalities, box) | native `info.primal_res` | `inf_pr` at the last iteration |
| `step_inf` | inf norm of the last computed step | last QP step | 0 | `d_norm` |
| `alpha` | last **accepted** line-search step length (`0.0` if no step was ever accepted, e.g. converged at `x0` or failed before globalization) | filter or l1 result | 0 | `alpha_pr` |
| `merit_penalty` | final merit penalty parameter | adaptive l1 penalty; `0.0` for filter globalization | 0 | 0 (filter line search) |
| `backtracks` | total rejected line-search trial points across the solve | counted directly | 0 | `ls_trials − 1` summed over iterations |
| `qp_iter` | QP iteration count accumulated across SQP iterations | summed PIQP `info.iter` | native `info.iter` (equals `iter`) | 0 |

IPOPT's callback-sourced fields (`primal_viol`, `step_inf`, `alpha`,
`backtracks`) record only regular-mode iterations: the restoration phase's
values describe the restoration subproblem, not the user's problem, and are
skipped.

### SQP per-iteration trace (`options={"trace": True}`)

`alloy-sqp` accepts a debug-only `trace` option, off by default. When enabled,
the generated wrapper prints per-SQP-iteration lines to **stderr**, prefixed
`[alloy-sqp <name>]`: a KKT line at the initial point and after each accepted
step, one line after each QP solve (QP status, QP iterations, primal violation,
step inf norm), and one after each globalization attempt (strategy, trial
objective, alpha, accepted flag, backtracks, and l1 penalty), plus a line when
the QP itself fails. The trace is deliberately not part of the stats ABI: it
is variable-length per solve, and a caller-provided buffer would need a new
ABI entry point and Python-side plumbing that nothing consumes yet — the
closed-loop harness records one `SolverStats` per step and its per-step
scalars already come from the stats struct, so stderr (line-buffered, greppable
by the step logs) is the cheaper sink until a consumer needs more. Since the
option is baked into the generated C, flipping it recompiles the solver (JIT
cache keyed on source).

### Sparse PIQP (`al.qp(..., sparse=True)`)

`sparse=True` routes the QP through PIQP's sparse interface. The structural
CSC patterns of `P` (upper triangle — PIQP's symmetric-P contract), `A_eq`,
and `G_ineq` are computed at build time (an entry is structurally nonzero if
it depends on a parameter or its constant value is nonzero) and baked into the
generated wrapper as static tables; the oracle emits compact CSC-ordered value
buffers, so runtime updates are values-only (`piqp_update_sparse`), with no
dense staging or transpose. `P` is assumed symmetric: exactly `triu(P)` is
baked, so an out-of-contract asymmetric `P` behaves as if symmetrized from its
upper triangle (the dense interface's behavior for such input is
solver-internal and may differ).

Two build-time caveats: constructing a sparse QP JIT-compiles and evaluates a
small pattern-probe function (so a working C toolchain is required at build
time, not just at first solve), and QP data computed from a nested solver
output is rejected with `NotImplementedError` — `SOLVER_CALL` outputs are
opaque to the dependency mask, so their pattern cannot be derived soundly.

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

There is exactly **one solve path**: the generated C wrapper (root
`ROADMAP.md` §3.4, L2). Every `SolverFunction.__call__` JIT-compiles a
self-contained C translation unit whose wrapper calls `piqp_c` /
`IpStdCInterface.h` directly with the generated oracle kernels and fills the
alloy-owned stats struct — the same artifact serves nested solves and AOT
deployment. The historical Python-interleaved backends (nanobind
`_piqp_ext`, ctypes IPOPT callbacks) were deleted on 2026-07-15; the plugin
packages now ship only the vendored native library, headers, and entry-point
metadata.

The pieces:

- `src/alloy/solvers/qp.py` / `nlp.py` — the user-facing builders: oracle /
  derivative-factory assembly and the `SolverDescriptor`.
- `src/alloy/solvers/_oracle.py` — `collect_free_inputs(exprs)` walks the
  expression graph and returns the unique `Ops.INPUT` exprs in deterministic
  name/id order.
- `src/alloy/solvers/solver_function.py` — `SolverFunction` opaque wrapper
  (a real `Function` whose outputs are `SOLVER_CALL` nodes); `__call__`
  dispatches through `jit.CompiledFunction` and refreshes
  `last_stats`/`last_status`.
- `src/alloy/solvers/registry.py` — entry-point discovery and the
  `SolverBackend` plugin protocol: packaging metadata (`name`, `kind`,
  `protocol_version`, `lib_stem`, `link_flags`, `header`, `lib_dir()`,
  `include_dir()`) plus the `render_wrapper(fun, ctx)` codegen hook. The
  full contract is documented in [`solver_plugins.md`](solver_plugins.md).
- `src/alloy/codegen/solver_c.py` — the codegen orchestration: it frames each
  plugin-rendered wrapper body with the alloy-owned stats storage/accessor
  and exposes the backend/include/link-flag queries the JIT and AOT
  consumers use.
- `plugins/alloy-piqp` / `plugins/alloy-ipopt` / `plugins/alloy-sqp` —
  vendored `libpiqpc` / `libipopt` + headers (SQP reuses PIQP's C library)
  and the
  per-solver C wrapper templates (`alloy_piqp/codegen.py`,
  `alloy_ipopt/codegen.py`, `alloy_sqp/codegen.py`). The SQP plugin supports
  exact Lagrangian and objective-Hessian modes and accepts native Alloy or
  externally generated C-ABI oracles.

### Sign and ordering conventions

- `lam_ineq`, `lam_box` are signed: positive ⇒ upper bound active, negative ⇒
  lower bound active. Both solvers report nonneg `z_l`/`z_u` separately
  internally; the conversion happens in the generated wrapper.
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

**Both solvers supported; the generated wrapper is the only solve path for
`al.qp` and `al.nlp`.** A SolverFunction renders to a self-contained C
function that drives PIQP (dense or sparse QP) or IPOPT (NLP) and fills the
`alloy_solver_stats` struct on every solve.

For PIQP, the generated wrapper calls the rendered oracle to fill QP data,
then either transposes `P`/`A`/`G` to PIQP's column-major layout (dense) or
hands compact CSC-ordered value buffers to static baked pattern tables
(sparse, see above), lazily sets up a static `piqp_workspace`, and dispatches
`piqp_update_{dense,sparse}` + `piqp_solve` on every call.

For IPOPT, the generated source emits a static context struct (parameter
pointers, the caller's workspace `w`, FE-time and evaluation counters),
static `const int` arrays for the sparse Jacobian and lower-triangular
Hessian patterns, five static `eval_*` callbacks that bridge IPOPT into the
rendered `base`/`grad`/`jac`/`hess` kernels (each timed and counted, each
receiving the caller's `w` — required scratch space, not optional), and an
intermediate callback that records the iteration count. The wrapper body
computes bounds via the rendered `bounds_raw`, clamps them to IPOPT's ±2e19
infinity convention, calls `CreateIpoptProblem`, applies the descriptor's
options, seeds `mult_g` from `lam_eq0`/`lam_ineq0` and `mult_x_L`/`mult_x_U`
from the sign-split `lam_box0`, runs `IpoptSolve`, writes the seven NLP
outputs (`x`, `f`, `h_eq`, `g_ineq`, `lam_eq`, `lam_ineq`, `lam_box`), and
maps `ApplicationReturnStatus` onto the alloy status enum via the vendored
header's enum constants (upstream drift breaks at compile time). The Hessian
callback gathers the lower-triangle via a precomputed static index table.
Problem-creation and option failures are checked: a rejected option surfaces
as `ERROR` stats (native status `Invalid_Option` /
`Invalid_Problem_Definition`) with defined outputs (`x = x0`, zeros
elsewhere).

Wrapper state (context struct, data buffers, PIQP workspace, latest stats)
lives in per-symbol statics: distinct solvers are isolated even within one
translation unit, but concurrent calls into the *same* compiled solver from
multiple threads clobber each other — the generated wrappers are
single-threaded, matching the non-reentrancy contract in `spec.md`.

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
