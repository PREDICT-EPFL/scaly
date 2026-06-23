# CT RK4 discrete-time CBF safety filter prototype

This directory is a minimal, self-contained closed-loop prototype extracted from
`~/dev/bumper_car_simulator` for comparing a CasADi implementation against an
Alloy implementation of the same centralized safety filter.

The intended use is not to publish a polished controller here, but to give us a
fast iteration loop for:

- closed-loop behavior with the colleague's latest continuous-time neural model,
- solver/runtime instrumentation,
- inspection of Alloy-generated C kernels,
- identifying missing Alloy features that block a fair replacement of CasADi.

## Model and filter being tested

The vehicle model is the fully nonlinear continuous-time `CTFullModel` trained in
`bumper_car_simulator` and stored by default at:

```text
/Users/tudoroancea/dev/bumper_car_simulator/ct_full_xlarge.pt
```

No `torch` dependency is required. The checkpoint is read with
`alloy.utils.load_torch_state_dict`, then evaluated with plain NumPy for the
simulator and with CasADi / Alloy expressions inside the filters.

Per car state and input are:

```text
x = [px, py, theta, vf, beta_f, beta_r, delta]
u = [u_m, u_s]
```

The continuous-time model is discretized with RK4. The same RK4 discrete-time
model is used both:

1. inside the safety-filter prediction constraint, and
2. as the closed-loop simulator.

The safety filter is a centralized one-step discrete-time position CBF:

```text
h_ij(x[k+1]) >= (1 - gamma_pair) h_ij(x[k])
```

with:

```text
h_ij(x) = ||p_i - p_j||^2 - safety_radius^2
```

Optional wall constraints use the same one-step DTCBF form on simple signed
wall-distance barriers. The NLP decision is all car inputs plus one shared slack:

```text
z = [u_0, ..., u_{N-1}, slack]
```

The objective tracks the nominal input and penalizes slack.

This CT-model + RK4 + discrete-time position-CBF combination is deliberate: it
avoids the relative-degree mismatch that arises when applying a position-only CBF
directly to an already-discrete black-box model.

## Running

From the repository root:

```bash
uv run examples/ct_dt_cbf_filter/run_closed_loop.py --filter casadi
uv run examples/ct_dt_cbf_filter/run_closed_loop.py --filter alloy
uv run examples/ct_dt_cbf_filter/run_closed_loop.py --filter both --dump-alloy-c
```

Common options:

```bash
--ncars 4
--horizon 80
--no-walls
--weights /Users/tudoroancea/dev/bumper_car_simulator/ct_full_xlarge.pt
--out-dir examples/ct_dt_cbf_filter/out
--show
```

Outputs are written under `<out-dir>/<filter>/`:

```text
rollout.npz        state/input trajectories
stats.csv          per-step solver + oracle instrumentation
summary.json       aggregate metrics and implementation metadata
trajectories.png   matplotlib trajectory plot
performance.png    solve/evaluation timing plot
alloy_c/           generated C kernels when --dump-alloy-c is used with Alloy
```

`out/` is intentionally git-ignored.

## Implementations

### CasADi

`CasadiDTCBFSafetyFilter` builds one MX NLP with IPOPT. By default it uses
IPOPT's limited-memory Hessian approximation:

```text
ipopt.hessian_approximation = limited-memory
```

This matches the Alloy prototype's Hessian mode. Passing `--exact-hessian`
removes that option and additionally constructs an exact Lagrangian Hessian
function for instrumentation.

Instrumentation recorded per step:

- IPOPT wall time,
- IPOPT status and iteration count,
- objective, min constraint value, slack,
- explicit `ca.Function` timings for `f`, `g`, `grad_f`, `jac_g`, and optionally
  `hess_lag`,
- CasADi's reported `n_call_*` counters.

Important caveat: the explicit instrumentation functions are close to, but not
necessarily identical to, the exact internal oracle functions CasADi wires into
IPOPT.

### Alloy

`AlloyDTCBFSafetyFilter` builds an Alloy oracle with outputs:

```text
cost(z, bar_x, u_des, weights)
g(z, bar_x, u_des, weights)
```

The RK4 neural dynamics are evaluated with `al.map_` over the car axis, so the
prototype exercises the mapped neural dynamics path we care about. The filter
then creates Alloy factories for:

- `grad:cost:z`,
- `spjac:g:z`.

These functions are evaluated through Alloy's normal `Function.eval_list(...)`,
which JIT-compiles each kernel to C and caches it. With `--dump-alloy-c`, the
same kernels are rendered to inspectable C files.

The IPOPT solve currently uses Alloy's low-level native IPOPT binding
(`alloy.solvers._ipopt.solve_ipopt`) with Python/ctypes callbacks. Each callback
calls a JIT-compiled Alloy function. So the oracle math is compiled C, but the
whole solve is not yet a single generated-C solver wrapper.

Instrumentation recorded per step:

- IPOPT wall time and status,
- objective, min constraint value, slack,
- callback counts,
- average time spent inside each JIT-compiled Alloy oracle function,
- total time spent inside each oracle function over the solve,
- Alloy build/JIT-compile timings,
- sparse Jacobian nnz.

## Interpreting the Alloy vs CasADi timings

The current comparison is useful, but not fully apples-to-apples yet.

Known discrepancies:

1. **Callback path.** CasADi's callbacks are internal to CasADi/IPOPT. Alloy's
   prototype uses IPOPT -> C callback shim -> Python/ctypes -> JIT-compiled Alloy
   function. The reported Alloy per-function time measures mostly the compiled
   kernel call, not the full callback transition and pointer/copy overhead.
2. **Warm-starting.** CasADi currently warm-starts primal variables and IPOPT
   multipliers (`lam_x`, `lam_g`). Alloy only warm-starts the primal decision
   vector in this prototype.
3. **Iteration visibility.** CasADi reports IPOPT iteration count. Alloy's
   low-level IPOPT binding does not currently expose iteration count, so the
   CSV has an empty iteration column for Alloy.
4. **Hessian mode.** The default comparison uses limited-memory Hessian on both
   sides. Exact Hessian is available for CasADi via `--exact-hessian`, but not
   for the mapped Alloy oracle yet.
5. **Success handling.** Both implementations accept a finite constraint-feasible
   solution if IPOPT exits on max iterations. This is convenient for closed-loop
   experimentation, but strict benchmarking should also report the raw IPOPT
   status.
6. **Model constants.** Most physical constants match the bumper-car defaults.
   The current Alloy decorated ODE function bakes in the default physical
   constants. Changing physics parameters should be made symbolic/parameterized
   before doing a broad tuning sweep.
7. **CasADi `expand`.** The CasADi path does not currently set `expand=True`.
   That option may improve CasADi runtime at the cost of larger build time and
   should be part of the next benchmark matrix.

Because of (1), it is possible for Alloy's reported function-evaluation kernels
to be faster while total solve time is not proportionally better. Total solve
time also depends on IPOPT line search behavior, callback overhead, warm starts,
and iteration count.

## Missing Alloy features exposed by this prototype

These are the main Alloy-side gaps that currently prevent a completely fair and
production-quality comparison.

### 1. Exact sparse Hessian through `Ops.MAP`

IPOPT's exact Hessian path would require the sparse Hessian of the Lagrangian
with respect to `z` through the mapped RK4 neural dynamics:

```text
sphess:lagrangian:z:z
```

The current oracle deliberately uses `al.map_` to evaluate the per-car neural RK4
model. Alloy's second-order / sparse Hessian AD path does not yet fully support
reverse/second-order propagation through `Ops.MAP`.

Possible paths:

- short-term: unroll the map for small-N experiments and use the existing sparse
  Hessian machinery;
- medium-term: build per-car Jacobian/Hessian factories and manually assemble the
  global sparse Lagrangian Hessian for this structured filter;
- proper Alloy fix: implement AD rules and sparse second-order lowering for
  `Ops.MAP`, preserving the compact mapped representation.

### 2. Native generated-C solver path for this exact use case

Alloy can represent solver calls, but this prototype intentionally uses the
low-level IPOPT wrapper so we can inject and time separate oracle callbacks. That
means it does not yet exercise a monolithic generated-C safety-filter solve.

For a fair end-state comparison, we want either:

- a generated C solver wrapper that calls the generated oracle kernels without
  Python callback overhead, or
- enough low-level instrumentation in the generated solver path to inspect the
  same timing breakdown.

### 3. IPOPT warm-start/status parity

The low-level Alloy IPOPT wrapper should expose and accept the same data we use
from CasADi:

- previous `lam_x`,
- previous `lam_g`,
- final multipliers,
- iteration count,
- raw callback counts if available.

This would remove one source of solve-time discrepancy and make closed-loop
runtime stats much easier to interpret.

### 4. Lower-overhead callback accounting

The current Alloy instrumentation times Python calls to JIT-compiled functions.
It does not separate:

- IPOPT callback transition overhead,
- pointer-to-numpy copying,
- JIT kernel execution,
- numpy-to-pointer copying.

A better benchmark would record these layers separately or avoid Python callbacks
entirely.

### 5. Parameterized model constants

The Alloy ODE function should take physical constants / `dt` as parameters or be
rebuilt explicitly when they change. `dt` is already passed into the mapped RK4
function, but physical constants are still baked into the decorated function.

## Next comparison matrix

Suggested next benchmarking pass:

1. CasADi MX baseline, limited-memory Hessian.
2. CasADi MX with `expand=True`, limited-memory Hessian.
3. CasADi exact Hessian.
4. Alloy current JIT-kernel callback path, limited-memory Hessian.
5. Alloy with exact Hessian if/when `MAP` Hessian support or manual assembly is
   available.
6. Alloy generated-C solver path, once instrumentation is good enough.

For each row, record:

- build time,
- JIT/compile time,
- IPOPT status and iterations,
- solver wall time,
- function/Jacobian/Hessian evaluation time,
- callback counts,
- final objective, slack, min constraint,
- trajectory-level min distance and tracking cost.
