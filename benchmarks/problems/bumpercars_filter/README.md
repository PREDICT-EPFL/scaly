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
`bumper_car_simulator` and vendored at:

```text
benchmarks/problems/bumpercars_filter/data/ct_full_xlarge.pt
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
uv run python benchmarks/run.py closed-loop --problem bumpercars --backend alloy --smoke
uv run python benchmarks/run.py closed-loop --problem bumpercars --backend alloy
uv run python -m benchmarks.problems.bumpercars_filter.run_closed_loop --filter both --dump-alloy-c
```

Both write `episode.mcap`; import this directory's hand-authored
`foxglove-layout.json` in Foxglove Desktop to view it. The direct module remains
useful for side-by-side Alloy/CasADi runs and advanced filter options.

Common options:

```bash
--ncars 4
--horizon 80
--no-walls
--weights benchmarks/problems/bumpercars_filter/data/ct_full_xlarge.pt
--out-dir benchmarks/results/bumpercars_filter
--show
```

Outputs are written under `<out-dir>/<filter>/`:

```text
rollout.npz                  state/input trajectories
stats.csv                    per-step solver + oracle instrumentation
summary.json                 aggregate metrics and implementation metadata
episode.mcap                 Foxglove telemetry and planar car scene (see below)
representative_fe_inputs.npz closed-loop oracle input for FE benchmarks
config/provenance/metadata   reproducibility data
trajectories.png             matplotlib trajectory plot
performance.png              solve/evaluation timing plot
alloy_c/                     generated C kernels when --dump-alloy-c is used with Alloy
```

Generated outputs belong under `benchmarks/results/`.

### Scene geometry

The 3D scene draws what the filter actually constrains, at the scale the filter
implies. Each car gets its own colour and shows:

- a body box `lf + lr + 0.33` m long and 0.9 m wide, centred `(lf - lr) / 2` ahead
  of the state position (the state tracks the centre of gravity, not the geometric
  centre);
- the keep-out circle of radius `safety_radius / 2 = 0.95` m, which is what the pair
  barrier `‖pᵢ - pⱼ‖² ≥ safety_radius²` enforces and which circumscribes the body;
- two arrows from the car, pointing along `theta + u[1] * max_delta` with length
  scaled by the throttle `u[0]`: the desired input in the car's colour and the
  applied input in black. A failed solve brakes (`u = [-1, 0]`), which shows up as a
  black arrow flipped to point backwards.

The arena is drawn twice: the walls, and the rectangle inset by `wall_margin`, which
is the region the wall barriers keep the car centres inside.

## Implementations

### CasADi

`CasadiDTCBFSafetyFilter` builds one MX NLP with IPOPT. By default it uses
CasADi `expand=True` and IPOPT's limited-memory Hessian approximation:

```text
ipopt.hessian_approximation = limited-memory
```

This matches the Alloy prototype's default Hessian mode. Passing
`--exact-hessian` removes that option and additionally constructs an exact
Lagrangian Hessian function for instrumentation.

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
cost(z, bar_x, u_des, weights, physics, dt)
g(z, bar_x, u_des, weights, physics, dt)
```

The RK4 neural dynamics are evaluated with `al.map_` over the car axis, so the
prototype exercises the mapped neural dynamics path we care about. The filter
then creates Alloy factories for:

- `grad:cost:z`,
- `spjac:g:z`,
- `sphess:gamma:z:z` with `gamma = lam:cost * cost + dot(lam:g, g)` when
  `--exact-hessian` is selected.

The whole solve is one `al.nlp(...)` SolverFunction: the derivative factories
above are built inside `al.nlp`, and the filter runs through the generated C
solver wrapper (single `.so` driving `IpStdCInterface.h` with generated
kernels — no Python/ctypes callbacks in the loop). Warm starts carry the
primal iterate plus the constraint and box multipliers between steps
(`lam_ineq0`/`lam_box0`). With `--dump-alloy-c`, the full solver module
(wrapper + kernels) is rendered to inspectable C files.

Instrumentation recorded per step (from the `alloy_solver_stats` struct):

- total solve time with the FE / solver / glue split, status (alloy + native),
  and iteration count,
- objective, min constraint value, slack,
- per-oracle-function evaluation counts,
- Alloy build/JIT-compile timings,
- sparse Jacobian nnz and lower-triangular Hessian nnz.

## Interpreting the Alloy vs CasADi timings

Both implementations use IPOPT and warm-start primal variables plus constraint
and box multipliers. Alloy runs entirely through the generated C solver wrapper:
IPOPT callbacks call generated kernels in the same `.so`, with no Python or
ctypes callback in the solve loop. Its stable stats ABI separates FE, native
solver, and wrapper/glue time and records evaluation counts. CasADi reports its
own solver and callback counters, so the categories are close but not guaranteed
to have identical accounting boundaries.

The default comparison uses limited-memory Hessians on both sides;
`--exact-hessian` selects exact Lagrangian Hessians. Both paths receive the same
symbolic model constants and `dt`, and CasADi expansion is enabled unless
`--no-casadi-expand` is passed. Strict comparisons should retain raw IPOPT
status in addition to the benchmark's feasible max-iteration acceptance rule.

## Alloy features closed by this prototype

The original prototype exposed the following gaps; all are now closed on the
Alloy path.

### 1. Exact sparse Hessian through `Ops.MAP` (closed)

IPOPT's exact Hessian path would require the sparse Hessian of the Lagrangian
with respect to `z` through the mapped RK4 neural dynamics:

```text
sphess:lagrangian:z:z
```

The oracle deliberately uses `al.map_` to evaluate the per-car neural RK4 model.
Alloy now propagates reverse and sparse second-order AD through `Ops.MAP` while
preserving the compact mapped representation. The filter builds
`sphess:gamma:z:z`, passes its lower-triangular sparsity to IPOPT, and evaluates
it from IPOPT's objective factor and constraint multipliers.

### 2. Native generated-C solver path (closed)

The filter is an `al.nlp(...)` SolverFunction. Its generated C wrapper calls
`IpStdCInterface.h` directly, routes IPOPT callbacks to generated oracle kernels,
and fills Alloy's stable stats struct with FE/solver/glue timings.

### 3. IPOPT warm-start/status parity (closed)

The low-level Alloy IPOPT wrapper now exposes and accepts the same data we use
from CasADi:

- previous `lam_x`,
- previous `lam_g`,
- final multipliers,
- iteration count,
- value-callback counts.

The closed-loop Alloy filter reuses these multipliers after successful solves
and reports iteration and callback statistics alongside CasADi's measurements.

### 4. Callback accounting (closed)

The generated wrapper avoids Python callbacks entirely and reports FE, native
solver, and wrapper/glue timing through `alloy_solver_stats`.

### 5. Parameterized model constants (closed)

The Alloy and CasADi ODE/RK4 paths take physical constants and `dt` symbolically.
`CarPhysics` and `ClosedLoopConfig` defaults fill those values for normal runs.

## Next comparison matrix

Suggested next benchmarking pass:

1. CasADi MX baseline, limited-memory Hessian.
2. CasADi MX with `expand=False`, limited-memory Hessian.
3. CasADi exact Hessian.
4. Alloy generated-C solver path, limited-memory Hessian.
5. Alloy generated-C solver path with exact sparse Hessian.

For each row, record:

- build time,
- JIT/compile time,
- IPOPT status and iterations,
- solver wall time,
- function/Jacobian/Hessian evaluation time,
- callback counts,
- final objective, slack, min constraint,
- trajectory-level min distance and tracking cost.
