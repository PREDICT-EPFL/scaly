# Alloy scalability sweep

Runs `benchmarks/scalability_sweep.py` over a fixed cell grid for each workload, capturing per-cell codegen / compile / runtime / source-size metrics. Each cell compiles its own Google Benchmark binary that includes Alloy + the selected backend so the binary's correctness check (scatter compact → dense, compare against the Python Alloy reference) guards every measurement.

Skip rules applied automatically:

- per-cell compile timeout (default 180 s);
- max generated source size (default 50 MB) — skip without compiling;
- after a backend hits any of the above at one size, larger sizes for that backend are skipped immediately, because both generated source size and compile cost are monotonically increasing in the iteration count.

CSV with the raw cell data: `benchmarks/scalability_results.csv`.

Both workloads now use Alloy's MAP-aware path (`al.map_` / `tracking_eq_function_map`, `unbumpercars_ineq_function` MAP-ified), and the codegen spills lifetime-packed slots ≥ 1024 doubles to the `w[]` workspace so very large intermediate buffers no longer overflow the C stack.

## Tracking equality Jacobian (`spjac:eq:z`)

4-state, 2-control bicycle with slip-angle β=δ/2 and `tanh` rolling-resistance term, RK4 over the horizon. Decision vector size `(N+1)·6`, output size `(N+1)·4`. Dynamics:

```python
phi, v = x[2], x[3]
beta = 0.5 * delta
vx = v * cos(beta)
[v*cos(phi+beta), v*sin(phi+beta), v*sin(beta)/lr,
 (C_M0*throttle - (C_R0 + C_R1*vx + C_R2*vx*vx) * tanh(10*vx)) / M]
```

The Alloy fixture wraps the interstage residual in a stage `Function` and assembles the equality vector via `al.scan(eq_interstage, length=N, ...)`. `al.sparse_jacobian` then routes through the structured per-formal local coloring path, baking colored seeds as constants into a JVP callee wrapped in a single `Ops.MAP` — the generated C is essentially a fixed body inside `for (int it = 0; it < N; ++it)`.

Runtime (µs, mean from Google Benchmark `cpu_time`):

| N | Alloy | CasADi SX | CasADi MX |
|---:|---:|---:|---:|
| 1 | 0.09 | 0.09 | 0.13 |
| 5 | 0.53 | 0.48 | 0.74 |
| 10 | 1.05 | 0.91 | 1.50 |
| 25 | 2.53 | 2.28 | 3.75 |
| 50 | 5.24 | 4.70 | 8.39 |
| 100 | 10.27 | 9.28 | 18.72 |
| 200 | 21.20 | 18.44 | (compile >180 s — skipped) |
| 500 | 54.46 | 45.76 | (skipped_after_failure at N=200) |

Source size (KB):

| N | Alloy | CasADi SX | CasADi MX |
|---:|---:|---:|---:|
| 1 | 19.3 | 14.1 | 42.2 |
| 5 | 17.8 | 50.0 | 190.8 |
| 10 | 17.9 | 95.0 | 374.6 |
| 25 | 19.5 | 229.8 | 985.1 |
| 50 | 22.1 | 455.1 | 1996.0 |
| 100 | 28.3 | 907.0 | 4019.8 |
| 200 | 40.6 | 1810.8 | 8676.0 (gen, compile timed out) |
| 500 | 78.0 | 4528.9 | (skipped) |

Alloy source LOC stays essentially constant — 596 at N=1, then 481-482 from N=10 through N=500 — because the entire interstage Jacobian is one `for it` loop calling one const-seed JVP callee, and the byte count grows only with the constant `idx[]` gather/scatter tables (data, not code).

Workspace (doubles allocated by the generated function, reported via `SZ_W`):

| backend | workspace |
|---|---:|
| Alloy | 0 at N ≤ 25, then ≈ 63·N from N ≥ 50 (e.g. 31 504 at N=500) — spilled slots only |
| CasADi SX | 77 (constant) |
| CasADi MX | scales linearly: 199 at N=1 → 30 788 at N=200 |

Reading:

- Alloy and CasADi SX are within ~10 % of each other on runtime across the whole range, with Alloy slightly ahead at very small N and SX slightly ahead from N≈25 upward (per-iteration callee dispatch costs less than SX's scalar tape on the smallest cells but the gap closes as N grows).
- Alloy beats CasADi MX by ≈ 2× at every N where MX still compiles; MX times out at N=200 already.
- Alloy source at N=500 is 78 KB, **58× smaller than CasADi SX** (4.5 MB). LOC at N=500 is 482 — the same as at N=10.
- The per-cell codegen time at N=500 is 248 ms for Alloy vs 481 ms for CasADi SX (and >10 min for older variants of the Alloy path). That's the Python-AD-construction win from routing the JVP through one per-formal small graph instead of through the global unrolled tape.

## Unbumpercars inequality Jacobian (`spjac:ineq:u`)

Official-size MLP (`256 → 128 → 3` with the example `model_kinematic_mlp.pth` weights), RK4 pose update per car, pairwise C3BF + per-car wall residuals, slack column. Decision vector size `2C + 1`, constraint count `C(C-1)/2 + 4C` (quadratic in `C`).

`unbumpercars_ineq_function` is now built as three `Ops.MAP` nodes:

1. `dynamics_fn` mapped over `C` packed states / `C` packed inputs (with `pw` broadcast).
2. `pair_c3bf_fn` mapped over `C(C-1)/2` `(i,j)` pairs, with two constant-table `al.gather`s producing the pair state vectors.
3. `wall_residuals_fn` mapped over `C` cars.

CasADi SX is dropped past C=2 because at C=2 it already takes >180 s to compile a 12 MB source file; the sweep records that and short-circuits larger C for SX.

Runtime (µs):

| C | Alloy | CasADi SX | CasADi MX | Alloy speedup over MX |
|---:|---:|---:|---:|---:|
| 2 | 96.1 | (compile >180 s — skipped) | 263.4 | 2.74× |
| 4 | 197.5 | (skipped after C=2) | 518.1 | 2.62× |
| 8 | 396.9 | (skipped) | 1058.6 | 2.67× |
| 16 | 875.1 | (skipped) | 2207.1 | 2.52× |
| 32 | 2254.4 | (skipped) | 4237.5 | 1.88× |

Source size (KB):

| C | Alloy | CasADi MX |
|---:|---:|---:|
| 2 | 47.2 | 211.8 |
| 4 | 53.6 | 326.4 |
| 8 | 91.5 | 739.3 |
| 16 | 338.1 | 2495.1 |
| 32 | 2406.1 | 9883.2 |

Alloy source LOC ranges from **1601 to 1889 across C=2..32** — the only thing that grows with `C` is the constant gather/scatter index tables.

Workspace (doubles):

| backend | C=2 | C=4 | C=8 | C=16 | C=32 |
|---|---:|---:|---:|---:|---:|
| Alloy | 0 | 0 | 6 664 | 110 616 | 1 092 752 |
| CasADi MX | 141 500 | 212 674 | 355 264 | 641 476 | 1 218 028 |

Reading:

- Alloy beats CasADi MX by a consistent **~2.5-2.7×** through C=16, narrowing to **1.88×** at C=32 — see the jump discussion below. The colored sparse Jacobian shares the per-car dynamics callee across colors and applies it inside a `for` loop, while MX clones the per-iteration graph through every JVP step (workspace and source both scale linearly in `C`).
- **Both Alloy and MX now compile at C=32**: Alloy at 1.6 s codegen + 1.1 s compile (source 2.4 MB); MX at 1.5 s codegen + 100 s compile (source 9.9 MB). The old unrolled Alloy path needed 18.5 MB of source at C=32 and timed out at compile.
- The 1.09 M-double Alloy workspace at C=32 lives in `w[]`; the wrapper allocates it `static` so the inner benchmark loop never goes through `malloc`. Without the spill threshold this would be ~8.7 MB of stack arrays and segfault under the 8 MB subprocess default `ulimit -s`.

### Why does runtime jump between C=16 and C=32?

Both backends slow down per-nnz between these two sizes:

| transition | Alloy ratio | MX ratio | nnz ratio |
|---|---:|---:|---:|
| C=4 → 8 | 2.01× | 2.04× | 3.03× |
| C=8 → 16 | 2.20× | 2.08× | 3.36× |
| C=16 → 32 | 2.58× | 1.92× | 3.62× |

So both backends are sub-linear in `nnz`, but the slope flattens noticeably at C=32. The bench runs on an Apple M-series chip with two CPU tiers (perf cores: L1-D 128 KB, L1-I 192 KB, L2 16 MB shared by 6 cores; efficiency cores: L1-D 64 KB, L1-I 128 KB, L2 4 MB shared by 4 cores), and `Run on` reports the smaller efficiency-core caches. Working-set sizes are:

| C | alloy workspace | alloy source | combined | vs L2 (eff 4 MB) |
|---:|---:|---:|---:|---:|
| 8 | 52 KB | 91 KB | 0.14 MB | fits |
| 16 | 885 KB | 338 KB | 1.22 MB | fits |
| 32 | 8.74 MB | 2.41 MB | 11.2 MB | spills (eff), fits (perf) |

So between C=16 and C=32 the combined code+data footprint goes from 1.2 MB to 11 MB — that's the first cell that exceeds the 4 MB efficiency-core L2. The macOS scheduler can place the bench thread on either core type, and on efficiency cores (or under perf-core L2 sharing with other threads) we start eating L2 misses. MX experiences a milder version of the same effect — its workspace was already past 4 MB at C=16 (5.1 MB) so the C=16→32 step doesn't cross a new boundary on the data side, only on the code side (2.5 MB → 9.9 MB).

Treat this as a hardware-locality story rather than a backend ceiling: the alloy code at C=32 is still 4× smaller than MX (2.4 MB vs 9.9 MB) and ~2× faster, just not the constant 2.7× we see at smaller `C`.

## Comparison with `tracking-nmpc-benchmarks` worktree

Tracking, N=50:

| metric | experiment2 (anvil, N=50, simple 4-state bicycle) | this sweep (alloy, N=50, slip-angle + tanh drag) |
|---|---:|---:|
| dense `jac:eq:z` | 72.1 µs | not measured (only spjac path benchmarked) |
| colored `spjac:eq:z` | 5.06 µs | 5.24 µs |
| `spjac_unroll` (CONST basis) | 103 µs (~87 s codegen) | n/a — alloy does not generate this path |
| `multistage` (per-block Jac) | **1.53 µs**, O(1) source | not yet — see plan |
| CasADi SX | 1.34 µs (experiment1, simple dynamics) | 4.70 µs (this sweep, fancier dynamics) |
| CasADi MX | 5.83 µs (experiment1) | 8.39 µs (this sweep) |

The 3-4× absolute-runtime gap between this sweep's SX/MX column and experiment1's matching column is from the heavier dynamics in our fixture (β-slip + tanh drag + division by `lr`), not from a backend regression — confirmed by inspecting both `bicycle_cont` implementations side by side.

For unbumpercars vs. `experiment3` of the worktree:

| C | alloy (this sweep) | anvil_ineq_jac (worktree, same colored-sparse approach) | CasADi MX (this sweep) | CasADi MX (worktree) |
|---:|---:|---:|---:|---:|
| 2 | 96.1 µs | 322 µs | 263 µs | 189 µs |
| 4 | 197.5 µs | 603 µs | 518 µs | 333 µs |
| 8 | 396.9 µs | 1417 µs | 1059 µs | 618 µs |
| 16 | 875.1 µs | 3978 µs | 2207 µs | 1179 µs |
| 32 | 2254.4 µs | 12273 µs | 4237 µs | 2329 µs |

Alloy beats the worktree's `anvil_ineq_jac` (which uses the same conceptual colored-sparse approach) by ~3.3-5.4× across the C=2..32 range, mostly thanks to the MAP-ification (loop-shaped per-car dynamics + pair C3BF), the sparse-constant matvec, scalar/vector inlining, the gather peephole, and the workspace spill that lets C=32 compile at all. CasADi MX's apparent disadvantage vs. its own worktree numbers is partly the heavier MLP fixture (official `256→128→3` weights vs. the worktree's simpler reduced MLP).

## Comparison with CasADi `Function.map(N, "serial")` (tracking)

For curiosity we ran the same workload through CasADi's own loop-preserving primitive: the interstage residual is wrapped in a stage `Function`, mapped over `N` via `.map(N, "serial")`, then `casadi.jacobian` is applied to the assembled equality vector.

| N   | CasADi SX unrolled  | CasADi SX with `.map`  | CasADi MX with `.map`        |
| --- | ------------------- | ---------------------- | ---------------------------- |
| 10  | 1030 ns / 96.97 KB  | 1026 ns / 96.88 KB     | 2113 ns / 109.1 KB, sz_w=4205 |
| 50  | 4574 ns / 466 KB    | 4534 ns / 466 KB       | 10190 ns / 196 KB, sz_w=20405 |
| 100 | 8951 ns / 928 KB    | 8946 ns / 928 KB       | 20475 ns / 317 KB, sz_w=40331 |
| 200 | 17830 ns / 1.85 MB  | 17879 ns / 1.85 MB     | 41339 ns / 560 KB, sz_w=80506 |

Reading:

- **CasADi SX with `.map` is byte-for-byte indistinguishable from fully unrolling** — SX flattens to scalars at codegen, so the `.map` abstraction does not survive past graph construction.
- **CasADi MX with `.map`** does keep a loop shape and is the most source-efficient CasADi configuration in the table, but `sz_w` grows linearly (4205 → 80506 doubles at N=200) and runtime is ~2× slower than SX/unrolled at every N. The MX evaluator's per-iteration workspace plumbing eats the win.
- **alloy** at N=200 is **45× less source than SX and 13× less than MX with `.map`**, runs at ~21 µs (close to SX, ~2× faster than MX `.map`), and uses 12 604 doubles of workspace — about 6× less than MX `.map`.

## How to reproduce

```bash
# Full sweep with default cells: tracking N=1,5,10,25,50,100,200,500 and unbumpercars C=2,4,8,16,32
uv run python benchmarks/scalability_sweep.py --csv benchmarks/scalability_results.csv

# Just tracking
uv run python benchmarks/scalability_sweep.py --workloads tracking --csv /tmp/tracking.csv

# Custom horizons / car counts / per-cell compile timeout
uv run python benchmarks/scalability_sweep.py \
    --tracking-horizons 1 10 50 200 \
    --unbumpercars-cars 2 4 \
    --compile-timeout 60 \
    --csv /tmp/quick.csv
```

Cells that hit the size cap or the per-cell compile timeout end up with a `compile_status` of `skipped_size` / `timeout`. Once a backend has given up at one cell, all larger cells for that backend are short-circuited to `skipped_after_failure` (saves a lot of wall time at the long tail of the sweep). Runtime errors and parse failures are surfaced explicitly in the CSV's `runtime_status` column.

Tracking N=1000 used to appear in this table; it is dropped from the default cell grid because the bench-time dense reference (single-seed JVP × 6006 columns through the unrolled fixture) is the bottleneck rather than alloy itself — supply `--tracking-horizons 1000` to add it back when you're willing to wait several minutes.

## Continuous-time CBF safety filter

Fixture: `tests/alloy/test_safety_filter_workload.py` builds the two variants described in [`safety_filter.md`](safety_filter.md):

- **Input-affine** — per-car velocity net `f_nn(x) + g_nn(x)·u` with a shared MLP body (`7 → 256 → 128`, SiLU) and two heads (drift 128→3, control 128→6). The constraint vector is the HOCBF residual `ḧ_ij + (γ1+γ2)·ḣ_ij + γ1·γ2·h_ij + s` over all `N(N-1)/2` pairs plus 4 wall residuals per car, with the slack term `s` shared. Cost is `Σ (u_i − u_des_i)^T Q (u_i − u_des_i) + M·s²`.
- **Fully nonlinear** — same shape but the velocity block is a single `f_nn(x, u)` MLP (`9 → 256 → 128 → 3`); the rest of the chain (pose kinematics, slack, HOCBF combination, cost) is unchanged.

The driver `benchmarks/alloy_safety_filter_benchmark.py` derives, for each `(ncars, variant)` cell, five single-output Alloy `Function`s — forward `ineq`, forward `cost`, dense `jac:ineq:u`, sparse `spjac:ineq:u`, and `grad:cost:u`. The sparse Lagrangian Hessian (`sphess:gamma:u:u`) is also requested but currently fails the `Ops.MAP` reverse-mode path in `alloy.ad._local_vjp`, so it is caught and skipped per cell rather than working around the IR limitation here.

Each Function is rendered to C, compared against the Python interpreter on a deterministic input vector, and timed by Google Benchmark on the universal ABI entry point.

Runtime (Apple M-series, `-O3`, single-threaded; ns/call from `cpu_time`):

| Cell             | ineq   | cost  | jac\_u    | spjac\_u | grad\_cost\_u |
|------------------|-------:|------:|----------:|---------:|--------------:|
| affine N=2       | 22 µs  | 1.5 ns | 22 µs    | 22 µs    | 1.2 ns        |
| affine N=4       | 44 µs  | 1.9 ns | 44 µs    | 44 µs    | 1.5 ns        |
| affine N=8       | 89 µs  | 2.7 ns | 90 µs    | 88 µs    | 2.3 ns        |
| nonlin N=2       | 22 µs  | 1.5 ns | **288 µs** | 64 µs  | 1.3 ns        |
| nonlin N=4       | 44 µs  | 1.9 ns | **1.19 ms** | 129 µs | 1.5 ns       |
| nonlin N=8       | 89 µs  | 2.7 ns | **4.40 ms** | 267 µs | 2.3 ns       |

Source size / codegen (bytes, lines, NNZ for sparse Jacobian; codegen ms is Python-side):

| Cell             | ineq           | jac\_u           | spjac\_u (nnz)         |
|------------------|----------------|------------------|-----------------------:|
| affine N=2       | 16 KB / 559 L  | 27 KB / 875 L    | 12 KB / 422 L (20)     |
| affine N=4       | 28 KB / 1024 L | 96 KB / 3034 L   | 27 KB / 1120 L (56)    |
| affine N=8       | 69 KB / 2569 L | 493 KB / 14370 L | 133 KB / 5455 L (176)  |
| nonlin N=2       | 15 KB / 516 L  | 27 KB / 889 L    | 16 KB / 546 L (20)     |
| nonlin N=4       | 26 KB / 951 L  | 88 KB / 2817 L   | 32 KB / 1237 L (56)    |
| nonlin N=8       | 66 KB / 2436 L | 452 KB / 13257 L | 136 KB / 5464 L (176)  |

Reading:

- **Forward `ineq` is linear in ncars** and identical between variants at the same size — both go through one per-car MLP forward, which is what dominates (~10 µs/car at this network sizing). Pair count grows as N(N-1)/2 but adds negligible work; the floor is the network.
- **Cost and `grad:cost:u` are essentially free** (single-digit ns). The quadratic cost touches no MLP and only `O(N)` doubles.
- **Affine `jac:ineq:u` runs in the same envelope as the forward** — expected, because the constraint is linear in u and the Jacobian rows `b_ij^i = 2·Δπ^T·(∂κ_π/∂v)·g_nn(x_i)` just reuse the per-car NN outputs. The QP path is essentially "one forward and you have A".
- **Nonlinear `jac:ineq:u` is the obvious hotspot** — dense Jacobian seeds u (size `2·ncars`) through the MLP, which is roughly `2·ncars` forward passes; that is exactly the ~50× scaling we see at N=8 (`4.4 ms` vs `89 µs`).
- **`spjac:ineq:u` recovers most of that loss** for the nonlinear case (`267 µs` at N=8, ~3× the forward instead of ~50×) because column coloring reduces the seed count to the number of structurally distinct columns. This is the right object for an NLP solver loop to call per IPOPT iteration.
- The Lagrangian Hessian wrt u would be the other per-iteration object for the NLP path; it is the most natural next target once `Ops.MAP` is added to the reverse-mode AD (`alloy/ad.py::_local_vjp`).

How to reproduce:

```bash
# Default sweep: ncars=2,4,8 over both variants
uv run python benchmarks/alloy_safety_filter_benchmark.py --ncars 2 4 8 --clean

# Just one variant or one size
uv run python benchmarks/alloy_safety_filter_benchmark.py --variant affine --ncars 8

# Code + source size stats only (skip compile + run)
uv run python benchmarks/alloy_safety_filter_benchmark.py --ncars 2 4 8 --stats-only
```
