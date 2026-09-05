# Alloy scalability sweep

The per-cell measurements behind [the results overview](index.md), which is the page to read first
if you want the summary rather than the tables.

**Sections marked *historical* describe formulations that no longer exist.** They are kept because
the measurements were real and the reasoning is still useful, not because the numbers describe
alloy today. Everything else is current.

Runs `benchmarks/run.py sweep` over a fixed cell grid for each workload, capturing per-cell codegen / compile / runtime / source-size metrics. Each cell compiles its own Google Benchmark binary that includes Alloy + the selected backend. The binary scatters the compact result into a dense matrix and compares it with an independent reference before it records a timing.

The unsuffixed workloads measure the exact sparse Lagrangian Hessian from the solver descriptor.
The `_jac` workloads retain the constraint Jacobian rows for the long paper and do not run by default.

Skip rules applied automatically:

- per-cell compile timeout (default 180 s);
- max generated source size (default 50 MB) — skip without compiling;
- after a backend hits any of the above at one size, larger sizes for that backend are skipped immediately, because both generated source size and compile cost are monotonically increasing in the iteration count.

CSV with the raw cell data: `benchmarks/results/sweep/scalability.csv`. Each cell's generated code, samples, binary, and logs live beside it under `benchmarks/results/sweep/repeat_<n>/<workload>/<backend>_<axis><size>/`.

`coloring_width` is an Alloy-owned construction metric, not a cross-backend comparison. It counts
the compressed tangent directions that Alloy executes. Structured Jacobian rows add the independently
executed per-formal or local batches, while Hessian rows use the global star-color count. CasADi
leaves the field blank because its generated code exposes no internal derivative count.

Both workloads route through Alloy's VMAP-aware path where the structure allows it — fully for the race car, per car but not per pair for unbumpercars, whose pair rows are still built by a Python loop (see that section for what this costs) — and the codegen spills lifetime-packed slots ≥ 1024 doubles to the `w[]` workspace so very large intermediate buffers no longer overflow the C stack.

## Why a C++ harness (and not the Google Benchmark Python bindings)

Every cell in the sweep codegens C, compiles a focused Google Benchmark binary, and calls the generated symbol from a tight C++ loop (`for (auto _ : state) fn(arg, res, ...)`). There is **zero Python in the timed region** — this is on purpose. The numbers above are a *codegen-quality* comparison: how fast is the generated C, with both backends measured identically.

The obvious simplification is to drop the per-cell C++ compile and instead drive [the google-benchmark Python bindings](https://pypi.org/project/google-benchmark/) (`@gb.register` + `while state:`) over `Function.__call__` and CasADi's `caf(DM)`. We measured whether that's viable:

- **The binding's own loop floor is negligible.** An empty `while state:` body benchmarks at ~14 ns/iter; a trivial Python call at ~24 ns. So the bindings do *not* add meaningful overhead on their own — whatever you call inside the loop is what you measure.
- **Per-call dispatch is the catch, and it is not symmetric between backends.** Anchoring against the race-car C-level numbers above (mean over the loop, M-series, default args):

  | N | Alloy C kernel | Alloy `cf.run()` (Python) | CasADi SX C kernel | CasADi `caf(DM)` (Python) |
  |---:|---:|---:|---:|---:|
  | 5 | 0.53 µs | 3.9 µs | 0.48 µs | 5.2 µs |
  | 100 | 10.3 µs | 15.1 µs | 9.3 µs | **79.5 µs** |

  Alloy's ctypes dispatch (`jit.py::CompiledFunction.run`: `np.asarray` inputs, allocate outputs + workspace, build pointer arrays, one FFI call, reshape) is a roughly **fixed ~3.3 µs floor** — its *relative* weight shrinks as the kernel grows (+47 % at N=100). CasADi's `caf(DM)` overhead instead **scales with nnz**, because it materializes a sparse `DM` return each call (≈ +70 µs at N=100). `caf(np, np)` is worse still (~120 µs at N=100) due to input conversion.

**Conclusion — use the right harness per question:**

- **Codegen-quality / scalability-vs-CasADi (these tables): keep the C++ harness.** A Python-level benchmark would report Alloy ~5× faster than CasADi SX at N=100 (15 vs 80 µs) when the generated code is actually within ~10 % (10.3 vs 9.3 µs). The 8× distortion is pure binding overhead, so the C++ harness is load-bearing here, not overhead-paranoia.
- **Alloy's own end-to-end Python latency, dispatch budgeting, and per-commit regression tracking: the Google Benchmark Python bindings are a great fit** — no per-cell compile, no 180 s timeouts, no source-size caps, and they measure the *realistic* cost paid when Alloy runs inside a Python solver loop. As a bonus they surface a genuinely favorable (and true) axis the C-only tables hide: Alloy's end-to-end Python dispatch is far lighter than CasADi's (15 vs 80 µs at N=100). The ~3.3 µs `cf.run()` floor is itself worth optimizing (preallocate workspace/outputs, cache the ctypes pointer arrays).

A minimal worked example used to live in `test_tracking_workload.py::test_tracking_eq_jac_python_gbench` (opt-in via `ALLOY_GBENCH=1`): it reproduced the equality-Jacobian cell through the Google Benchmark Python bindings, checked the result against the CasADi dense reference outside the timed loop, and recorded the dispatch time via `record_property`. It was dropped when the suite moved in-repo in `8b1dd3f`.

## Race-car equality Jacobian (`spjac:eq:z`) — historical

4-state, 2-control bicycle with slip-angle β=δ/2 and `tanh` rolling-resistance term, RK4 over the horizon. Decision vector size `(N+1)·6`, output size `(N+1)·4`. Dynamics:

```python
phi, v = x[2], x[3]
beta = 0.5 * delta
vx = v * cos(beta)
[v*cos(phi+beta), v*sin(phi+beta), v*sin(beta)/lr,
 (C_M0*throttle - (C_R0 + C_R1*vx + C_R2*vx*vx) * tanh(10*vx)) / M]
```

The Alloy fixture wrapped the interstage residual in a stage `Function` and assembled the equality
vector via `al.vmap(eq_interstage, length=N, ...)`. The current
`RaceCarConstraintJac` cell instead takes the full equality-plus-corridor Jacobian from the solver
descriptor. The table in this section predates that correction and remains an equality-only result
until the reference machine reruns the sweep.

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

## Neural-process MPC on the Furuta pendulum (`npmpc`)

A conditional-neural-process decoder — `9 → 32 → 32 → 2`, sigmoid, weights and latent code read out
of the parameter tail — evaluated at **every node of a prediction horizon**. This is the one workload
in the suite where a dense matmul sits inside the VMAP stage body, so it is the one that separates
loop-preserving lowering from scalar expansion most sharply. Formulation, vendored data and
closed-loop numbers live with the problem, in `benchmarks/problems/npmpc/README.md`; this section is
the sweep.

Both axes are gated per cell against a dense reference before any timing is recorded, and both
backends compile at `-O3` with the same compiler.

### Equality Jacobian (`spjac:eq:z`), horizon axis — historical

| N | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | SX/alloy | MX/alloy |
|---|---|---|---|---|---|---|---|---|
| 6 | **417** | 80 111 | 2 471 | 17 087 | 26 552 | 15 299 | 1.55 | 0.90 |
| 12 | **417** | 158 646 | 4 294 | 34 291 | 66 382 | 28 610 | 1.94 | 0.83 |
| 25 | **417** | 328 807 | 8 243 | 71 850 | 213 096 | 64 810 | 2.97 | 0.90 |
| 50 | **417** | 656 038 | 15 837 | 143 016 | 637 475 | 158 745 | 4.46 | 1.11 |
| 100 | **417** | 1 310 500 | 31 024 | 287 899 | *compile > 900 s* | 288 986 | — | 1.00 |
| 200 | **417** | *skipped* | 61 400 | 553 644 | — | *compile > 900 s* | — | — |

### Exact Lagrangian Hessian (`sphess:gamma:z`), horizon axis

| N | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | alloy compile | MX compile |
|---|---|---|---|---|---|---|---|---|
| 6 | **1041** | 293 452 | 4 924 | 31 969 | 62 620 | 26 512 | 0.8 s | 5 s |
| 12 | **1041** | 584 683 | 8 834 | 64 358 | 136 167 | 52 956 | 0.8 s | 13 s |
| 25 | **1041** | 1 215 891 | 17 355 | 134 614 | 273 704 | 116 385 | 0.8 s | 47 s |
| 50 | **1041** | *53 MB > cap* | 33 688 | 273 575 | — | 237 982 | 0.8 s | **227 s** |
| 100 | **1041** | *skipped* | 66 358 | 542 518 | — | *compile > 900 s* | 0.9 s | — |
| 200 | **1041** | *skipped* | — | 1 067 826 | — | — | 0.9 s | — |

### Decoder-width axis at N = 12

Untrained weights at every width, including 32, so the axis stays homogeneous: kernel timing and
generated code size depend on the graph's shape rather than on the numbers in it. These are code-size
and timing cells only, never accuracy cells.

Equality Jacobian — historical:

| W | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | MX/alloy |
|---|---|---|---|---|---|---|---|
| 16 | **417** | 45 496 | 4 156 | 10 054 | 9 078 | 8 457 | 0.84 |
| 32 | **417** | 158 646 | 4 294 | 33 201 | 52 663 | 27 023 | 0.81 |
| 64 | **417** | 590 768 | 4 762 | 145 176 | *compile > 900 s* | 126 174 | 0.87 |
| 128 | **417** | *skipped* | 6 466 | 597 719 | — | 815 594 | **1.36** |
| 256 | **417** | *skipped* | 12 946 | 2 890 248 | — | 7 539 505 | **2.61** |

Exact Lagrangian Hessian:

| W | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | MX/alloy |
|---|---|---|---|---|---|---|---|
| 16 | **1043** | 160 550 | 8 696 | 20 981 | 30 382 | 16 736 | 0.80 |
| 32 | **1041** | 584 612 | 8 834 | 65 064 | 131 291 | 52 576 | 0.81 |
| 64 | **1041** | *55 MB > cap* | 9 302 | 271 564 | — | 240 323 | 0.88 |
| 128 | **1041** | *skipped* | 11 006 | 1 284 817 | — | 1 382 204 | **1.08** |
| 256 | **1041** | *skipped* | 17 486 | 7 186 310 | — | 12 468 110 | **1.73** |

Reading, in descending order of confidence:

- **The code-size and compile-time result is unambiguous and large.** Alloy's source is 417 lines for
  the Jacobian at *every* point on *both* axes, and 1041 for the Hessian at every point but W = 16
  (1043 there), because the weights are read
  out of the parameter tail rather than baked in as literals and the stage body uses VMAP rather than
  unrolled. SX reaches 1.31 million lines at N = 100 — a factor of 3142 — and stops being compilable
  at all: clang exceeds a 15-minute budget there, and MX joins it at N = 200 for the Jacobian and
  N = 100 for the Hessian, where it already needs 227 s at N = 50. Alloy compiles the N = 200 Hessian
  in 0.94 s.
- **Against SX the runtime advantage is real**, and on the Jacobian it grows with the horizon: 1.55×
  at N = 6 to 4.46× at N = 50, after which SX drops out. On the Hessian it is flat instead, 1.96–2.12×
  over N = 6…25. The one cell where SX is ahead is the narrowest decoder, W = 16, at 0.90.
- **Against MX the horizon axis is a tie** — 0.82–1.11 with no trend — and the *width* axis is where
  alloy pulls ahead. Alloy's runtime grows 3.3×, 4.4×, 4.1×, 4.8× per doubling, which is the quadratic
  cost a matmul-dominated kernel should have. MX grows 891× over a 16× width increase against a ~256×
  quadratic expectation, so it crosses from 1.2× faster than alloy at the shipped width to 2.6×
  *slower* at W = 256. MX's source barely grows, because it keeps the matmuls as operations, so the
  blowup is in what its Jacobian does at runtime rather than in code size.

Alloy is *not* scalar-expanding these matmuls: the generated C contains real loop nests. The small
fixed handicap at narrow decoders and short horizons (0.80–0.90 across W = 16–64 and N = 6–25; by
N = 50 and N = 100 on the Jacobian axis it is gone, at 1.11 and 1.00) is ours, and the likeliest
cause is the AD mode — Alloy's `sparse_jacobian` colours columns only, and the per-stage block here is wider
than it is tall, which is the regime where a row-coloured or reverse sweep needs fewer passes. That
is a hypothesis with supporting structure, not a measured cause.

## Discrete-time HCBF safety filter (`unbumpercars`)

The Phase 5 driving workload: a centralized one-step CBF filter over `C` cars, with
`C(C-1)/2` hyperbolic pair rows plus four order-1 velocity wall rows per car, one L1 slack
per row, and a neural vehicle model inside the constraint. The formulation and its provenance live with the problem, in
`benchmarks/problems/unbumpercars/README.md`; this section is only the numbers.

> **Measurement date:** the tables below were recorded before the 2026-08-12 migration from
> position wall rows to order-1 velocity wall rows. They remain the latest backend-scaling
> measurements, but absolute oracle and solve times describe the older wall graph and need to
> be regenerated before being quoted for the current formulation.

**Two model choices matter for cost.** The default, `--filter-model dt`, predicts with the
natively discrete MLP; `--filter-model ct` predicts with an RK4 map of a
continuous-time network (`6 → 64 → 64 → 3`, SiLU, 4,803 weights, four evaluations per step for
the RK4 stages). The discrete MLP is `6 → 256 → 128 → 3`, smoothed ReLU, 35,075 weights, and
**one** evaluation per step because it is already a one-step map — 7.3x the weights but only
1.86x the multiply-accumulates.

Two different things are measured below, and they do not say the same thing: one isolated
kernel, and the whole oracle inside a real IPOPT loop.

### Isolated exact Lagrangian Hessian (`sphess:gamma:z:z`) — current

The current C=2/4/8 sweep uses the canonical discrete-time model and evaluates
the compact structurally sparse exact Lagrangian Hessian. Each cell checks
the reconstructed dense matrix against a CasADi MX reference before timing.
At C=8 it consumes the complete canonical closed-loop handoff: primal,
objective factor, constraint multipliers, state, desired input, model weights,
physics, and time step.

Measured 2026-08-20 on the current formulation (AMD Ryzen 9 7940HS, clang 20 at `-O3`,
single-threaded, both backends through the same Google Benchmark harness). CasADi SX is not run: it
was already past a 180 s compile budget at C=4 on the easier Jacobian kernel.

| C | backend | runtime | source | lines | workspace | compile |
|---:|---|---:|---:|---:|---:|---:|
| 2 | alloy | **654 µs** | 83.9 KB | 2 294 | 34 304 | 1.0 s |
| 2 | casadi_mx | 760 µs | 288.0 KB | 8 937 | 214 251 | 1.4 s |
| 4 | alloy | **1 315 µs** | 164.6 KB | 4 015 | 34 304 | 1.8 s |
| 4 | casadi_mx | 2 333 µs | 619.3 KB | 19 851 | 356 642 | 4.3 s |
| 8 | alloy | **2 681 µs** | 471.8 KB | 10 559 | 36 736 | 7.8 s |
| 8 | casadi_mx | 8 135 µs | 2 005.6 KB | 62 466 | 641 811 | 33.6 s |

**This is the suite's cleanest compiled-against-compiled oracle win, and it grows with the car
count**: 1.16× at C=2, 1.77× at C=4, **3.03× at C=8**, against 3.4–4.3× less source, 4–17× less
workspace and 1.4–4.3× less compile time. Alloy's workspace is essentially flat (34 304 → 36 736
doubles over a 4× car count) because the per-car neural dynamics stay a loop; CasADi MX's triples.

Alloy's advantage growing in C while its own workspace does not is the VMAP Hessian doing what it
is for. Note that alloy's source still grows here, because the `C(C-1)/2` pair rows are built by a
Python loop rather than `al.vmap` — the one place in the suite where *we* write the code-size growth
that the code-size claim argues against (tracked internally).

This kernel is also the right anchor for reading the problem's closed-loop numbers. At C=8 the closed
loop is 93% function evaluation, and CasADi's interpreted MX oracle set costs 126 ms per solve
against alloy's 50 ms — so the ~2.6× there and the 3.03× here are the same effect, and compiling
CasADi's oracles would close only part of it. The [fairness audit](fairness.md#unbumpercars-c8-40-steps-exact-lagrangian-hessian) has the
per-configuration closed-loop table.

### Isolated constraint Jacobian (`jac:g:z`) — historical

Historical sweep cells, Apple M-series, `-O3`, single-threaded. These cells pin the continuous-time model
(`FilterConfig(model="ct")`) rather than following the default, so they stay comparable with the
numbers recorded before the default changed:

| C | backend | runtime | source | lines | compile |
|---:|---|---:|---:|---:|---:|
| 2 | alloy | 61.0 µs | 59 KB | 849 | 0.6 s |
| 2 | casadi_sx | 64.3 µs | 5.4 MB | 236,847 | 144 s |
| 2 | casadi_mx | 65.1 µs | 185 KB | 5,619 | 2.8 s |
| 4 | alloy | 128.8 µs | 81 KB | 1,407 | 0.7 s |
| 4 | casadi_sx | — | 11.8 MB | 470,243 | **>180 s, timeout** |
| 4 | casadi_mx | 131.4 µs | 460 KB | 14,487 | 6.4 s |
| 8 | alloy | 263.4 µs | 170 KB | 3,459 | 2.6 s |
| 8 | casadi_mx | 263.3 µs | 1.4 MB | 45,312 | 18.4 s |

On this kernel **Alloy and CasADi MX tie on runtime** at every size, to within 1.5%. The win is
in the artefacts around it: 3–8x smaller C, 4–7x faster to compile, and zero workspace against
MX's 55k–179k doubles. CasADi SX is not viable here at all — 5.4 MB and 144 s to compile at
`C=2`, and past the 180 s budget by `C=4`, so the sweep short-circuits the larger cells.

### Per-solve, inside IPOPT

The kernel above is not what a solve actually calls: the solve wants `f`, `g`, `grad_f`, a
*sparse* `jac_g` and an exact sparse Lagrangian Hessian, several times per iteration. Mean per
solve over a 60-step episode, plant matched to the filter's model, exact Hessians:

| filter model | C | Alloy | CasADi | Alloy speedup | IPOPT iters (both) |
|---|---:|---:|---:|---:|---:|
| ct | 2 | 3.13 ms | 8.65 ms | 2.8x | 5.5 |
| ct | 4 | 8.09 ms | 25.31 ms | 3.1x | 7.8 |
| ct | 8 | 21.69 ms | 84.26 ms | 3.9x | 10.6 |
| dt | 2 | 3.78 ms | 17.13 ms | 4.5x | 5.9 |
| dt | 4 | 10.63 ms | 63.91 ms | 6.0x | 9.2 |
| dt | 8 | **32.02 ms** | **289.10 ms** | **9.0x** | 14.0 |

Iteration counts are identical between the two backends in every cell, so IPOPT walks the same
path and only the oracle provider differs.

Reading:

- **Alloy's advantage grows along both axes** — with car count (2.8x → 3.9x for `ct`, 4.5x → 9.0x
  for `dt`) and with network size (3.9x → 9.0x at `C=8`). The bigger the constraint graph, the
  more the oracle provider matters.
- **Function evaluation is where the solve lives**: 30.0 of Alloy's 32.0 ms and 279.5 of CasADi's
  289.1 ms at the largest cell. So the gap is essentially all oracle, not solver.
- **The gap is not in the dense Jacobian**, which ties above. It is in the exact Lagrangian
  Hessian and the call path: Alloy's generated C wrapper calls the kernels directly, where CasADi
  re-enters its own machinery per call. Measured per-call at `C=8` on the `ct` model, CasADi's
  `hess_lag` alone is 3.7 ms against 0.2–0.8 ms for its other oracle outputs.
- **The heavier network costs Alloy 1.5x and CasADi 3.4x** per solve at `C=8` (21.7 → 32.0 ms
  against 84.3 → 289.1 ms), for 7.3x the weights.
- **Superlinear in `C` for both**, as expected — the pair rows grow as `C(C-1)/2` and the
  iteration count grows too: Alloy ≈2.6x (`ct`) and ≈2.8x (`dt`) per doubling of `C`, CasADi
  ≈3.1x and ≈3.9x.
- **Build cost** is the one place Alloy pays: 5.0 s against CasADi's 2.2 s at `C=8` on `ct`, and
  3.8 s against 4.3 s on `dt`. It is a once-per-configuration cost, and the `.so` is cached.

One caveat on the `dt`-versus-`ct` rows: each is measured against *its own* plant, so they are
two coherent configurations rather than a controlled A/B. Against the faithful (discrete) plant,
which is the default, the `dt` filter is both safer and *faster* than the `ct` one — 31.7 ms
against 37.8 ms — because the mismatched filter needs 19.3 iterations to the matched filter's
14.1. The problem README has that comparison.

### How to reproduce

```bash
# Current isolated exact-Hessian cells
uv run python benchmarks/run.py sweep --workloads unbumpercars --out /tmp/sweep.csv

# Per-solve (the second table), one cell per invocation
uv run python -m benchmarks.problems.unbumpercars.run_closed_loop \
  --solver ipopt --oracle both --filter-model dt --ncars 8 --steps 60
```

Canonical unified-runner episodes write under
`benchmarks/results/closed-loop/unbumpercars/<solver>+<oracle>/`; unified `--smoke`
episodes use `benchmarks/results/smoke/closed-loop/unbumpercars/<solver>+<oracle>/` so
they cannot replace the canonical handoff. The direct problem module writes to
the canonical location unless given another output directory.

## Unbumpercars inequality Jacobian (`spjac:ineq:u`) — historical

> **The workload measured here no longer exists.** Its implementation and
> the retired `test_unbumpercars_workload.py` were removed in `1b03820`
> (2026-08-10): the fixture was the only thing exercising gather-fed and chained VMAPs,
> and its two numeric tests had been silently skipping because the
> `model_kinematic_mlp.pth` checkpoint is not in the repo — so it read as coverage
> without being any. The pattern moved to
> `tests/integration/test_vmap.py::test_gather_fed_chained_vmaps_spjac_and_sphess_match_dense`,
> which runs unconditionally on small artificial cases.
>
> The current workload reuses the canonical `unbumpercars` ID at
> `benchmarks/problems/unbumpercars/`. It keeps `al.vmap` for
> the per-car neural dynamics but builds its `C(C-1)/2` pair rows with an unrolled Python
> loop, so the constant-LOC property below does **not** hold for it: its `spjac:g:z`
> kernel goes 845 → 1403 → 3455 lines for `C = 2 → 4 → 8`. Porting it back onto the
> gather-fed shape is tracked internally; the numbers below are what that
> port is expected to recover, and are kept for that reason.

Official-size MLP (`256 → 128 → 3` with the example `model_kinematic_mlp.pth` weights), RK4 pose update per car, pairwise C3BF + per-car wall residuals, slack column. Decision vector size `2C + 1`, constraint count `C(C-1)/2 + 4C` (quadratic in `C`).

`unbumpercars_ineq_function` was built as three `ExprOp.VMAP` nodes:

1. `dynamics_fn` mapped over `C` packed states / `C` packed inputs (with `pw` broadcast).
2. `pair_c3bf_fn` mapped over `C(C-1)/2` `(i,j)` pairs, fed by `al.gather` from two constant index tables — one per side of the pair, each built as `concatenate([arange(NSTATE) + k * NSTATE for k in bodies])` over the strict upper triangle, so a gather produces exactly the contiguous `NSTATE` block per iteration that a VMAP wants. The same two tables gather both the parameter states and the first VMAP's output, which is what makes it a VMAP → gather → VMAP chain.
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

- Alloy beats CasADi MX by a consistent **~2.5-2.7×** through C=16, narrowing to **1.88×** at C=32 — see the jump discussion below. Alloy's colored sparse Jacobian shares the per-car dynamics callee across colors and applies it inside a `for` loop, while MX clones the per-iteration graph through every JVP step (workspace and source both scale linearly in `C`).
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

Alloy beats the worktree's `anvil_ineq_jac` (which uses the same conceptual colored-sparse approach) by ~3.3-5.4× across the C=2..32 range, mostly thanks to the VMAP structure (loop-shaped per-car dynamics + pair C3BF), the sparse-constant matvec, scalar/vector inlining, the gather peephole, and the workspace spill that lets C=32 compile at all. CasADi MX's apparent disadvantage vs. its own worktree numbers is partly the heavier MLP fixture (official `256→128→3` weights vs. the worktree's simpler reduced MLP).

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
# Full exact-Hessian sweep with default cells
uv run benchmarks/run.py sweep --out benchmarks/results/sweep/scalability.csv

# Long-paper race-car Jacobian row
uv run benchmarks/run.py sweep --workloads race_cars_jac --out /tmp/race_cars-jac.csv

# Custom horizons / car counts / per-cell compile timeout
uv run benchmarks/run.py sweep \
    --workloads race_cars \
    --sizes 1,10,50,200 \
    --compile-timeout 60 \
    --out /tmp/quick.csv
```

Cells that hit the size cap or the per-cell compile timeout get `skipped_size` or `timeout` in
`compile_status`. A backend that does not apply to a workload gets `not_applicable`. After a backend
times out or exceeds the size cap, larger cells get `skipped_after_failure`. Runtime errors and parse
failures appear in `runtime_status`.

Race-car N=1000 used to appear in this table; it is dropped from the default cell grid because the bench-time dense reference (single-seed JVP × 6006 columns through the unrolled fixture) is the bottleneck rather than alloy itself — supply `--workloads race_cars --sizes 1000` to add it back when you're willing to wait several minutes.

## Continuous-time CBF safety filter — historical

> **The fixture measured here no longer exists.** Both
> the retired `test_safety_filter_workload.py` and
> `benchmarks/alloy_safety_filter_benchmark.py` are gone, so nothing below can be
> regenerated. It is kept because the input-affine-versus-fully-nonlinear reading still
> explains why the live workload is shaped the way it is: the dense `jac:ineq:u` blow-up on
> the nonlinear variant (4.4 ms at N=8 against an 89 µs forward) is exactly why
> `unbumpercars` calls `spjac`/`sphess` rather than dense factories. The successor is
> [Discrete-time HCBF safety filter](#discrete-time-hcbf-safety-filter-unbumpercars)
> above, whose per-solve numbers supersede these per-kernel ones. Note also that the
> Lagrangian-Hessian limitation this section records as blocking has since been closed —
> `sphess` through `ExprOp.VMAP` works and is what the live workload uses.

Fixture: the retired `test_safety_filter_workload.py` built the two variants of the retired
continuous-time HOCBF design study (both the fixture and that design are gone; see
`benchmarks/problems/unbumpercars/README.md` for the filter that exists):

- **Input-affine** — per-car velocity net `f_nn(x) + g_nn(x)·u` with a shared MLP body (`7 → 256 → 128`, SiLU) and two heads (drift 128→3, control 128→6). The constraint vector is the HOCBF residual `ḧ_ij + (γ1+γ2)·ḣ_ij + γ1·γ2·h_ij + s` over all `N(N-1)/2` pairs plus 4 wall residuals per car, with the slack term `s` shared. Cost is `Σ (u_i − u_des_i)^T Q (u_i − u_des_i) + M·s²`.
- **Fully nonlinear** — same shape but the velocity block is a single `f_nn(x, u)` MLP (`9 → 256 → 128 → 3`); the rest of the chain (pose kinematics, slack, HOCBF combination, cost) is unchanged.

The driver `benchmarks/alloy_safety_filter_benchmark.py` derives, for each `(ncars, variant)` cell, five single-output Alloy `Function`s — forward `ineq`, forward `cost`, dense `jac:ineq:u`, sparse `spjac:ineq:u`, and `grad:cost:u`. The sparse Lagrangian Hessian (`sphess:gamma:u:u`) is also requested but currently fails the `ExprOp.VMAP` reverse-mode path in `alloy.ad.reverse._local_vjp`, so it is caught and skipped per cell rather than working around the IR limitation here.

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
- **`spjac:ineq:u` recovers most of that loss** for the nonlinear case (`267 µs` at N=8, ~3× the forward instead of ~50×) because Alloy's column coloring reduces the seed count to the number of structurally distinct columns. This is the right object for an NLP solver loop to call per IPOPT iteration.
- The Lagrangian Hessian wrt u would be the other per-iteration object for the NLP path; it is the most natural next target once `ExprOp.VMAP` is added to the reverse-mode AD (`alloy/ad/reverse.py::_local_vjp`).

How these were reproduced, at the time. **None of these commands work now** —
`alloy_safety_filter_benchmark.py` was removed along with the formulation it measured, and is
recorded here only so the numbers above can be read in context:

```bash
# Default sweep: ncars=2,4,8 over both variants
uv run python benchmarks/alloy_safety_filter_benchmark.py --ncars 2 4 8 --clean

# Just one variant or one size
uv run python benchmarks/alloy_safety_filter_benchmark.py --variant affine --ncars 8

# Code + source size stats only (skip compile + run)
uv run python benchmarks/alloy_safety_filter_benchmark.py --ncars 2 4 8 --stats-only
```
