# Closed-loop timings with `zig cc` against gcc, 2026-10-09

The measurement for #82 on la015: Scaly-oracle closed loops for race_cars, unbumpercars and npmpc
with IPOPT and SQP, five fresh processes per cell, `SCALY_VECTOR_LIBM=glibc`, performance governor,
boost off. One arm compiled the JIT with gcc 13.3 (`SCALY_CC=gcc`), the other with `zig cc` from
`ziglang` 0.16.0 (clang 21.1.0). A three-process diagnostic arm used the system clang 20.1.8 on
unbumpercars and npmpc. The raw episodes are under `benchmarks/results/zig-cc-2026-10-09/` and
`benchmarks/results/zig-cc-2026-10-09-diag/` on la015, not in the repository.

`compare.py` prints the gcc and zig table from the episodes' `telemetry.csv`, averaging every step
after the first, and `build_ms` from `summary.json`:

```bash
uv run internal/notes/perf_2026_10_09_zig_cc/compare.py benchmarks/results/zig-cc-2026-10-09
```

Mean function-evaluation time per step, in ms:

| Problem | Solver | gcc 13.3 | clang 20.1 | zig cc (clang 21.1) | zig / gcc |
|---|---|---:|---:|---:|---:|
| race_cars | ipopt | 0.145 | | 0.126 | 0.87 |
| race_cars | sqp | 0.0413 | | 0.0373 | 0.90 |
| unbumpercars | ipopt | 16.6 | 21.8 | 25.5 | 1.54 |
| unbumpercars | sqp | 9.49 | 12.8 | 14.9 | 1.57 |
| npmpc | ipopt | 0.957 | 1.02 | 1.10 | 1.16 |
| npmpc | sqp | 0.455 | 0.488 | 0.526 | 1.16 |

Every coefficient of variation is at most 2.3%, and iteration counts agree between arms (unbumpercars
SQP 7.91 against 7.96). The unbumpercars regression is mostly Clang against GCC, with zig's
clang 21 another 16% slower than clang 20. Builds are 3 to 26% faster with zig. A cold zig cache
adds about 2.4 s to the first compilation on the machine; afterwards a first call compiles in
0.11 s against 0.05 s with gcc.
