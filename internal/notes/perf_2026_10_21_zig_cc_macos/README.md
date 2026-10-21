# Closed-loop timings with `zig cc` against Apple Clang on macOS, 2026-10-21

The macOS counterpart of `perf_2026_10_09_zig_cc` for #82, on a MacBook Pro (Mac16,6) with an
Apple M4 Max (12 performance and 4 efficiency cores), 48 GB of RAM, on mains power, macOS 27.0.1.
Scaly-oracle closed loops for race_cars, unbumpercars and npmpc with IPOPT and SQP, one fresh
process per cell. One arm compiled the JIT with Apple Clang 21.0.0 (`SCALY_CC=cc`), the other with
`zig cc` from `zig-bin` 0.17.0 (clang 22.1.8). Both use scalar libm, because vector libm is
glibc-only. The raw episodes are under `benchmarks/results/zig017-macos/` on that Mac, not in the
repository.

```bash
G="--problem race_cars,unbumpercars,npmpc --solver ipopt,sqp --oracle scaly --repetitions 1"
SCALY_CC=cc uv run benchmarks/run.py closed-loop $G --out-dir benchmarks/results/zig017-macos/appleclang
env -u SCALY_CC uv run benchmarks/run.py closed-loop $G --out-dir benchmarks/results/zig017-macos/zig
uv run internal/notes/perf_2026_10_21_zig_cc_macos/compare.py benchmarks/results/zig017-macos
uv run internal/notes/perf_2026_10_21_zig_cc_macos/first_compile.py
```

`compare.py` prints the table from the episodes' `telemetry.csv`, averaging every step after the
first, with `build_ms` from `summary.json`, and writes `ratios.png`. `first_compile.py` times the
first call of a small JIT function in five fresh processes each, with an empty `SCALY_CACHE_DIR`.

Mean time per step, in ms:

| Problem | Solver | Function evaluation, Apple Clang | zig cc | zig / Apple Clang | Solver, zig / Apple Clang |
|---|---|---:|---:|---:|---:|
| race_cars | ipopt | 0.158 | 0.157 | 0.995 | 0.988 |
| race_cars | sqp | 0.0416 | 0.0413 | 0.994 | 0.991 |
| unbumpercars | ipopt | 12.6 | 12.0 | 0.947 | 0.952 |
| unbumpercars | sqp | 7.40 | 7.11 | 0.960 | 0.963 |
| npmpc | ipopt | 0.401 | 0.404 | 1.007 | 1.029 |
| npmpc | sqp | 0.198 | 0.209 | 1.058 | 1.014 |

Iteration counts agree everywhere except unbumpercars SQP (7.97 against 7.92 per step). On this
Mac, unlike la015 against gcc, zig cc is within about 6% of the system compiler in every cell, and
5% faster on unbumpercars. Only npmpc SQP function evaluation is more than 5% slower, from one
process per arm. Builds take 0 to 25% longer with zig. A first call compiles in 2.0 s with an
empty zig cache, 0.44 s with a warm one and 0.26 s with Apple Clang.
