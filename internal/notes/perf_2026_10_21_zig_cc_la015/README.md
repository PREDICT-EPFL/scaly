# Closed-loop timings with `zig cc` 0.17 on la015, 2026-10-21

The la015 half of the #82 measurement after the switch to Zig 0.17, at the branch revision that
includes main through #181. Scaly-oracle closed loops for race_cars, unbumpercars and npmpc with
IPOPT and SQP, one fresh process per cell (the protocol's rule since 2026-10-13),
`SCALY_VECTOR_LIBM=glibc`, performance governor, boost off. Three arms: gcc 13.3 (`SCALY_CC=gcc`),
`zig cc` from `zig-bin` 0.17.0 (clang 22.1.8), and the system clang 20.1.8 (`SCALY_CC=clang`). The
raw episodes are under `benchmarks/results/zig017-2026-10-21/` on la015, not in the repository.

```bash
uv run internal/notes/perf_2026_10_21_zig_cc_la015/compare.py benchmarks/results/zig017-2026-10-21
```

Mean function-evaluation time per step after the first, in ms:

| Problem | Solver | gcc 13.3 | clang 20.1 | zig cc 0.17 | zig / gcc | zig / clang 20 |
|---|---|---:|---:|---:|---:|---:|
| race_cars | ipopt | 0.185 | 0.172 | 0.165 | 0.89 | 0.96 |
| race_cars | sqp | 0.0552 | 0.0491 | 0.0466 | 0.84 | 0.95 |
| unbumpercars | ipopt | 16.7 | 21.7 | 25.3 | 1.52 | 1.17 |
| unbumpercars | sqp | 9.40 | 12.8 | 14.7 | 1.56 | 1.15 |
| npmpc | ipopt | 0.944 | 1.02 | 1.09 | 1.16 | 1.07 |
| npmpc | sqp | 0.451 | 0.480 | 0.522 | 1.16 | 1.09 |

The ratios match the zig 0.16 run of `perf_2026_10_09_zig_cc/` within a few percent, so Zig 0.17 and
main's codegen changes since then leave the gap in place. Iteration counts agree across arms
(unbumpercars SQP 7.910 for gcc, 7.925 for both clangs). On an M4 Max the same zig is within about
5% of Apple Clang 21 and faster on unbumpercars (`perf_2026_10_21_zig_cc_macos/`), so the
regression is specific to x86-64 against gcc, which this machine's earlier published timings use.

A first JIT call in a fresh process takes 2.85 s with an empty zig cache, 0.11 s with a warm one,
and 0.05 s with gcc.
