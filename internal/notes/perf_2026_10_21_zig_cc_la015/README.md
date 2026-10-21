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

## Where the gap came from: AVX-512 gathers

The Hessian sweep does not show the gap (g++ against clang++, N=8: unbumpercars 1.00, npmpc 1.02,
race_cars 1.00), so it lives in the closed loop's own module. `capture_sitecustomize.py` (on
`PYTHONPATH`, with `ZIGPROF_OUT` set) saves 30 mid-episode calls of the unbumpercars SQP solver.
`gen_replay.py` writes a C harness that replays them, and `oracle_bench.c`, appended to the
generated module, times each oracle in the same translation unit. `perf` needs privileges on la015,
so the profile came from gprof (`-pg`) on the clang build: 91% of oracle time sits in the per-car
network step and its derivatives, which are dense matrix-vector loops summed in order over the
input index.

`-O3`, `-fveclib=libmvec` and `-mprefer-vector-width=512` change nothing. LLVM vectorizes those
loops across output rows with `vgatherqpd` loads of the strided weights (311 gathers in clang 20's
module, 367 in upstream clang 22.1.8's, 263 in zig's), and gathers are slow on Zen 4. gcc keeps the
loops scalar and emits none. Apple silicon has no gathers, which is why the M4 Max run saw no gap.
`zig cc` accepts `-mno-gather` and ignores it, because it hands LLVM its own CPU feature list; it
needs `-mcpu=native+prefer_no_gather+prefer_no_scatter`.

Replay of the 30 calls, function evaluation per solve in ms:

| Compiler | Default | Gathers off |
|---|---:|---:|
| gcc 13.3 | 6.61 | |
| clang 20.1.8 | 8.84 | 6.56 |
| clang 22.1.8, upstream release | 10.25 | 6.77 |
| zig cc 0.17 (clang 22.1.8) | 10.30 | 6.63 |

Upstream clang 22 and zig agree, so zig's extra cost over clang 20 is LLVM 22's, not zig's
configuration. The JIT now passes `-mno-gather -mno-scatter` to clang on x86-64 and the feature
form to zig (`toolchain.no_gather_flags`).

Closed loops with that fix, one fresh process per cell, means per step after the first in ms:

| Problem | Solver | Function evaluation, gcc | zig cc | zig / gcc | Solver step, gcc | zig cc | zig / gcc |
|---|---|---:|---:|---:|---:|---:|---:|
| race_cars | ipopt | 0.185 | 0.162 | 0.87 | 5.43 | 5.38 | 0.99 |
| race_cars | sqp | 0.0542 | 0.0483 | 0.89 | 3.21 | 3.16 | 0.99 |
| unbumpercars | ipopt | 16.7 | 16.5 | 0.99 | 20.6 | 20.5 | 1.00 |
| unbumpercars | sqp | 9.70 | 9.54 | 0.98 | 11.0 | 10.8 | 0.99 |
| npmpc | ipopt | 0.943 | 0.763 | 0.81 | 3.50 | 3.37 | 0.96 |
| npmpc | sqp | 0.446 | 0.354 | 0.79 | 1.21 | 1.12 | 0.93 |

Iteration counts agree (unbumpercars SQP 7.910 against 7.925). Storing each layer's weights
transposed would let every compiler vectorize these loops with plain loads and the same summation
order; that belongs to #45 and #71.
