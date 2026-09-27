# Integrator benchmarks, 2026-09-27

Studies behind the integrator PR reports (`../integrators_i*_report.html`, todo API-142 onward). Run
each from the repository root. Timings come from an Apple M3 Max with Apple clang 21 at the JIT's
flags (`-O2 -mcpu=native -fno-math-errno`). They are indicative, not reference-machine results, and
no page under `docs/` cites them.

| Script | Measures |
| --- | --- |
| `bench_explicit.py` | I1: `si.rk4` against the hand-written scaly RK4 of `examples/nmpc_cartpole.py` and CasADi 3.8 (SX expanded, MX with a serial map; the faster is reported and named), per piece: the step, its Jacobians, the 40-stage shooting defects and their sparse Jacobian. Each cell is compiled to a shared library and timed in C by `time_entry.c`, rounds interleaved, fastest sample kept; the library's values are checked against the hand-written ones first |
| `time_entry.c` | the C driver: `../perf_2026_09_27_ipm_speed/time_entry.c` plus an integer workspace (CasADi MX needs one) and an inner repeat count, since macOS's clock ticks every 41.7 ns |
| `results_2026_09_27.md` | the tables the reports quote |
