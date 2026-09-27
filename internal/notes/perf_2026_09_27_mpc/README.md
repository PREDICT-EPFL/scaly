# MPC benchmarks, 2026-09-27

Studies behind the MPC PR reports (`../mpc_m*_report.html`, todo API-147 onward). Run each from the
repository root. Timings come from an Apple M3 Max with Apple clang 21 at the JIT's flags. They are
indicative, not reference-machine results, and no page under `docs/` cites them.

| Script | Measures |
| --- | --- |
| `bench_ocp.py` | M1: the cart-pole OCP built by `scaly.mpc` against the same NLP written by hand (and, for context, with collocation and pseudospectral transcriptions), solved by IPOPT from one cold start: iterations, IPOPT's own time, the Python call, the build |
| `results_2026_09_27.md` | the tables the reports quote |
