# Interpolation benchmarks, 2026-09-28

Studies behind the interp PR reports (`../interp_sp*_report.html`, todo C-170 onward; plan
`../interp_plan_2026_09_28.md`). Run each from the repository root. Timings come from an Apple M3 Max
with Apple clang 21 at the JIT's flags (`-O2 -mcpu=native -fno-math-errno`). They are indicative, not
reference-machine results, and no page under `docs/` cites them. Every C-side timing uses
`../perf_2026_09_27_integrators/time_entry.c`: rounds interleaved, the fastest sample kept, values
checked against SciPy before timing.

| Script | Measures |
| --- | --- |
| `bench_spike.py` | SP0: a spline evaluation composed from existing ops (a prototype, not the library), one point per call, against CasADi 3.8's generated `interpolant` at each lookup mode: a 1-D cubic on 1 000 clustered knots (value and derivative), a 2-D bicubic on 64 x 64 (value and gradient), a 3-D trilinear on 20 x 20 x 20 (value). The kill criterion's inputs: time ratio, C lines, lowering time |
| `bench_eval.py` | SP1 onward: `interp.interpolant` itself, one point per call, value and derivative (gradient in n-D), every `search` and `strategy` an axis allows and the one `"auto"` picks, against CasADi's `interpolant` at each `lookup_mode`: 1-D linear and cubic at 32 and 1 024 sites, uniform and clustered; a 64 x 64 bicubic; a 20 x 20 x 20 trilinear |
| `bench_fit.py` | SP2 onward: what fitting costs when the graph is built: `interpolant` per kind at 100 to 1e6 sites against SciPy's constructors, the first traced evaluation (the per-cell tables), `smoothing` against `make_smoothing_spline` |
| `bench_param.py` | SP3 onward: the Jacobian in the coefficients at symbolic points, through `at()` and in CasADi's inlined `bspline`; the in-graph cubic fit as a dense map against the tridiagonal scans |
| `bench_batch.py` | SP4: batches of 1e3 to 1e6 random points per call against CasADi's mapped `interpolant` and SciPy; per-cell polynomials against local bases on a table past the caches |
| `bench_codegen.py` | SP4: large tables (1e6 1-D, 256^2 2-D, 64^3 3-D): fitting, rendering and compile time and C size, against CasADi's code generator |
| `_harness.py` | building a Scaly or CasADi Function into a shared library at the JIT's flags, and timing it with `time_entry.c` |
| `results_2026_09_28.md` | the tables the reports quote |
