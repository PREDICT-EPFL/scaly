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
| `results_2026_09_28.md` | the tables the reports quote |
