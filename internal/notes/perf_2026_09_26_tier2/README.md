# Tier 2 timings, 2026-09-26

Wall-clock studies behind the Tier 2 reports (`../tier2_pr*_report.html`). Each script builds the
functions it compares, times the first call (render, compile, load) and the median of repeated
calls through the JIT, and prints a table. The numbers in the reports were taken on an aarch64
Linux VM with gcc and the JIT's default flags; they are indicative, not reference-machine results,
and no page under `docs/` cites them.

| Script | Compares |
| --- | --- |
| `pr1_index.py` | a time-varying RK4 rollout with the step number as a body input (`index`) against a stored float time table (`table`) and time as an extra carry entry (`carry`): value, gradient, Hessian |
| `pr2_take.py` | `A x` and `A^T y` with run-time column indices (`take`, a row scan with `take`, a row scan accumulating with `put_add`) against baked-in indices (`gather` + `segment_sum`) and SciPy; `scan-put` is also the T2-3 in-place case |
| `pr4_sparse_matrix.py` | `SparseMatrix` KKT assembly, `K @ z` and `A^T A` through the JIT against SciPy in Python |
| `pr5_symbolic.py` | symbolic analysis per matrix and ordering: fill against SuperLU, update lanes, tree height, supernodes, segments and analysis time |

Run from this directory, with a fresh cache so first-call times include compilation:

```bash
SCALY_CACHE_DIR=$(mktemp -d) uv run python pr1_index.py 50 400
```
