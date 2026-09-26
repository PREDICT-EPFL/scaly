# Tier 1 primitive timings, 2026-09-25

Wall-clock studies behind the Tier 1 reports (`../tier1_pr*_report.html`). Each script builds the
functions it compares, times the first call (render, compile, load) and the median of repeated
calls through the JIT, and prints a table. The numbers in the reports were taken on an aarch64
Linux VM with gcc and the JIT's default flags; they are indicative, not reference-machine results,
and no page under `docs/` cites them.

| Script | Compares |
| --- | --- |
| `pr1_select.py` | `where` against `fmax`/`fmin` and NumPy, single and nested |
| `pr2_reductions.py` | `reduce_max`, `norm_inf` and their tie-aware gradients against `sum` and NumPy |
| `pr3_sparse.py` | SpMV from `gather` and `segment_sum` against dense `matmul`, SciPy and NumPy; `gather` adjoint build cost before and after |
| `pr4_scan.py` | an RK4 rollout and its gradient as `scan` against the unrolled chain of calls |
| `pr5_while.py` | a Newton solve as `while_loop` against a fixed-length `scan`, with and without gradient |
| `pr6_inplace.py` | a scan updating 4 entries of a large carry in place against the two-slot carry |
| `pr7_custom.py` | the gradient of a solve through its steps against an implicit rule from `sc.custom_derivative` |
| `pr8_multiseed.py` | dense Hessians through a scan with one tangent loop for all seeds against one per seed (`mpc`/`rk4`, `new`/`old`, sizes) |

Run from the repository root, with a fresh cache so first-call times include compilation:

```bash
SCALY_CACHE_DIR=$(mktemp -d) PYTHONPATH=internal/notes/perf_2026_09_25_tier1 uv run internal/notes/perf_2026_09_25_tier1/pr4_scan.py
```
