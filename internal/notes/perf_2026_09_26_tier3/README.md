# Tier 3 timings, 2026-09-26

Studies behind the Tier 3 reports (`../tier3_pr*_report.html`). Run each from the repository root
with `uv run internal/notes/perf_2026_09_26_tier3/<script>.py`. The numbers in the reports come
from an Apple M-series Mac with Apple clang 21 and are indicative, not reference-machine results;
no page under `docs/` cites them.

| Script | Measures |
| --- | --- |
| `t3_0_piqp_baseline.py` | vendored PIQP 0.6.2 on the stored Maros–Mészáros subset and three MPC sizes, sparse and dense backends: status, iterations, PIQP's own setup and solve times (median of 5). The baseline for the Scaly IPM |
