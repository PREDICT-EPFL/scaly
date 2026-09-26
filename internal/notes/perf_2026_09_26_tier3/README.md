# Tier 3 timings, 2026-09-26

Studies behind the Tier 3 reports (`../tier3_pr*_report.html`). Run each from the repository root
with `uv run internal/notes/perf_2026_09_26_tier3/<script>.py`. The numbers in the reports come
from an Apple M-series Mac with Apple clang 21 and are indicative, not reference-machine results;
no page under `docs/` cites them.

| Script | Measures |
| --- | --- |
| `t3_0_piqp_baseline.py` | vendored PIQP 0.6.2 on the stored Maros–Mészáros subset and three MPC sizes, sparse and dense backends: status, iterations, PIQP's own setup and solve times (median of 5). The baseline for the Scaly IPM |
| `t3_0b_reference_gate.py` | the NumPy PIQP reference against vendored PIQP per problem (MM subset, infeasible set, MPC, random QPs and LPs): iterations with each backend and the reference, decision-trace matches against each backend, whether the backends agree (rounding sensitivity), the reference's wall time |
| `t3_1_while_params.py` | C-112: `SparseLDL` adaptive refinement (a `while_loop` whose factor, matrix and right-hand side became params) against the plain solve, on KKT systems of 75 to 1200 unknowns; run on two checkouts to compare |
| `t3_2_ruiz.py` | T3-2: generated Ruiz equilibration against the NumPy reference and PIQP's setup time, with first-call (render and compile) time |
| `t3_3_kkt.py` | T3-3: one generated factorization and two solves per backend against PIQP's time per iteration with the same backend, and first-call time |
| `t3_4_ipm.py` | T3-4/T3-5: the generated solver end to end per problem and backend against vendored PIQP with the same backend: status, iterations, the generated solve's time against PIQP's solve and setup + solve times, time per iteration, graph-build and first-call (cold compile with an empty `SCALY_CACHE_DIR`) times; geometric means per backend |
