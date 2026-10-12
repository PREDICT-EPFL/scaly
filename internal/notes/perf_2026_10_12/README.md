# Changes since 0.1.0a1, 2026-10-12

The study `benchmarks/results/study-2026-10-12-post-0.1.0a1` reran the frozen grid at `1c955a07` on
the reference machine, with the same toolchain, CPU policy and vector-math policy as the September 23
libmvec study. The September 23 raw files are gone, so the baseline is the summary files in
`docs/assets/benchmarks/`, which keep means and coefficients of variation only.

`uv run internal/notes/perf_2026_10_12/compare.py <study-dir>` prints the Scaly Hessian means and
the closed-loop totals and function-evaluation times of both studies side by side. The results and
their reading are in [the comparison history](../benchmark_comparison_history.md).
