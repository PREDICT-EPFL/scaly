# Post-179 study analysis

`compare.py` audits the complete study at `38bd5896` and compares its summaries with
the October 12 summaries retained at main's `d9755969`. It checks grid coverage,
command results, CPU and numerical policy, solver-step success and paired-step
coverage before producing `analysis.md`. The full report is retained there too.

```bash
uv run internal/notes/perf_2026_10_13/compare.py benchmarks/results/study-2026-10-13-post-179
```

`--publish` copies this study's sweep and closed-loop summaries into
`docs/assets/benchmarks/` and derives `closed_loop_speedup.summary.csv` from the
report's paired ratios. NumPy's default percentile interpolation computes the
10th, 50th and 90th percentiles. No timings from another study enter those files.

The comparison is observational. The fresh process does not estimate measurement
noise, and the CasADi encoding changes prevent attributing closed-loop changes to
Scaly's compiler alone. The protocol and comparison history record the conclusions.
