# October 14 study

The first study with the current sweep format: compiled kernel sizes from the object, constant data
in source and binary, kernels compiled as C, a single GEMM encoding, and closed-loop cold starts
without the result-check oracles. It ran the harness of pull request #181 from a clean tree in
80 minutes, against 103 for the October 13 study.

`publish.py` audits the study (commands, CPU and numerical policy, cell outcomes, solver-step success
and paired-step coverage) and writes the five files under `docs/assets/benchmarks/` that the pages
read. NumPy's default percentile interpolation gives the paired ratios' 10th, 50th and 90th
percentiles.

```bash
uv run internal/notes/perf_2026_10_14/publish.py benchmarks/results/study-2026-10-14
```

Against the October 13 study, cell by cell:

- 116 sweep attempts: 87 timed, 12 compile timeouts, the rest skipped after a failure. The timeouts
  are October 13's, minus the four of the removed GEMM duplicates.
- Every generated source and compiled kernel has the same size.
- Kernel compiles as C took a median 0.97 of their C++ time, with a tenth of cells at 0.78 or less.
- Runtimes changed by a median factor of 1.000, with 80% of cells within 2%. Race-car N=500 Scaly
  measured 16% slower, but rerunning either study's binary gives 29 to 34 µs, so that cell is noisy
  rather than changed.
- Closed-loop iterations are identical and step times within 3%. Scaly cold starts fell by 0.2 to
  1.9 s, the compile of the result-check oracles. CasADi cold starts moved by at most 0.3 s.
