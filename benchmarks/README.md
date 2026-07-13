# Benchmarks

This package owns Alloy's reproducible benchmark workloads and correctness-gated harness. Generated sources, binaries, logs, CSV files, and provenance live under `benchmarks/results/` and are not committed.

## Layout

- `problems/` contains importable chain-of-masses, tracking NMPC, unbumpercars, and bumpercars-filter formulations.
- `harness/` contains Google Benchmark wrapper generation, dense-reference checks, sweep mechanics, and provenance capture.
- `run.py` is the entry point for CI smoke gates and scalability sweeps.

## Smoke gates

```bash
uv run python benchmarks/run.py smoke
uv run python benchmarks/run.py smoke --skip solver_call
uv run python benchmarks/run.py smoke --select solver_call
```

Smoke checks Python and compiled-C Jacobians before recording runtime, enforces sparsity/workspace and loop-preservation invariants, and exercises the QP/IPOPT solver-call ABI when vendored solver libraries are present. The unbumpercars gate is reported as skipped when its optional checkpoint is absent. Any non-skipped failure produces a nonzero exit status.

## Sweeps

```bash
uv run python benchmarks/run.py sweep
uv run python benchmarks/run.py sweep --workloads tracking --sizes 1,5,10,50 --backends alloy,casadi_sx
uv run python benchmarks/run.py sweep --out benchmarks/results/my-sweep.csv
```

Each `(workload, size, backend)` cell retains its generated C/header, raw float64 samples, wrapper, binary, and compile log under `benchmarks/results/gen/`. Rows stream to CSV as cells finish; a sibling `.provenance.json` records the exact CLI, git state, package/compiler versions, platform, Python, and timestamp.

The doctrine is claims-first: broad sweeps establish scaling and canonical points support comparisons; correctness gates always run before speed is measured; every result carries enough provenance to reproduce it. See [ROADMAP.md §2](../ROADMAP.md#2-benchmark-suite) for the governing claim matrix.

## Problems

| problem | scaling axis | reference |
|---|---|---|
| chain of masses | number of masses | laopt/acados chain-mass formulation; `M=5` is the canonical point |
| tracking NMPC | horizon | CasADi SX/MX |
| unbumpercars | number of cars | CasADi SX/MX |
