# Benchmarks

This package owns Alloy's reproducible benchmark workloads and correctness-gated harness. Generated sources, binaries, logs, CSV files, and provenance live under `benchmarks/results/` and are not committed.

## Layout

- `problems/` contains importable chain-of-masses, tracking NMPC, and bumpercars CT-DTCBF formulations. The old input-affine unbumpercars module is retained only as a compiler regression fixture and is not part of the benchmark runner.
- `harness/` contains Google Benchmark wrapper generation, dense-reference checks, sweep mechanics, and provenance capture.
- `run.py` is the entry point for CI smoke gates and scalability sweeps.

## Smoke gates

```bash
uv run python benchmarks/run.py smoke
uv run python benchmarks/run.py smoke --skip solver_call
uv run python benchmarks/run.py smoke --select solver_call
```

Smoke checks Python and compiled-C Jacobians before recording runtime, enforces sparsity/workspace and loop-preservation invariants, and exercises the QP/IPOPT solver-call ABI when vendored solver libraries are present. Any failure produces a nonzero exit status.

## Sweeps

```bash
uv run python benchmarks/run.py sweep
uv run python benchmarks/run.py sweep --workloads tracking --sizes 1,5,10,50 --backends alloy,casadi_sx
uv run python benchmarks/run.py sweep --out benchmarks/results/my-sweep.csv
```

Each `(workload, size, backend)` cell retains its generated C/header, raw float64 samples, wrapper, binary, and compile log under `benchmarks/results/gen/`. Rows stream to CSV as cells finish; a sibling `.provenance.json` records the exact CLI, git state, package/compiler versions, platform, Python, and timestamp. After canonical closed-loop runs, the chain M=5, tracking N=30, and bumpercars C=4 cells automatically consume their harvested `representative_fe_inputs.npz` rather than synthetic samples.

The doctrine is claims-first: broad sweeps establish scaling and canonical points support comparisons; correctness gates always run before speed is measured; every result carries enough provenance to reproduce it. See [ROADMAP.md §2](../ROADMAP.md#2-benchmark-suite) for the governing claim matrix.

## Closed-loop episodes and Foxglove

Run a short end-to-end episode, including the generated Alloy/IPOPT path and MCAP recording:

```bash
uv run python benchmarks/run.py closed-loop --problem tracking --smoke
```

Canonical runs omit `--smoke`:

```bash
uv run python benchmarks/run.py closed-loop --problem chain
uv run python benchmarks/run.py closed-loop --problem tracking
uv run python benchmarks/run.py closed-loop --problem bumpercars --backend alloy
```

Each run prints the exact artifact directory. It contains `episode.mcap`,
`rollout.npz`, configuration/summary/provenance JSON, and
`representative_fe_inputs.npz`. The MCAP includes `/tf` (`world → scene`),
`/scene`, control, state, and solver telemetry channels. Planar scenes are
centered through the transform and retain trajectory trails; the bumpercars scene
also contains a persistent arena boundary.

Layouts are **not** generated. Each problem keeps one hand-authored layout,
exported from Foxglove Desktop, next to its runner:

```text
benchmarks/problems/chain_of_masses/foxglove-layout.json
benchmarks/problems/tracking_nmpc/foxglove-layout.json
benchmarks/problems/bumpercars_filter/foxglove-layout.json
```

In Foxglove Desktop:

1. Open the Layouts menu and import the problem's `foxglove-layout.json`.
2. Select that layout.
3. Open `episode.mcap` with **Cmd/Ctrl+O**, or run
   `foxglove-studio /absolute/path/to/episode.mcap`.

To change a layout, edit it in Desktop, export it, and overwrite the checked-in
file. Only Desktop's own export is trusted here — the SDK's `foxglove.layouts`
builder emits a different `version`/`content` envelope, and an earlier
hand-rolled `configById` generator produced files Desktop refused to import.

Canonical operating points are deterministic and intentionally modest:

| problem | canonical point | scene |
|---|---|---|
| chain of masses | `M=5`, controller `N=12`, 20 plant steps at 0.2 s | 3D chain |
| tracking NMPC | controller `N=30`, 260 plant steps at 0.05 s | planar vehicle + reference, with lap progress |
| bumpercars CT-DTCBF | 4 cars, 80 plant steps at 0.1 s, seed 42 | planar cars |

The midpoint successful closed-loop oracle input is harvested for future
Google Benchmark cells. Closed-loop runs are manual-only; CI keeps using the
faster correctness and code-size smoke gates.

## Problems

| problem | scaling axis | reference |
|---|---|---|
| chain of masses | number of masses | laopt/acados chain-mass formulation; `M=5` is the canonical point |
| tracking NMPC | horizon | CasADi SX/MX; reference stages followed by symbolic vehicle parameters in `p` |
| bumpercars CT-DTCBF filter | number of cars | CasADi MX; neural weights followed by symbolic vehicle parameters and `dt` |
