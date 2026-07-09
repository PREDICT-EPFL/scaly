# Alloy benchmark-suite plan

This note summarizes the benchmark-suite discussion around `fast_benchmarks/`, FastBench, and how a paper-grade benchmark effort should relate to Alloy.

## Motivation

Alloy is scientifically interesting, but its current novelty is mostly an engineering/compiler/code-generation contribution rather than a new optimization algorithm. A paper therefore needs a comprehensive, fair, and reproducible benchmark suite showing where Alloy is better than, comparable to, or different from CasADi, acados, l4casadi/l4acados, FATROP, and QP/NLP solver baselines.

The benchmark suite should cover representative optimization-based-control workloads used by practitioners, while still preserving enough generated artifacts and raw traces to explain why one tool wins or loses.

## What is currently in `fast_benchmarks/`

`fast_benchmarks/FastBench/` is a prototype benchmark harness for closed-loop MPC/OCP comparisons. It already has useful infrastructure:

- a `Problem` interface with dynamics, costs, bounds, references, path constraints, plant simulation, initial-condition sampling, and LTI/QP metadata;
- solver adapters for CasADi+IPOPT, acados, OSQP, PIQP, ProxQP, do-mpc, GRAMPC, and an experimental C++ FastSQP;
- closed-loop simulation with identical episode seeds across solvers;
- control-quality, solve-quality, reliability, timing, and codegen metrics;
- JSON results and CSV/Markdown/HTML reports;
- a setup script and Dockerfile.

The 14 current problems are useful smoke/cross-validation cases, not yet a paper-grade suite:

| problem | category | role |
|---|---|---|
| `double_integrator` | LTI/QP | minimal QP sanity check |
| `dc_motor` | LTI/QP | classical position tracking |
| `rocket_landing_1d` | affine LTI/QP | simple powered-descent toy |
| `chain_mass` | LTI/QP | sparse banded QP cross-check |
| `oscillating_masses` | scalable LTI/QP | larger sparse QP stress test |
| `van_der_pol` | nonlinear OCP | small nonlinear regulation |
| `pendulum_swingup` | nonlinear/nonconvex | underactuated NMPC |
| `cstr` | nonlinear/stiff process | process-control example |
| `unicycle` | nonlinear/nonholonomic | NMPC discriminator |
| `kinematic_vehicle` | nonlinear tracking | simple vehicle tracking, overlaps with Alloy tracking NMPC |
| `vehicle_obstacle` | nonlinear path constraints | obstacle-avoidance NLP |
| `two_link_arm` | nonlinear manipulation | manipulator dynamics; for Alloy, write the 2x2 inverse explicitly |
| `quadruple_tank` | nonlinear MIMO process | classical process benchmark |
| `planar_quadrotor` | nonlinear aerospace | placeholder before full quadrotor |

`fast_benchmarks/REAL_BENCHMARK_CANDIDATES.md` identifies stronger future candidates:

1. agile quadrotor with learned aerodynamics;
2. BOPTEST building HVAC;
3. AWEbox airborne wind energy;
4. PMSM / electric-drive NMPC;
5. whole-body legged MPC;
6. OpenFAST + ROSCO wind turbine;
7. Tennessee Eastman process;
8. PGLib-OPF as a separate static NLP track.

The consensus is that trying to launch with all real candidates would look paper-shaped but likely be unmaintainable. We should start with a coherent core and add real flagship cases gradually.

## Repository strategy

The paper benchmark suite should live in a **separate repository from the beginning**.

Reasons:

- avoid polluting Alloy with heavy dependencies such as acados, FATROP, l4casadi, l4acados, PyTorch, GPyTorch, BOPTEST, OpenFAST, etc.;
- test Alloy through user-facing installed APIs, not internal source-tree imports;
- keep benchmark CI, Docker images, datasets, and artifact storage separate from Alloy's fast development CI;
- make the suite credible as a reproducible benchmark rather than merely an Alloy internal test folder.

Suggested split:

```text
alloy/
  benchmarks/                 # small internal regression/codegen benchmarks only

alloy-benchmarks/
  pyproject.toml
  uv.lock
  Dockerfile
  docker/
  problems/
  backends/
  harness/
  results/
  artifacts/
```

## Alloy dependency workflow in the benchmark repo

For fairness, CI and paper runs should use an installed Alloy wheel or a pinned Git revision, not editable source-tree imports.

### Default/paper mode

Use `uv.lock` plus either a versioned package, wheel URL, private package index, or pinned Git revision.

Example wheel URL source:

```toml
[project]
dependencies = ["alloy", "casadi", "numpy"]

[tool.uv.sources]
alloy = { url = "https://github.com/PREDICT-EPFL/alloy/releases/download/v0.3.0/alloy-0.3.0-py3-none-any.whl" }
```

Example Git pin:

```toml
[tool.uv.sources]
alloy = { git = "https://github.com/PREDICT-EPFL/alloy.git", rev = "abc123..." }
```

Then:

```bash
uv lock
uv sync --locked
```

### Local Alloy development mode

Keep the committed benchmark lockfile pinned, then overlay a local editable Alloy only for development:

```bash
cd alloy-benchmarks
uv sync --locked
uv pip install --reinstall --no-deps -e ../alloy
```

A future `benchctl` command can wrap this:

```bash
benchctl alloy use-local ../alloy
benchctl alloy use-wheel
benchctl alloy show
```

Every result should record Alloy install provenance:

```json
{
  "alloy_install": {
    "mode": "wheel|git|editable",
    "version": "...",
    "commit": "...",
    "path": "...",
    "wheel_url": "..."
  }
}
```

Editable Alloy results are useful for development, but not paper-grade unless the exact commit and environment are recorded.

## Environment and build system

Docker/locked environments should be the default, including CI.

Suggested profiles:

- `core`: Alloy wheel, CasADi, acados, FATROP, l4casadi, l4acados, PyTorch CPU, Google Benchmark, compilers;
- `full`: core plus external/heavy simulators such as BOPTEST, OpenFAST, large datasets, etc.;
- optional GPU profile later.

The current `FastBench/setup.sh` should eventually be replaced. Preferred direction:

- package the benchmark harness with Hatchling;
- keep Python dependencies in `pyproject.toml` + `uv.lock`;
- use Dockerfiles for full native provisioning;
- expose setup and diagnostics through a CLI, e.g. `benchctl`, rather than a raw shell script.

Example commands:

```bash
benchctl doctor
benchctl setup core
benchctl setup acados
benchctl setup l4casadi
benchctl run smoke
```

Avoid surprising users by building heavy native dependencies as an unconditional PEP 517 wheel-build side effect unless those artifacts are truly part of the benchmark package.

## CI strategy

Use a tiered CI setup.

### GitHub-hosted CI

Purpose: correctness and packaging only.

- build/sync the locked environment;
- build Docker image(s);
- run smoke problems;
- run tiny generated-code correctness checks;
- do not use hosted-runner timings as paper evidence.

### Self-hosted Linux CI

Purpose: real timing and regression tracking.

- run pinned Docker images;
- isolate CPU cores where possible;
- pin thread counts (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, etc.);
- fix CPU governor/turbo policy where possible;
- archive generated artifacts and raw traces.

Hosted runners are too noisy and weak for serious latency claims, especially compared with a stable self-hosted Linux machine.

## Benchmark tracks

Do not use one global leaderboard for all tools. Different tools have different deployment models and solver semantics. Split the suite into explicit tracks.

### 1. Native generated-code/kernel track

For backends that can produce native code or native callable artifacts:

- Alloy generated C;
- CasADi SX generated C;
- CasADi MX generated C;
- CasADi/FATROP generated functions where applicable;
- acados generated solvers where applicable.

Measure, with no Python in the timed region:

- dynamics/cost/constraint evaluations;
- Jacobians and Hessians;
- OCP residual/oracle assembly;
- generated source bytes and LOC;
- workspace size;
- compile time;
- median/p95/p99 runtime;
- correctness diff against a reference before timing.

Alloy's existing C++ Google Benchmark harness style should be reused.

### 2. Python deployment track

This track is not second-class: for some tools, especially l4acados and some l4casadi modes, Python is the realistic deployment path.

Measure:

- warmed-up solve/evaluation time;
- first-call/warmup time separately;
- Python overhead intentionally included;
- dependency/runtime footprint;
- CPU/GPU device;
- whether derivatives are exact, approximated, finite-difference, or locally approximated.

Relevant tools:

- l4acados;
- l4casadi main/libtorch mode;
- JAX/Torch/XLA-style Python runtimes if included;
- high-level Python MPC stacks.

### 3. Solver/oracle track

Measure complete optimization steps and solver-specific behavior:

- total solve time;
- KKT/primal/dual residuals where available;
- constraint violation;
- iteration counts;
- warm-start behavior;
- oracle evaluation time;
- status/failure modes.

### 4. Closed-loop control track

Measure controller quality and robustness:

- closed-loop cost;
- tracking RMSE;
- terminal error;
- settling time;
- control effort and input rate;
- success rate;
- deadline-miss rate;
- state/input/path-constraint violations;
- domain-specific metrics.

## Solvers and backends

Core backends/solvers to consider:

- Alloy;
- CasADi SX/MX;
- CasADi+IPOPT;
- CasADi+FATROP;
- acados SQP_RTI / SQP;
- l4casadi;
- l4acados.

QP subtrack:

- PIQP;
- OSQP;
- ProxQP.

Optional/non-core unless they become important:

- do-mpc;
- GRAMPC;
- JAX/Torch native runtimes;
- external simulator stacks.

FastBench's internal `fastsqp` is useful as a prototype/harness experiment, but should not be central to an Alloy paper unless it becomes part of the contribution.

## l4casadi and l4acados integration

Earlier phrasing that implied l4casadi/l4acados are ordinary self-contained codegen backends was too loose.

### l4casadi

- Integrates PyTorch models with CasADi, primarily through MX/external-function style workflows.
- Depends on PyTorch/libtorch machinery in the main path.
- Does not provide the same self-contained generated C story as CasADi SX or Alloy C.
- Has a limited naive mode for small MLPs that recreates supported PyTorch models as CasADi operations; this can be closer to pure generated code but is not general.
- Real-time l4casadi is complementary and Python-restricted, not a one-to-one codegen replacement.

### l4acados

- Integrates learned residual models into acados via a Python-facing controller object.
- Should be benchmarked in the Python deployment track.
- Not a standalone generated C deployment backend.

Useful l4casadi/l4acados benchmark cases should include more than agile quadrotor:

1. agile quadrotor with learned aerodynamic residual;
2. fully nonlinear neural safety filter;
3. unbumpercars / neural CBF workload;
4. learned residual racing model;
5. l4acados inverted-pendulum/GP residual as a small smoke case.

Record deployment mode for every backend:

```text
standalone_c
c_with_runtime_library
python_runtime
python_with_native_solver
python_with_torch_runtime
```

## JAX/Torch/XLA comparison

JAX, Torch, XLA, and related systems can be useful comparison points, but they do not have a single simple, universal, self-contained C OCP codegen path comparable to Alloy/CasADi generated C.

If included, they should usually be measured in the Python/native-runtime track, with clear accounting for:

- first compile time vs warm runtime;
- runtime dependencies;
- CPU/GPU device;
- export path limitations;
- whether the output is self-contained.

## FATROP

CasADi+FATROP should be a serious structured-OCP baseline.

Reasons:

- it targets structured OCPs;
- it has a CasADi interface;
- it can exploit manual OCP structure;
- it has a code-generation/linking story through CasADi/FATROP;
- it is a stronger structured-OCP comparison than generic IPOPT for many problems.

A future Alloy+FATROP path would be interesting if Alloy can generate the required structured callbacks/data.

## Fairness policy

Do **not** force every backend through a single neutral formulation if that prevents each tool from being used well.

Instead:

- maintain a canonical mathematical problem specification;
- allow backend-specific best-practice implementations;
- validate equivalence through numerical checks, common episodes, and quality gates.

Suggested problem layout:

```text
problems/cartpole/
  spec.yaml
  README.md
  numpy_reference.py
  alloy_impl.py
  casadi_sx_impl.py
  casadi_mx_impl.py
  acados_impl.py
  fatrop_impl.py
```

Fairness comes from:

- same documented math;
- same initial-condition/disturbance episodes;
- same success criteria;
- backend output correctness checks;
- quality gates before speed comparison;
- raw traces and generated artifacts for inspection;
- deployment-mode labels.

This is better than a generic `problem.to_alloy()` / `problem.to_casadi()` abstraction that may be suboptimal for every backend.

## Maintainability

Use:

- separate benchmark repo;
- Docker by default;
- `uv.lock`;
- Hatchling package + `benchctl` CLI;
- required core dependency profile plus optional heavy profiles;
- tiered CI;
- backend-specific implementations;
- stable result schema;
- per-cell artifact directories.

Core dependencies such as CasADi, acados, FATROP, l4casadi, and l4acados should be required in the core benchmark image. Heavy external simulators can remain optional plugins.

## Metrics and artifacts

Each benchmark cell should produce an artifact directory such as:

```text
artifacts/<problem>/<backend>/<config>/<seed>/
  manifest.json
  generated.c
  generated.h
  generated.o
  generated.so
  benchmark.cpp
  compile.log
  correctness.json
  metrics.json
  raw_trace.jsonl
```

Important metrics:

### Build/codegen metrics

- symbolic construction time;
- derivative construction time;
- sparsity/coloring time;
- code generation time;
- compile/link time;
- source/header bytes and LOC;
- object/shared-library size;
- workspace size;
- static/global memory where available.

### Runtime kernel metrics

- dynamics/cost/constraint runtime;
- Jacobian/Hessian runtime;
- OCP residual/oracle runtime;
- median/p95/p99/min/max;
- optional cycles/cache misses on self-hosted Linux.

### Solver metrics

- solve time;
- objective;
- KKT/primal/dual residuals where available;
- constraint violation;
- iterations;
- line-search or QP iteration details where available;
- status and failure reason.

### Closed-loop metrics

- cost;
- RMSE;
- terminal error;
- settling time;
- control effort;
- input rate;
- success rate;
- deadline misses;
- state/input/path-constraint violations;
- domain-specific metrics.

### Memory metrics

Be careful measuring Python peak memory: it includes the benchmark harness. Use isolated subprocesses and subtract/record a harness baseline where possible. Prefer explicit native workspace sizes and per-process peak RSS for codegen/compile/runtime.

Alloy's visualizer can be useful for internal debugging, but the paper should not depend on it.

## Problem roadmap

### Permanent smoke/sanity problems

Keep small problems for fast CI and cross-checking:

- double integrator;
- chain mass;
- Van der Pol;
- pendulum swing-up;
- two-link arm with explicit 2x2 inverse;
- maybe CSTR.

### First serious medium problems

- structured tracking NMPC based on the existing Alloy tracking formulation;
- vehicle obstacle/path constraints;
- neural/unbumpercars CBF workload;
- fully nonlinear neural safety filter;
- planar quadrotor only as an intermediate placeholder.

The simple FastBench kinematic vehicle is likely too close to the existing tracking NMPC benchmark; use the existing tracking formulation as the starting point because Alloy sparse-Jacobian optimizations already exploit it.

### Flagship upgrades

Prioritize:

1. racing MPCC with complex bicycle model, Pacejka tires, contouring/lag errors, track constraints, and later learned residuals;
2. fully nonlinear neural safety filter;
3. agile full quadrotor with learned aerodynamic residual;
4. possibly PMSM for embedded-speed/electric-drive coverage.

AWEbox and BOPTEST remain interesting, but racing MPCC is more aligned with Alloy's current direction and should be prioritized before those broader-domain examples.

## Alloy OCP builder direction

If implementing an Alloy multiple-shooting/OCP builder, it should not be a naive dense generic transcription. It should preserve the structured formulation used in the existing tracking NMPC benchmark:

- stage/interstage function boundaries;
- repeated stage maps/scans;
- sparse Jacobian construction over structured residuals;
- compact sparse outputs;
- no fully unrolled global expression unless explicitly benchmarking that mode.

This is essential because Alloy has already made sparse-Jacobian optimizations that leverage this formulation.

## Short revised answers

- **Different repo?** Yes, from the beginning.
- **Python or C/C++ driven?** Python orchestrated, Docker/uv locked; native C/C++ timing where the backend supports native deployment; Python timing where that is the realistic deployment mode.
- **How many solvers/backends?** Core: Alloy, CasADi SX/MX, CasADi+IPOPT, CasADi+FATROP, acados, l4casadi, l4acados. QP subtrack: PIQP, OSQP, ProxQP.
- **Fairness?** Same problem spec, backend-specific best-practice implementations, same episodes, quality gates, correctness checks, deployment-mode labels, artifact inspection.
- **Maintainability?** Separate repo, Docker by default, uv lock, Hatchling package + CLI, tiered CI, stable schema, required core profile plus optional heavy profiles.
- **Granular metrics?** Per-cell artifact directories, raw traces, generated code and compile commands archived, native gbench results, isolated memory measurements, deployment footprint recorded.
