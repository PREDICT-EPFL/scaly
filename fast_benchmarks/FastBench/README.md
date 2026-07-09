# FastBench

**A reproducible benchmark for optimization-based (embedded) control.**

FastBench lets you pose an optimal-control / MPC problem **once** and solve it
with **many backends** — CasADi+IPOPT, acados, OSQP, PIQP, ProxQP, do-mpc,
GRAMPC, and **FastSQP** (FastBench's own codegen C++ real-time SQP) — then run
closed-loop (or open-loop) simulations and produce comparable benchmark
metrics: solve quality, control quality, success/failure rates, real-time
timing, and code-generation/compilation size and time.

It is biased toward **fast / embedded** control (acados, CasADi, HPIPM-class
QP solvers) but is solver-agnostic by design. New problems and solvers are
added by implementing one small interface.

```
                 ┌──────────────┐
   Problem  ───► │  symbolic     │ ──►  CasADi model, cost, constraints,
 (one class)     │  description  │      references, plant, Monte-Carlo x0
                 └──────┬───────┘
                        │  the same problem feeds every adapter
        ┌─────────────┬─────────────┬──────────────┬─────────────┐
        ▼             ▼             ▼              ▼             ▼
   casadi_ipopt    acados       osqp/piqp/      fastsqp      do_mpc /
   (reference NLP) (embedded    proxqp          (codegen     grampc
                    RTI)        (QP, LTI only)   C++ SQP)     (NMPC)
        └─────────────┴─────────────┴──────────────┴─────────────┘
                        │  closed-loop simulation + metrics
                        ▼
                  results.json ──► report (CSV, REPORT.md, HTML, plots)
```

## Install

Core backends (CasADi+IPOPT, OSQP, PIQP, ProxQP, and **FastSQP** if a C++
compiler is present) are pure pip and install in seconds:

```bash
./setup.sh                 # venv .venv + core solvers + reporting
source .venv/bin/activate
fastbench doctor           # show which backends are available
```

FastSQP needs only `g++`/`gcc`; the Eigen and ProxQP headers it compiles
against are bundled in the CasADi wheel, so no system packages are required.

Optional backends:

```bash
./setup.sh --do-mpc        # do-mpc NMPC
./setup.sh --acados        # build acados + HPIPM (needs cmake + C compiler)
./setup.sh --grampc        # GRAMPC via pygrampc
./setup.sh --all
```

Or use the pinned Docker image for exact reproduction (includes acados):

```bash
docker build -t fastbench .
docker run --rm -v "$PWD/results:/app/results" fastbench fastbench run --episodes 10
```

## Quickstart

```bash
fastbench list                                   # problems & solvers
fastbench run --episodes 10                      # all problems × available solvers
fastbench run --problems chain_mass --solvers osqp piqp casadi_ipopt
fastbench report --results results/results.json  # rebuild the report
```

Open `results/report/index.html` (or `REPORT.md`) for the leaderboards, timing
plots, performance profile and codegen sizes.

## What gets measured

| family | metrics |
|--------|---------|
| **control quality** | closed-loop cost, tracking RMSE, terminal error, settling time, control effort, input rate |
| **solve quality** | suboptimality vs. the high-accuracy IPOPT reference, KKT residual, OCP & closed-loop constraint violation, step success rate |
| **reliability** | episode success rate, **deadline-miss rate** (solve time > sampling period) — Monte-Carlo over initial conditions, disturbances and seeds |
| **timing** | solve-time min / median / mean / p95 / p99 / max, iteration counts |
| **code (codegen backends)** | build time, generation time, compile time, generated source size & LOC, compiled-artifact size |

**Fairness & reproducibility built in.** Every solver sees the *same* episodes
(initial conditions + identically-seeded plant disturbances). The plant model
deliberately differs from the prediction model (parameter mismatch + noise) so
closed-loop numbers reflect robustness, not nominal accuracy. Each run stamps
the host CPU, package versions, git hash and thread settings into
`results.json`. Pin `OMP_NUM_THREADS=1` (and `taskset`/turbo settings) for
clean timing.

## Problems

14 problems spanning LTI/QP, nonlinear NMPC, tracking, process, aerospace,
automotive and robotics. LTI/QP problems are solved by *every* backend (and
serve as cross-validation cases); nonlinear problems by the NLP/SQP backends;
`vehicle_obstacle` additionally has a nonlinear path constraint (IPOPT only).

| name | class | nx·nu | structure |
|------|-------|-------|-----------|
| [`double_integrator`](fastbench/problems/double_integrator) | regulation | 2·1 | LTI / QP |
| [`dc_motor`](fastbench/problems/dc_motor) | tracking | 3·1 | LTI / QP |
| [`rocket_landing_1d`](fastbench/problems/rocket_landing_1d) | landing | 2·1 | affine-LTI / QP |
| [`chain_mass`](fastbench/problems/chain_mass) | regulation | 8·1 | LTI / QP |
| [`oscillating_masses`](fastbench/problems/oscillating_masses) | regulation | 12·2 | LTI / QP (scalable) |
| [`van_der_pol`](fastbench/problems/van_der_pol) | regulation | 2·1 | nonlinear |
| [`pendulum_swingup`](fastbench/problems/pendulum_swingup) | swing-up | 4·1 | nonlinear, nonconvex |
| [`cstr`](fastbench/problems/cstr) | stabilization | 2·1 | nonlinear, stiff |
| [`unicycle`](fastbench/problems/unicycle) | parking | 3·2 | nonlinear, nonholonomic |
| [`kinematic_vehicle`](fastbench/problems/kinematic_vehicle) | tracking | 4·2 | nonlinear, time-varying ref |
| [`two_link_arm`](fastbench/problems/two_link_arm) | tracking | 4·2 | nonlinear, robotics |
| [`quadruple_tank`](fastbench/problems/quadruple_tank) | tracking | 4·2 | nonlinear, MIMO |
| [`planar_quadrotor`](fastbench/problems/planar_quadrotor) | stabilization | 6·2 | nonlinear, aerospace |
| [`vehicle_obstacle`](fastbench/problems/vehicle_obstacle) | tracking+avoid | 4·2 | nonlinear + path constraint |

Each problem directory has a README with the full model, parameters, cost,
constraints and original sources.

## Solvers

| name | kind | needs | notes |
|------|------|-------|-------|
| `casadi_ipopt` | NLP (IPOPT) | pip | reference / ground-truth; handles path constraints |
| `fastsqp` / `fastsqp_full` | **codegen C++ SQP** | g++ | FastBench's own real-time / full SQP (see below) |
| `osqp`, `piqp`, `proxqp` | sparse QP | pip | LTI/QP problems only |
| `acados`, `acados_sqp` | embedded RTI/SQP | acados build | `setup.sh --acados` |
| `do_mpc` | NMPC (CasADi/IPOPT) | pip | `setup.sh --do-mpc` |
| `grampc` | augmented-Lagrangian NMPC | pygrampc | `setup.sh --grampc` |

## FastSQP — the built-in C++ solver

`fastsqp` is FastBench's own real-time SQP, written in C++ (`cpp/fastsqp.cpp`).
On build it (1) generates C code with CasADi for the discrete dynamics +
Jacobians and the cost gradients/Hessians of the problem, (2) compiles that
together with the SQP core and a QP backend into a shared library, and (3)
drives it from Python via ctypes — so it produces real **code-generation and
compilation metrics** (generated LOC/size, compile time, compiled size) the
same way an embedded toolchain would, with no acados dependency.

The SQP assembles the sparse banded OCP-QP

```
min 0.5 dz' H dz + g' dz
 s.t.  dx_0 = x_meas − x_nom_0
       dx_{k+1} − A_k dx_k − B_k du_k = fd(x_nom_k,u_nom_k) − x_nom_{k+1}
       lbx − x_nom_k ≤ dx_k ≤ ubx − x_nom_k   (and similarly for u)
```

each iteration and solves it with a pluggable QP backend. `fastsqp` performs a
single real-time iteration; `fastsqp_full` runs to convergence. Backends are
selected at compile time (`-DHAVE_PROXQP/HAVE_PIQP/HAVE_OSQP/HAVE_QPOASES`);
**ProxQP** is the default and ships its headers (and Eigen) inside the CasADi
wheel, so no extra system packages are needed. The C++ core compiles once and
is cached; only the per-problem generated code is recompiled.

On the linear problems FastSQP reproduces the IPOPT/OSQP/ProxQP cost exactly;
on nonlinear problems `fastsqp_full` matches IPOPT quality while running several
times faster (single-iteration `fastsqp` trades a little optimality for the
lowest latency).

## Adding a problem

Subclass `fastbench.core.problem.Problem`, provide the symbolic model and a
plant, and register it:

```python
from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem

class MyProblem(Problem):
    def __init__(self):
        self.meta = ProblemMeta(name="my_problem", nx=2, nu=1, dt=0.05,
                                N=20, n_sim=100)
    def dynamics_ct(self, x, u): ...          # CasADi xdot
    def stage_cost(self, x, u, xref, uref): ...
    def terminal_cost(self, x, xref): ...
    def x0_nominal(self): ...

@register_problem("my_problem")
def _factory():
    return MyProblem()
```

Add `is_lti=True` and a `lti_matrices()` returning `{A,B,Q,R,Qf}` to make it
solvable by the QP backends too.

## Adding a solver

Subclass `fastbench.solvers.base.SolverAdapter` and implement
`available()`, `build(problem) -> BuildInfo`, and `solve(x, k) -> SolveStats`;
register with `@register_solver("name")`. Adapters degrade gracefully: if a
backend is not installed, `available()` returns `False` and the runner skips it
with a logged reason, so the rest of the benchmark still runs.

## Repository layout

```
fastbench/
  core/      problem interface, simulator, metrics, runner, env capture, registry
  solvers/   casadi_ipopt, osqp, piqp, proxqp, acados, do_mpc, grampc,
             fastsqp (+ qp_mpc builder)
  problems/  14 problem packages, each with a README
  report/    CSV / markdown / HTML report + plots + Dolan-Moré performance profile
  cli.py     `fastbench list|doctor|run|report`
cpp/         fastsqp.cpp (C++ SQP core) + casadi_abi.h
scripts/install_acados.sh   setup.sh   Dockerfile   tests/   examples/
```

## Cleaning

Build/run artifacts (caches, generated C code, compiled FastSQP libraries,
`results/`) are regenerable and never need to be kept or shipped. Two levels:

```bash
make clean          # or: fastbench clean
make distclean      # or: fastbench clean --all

fastbench clean --dry-run     # preview exactly what would be removed
```

`clean` removes `__pycache__`, `fastsqp_build/`, `results/`, codegen,
`.pytest_cache`, `*.egg-info`, `build/`/`dist/` — but keeps `.venv` and the
acados build so you don't have to reinstall. `distclean` (`--all`) additionally
removes `.venv`, `third_party/` (acados) and `examples/`, resetting the tree to
source-only (~300 KB) — exactly what you want before zipping it up to share.
Deletion is scoped to the project root and current directory; anything outside
them is refused. After a `distclean`, re-run `./setup.sh`.

## Roadmap / extension points

* **More FastSQP QP backends compiled in**: the PIQP/OSQP/qpOASES backends are
  written behind `-DHAVE_*` guards; ProxQP is the default verified one. Wiring
  the others into the build (and condensing for dense QP) is the next step.
* **FastSQP general inequalities**: today FastSQP handles box constraints
  (like acados Gauss-Newton RTI); linearized nonlinear path constraints
  (e.g. `vehicle_obstacle`) are an extension point.
* **GRAMPC** CasADi→ProblemDescription codegen is wired (`grampc_solver.py`)
  and activates when `pygrampc` is installed.
* **On-target embedded metrics**: ARM cross-compilation for flash/stack
  footprint (host x86 codegen size/time is measured today).

## License

MIT (framework). Individual problems cite their original sources; respect the
upstream licenses when redistributing derived data.
