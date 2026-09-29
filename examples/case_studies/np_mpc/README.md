# Case study E9: Neural Process MPC

*Neural Process Model Predictive Control* (Waibel, Mello Rella, Jones, 2026) swings up a Furuta
pendulum with an MPC whose model is a conditional neural process (CNP). The CNP is meta-trained across
a family of pendulums. At deployment its encoder turns a short context trajectory of the pendulum at
hand into a four-number latent code z, and its decoder, conditioned on z, is the one-step model inside
the MPC. The authors' code, [jowaibel/neural_process_mpc](https://github.com/jowaibel/neural_process_mpc)
at `2d8de66`, has the controller in Python (CasADi Opti, IPOPT) and in C++ (CasADi Opti with IPOPT; laOPT
with IPOPT or with its SQP and PIQP), each with the neural model and an equation-based baseline.

This study runs all of them as published, reimplements the encoder, the decoder and both MPCs in Scaly,
and compares them on identical problems.

```bash
examples/case_studies/np_mpc/baseline/setup.sh     # the upstream, laOPT, PIQP, Eigen 3.4, yaml-cpp, its Python env; ~70 s
uv run examples/case_studies/np_mpc/compare.py all  # everything below, into results/; ~20 min
uv run --with jupyterlab jupyter lab examples/case_studies/np_mpc/np_mpc.ipynb
```

`np_mpc.ipynb` explains the study, runs the Scaly side live (the encoder against the authors' z, 20 of
their OCPs replayed iteration for iteration, a closed-loop episode) and plots the recorded results. Its
`RUN_BASELINES` flag reruns `setup.sh` and `compare.py`.

## Results

Apple M3 Max, macOS 26.6, Apple clang 21. Upstream C++ at the upstream's Release flags, Scaly's JIT at
`-O2`. This is not the reference machine of [fairness](../../../docs/results/fairness.md); none of
this belongs on the results pages. Each timed process started on a quiet machine (`_common.py`).

### Replay: the authors' 100 OCPs, median ms per solve

The first instance is left out (CasADi builds its solver there). Iterations are the mean; "as ref." counts
the instances where the iteration count equals the authors' Python controller's. Max |Δu| is against its
solutions.

| Implementation | IPOPT build | Neural: ms | iter (as ref.) | converged | max \|Δu\| | Equation: ms | iter (as ref.) | converged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CasADi Python, as published | CasADi's | 17.45 | 13.76 (100) | 100 | – | 8.95 | 11.97 (100) | 100 |
| CasADi Python, `expand` | CasADi's | 27.95 | 13.76 (100) | 100 | 7e-16 | 5.74 | 11.97 (100) | 100 |
| CasADi Python, `jit` | CasADi's | 8.93 | 13.76 (100) | 100 | 0 | 5.50 | 11.97 (100) | 100 |
| CasADi C++, as published | CasADi's | 15.13 | 13.76 (100) | 100 | 0 | 8.34 | 11.97 (100) | 100 |
| CasADi C++, `jit` | CasADi's | 8.93 | 13.76 (100) | 100 | 0 | 5.43 | 11.97 (100) | 100 |
| laOPT IPOPT | CasADi's | 36.14 | 11.31 (2) | 100 | 3e-7 | 6.24 | 9.10 (0) | 100 |
| laOPT IPOPT | Scaly's | 32.56 | 11.31 (2) | 100 | 3e-7 | 3.41 | 9.10 (0) | 100 |
| laOPT SQP-PIQP, as published | – | 1.57 | 6.30 | 94 | 1e-2 | 0.40 | 2.62 | 100 |
| **Scaly, Opti form, IPOPT** | Scaly's | **3.83** | 13.76 (100) | 100 | 6e-16 | **2.38** | 11.97 (100) | 100 |
| Scaly, Opti form, IPOPT | CasADi's | 7.73 | 13.76 (100) | 100 | 7e-16 | 5.31 | 11.97 (100) | 100 |
| Scaly, laOPT form, IPOPT | Scaly's | 2.88 | 11.25 (7) | 100 | 4e-7 | 1.66 | 9.11 (0) | 100 |
| Scaly, laOPT form, SQP-PIQP | – | 1.25 | 6.06 | 88 | 5e-2 | 0.36 | 2.62 | 97 |

- **Same NLP, same iterates.** Scaly's Opti mirror takes the authors' IPOPT iteration count on every
  instance of both models, with solutions equal to 6e-16. It is 4.6× faster than their controller as
  published on the neural model and 3.8× on the analytic one, and 2.3× faster than CasADi's `jit`.
- **Half of that is the IPOPT build.** Pointed at CasADi's wheel IPOPT through the loader, the same
  generated solver takes 7.73 ms. Against CasADi's `jit` (8.93 ms) that isolates the oracles and the
  glue (1.16×); against Scaly's IPOPT build (3.83 ms) it isolates the build (2.0×). laOPT shows the same build
  effect on the analytic model (6.24 against 3.41 ms), as [fairness](../../../docs/results/fairness.md)
  found for the benchmark problem.
- **laOPT's IPOPT is the slowest on the network** (36 ms, about 3 ms per iteration), so the authors'
  laOPT clients use its SQP, which needs only the objective's Hessian. That SQP is the fastest on the
  neural model and converges on 94 of 100 within 15 iterations.
- **Scaly's SQP converges on fewer**, 88 and 97. It fails on the swing-up's first problems, where the
  objective's own Hessian is indefinite, and its convexification leaves a QP PIQP does not solve
  (`internal/todo.md` CS-19; 95 with `hessian="exact"`).
- The laOPT mirror is close to laOPT, not identical. It takes the same iteration count on 99 of 100 analytic-model
  instances and 67 of 100 neural ones, solutions within 4e-7. laOPT passes IPOPT six rows per node with
  both bounds infinite, which the mirror leaves out.

Compile and setup: Scaly generates and compiles the neural solver in 3.5 s and the analytic one in 0.8 s,
once for every latent code. CasADi's `jit` compiles for 23 s (neural) on its first solve, as published it
builds in 38 ms; laOPT tapes in 25 ms, after a C++ build of about 30 s for all its targets.

### Lockstep episodes

System 3, z from a 100-step context, 100 steps.

| Controller | Loop | Neural: cost | settles | ms per solve (median) | Equation: cost | settles | ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Authors' (CasADi Python) | delayed | 325.81 | 28 | 17.61 | 161.35 | 14 | 9.05 |
| Scaly, Opti form, IPOPT | delayed | 325.81 | 28 | 3.84 | 161.35 | 14 | 2.41 |
| Scaly, laOPT form, IPOPT | delayed | 325.81 | 28 | 2.94 | 161.35 | 14 | 1.68 |
| Scaly, laOPT form, SQP | delayed | 305.04 | 30 | 1.27 | 234.56 | 27 | 0.37 |
| Authors' (CasADi Python) | immediate | 199.67 | 24 | 17.56 | 151.55 | 13 | 9.12 |
| Scaly, Opti form, IPOPT | immediate | 199.67 | 24 | 3.88 | 151.55 | 13 | 2.41 |

Scaly's IPOPT episodes follow the authors' to 7e-5 in the state (their model runs in float32). The SQP
leaves 15 to 18 steps unconverged in each episode and swings up along a different path.

### The adaptation study (paper figure 8)

Mean closed-loop cost over five context experiments; the equation-based controller uses the true plant.
The authors' controller and Scaly's ran both loops. In all 126 episodes their costs agree to 6e-6 and
they settle at the same step. Scaly's numbers:

| System | Loop | Equation | ctx 1 | ctx 20 | ctx 50 | ctx 100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sys1 | delayed | 159.3 | 245.0 | 173.3 | 176.2 | 184.4 |
| sys2 | delayed | 163.5 | 231.8 | 201.8 | 223.9 | 265.6 |
| sys3 | delayed | 161.4 | 334.7 | 304.8 | 319.5 | 323.5 |
| sys1 | immediate | 150.4 | 220.2 | 166.2 | 162.2 | 166.8 |
| sys2 | immediate | 153.6 | 201.6 | 161.1 | 173.1 | 186.3 |
| sys3 | immediate | 151.5 | 257.4 | 176.3 | 193.1 | 205.1 |
| sys3, the paper (real time) | | 182.1 | 597.6 | 331.6 | 290.1 | 188.6 |

With the authors' delayed loop, system 3's neural controller overshoots to 4π and costs about 320
whatever the context. Their own real-time deploy of this release lands there too (below). The paper's
numbers (`datasets/furuta_mpc/all experiments/`, real-time runs from June 2026) are closer to the
immediate loop.

Mean solve over the grid: 17.4 ms for the authors' controller, 3.7 ms for Scaly's. After a new latent
code, the authors' controller rebuilds its Opti and its first solve builds the NLP solver, 38 ms in
all. Scaly solves the Riccati equation for the terminal weight (3 ms) and solves (5 ms) on the solver it
already has.

### Real time, the authors' server

Median of three 1 s runs (`sim_steps` 50), scored like `compute_metrics` on the server's log.

| Agent | Neural: cost | settles (s) | solves | mean ms | Equation: cost | settles (s) | solves | mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Authors' laOPT SQP client, as published | 202.6 | 0.20 | 201 | 1.52 | 158.1 | 0.24 | 196 | 0.62 |
| Authors' laOPT client with IPOPT | 5309 | – | 9 | 121 | 285.9 | 0.60 | 138 | 7.21 |
| same, Scaly's IPOPT build | 8749 | – | 10 | 107 | 152.0 | 0.24 | 222 | 3.84 |
| Authors' CasADi C++ client, as published | 33643 | – | 25 | 45.2 | 39603 | – | 52 | 20.7 |
| Authors' Python deploy, as published | 635.4 | – | 41 | 22.0 | 162.4 | 0.24 | 45 | 9.94 |
| **Scaly client, laOPT form, IPOPT** | **157.1** | **0.26** | 363 | 2.69 | **148.9** | 0.28 | 496 | 1.88 |
| Scaly client, Opti form, IPOPT | 181.8 | 0.20 | 254 | 3.88 | 148.9 | 0.26 | 367 | 2.67 |
| Scaly client, laOPT form, SQP | 539.7 | – | 35 | 11.1 | 1170 | – | 1 | – |

- The authors' C++ CasADi clients do not swing up on this machine. With the wheel's IPOPT their solves
  average 20 to 45 ms, the clients have no delay compensation, and an unconverged iterate is applied as
  is: Opti poses the torque bound as a row, which an IPOPT iterate may violate, and they sent up to
  1.68 N m against the 0.05 bound. Their source's `kMinSolveMs = 4.3` comment puts a CasADi solve at
  4.3 ms on the authors' machine (Linux, Ryzen 9 7940HS).
- The Python deploy settles within the 1 s in one run of three (the delayed loop's overshoot); it settled
  at 0.58 s in the run before these. Its z comes from a real-time context experiment each run.
- Scaly's IPOPT clients swing up in every run at 250 to 500 solves per second. Scaly's SQP client
  repeats its first solve until it converges, as the laOPT clients do, and from the hanging state that
  takes 0.5 to 8 s (CS-19); the controller never recovers.

## Changes to the authors' code

One patch, and it touches logging only. Their C++ CasADi clients crash printing a `casadi::DM`, because the CasADi wheel
runs on its own `libc++` and the clients on the system's (`baseline/patches/casadi_clients_print.patch`).
The laOPT clients' IPOPT variants are copies with the `kSolver` line changed, made by `CMakeLists.txt`.
Everything else runs as published, from their classes: `upstream.py` builds their controller with
`MPCController._build_optimization` and `configure_solver` but without the runtime, which would start
their simulator.

## What is compared

Three harnesses, each running the authors' controller code unchanged beside Scaly's:

- **Replay.** The authors' Python controller runs one closed-loop swing-up (system 3, z from a 100-step
  context), and every OCP it meets is recorded with the point its solve started from. Every
  implementation then solves those 100 OCPs from those points, in three fresh processes, keeping the
  fastest solve of each. Same problems and starting points, so the same algorithm should take the same
  iterations, and time differences are the implementations'. The C++ drivers
  (`baseline/replay_{laopt,casadi}.cpp`) use the upstream's OCP classes, transcriptions and
  `configureSolver`; `baseline/upstream.py replay` uses its Python classes.
- **Lockstep.** `protocol.lockstep` reproduces the authors' controller loop (`run_controller` over
  `RuntimeBase`) against a deterministic plant. The plant is the analytic Furuta model integrated by
  explicit midpoint at 4 kHz, as their simulator does, and it advances in lockstep with the controller. The loop
  compensates a delay. At step k it predicts x_{k+1} with its own model under the torque already
  acting, solves from there, and its torque acts from t_{k+1}, which is what their real-time loop does
  when every solve fits in its 20 ms. `--no-delay` solves from the measurement and applies the torque at
  once. The adaptation study runs on it. It covers the paper's three pendulums, z from contexts of 1,
  20, 50 and 100 steps with five context experiments each, and the equation-based controller with the
  true plant parameters.
- **Real time.** The authors' `scripts/run_qube_server.py` integrates the plant in its own process at
  4 kHz on wall-clock time and streams the state at 500 Hz. Their C++ clients solve as fast as they can
  from the newest state (the laOPT ones wait until 4.3 ms after a solve started before sending);
  their Python deploy paces at 20 ms with the delay compensation. `realtime_scaly.py` is a client like
  their C++ ones. `baseline/run_realtime.py` runs their agents; each agent runs three times for 1 s.

## The Scaly implementation

`scaly_impl.py`, NumPy and Scaly only:

- The checkpoint is read with `nn.load_torch_state_dict`: the encoder (7 → 64 → 128 → 128 → 64 → 4,
  GELU, written with `erf`), a masked mean over the context as one `vmap` and a matrix product, and the
  decoder's mean (9 → 32 → 32 → 2, sigmoid). The file is the benchmark problem's vendored
  `bench/problems/npmpc/data/cnp_model.pth`, byte for byte the upstream's `model/model.pth`.
- Both OCPs, the neural one and the equation-based one (implicit midpoint), each in two forms:
  `opti` mirrors `MPCController._build_optimization` (every constraint a row, four slacks) and `laopt`
  mirrors `MultipleShootingXDiff` over `FurutaNPOCP`/`FurutaEqOCP` (bounds as variable bounds, the unused
  slacks fixed at zero). The dynamics and the stage cost are each one `vmap` over the horizon.
- The initial state, the model parameter (z or the plant parameters) and the terminal weight are runtime
  parameters, so one compiled solver serves every latent code. The terminal weight comes from the model's
  Jacobian at upright (`sc.jacobian`) and SciPy's discrete Riccati solver, as the upstream's does from
  torch's autograd.
- Solvers: IPOPT with the upstream's settings (tol 1e-6, 50 iterations), and Scaly's SQP with PIQP
  subproblems with laOPT's (objective-only Hessian, filter line search, 15 iterations).

### The Opti mirror has to put constants where Opti puts them

Written naturally, the `opti` form posed the same NLP as CasADi's but IPOPT took different iterations
from the first step. The cause: Opti turns a side free of variables into the row's bound (`x_0 <= x0 +
1e-3` is the row `x_0` with upper bound `x0 + 1e-3`) and writes `a <= b` otherwise as `a - b <= 0`, and
IPOPT pushes its slacks off a bound by `1e-2 max(1, |bound|)` and relaxes each bound by
`1e-8 max(1, |bound|)`. Swapping Scaly's generated solver onto CasADi's IPOPT build (through the loader)
changed nothing, which ruled the build out. Written Opti's way, Scaly's IPOPT follows CasADi's iteration
for iteration on all 100 replayed OCPs, with solutions equal to 6e-16.

## Files

| File | What it is |
| --- | --- |
| `baseline/setup.sh` | pins and builds the upstream (2d8de66), laOPT (8f9820a), PIQP (v0.6.2), Eigen 3.4.0, yaml-cpp 0.8.0 and the upstream's Python 3.12 env (torch 2.5.1, casadi 3.7.0) in `baseline/third_party/` |
| `baseline/CMakeLists.txt` | the upstream's C++ programs, IPOPT variants of its laOPT clients, the replay drivers; laOPT's IPOPT targets twice, against CasADi's IPOPT and Scaly's |
| `baseline/replay_laopt.cpp`, `replay_casadi.cpp`, `replay_io.hpp` | the replay drivers |
| `baseline/upstream.py` | the authors' Python controller in the lockstep, grid and replay harnesses, built from their classes |
| `baseline/run_realtime.py` | their real-time agents against their server |
| `baseline/patches/` | one logging-only patch to their C++ CasADi clients (see the file) |
| `protocol.py` | the plant, the context experiment, the lockstep loop and the metrics; NumPy only, imported by both sides |
| `scaly_impl.py` | the CNP and both MPCs in Scaly |
| `run_scaly.py` | the Scaly side of the lockstep, grid and replay harnesses |
| `realtime_scaly.py` | a Scaly client for their real-time server |
| `compare.py` | runs everything in fresh processes on a quiet machine, into `results/` |
| `np_mpc.ipynb` | the study as a notebook |

`tests/integration/test_case_study_np_mpc.py` checks the encoder (GELU through `erf`, the masked mean)
against NumPy and finite differences, initial-state bands that follow a runtime parameter across
consecutive solves of one generated solver (as rows and as variable bounds, under IPOPT and SQP), and the
two mirrors agreeing on a miniature of the OCP.
