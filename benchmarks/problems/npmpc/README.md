# Neural process MPC on the Furuta pendulum

The model-predictive controller from *Neural Process Model Predictive Control* (Waibel, Mello Rella,
Jones; European Journal of Control, 2026). A conditional neural process is meta-trained across a
family of Furuta pendulums; at deployment its encoder turns a short open-loop context trajectory
into a four-number latent code, and its decoder — conditioned on that code — is the one-step
dynamics model inside a nonlinear MPC that swings the pendulum up and holds it upright.

Reference implementation: <https://github.com/PREDICT-EPFL/neural_process_mpc>. Only the controller
is reproduced here; the training pipeline, the dataset generators and the hardware driver are out of
scope.

## Why it is in the suite

The other three problems cover analytic dynamics over a horizon (`race_cars`), neural dynamics
without a horizon (`unbumpercars`, a one-step filter mapped over cars) and a large sparse structured
NLP (`chain`). This is the missing combination: **a dense neural network evaluated at every node of
a prediction horizon**, which is the shape most deployed learning-based MPC actually has.

It is also the one problem whose reference implementation states a limit in its own paper. The
authors report a 20 ms sampling time, 10–20 ms measured solve times, and that they shrank the
decoder to two layers of 32 units — "NP dimensions are chosen as small as possible while providing
the necessary performance" (§5.3). With 65 decision variables there is almost no linear algebra for
the solver to do, so function evaluation is most of the solve, and the interesting question is not
whether a kernel is faster but how much network fits inside the real-time budget.

## Formulation

State `x = (θ, φ, θ̇, φ̇)` with θ = 0 upright and θ = π hanging, φ the arm angle; input `u = τ`, an
arm torque. `dt = 0.02`, horizon `N = 12`.

Their `FurutaDynamics` docstring lists the state order as `(θ, θ̇, φ, φ̇)`; the code and the paper
both use `(θ, φ, θ̇, φ̇)`. The docstring is wrong and the code is followed.

### The learned model

Decoder features are `(sin θ, cos θ, θ̇, φ̇, τ)`, scaled elementwise, concatenated with the length-4
latent code, then `9 → 32 → 32 → 2` with sigmoids on the two hidden layers, a bias on the last, and
the output rescaled. The full one-step map closes the position rows by explicit trapezoidal
integration (paper eq. 17):

```text
x_next = x + [ dt * (x[2:4] + Δx_vel / 2),  Δx_vel ]
```

### The optimal control problem

Decision variables are blocked rather than interleaved, so both VMAP windows stride cleanly and
there is no dead trailing control:

```text
z = [ x_0 … x_N (4 each) | u_0 … u_{N-1} (1 each) | s ]     4(N+1) + N + 1 = 65 at N = 12
```

Cost, with `χ = (2 sin(θ/2), φ, θ̇, φ̇)` — the half-angle lift makes every upright configuration cost
the same, since `(2 sin(θ/2))² = 2(1 − cos θ)`:

| term | weights |
|---|---|
| stage, `Σ_{i<N} χ_iᵀ Q χ_i + R τ_i²` | `Q = diag(2.5, 1.5, 0.005, 0.01)`, `R = 1` |
| inter-stage, `Σ_{i<N} Δx_iᵀ Q_Δ Δx_i` | `Q_Δ = diag(0, 0, 0.05, 0.125)` |
| terminal, `χ_Nᵀ P χ_N` | `P` from the Riccati solve below |
| slack penalty | `1000 · 0.5 · (s² + s)` |

Constraints: the `4N` dynamics equalities; `φ_i ∈ [−2, 2]` at all `N+1` nodes, softened by the
single slack with `s ∈ [0, 5]`; `τ_i ∈ [−0.05, 0.05]`; and the initial state, imposed as a ±1e-3
band of inequalities rather than an equality, as theirs is.

`P` solves the discrete algebraic Riccati equation for the learned dynamics linearized at the upright
equilibrium with `Q_f = diag(1, 10, 0.1, 0.1)` and `R_f = 1`. Their code takes `A`, `B` from
`torch.autograd`; `linearize` takes them from one typed Alloy derivative function instead, which
removes the torch dependency and exercises Alloy's own differentiation in the problem's setup. The
Riccati residual is gated regardless, so a pinned `P` cannot drift from the linearization it claims
to come from.

Both the dynamics and the per-stage cost use `al.vmap`. Using VMAP for the cost is not cosmetic: written
as a Python loop over stages it unrolls, which grows the Lagrangian Hessian's generated source
linearly in the horizon and, past roughly 75 stages, exceeds the Program IR passes' recursion depth
during lowering.

### The parameter vector

At the shipped decoder width, both generated backends accept the same 1427-entry runtime vector:

```text
p = [xstart (4) | pw (1396) | dt (1) | cost (10) | P (16)]
```

The cost block contains the four state weights, four inter-stage weights, the input weight, and the
slack weight. `P` is row-major. The decoder tail is, in order:

```text
x_scale_w 5 | x_scale_b 5 | y_inv_w 2 | y_inv_b 2 | W0 288 | W1 1024 | W2 64 | b2 2 | z 4
```

Both backends read all numerical tuning data from `p`. Generated C can therefore change the model,
sample time, cost, and terminal weight without recompilation. The horizon and decoder architecture
still specialize the generated function because they determine loop bounds and buffer shapes. This
benchmark assumes that the packed decoder-vector length identifies its architecture. Arbitrary MLP
layouts can share a total parameter count, so a general `FunctionTemplate` must use the individual
layer shapes instead.

The shipped inter-stage weights start with two zeros. Keeping those weights runtime-configurable
means the generated Hessian retains the corresponding structural entries instead of folding them
away: 12 entries at N = 6 and 200 at N = 100 are zero for the default configuration. Both backends
receive the same runtime data and retain the same structure.

### The plant

The analytic rigid-body Furuta model — the true physics the learned model approximates, which is
what makes the closed loop meaningful. System 3, the pendulum the paper's closed-loop figure and the
recovered latent code both belong to: `l_p = 0.1378`, `m_p = 0.00881`, `l_r = 0.0895`,
`m_r = 0.0273`. Episode start `x_0 = (π, 0, 0, 0)`, hanging down, 100 steps.

## Vendored data

| file | what |
|---|---|
| `data/cnp_model.pth` | their trained checkpoint, read without torch by `alloy.utils.load_torch_state_dict` |
| `data/reference_config.json` | their `model/furuta_mpc.json`, verbatim — the authoritative source for every weight and bound |
| `data/reference_episode.npz` | their released `experiment_np_m3.npz`, trimmed |

The episode is trimmed twice over: the 4 MB `x_mc` Monte Carlo rollouts and the pickled metrics are
dropped, and every step-indexed array is truncated to the `k + 1 = 100` steps their run actually
completed, so there is no trailing all-zero row to mistake for data. What is left is 96 kB: `x0`
measured states, `u0` the control applied one step earlier, `x`/`u` the solved horizon, `x_np` the
neural-process open-loop rollout behind it, `x_oracle` the same rollout under the analytic plant, and
`mpc_t` their measured per-step loop times, which run 14.1–19.3 ms against their 20 ms budget — plus
the scalars `steps`, `dt` and `horizon_steps`.

Their config records `sim_steps: 50` while the released episode ran 100 steps; the canonical episode
here runs 100 too, following the artifact rather than the config.

The latent code is pinned data rather than something computed. Their code computes it at startup by
encoding a context trajectory and never saves it, and the trajectory is not reproducible: their
plant runs in a separate process at 4 kHz on wall-clock time steps. It is only four numbers, though,
and `x_np` is a full neural-process rollout, so the four unknowns were fitted to it by multistart
Levenberg–Marquardt over ~1200 one-step residuals. The fit converges from many starts to

```text
z = [-2.34141442, 0.16122490, -10.00758909, -1.65464341]
```

with a largest one-step residual of 1.7e-5 against velocity changes of magnitude up to 11 — agreement
at the precision of their float32 checkpoint. That single number simultaneously pins the latent code
for system 3 and establishes that this re-implementation of their decoder is faithful.

## Deliberate departures from the reference implementation

- **One slack instead of four.** Their `slack` is a 4-vector but only component 1, the arm angle, is
  ever constrained, penalized, or referenced; the other three are free, cost-free and appear in no
  constraint. IPOPT survives this by regularizing, but three variables absent from the Lagrangian
  make the exact KKT matrix singular, which an exact-Hessian SQP has no reason to tolerate.
  Allocating one slack is mathematically equivalent, since the dead variables cannot influence the
  solution.
- **A deterministic plant.** Theirs integrates in a separate process at a nominal 4 kHz driven by
  wall-clock time steps, so no two runs agree and neither is reproducible. Ours integrates in
  process with 80 fixed midpoint substeps over each 20 ms interval — the same nominal rate. This is
  why their saved episode is used as a per-step gate rather than as a trajectory to match.
- **No delay compensation.** This is the one departure that changes behaviour rather than only
  reproducibility, so it is worth spelling out. Their controller pins the first horizon node to the
  measurement pushed one step forward through the learned model, because their plant keeps moving in
  another process while the solve runs. This loop is synchronous: the control the solve returns acts
  over the very next interval, so pinning the first node one step ahead does not remove a delay, it
  adds one. On this pendulum one 20 ms interval at the torque limit is worth about 10 rad/s, and the
  effect is not subtle — measured over 100 steps:

  | first node pinned to | applied over | settles | final offset from upright |
  |---|---|---|---|
  | the measurement | the next interval | **step 10** | 0.01° |
  | the measurement pushed one step on | the next interval | never — spins through six revolutions | 179.7° |
  | the measurement pushed one step on | the interval after next | step 31, after overshooting to 4π | 0.25° |

  The runner uses the first. Their own episode settles at step 14, and the difference is exactly the
  delay their compensation exists to cover.
- **The encoder is not reproduced.** It is `GELU[64, 128, 128, 64]`, and GELU needs `erf`, which
  Alloy's expression surface does not have. Nothing in this benchmark needs the encoder: the latent
  code is pinned. An in-graph encoder is only needed for the online adaptation their conclusion
  points at, and the clean route there is an `erf` op in Alloy rather than a benchmark workaround —
  CasADi has `ca.erf` natively, so only Alloy is missing a primitive. A tanh approximation must not
  be used on any path that claims to reproduce the paper, because it changes the model and so moves
  the latent code.

## Benchmark results

This README owns the formulation, reference data, and correctness gates. It does not copy timing
tables. The [current benchmark results](../../../docs/results/index.md) contain the canonical
closed-loop comparison, and the [scalability tables](../../../docs/results/scalability.md) contain
every horizon cell from the latest study.

## Gates

This problem contributes sixteen gates to `benchmarks/run.py smoke --select problems`, which runs
every problem's. Each was shown to fail by perturbing what it checks, and the perturbations that
did *not* fire are recorded below too, because a gate that cannot fail reads as coverage without
being any.

| gate | what it holds |
|---|---|
| `dims_and_checkpoint` | declared dimensions, the vendored checkpoint's layer shapes, the pinned latent code |
| `reference_config` | every weight, bound and constant read out of *their* config file |
| `parameter_tail_order` | the tail's order, against an evaluation that indexes it without asking `Decoder` |
| `decoder_rollout` | the learned model against their released neural-process rollout, and Alloy against NumPy |
| `plant_rollout` | the analytic plant against their analytic rollout, plus substep convergence and both equilibria |
| `terminal_riccati` | `A`, `B` against finite differences, and `P` against the Riccati equation |
| `constraint_rows` | the inequality rows and box bounds against a hand-written evaluation |
| `runtime_tuning_parameters` | one compiled Alloy function responds to runtime changes in `dt`, cost weights and `P` |
| `casadi_runtime_parameters` | CasADi reads the same runtime fields and matches Alloy after each change |
| `initial_guess` | the cold start rotates *forward* to upright, which picks the swing-up direction |
| `exact_hessian` | the IPOPT column really consumes the generated exact Lagrangian Hessian |
| `episode_artifacts` | shapes, finiteness, the first node inside the band, the plan's first control applied |
| `episode_swings_up` | the closed-loop claim: up from hanging and held, slack and bounds inactive |
| `reference_episode` | the cross-implementation gate, below |
| `sqp_matches_ipopt` | per-step agreement between the two solvers, tolerances from measured healthy steps |
| `oracles_agree` | identical iteration counts and trajectories from both oracle providers under IPOPT |
| `sqp_oracles_agree` | the same under the SQP, where both providers are compiled C |
| `recorded_scene` | the runner feeds the 3D scene builders this problem's own data |

### What the cross-implementation gate establishes, and what it does not

For every step of their released episode, `reference_episode` solves our NLP from the state and warm
start their run recorded and compares. Three things, in increasing strength of claim:

1. **Their recorded solution is feasible under our dynamics**, to 5.4e-6 — their solver tolerance
   plus their checkpoint's float32 noise. This is the strong result: it validates the model, the
   recovered latent code, the state ordering and the parameter layout against an independent
   implementation, and it fires on a latent-code bias of 1e-2.
2. **Our objective is never worse than theirs**, evaluated with our own cost — measured strictly
   better at every step, by 0.29 to 10.7 on objectives of 1.5 to 170.
3. **The applied torque agrees to 2.5e-3** after their own reported settling step (14), against a
   measured worst case of 9.4e-4 on a torque bounded by ±0.05.

Result 2 is a one-sided check, and result 3 is weaker than it looks. Their recorded solutions are
feasible but well short of stationary for the problem either implementation writes down: restarting
our IPOPT from their step-50 solution moves the horizon by 0.46 and drops the objective from 2.11 to
1.52, almost all of it in the horizon's tail. The applied torque is insensitive to that tail, which
is why it agrees so well — and equally insensitive to the cost weights, which can be moved from 1.5
to 6.0 on the arm angle without this gate noticing. **`reference_config` is what pins the weights**,
by reading them out of their own configuration file, and that is why that file is vendored.

The swing-up window is excluded from result 3 on purpose. There the problem admits more than one
local minimum and the two runs pick different ones: at step 8 their horizon and ours differ by
2.7 rad/s in pendulum velocity while both remain feasible and ours costs 10.7 less. Loosening the
tolerance until that passed would have left a gate that could not fail, so the window is reported
instead and the first diverging step is named on failure.

### What each gate was perturbed with

Firing perturbations, one example each where the gate is not obvious: zeroing the terminal weight or
biasing the latent code by 0.3 stops `episode_swings_up`, and so does widening or tightening the
torque bound tenfold; handing IPOPT a `limited-memory` Hessian approximation trips `exact_hessian`
(via the Hessian oracle's call count dropping to zero, since `al.nlp` always *attaches* one);
opening the arm-angle upper family downwards trips `constraint_rows`; a 0.1% change in the terminal
weight moves `sqp_matches_ipopt`; dropping the inter-stage cost from the CasADi mirror alone trips
both `oracles_agree` gates; swapping the two recorded angles trips `recorded_scene`.

Perturbations that did **not** fire, which is worth knowing:

- A **50% heavier plant** does not stop `episode_swings_up` — the controller still brings the
  pendulum up. That is a robustness result, not a gap.
- **Reversing the cold start** does not stop it either: the pendulum swings up the other way and
  settles just the same. `initial_guess` is what pins the direction.
- **Moving the arm-angle stage weight from 1.5 to 6.0** does not move `reference_episode`, for the
  reason given above. `reference_config` is what pins the weights.

### One thing worth telling the authors

Their reported closed-loop cost metric (`metrics['cost'] = 481.49` in the released episode) cannot be
reproduced from the trajectory they saved with the weights their own `stage_cost` uses — 189.6 with
`cost['x']`, 658.5 with `cost['x_end']`, neither matching. It is a reporting quantity computed after
the run rather than part of the controller, so nothing here depends on it and nothing gates it — and
because the pickled metrics are dropped from the trimmed episode, checking this needs their original
`experiment_np_m3.npz` rather than the vendored copy.

## Scene

`NpmpcRecorder` subclasses `Recorder` directly, because this is a rotating linkage rather than a
planar vehicle: the pose is two angles and the geometry hangs off a `scene → arm → pendulum`
transform chain, below the `world → scene` transform every problem's recorder publishes. The arm frame rotates by φ about the vertical at the top of the shaft; the pendulum
frame sits at the arm tip and rotates by θ about the arm's own axis. Both rods are frame-locked and
logged once per step in body coordinates, so the transforms do the moving.

Topics: `/furuta/state` (θ, φ, τ), `/furuta/plan` (the solved horizon's two angles), `/scene`,
`/scene/horizon`, `/scene/static`, `/tf`, plus the shared `/control` and `/telemetry`. The base, the
shaft and the ±2 rad arm-angle bound go on `/scene/static` once at time zero — the bound as two
radial lines, so an active bound is visible in the picture and not only in the telemetry. The tip
trail and the predicted tip path are in the `scene` frame rather than a body frame, which is why
`furuta_tip` exists and why `tests/viz/test_recording.py` pins it against the transform chain: if
the two ever disagree, the plan floats away from the pendulum it belongs to.

The layout is not generated. In Foxglove Desktop, enable `/scene`, `/scene/static` and
`/scene/horizon` in a 3D panel with the display frame set to `scene`, add plots for the
`upright_error_deg`, `arm_angle` and `slack` telemetry scalars and for `/control`, then export and
check the file in as `foxglove-layout.json`. A scene topic is invisible until the layout turns it
on, so `/scene/horizon` in particular shows nothing until it is enabled.

## Compiler coverage

The IR shape this problem leans on — a dense-matmul stage body with a broadcast weight tail, used
through VMAP over a horizon and differentiated to second order — has a self-contained reproduction in
`tests/integration/test_vmap_mlp.py`: a small MLP used through VMAP over a few stages, with `spjac` and
`sphess` checked against an unrolled twin, a NumPy-scattered dense reference, and finite differences
of the Lagrangian's gradient. It runs unconditionally, so retiring this benchmark cannot silently
drop the coverage.
