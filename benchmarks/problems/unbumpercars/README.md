# Unbumpercars safety filter

This directory is a *representative reproduction* of the centralized safety filter in
`~/dev/bumper_car_simulator`, kept only so Alloy has a realistic workload to be fast on: a
neural model inside pairwise constraints, exact sparse Lagrangian Hessians through `Ops.MAP`,
and a CasADi implementation of the same NLP to compare against.

**It is not the development center for the safety filter itself.** Controller research —
better barriers, better envelopes, new scenarios — happens in `bumper_car_simulator`; what
this benchmark needs from the formulation is only that it is faithful enough to be
representative and behaves well enough (no collisions, no solver failures at the canonical
point) that the measurements mean something. When a formulation choice arises, the tie-break
is benchmark stability, not filter quality. Concretely this gives us a fast iteration loop
for:

- solver/runtime instrumentation on a realistic closed loop,
- inspection of Alloy-generated C kernels,
- identifying missing Alloy features that block a fair replacement of CasADi.

## Model and filter being tested

Two checkpoints from `bumper_car_simulator` are vendored:

```text
benchmarks/problems/unbumpercars/data/dt_kinematic_mlp.pt  # the natively discrete MLPModel (default)
benchmarks/problems/unbumpercars/data/ct_full_xlarge.pt    # the continuous-time CTFullModel
```

No `torch` dependency is required. Both are read with
`alloy.utils.load_torch_state_dict`, then evaluated with plain NumPy for the
simulator and with CasADi / Alloy expressions inside the filters.

Per car state and input are:

```text
x = [px, py, theta, vf, beta_f, beta_r, delta]
u = [u_m, u_s]
```

By default **both the plant and the filter run the discrete `MLPModel`**
(`common.dt_mlp_step_np` for the plant, `common.dt_mlp_step_smooth_np` and its two symbolic
mirrors for the filter) — the more faithful of the two models and the one the colleague's
own HCBF runs on. It is three pieces, of which only the middle is learned: RK4 pose
integration with the velocity block held over the step, the network for the velocity block,
and a first-order steering actuator at the checkpoint's own `tau = 0.155 s`.

`--plant ct` and `--filter-model ct` select the continuous-time `CTFullModel` instead,
discretized with RK4 into a one-step map at `dt = 0.1` (`rk4_step_np` /
`alloy_ctfull_rk4_fn`). Nothing in the barriers depends on how the next state is produced,
so either model can sit on either side; every measurement before 2026-08-11 was taken with
the CT model on both. The long study below of what a *mismatched* pairing costs (DT plant,
CT filter) is kept as the measured argument for the matched default.

### Pair constraint: the order-1 hyperbolic CBF

The pair barrier is the HCBF from `bumper_car_simulator`
(`control/algorithms.py::gradient_HCBF`). With `p = p_j - p_i`, `r = ‖p‖`,
`v = v_j - v_i` the relative world velocity, `v_x = p·v / r` the separation rate,
`v_y = (p x v) / r` the transverse rate, `R` the safety radius and `d = r - R`:

```text
d_eps = sqrt(d^2 + eps^2)                       smooth |d|
s     = d / d_eps                               smooth sign(d)
a_env = envelope_c d_eps^envelope_q             braking envelope V(d)
q     = sqrt(d_eps (r + R)) / R * v_y           lateral-miss allowance
b_ij  = v_x + s (a_env^4 + q^4)^(1/4)
```

`b_ij >= 0` reads "close no faster than you can brake, unless you have enough
transverse velocity to pass wide". Because `b` is built on velocity states, one
step of the discrete model already moves it — relative degree 1. That is what
replaced the earlier position barrier `‖p_i - p_j‖^2 - R^2`, whose relative degree
left the filter with almost no authority over it within one step and forced the
distance down to 1.55 m on the canonical episode; the HCBF holds it at 2.28 m with
a *lower* tracking cost, because it acts early instead of late.

The power-law envelope `V(d) = c_pair d^q` replaces the reference implementation's
tabulated inversion of the full-brake speed map, which is why that implementation
refuses HCBF in its discrete-time mode; the power law is smooth and branch-free. The
shipped `(c_pair, q)` is a single conservative fit covering both vendored models —
the tightest power law that never over-predicts either one's exact discrete pair
stopping envelope. See `common.HCBFConfig`, and the envelope sections below for the
per-model fits it summarizes.

### Wall constraints, slacks, and the NLP

The walls use the order-1 velocity barrier from
`ControllerCBF.gradient_walls_velocity`. For each axis-aligned wall, let `c_w` be
the signed clearance from the `wall_margin` inset, `n_w` its outward unit normal,
and `v_out = n_w dot v` the velocity toward it. The pair envelope is converted back
to its single-car form using `V_pair(d) = 2 V_single(d / 2)`:

```text
d_eps       = sqrt(c_w^2 + wall_eps^2)
s           = c_w / d_eps
c_single    = envelope_c / 2^(1 - envelope_q)
V_single(d) = c_single d^envelope_q
b_iw        = s V_single(d_eps) - v_out
```

Thus `b_iw >= 0` says that a car may approach a wall no faster than it can brake
within its clearance. Outside the inset, the smooth sign flips and demands inward
motion. The filter applies the same one-step DTCBF decrease condition to the pair
and wall velocity barriers. Every row gets its own non-negative slack, penalized in
L1:

```text
b_ij(x[k+1]) - (1 - gamma_pair) b_ij(x[k]) + s_ij   >= 0
b_iw(x[k+1]) - (1 - gamma_wall) b_iw(x[k]) + s_iw   >= 0

z    = [u_0, ..., u_{C-1}, s_0, ..., s_{m-1}],   s >= 0
cost = sum_i ‖u_i - u_des_i‖^2_R + w_slack * sum_k s_k
```

The L1 penalty relaxes only the rows that are actually infeasible; the single
shared slack it replaced was an L-infinity penalty, which let one hard row relax
every other row with it.

The reported `slack_l1` is the sum, so a *fully inactive* step reports a small negative
number rather than zero: IPOPT relaxes every bound by `bound_relax_factor` (default 1e-8)
before solving, and each slack settles just inside its relaxed lower bound at −9.1e−9.
Summed over the 60 rows at `C=8` that is −5.45e−7, which is a fixed floor set by the row
count, not drift. `slack_max` is the largest single slack and reads the same way.

Two radii are tracked separately, as in the reference implementation:
`collision_radius = 1.9 m` is where the body discs touch and is the only thing a
collision is counted against, while `safety_radius = 2.28 m` is what the filter
enforces.

The position-only wall rows previously used here had no control authority in the
default discrete model. Its pose step holds the current velocity block fixed, so
`d h_iw(x[k+1]) / d u` was exactly zero for throttle and steering. IPOPT returned
the desired input unchanged and paid wall slack while the car escaped. In the
one-car seed-42 reproduction (`200` steps, desired `[0.55, 0]`), the old row reached
`-9.627 m` signed inset clearance, spent 100 steps outside, and changed the desired
control by at most `2.5e-7`. The velocity row intervened on 43 steps and limited the
inset crossing to `0.00088 m`, with no slack or solver failure.

The result is not specific to one matching-model case. In one-car DT/DT, CT/CT,
DT/CT, and CT/DT runs, the worst inset crossing was `0.0031 m`. Across DT/DT seeds
`{42, 1, 2}` at the canonical `C=8`, 200-step size, it was `0.0036 m`, with zero
collisions and zero solver failures. Since the constraint rectangle is inset by
`1 m`, these millimetre-scale sampled crossings still leave every car about a metre
inside the physical arena.

## Putting the discrete MLP in the filter too: it fixes everything

Status: **the default.** Plant and filter both run the discrete MLP; `--plant ct` and
`--filter-model ct` select the continuous-time model for either side.

Everything below this section documents the filter predicting with the RK4'd continuous-time
model against a discrete-MLP plant, and the trouble that mismatch causes — it is kept because it
is the measured argument for the default being what it is. Giving the filter the plant's own model
removes all of it. `C=8`, 200 steps, seeds `{42, 1, 2, 3, 7}`, Alloy:

| | `--filter-model ct` (the old default) | `--filter-model dt` (now the default) |
|---|---|---|
| worst min pair distance | 1.775 m | **2.266 m** |
| per-seed min distance | 1.775–2.078 m | 2.266–2.273 m |
| steps inside the 1.9 m collision radius | 9 / 1000, in 2 of 5 episodes | **0 / 1000, in 0 of 5** |
| mean tracking cost | 32.29 | **4.07** |
| mean IPOPT iterations | 19.3 | **14.1** |
| mean solve, Alloy | 37.8 ms | **31.7 ms** |
| solver failures | 0 | 0 |

Safer, 8x cheaper to track, *and* faster despite a network with 7.3x the weights — because the
discrete model is already a one-step map, so the oracle evaluates it once where the RK4 path
evaluates its smaller network four times, and the better-conditioned problem needs a third fewer
iterations. This is the same 2.27 m and tracking ~4 the filter achieved back when its model *was*
the plant, so the mismatch penalty is simply gone rather than traded away.

**The chaos goes with it.** The sensitivity documented below is a symptom of the mismatch, not a
property of the plant: with the matched model a 1e-9 perturbation amplifies 1.015x per step
instead of 1.329x and is still 3.4e-9 after 80 steps. Episodes are reproducible again, and the
the two oracle providers agree on the whole rollout — identical tracking cost to four digits.

**And the envelope stops being load-bearing.** With the matched model, the honest DT-fitted
envelope `(1.0118, 0.84)` and the CT one `(1.456, 0.5)` are indistinguishable
(2.266 m / 4.07 tracking versus 2.270 m / 4.23), where against the mismatched filter the choice
between them was worth 24 colliding steps. That freedom is spent on stability: the shipped
default is a **single conservative fit `(1.00994, 0.8355)`** — the tightest power law that
never over-predicts *either* model's exact pair stopping envelope on `d ∈ [0.1, 3] m` — so
every plant/filter pairing runs under one set of constants. At the canonical point it is
collision- and failure-free for both matched pairings (dt/dt min distance 2.273 m, tracking
3.99; ct/ct 2.280 m, 3.58). This benchmark is a representative reproduction rather than the
filter's development center, so one stable, slightly conservative fit beats per-model tuning.

Two departures from the plant are deliberate, both gated by `dt_filter_model_matches_numpy`:
the ReLUs are smoothed (`common.DT_RELU_EPS`, worst 3 mm/s on the velocity block) because IPOPT
wants C², and the `vf` deadzone is dropped because it is a jump discontinuity that never fires
in these episodes. The one place they genuinely disagree is `vf` near zero under hard braking,
where the network extrapolates negative and the plant clamps to a standstill; no state in any
measured episode reached it.

**Alloy's margin widens.** Same NLP, same IPOPT, seed 42, 200 steps, `--filter-model dt`:
32.4 ms per solve for Alloy against **292.1 ms** for CasADi, a **9.0x** gap where the
continuous-time model gave 3.8x. The bigger network is where the oracle provider starts to
matter: FE is 41 ms of Alloy's solve and 374 ms of CasADi's.

## The discrete-time MLP plant

Status: **adopted as the plant, and since then as the filter's model too** (see the previous
section). This section documents the adoption as the plant and the mismatch era it opened —
running the more faithful model as the plant while the filter kept the RK4'd continuous-time
one is how we found out what the mismatch actually costs, and those measurements are why the
matched pairing is now the default.

The natively discrete `MLPModel` from the reference repo is the more faithful model — it is
what the colleague's own HCBF runs on, and it is a one-step map rather than an integrated
ODE, so it would need no RK4 inside the filter. Its checkpoint is **not** in the public
`bumper_car_simulator`; it came from `~/dev/unbumpercars/model_kinematic_mlp.pth` and is
vendored here as `data/dt_kinematic_mlp.pt`, byte-identical (143,104 bytes; keys
`model.{0,2,4}.{weight,bias}`; `6 -> 256 -> 128 -> 3` with two ReLUs, matching `MLPModel`'s
defaults exactly). No normalization is stored in the checkpoint — it comes from the
`x_scaling = [1.9, 0.6, 0.1, 2.012]` constant in code, kept as `common.DT_X_SCALE`.

Its forward pass, with `alpha_f = beta_f - delta` and `alpha_r = beta_r`:

```text
features = [vf/1.9, alpha_f/0.6, alpha_r/0.1, delta/2.012, u_s, u_m]
outputs  = [vf+, alpha_f+, alpha_r+]        absolute next-state values, not deltas
```

Two things to carry forward. **The controls are in the opposite order to ours** — `u_s`
before `u_m` — confirmed by the model's author and independently recovered by checking which
assignment produces sensible accelerate/brake behaviour; `check_dt_plant_control_order` gates
it, and the swapped order leaves under 0.035 m/s of throttle-versus-brake authority where the
correct one has 0.089–0.160. And the "DT model" is three pieces, only one of them learned:
analytic RK4 pose integration with the velocity frozen over the step, this network for the
velocity block (with a hard `vf+ = 0` below 0.03 m/s), and a first-order steering actuator at
`tau = 0.155 s` — a third of our `tau * 3 = 0.465 s`. The plant adopts that faster actuator,
because it is what the network was trained against; `check_dt_plant_pieces` gates the split
and the actuator's time constant.

### What the mismatch costs

`C=8`, 200 steps, IPOPT with Alloy oracles, exact Hessians, same filter throughout — only the plant
differs. **Aggregated over seeds `{42, 1, 2, 3, 7}`, because single episodes on the DT plant
are not decision-grade** — see the noise floor two sections down:

| | CT plant (`--plant ct`) | DT plant (`--plant dt`, default) |
|---|---|---|
| worst min pair distance over 5 seeds | 2.274 m | **1.775 m** |
| per-seed min distance | 2.274–2.280 m | 1.775–2.078 m |
| steps inside the 1.9 m collision radius | **0** / 1000 | **9** / 1000, in 2 of 5 episodes |
| steps inside the enforced 2.28 m | 340 / 1000 | 809 / 1000 |
| largest `slack_l1` | 0.324 | **2.08** |
| mean tracking cost | 4.05 | 32.29 |
| solver failures | 0 | 0 |

**The DT plant breaches the collision radius.** The filter never fails, but it is being asked
to protect a plant it does not describe: the barrier promises braking this plant cannot deliver
at close range, rows go infeasible, the L1 slack absorbs the difference — and in 2 of 5
episodes the body discs touch. The margin that was supposed to absorb this is
`safety_radius - collision_radius = 0.38 m`, and the worst case eats 130% of it. The plant is
also slower on average (mean speed 0.863 versus 1.323 m/s) because its low-speed creep and
12.5% lower top speed leave it further from the desired input, which is most of the
tracking-cost gap.

### Why the two oracle providers' trajectories differ

*This section describes `--filter-model ct`. Under `--filter-model dt` the loop is not sensitive
and none of it applies.*

With the mismatched filter, the Alloy and CasADi rollouts visibly separate — different paths,
different per-step slacks. **This is not an oracle disagreement.** Three measurements pin it down:

1. **Per step, on the same state, the two providers agree to 7e-14** in the commanded input,
   with identical IPOPT iteration counts, over 60 steps at `C=8` (largest disagreement across
   all steps: 7.3e-14). They are solving the same NLP to the same point. `oracles_solve_alike`
   gates this.
2. **The same provider against itself diverges identically.** Perturb one car's initial speed by
   1e-9 and run Alloy twice: with the DT plant the gap grows to 5.8e-4 by step 39 and 5.8 by
   step 79, a geometric mean amplification of **1.33x per step**. With the CT plant the same
   perturbation ends at 3.2e-8 — amplification 1.045x per step, i.e. flat.
3. **The plant map alone is not the amplifier.** Open loop under a fixed control sequence, a
   1e-12 perturbation grows 1.6x over 80 steps under the DT map and 4.5x under the CT map.
   Both are neutral; the velocity-block Jacobian's largest singular value averages 1.20 (DT)
   and 1.05 (CT).

A fourth, added later, identifies the cause: **giving the filter the plant's own model removes
the amplification entirely** (1.015x per step). So it is the mismatch that is unstable, not the
plant. The filter's
state-to-input gain has a long tail — median 1.0, p90 5.2, max 27 along a DT rollout — because
the plant brakes far less than the barrier assumes, so the rollout sits against the constraint
boundary with rows going in and out of activity and the L1 slack at its kink. Composed with the
plant, that is a positive Lyapunov exponent: **any** perturbation grows, and the difference
between two correct oracles' rounding is simply the smallest one available. It reaches O(1) by
step ~75, so beyond roughly 55 steps the two rollouts are independent samples of the same
closed loop rather than the same trajectory. The deadzone is not involved — no car reaches
exact standstill in these episodes.

**The consequence is a noise floor on every single-episode aggregate.** Same config, same seed,
initial speed of one car nudged by the amount shown, 200 steps:

| nudge | min pair distance | steps inside the collision radius | mean tracking cost |
|---|---|---|---|
| none | 1.775 m | 6 | 29.74 |
| 1e−12 | 1.829 m | 5 | 32.83 |
| 1e−9 | 2.060 m | **0** | 31.59 |
| 1e−6 | 1.918 m | 0 | 30.31 |

A perturbation twelve orders of magnitude below anything physical moves the collision count
between 0 and 6 and the tracking cost by 10%. So on this plant a single episode cannot support
a claim about either, and **comparing two configurations or two oracle providers by one rollout each is
measuring the noise**. Aggregate over seeds instead; the tables above and below do. (An earlier
version of this section compared the providers' single-episode aggregates and found them "under
1% apart" — that was one lucky pair of draws, not a property of the providers.) With
`--plant ct` the question does not arise: the two providers' rollouts are bit-identical and the
per-seed spread is 6 mm.

Two consequences: compare oracle providers **per step on a shared state**, not by rollout, and take any
head-to-head *timing* claim on `--plant ct`, where both walk the same path by construction.

### Why the envelope does not describe it

**The two models brake with opposite speed dependence, and `a_brake` is fitted to ours.**
Per-step speed loss under full brake (`u_m = -1`, straight, `dt = 0.1`):

| vf | 0.25 | 0.50 | 1.00 | 1.50 | 2.00 |
|---|---|---|---|---|---|
| CT (ours), loss per step | 0.069 | 0.064 | 0.054 | 0.044 | 0.035 |
| DT (MLP), loss per step | 0.032 | 0.055 | 0.067 | 0.099 | 0.181 |

Ours is near constant-deceleration and slightly *decreasing* with speed — the premise
the old CT-only fit rested on and the reason `V(d) = sqrt(2 a_brake d)` fits it to ~3%. The DT model
is drag-like: across the full 0.05–2.2 m/s span its per-step loss grows 40x, from 0.005 to
0.22 m/s, so at low speed it creeps almost indefinitely. Exact discrete stopping distance
`D(v) = dt v + D(F(v))` and its inverse `V(d)` diverge accordingly:

| stopping distance `D(v)` [m] | v = 0.25 | 0.50 | 1.00 | 1.50 | 2.00 |
|---|---|---|---|---|---|
| CT (ours) | 0.058 | 0.212 | 0.889 | 2.226 | 4.502 |
| DT (MLP) | 0.188 | 0.401 | 1.003 | 1.769 | 2.429 |

| single-car envelope `V(d)` [m/s] | d = 0.10 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|
| CT (ours) | 0.338 | 0.763 | 1.055 | 1.433 |
| DT (MLP) | **0.136** | 0.600 | 0.998 | 1.650 |

The crossover is the problem: near contact the DT model needs a far tighter envelope than
ours, and far away a looser one.

The barrier bounds a *pair's* closing speed, so what it needs is the pair envelope. With both
cars braking from half the closing speed, the gap closes by `2 D(v/2)`, so

```text
V_pair(d) = 2 V_single(d / 2)
```

which for `D(u) = u^2 / 2a` gives `sqrt(2 (2a) d)` — the `a_pair = 2 a_single` convention
`HCBFConfig` already uses. Fitting the *pair* envelope directly on `d ∈ [0.1, 3] m`:

| model | conservative-everywhere `a_pair` | best-fit `a_pair`, mean / max error |
|---|---|---|
| CT (ours) | **1.067** | 1.102, 1.7% / 3.5% |
| DT (MLP) | **0.114** | 1.026, 24.2% / 200.4% |

The CT column is the cross-check: the then-shipped `a_brake = 1.06` *is* that model's
conservative-everywhere fit, to three digits. So the same computation applied to the DT model
is trustworthy, and it says today's constant is **9.3x too bold** for it: `1.06` over-predicts
the DT pair envelope out to `d = 2.37 m` — essentially the whole live range — by up to 205% at
`d = 0.11 m`. (An earlier version of this section reported 1.19 m and 140% from comparing the
promised pair envelope against the *single-car* curve; the pair relation above is tighter near
contact, so the blocker is about twice as bad as first recorded.) Being conservative everywhere
needs `a_pair = 0.114`, which would make the filter uselessly timid. This is precisely why the
reference implementation tabulates the inversion and refuses HCBF in its discrete-time mode.

### What moving the filter onto it took

*All four items are now done; see the top of this file for the result. Kept because the
measurements behind each one are the reason the change was scoped the way it was.*

1. **A refitted envelope — tried, and it does not work. See "the envelope refit" below.**
   The form change itself was right and is shipped: `HCBFConfig` now carries
   `V(d) = envelope_c d^envelope_q`, and a power law is what can describe either model —

   | form | CT: constant, mean / max error | DT: constant, mean / max error |
   |---|---|---|
   | `sqrt(2 a d)` | `a = 1.10`, **1.7% / 3.5%** | `a = 1.03`, 24.2% / 200.4% |
   | `kappa d` | `kappa = 1.03`, 25.5% / 77.7% | `kappa = 0.98`, 13.2% / 35.5% |
   | `c d^q` | `c = 1.51, q = 0.486`, **1.0% / 6.5%** | `c = 1.13, q = 0.827`, **4.1% / 14.5%** |

   — but *refitting the constants to the DT plant made the closed loop less safe*, for a reason
   that has nothing to do with fit quality. The envelope does double duty, and only one of its
   two jobs is a braking claim.
2. **Smoothness — cheaper than it looked, and softplus was not the answer.** The
   `vf+ = 0 below 0.03 m/s` deadzone is simply left out of the filter: no sampled state over a
   60-step 8-car rollout comes near it (slowest `vf` 0.050 m/s). For the ReLUs, softplus needs
   `beta = 50` to keep the worst error in the predicted next `vf` to 0.019 m/s (at `beta = 10`
   it is 0.165 m/s, the whole signal). What shipped instead is the smooth-|x| form already used
   for `|d|`, `relu(x) ≈ (x + sqrt(x² + eps²)) / 2` at `eps = 0.01`: **0.005 m/s worst error, a
   quarter of softplus at `beta = 50`**, one `sqrt` instead of an `exp` and a `log` across 384
   activations per car per step, and no overflow to reason about — softplus at `beta = 50` needs
   `exp(50 x)`, which is safe only because the measured pre-activations stay under 1.4. In the
   event neither stiffness worried IPOPT: iterations went *down*. The plant is stepped rather
   than differentiated, so it keeps the reference's exact ReLU and deadzone.
3. **The steering actuator.** Already done on the plant side: it runs the checkpoint's
   `tau = 0.155 s`, which is what the network was trained against, while the filter still
   predicts with `tau * 3`. Moving the filter over *removes* that mismatch rather than adding
   one. It accounts for nearly all the raw `beta_f`/`delta` disagreement between the two
   models (74% of the CT model's own step change once matched, 234% when not).
4. **Oracle cost — a 2x forward graph, not 7x, and in the event it came out faster.** The DT
   network is `6 -> 256 -> 128 -> 3`, 35,075 weights against the CT model's 4,803. But it is
   already a one-step map, so the oracle evaluates it **once** where the CT path evaluates its
   small network four times for RK4: 34,688 versus 18,688 multiply-accumulates per car per step,
   a factor 1.86. The prediction of "roughly 2x slower" was wrong in the right direction —
   Alloy's mean solve *fell* from 37.8 to 31.7 ms, because the better-conditioned problem needs
   a third fewer IPOPT iterations, which more than pays for the heavier oracle. The cost shows up
   in build time instead (5.1 s to 10.9 s) and, far more sharply, in CasADi (72.6 ms to 292.1 ms
   per solve) — the scalability question this problem exists to surface, answered.

### The envelope refit: the constant does double duty

Status: **the failure below is a mismatch artifact, and it is why the refit was rejected at
the time.** With the matched filter model the choice stopped mattering, and the shipped
constants are now the single conservative both-model fit described in "Putting the discrete
MLP in the filter too". This section documents why refitting the constants *under mismatch*
made the closed loop less safe.

The barrier is `b = v_x + smooth_sign(d) V(d_eps)`, and the sign flip is the whole story:

```text
outside the safety radius (s = +1):  b >= 0  <=>  closing speed <= V(d)     a braking claim
inside  the safety radius (s = -1):  b >= 0  <=>  separation    >= V(d)     a recovery gain
```

One constant sets both. Fitting it honestly to the plant's true braking makes the *recovery*
demand collapse, and this plant spends 809 of 1000 steps inside the safety radius, so that is
the regime that dominates. Measured over the pair-steps spent inside the radius:

| | CT `sqrt`, `c=1.456 q=0.5` | honest refit, `c=1.012 q=0.84` |
|---|---|---|
| mean separation the barrier demands | 0.423 m/s | **0.164 m/s** |
| mean separation achieved | +0.119 m/s | +0.049 m/s |
| mean penetration depth | 0.065 m | **0.099 m** |
| max penetration depth | 0.220 m | **0.445 m** |

And the closed-loop outcome, over seeds `{42, 1, 2, 3, 7}` x 200 steps:

| envelope | worst min distance | steps inside collision radius | episodes with a breach | mean tracking |
|---|---|---|---|---|
| CT `sqrt`, `c=1.456 q=0.5` | 1.775 m | **9** | 2 / 5 | 32.29 |
| honest refit, `c=1.012 q=0.84` | 1.831 m | 33 | 4 / 5 | 27.22 |
| refit x1.5, `c=1.518 q=0.84` | 1.781 m | 23 | 2 / 5 | 27.60 |
| refit x2.0, `c=2.024 q=0.84` | 1.590 m | 55 | 4 / 5 | 30.62 |

Three things follow. **Scaling is not a way out** — the honest fit is worst on safety, and
scaling it up trades the weak-recovery failure for the over-promised-braking failure without
ever beating the incumbent. **The refit does buy tracking cost** (27.2 versus 32.3), so this is
a real trade-off rather than a strictly worse setting. And **no setting tested makes this plant
safe**: even the incumbent breaches in 2 of 5 episodes. A better-fitted envelope cannot fix a
filter that predicts with the wrong model, which is what pointed at the model swap.

That was the right read. Under `--filter-model dt` the conflation stops mattering — the honest
refit and the incumbent land within 4 mm of each other and neither collides — because the filter
no longer needs the recovery role to compensate for a prediction it cannot trust. **So the
two-constant split is not worth doing.** It would decouple a knob whose setting only matters when
the model is wrong. Keeping the note here because the mechanism is real and would return with any
other mismatched plant: the barrier's envelope is a braking claim outside the radius and a
recovery gain inside it, and those two jobs only agree when the prediction is honest.

### Where they already agree

Identical state and control conventions (modulo the swap), identical analytic pose
kinematics, the same three learned channels with the same meaning, and acceleration that
matches closely mid-range: per-step `vf` disagreement of 0.024–0.035 m/s on states drawn
from either model's own rollout, about half the CT model's own step change, and near-exact
agreement at `vf = 1.0–1.5` under `u_m = +1`. Top speed differs by 12.5% (1.797 vs 2.053
m/s). So moving the *filter* onto it is a scoped formulation change, not a rewrite — but it
*is* a formulation change, not a checkpoint swap, and it has to be re-measured against the
same collision/failure gates. `ROADMAP.md` §2.6 carries the roadmap-level version.

### Reproducing the numbers

Both checkpoints are now vendored and both one-step maps are in `common.py`
(`rk4_step_np`, `dt_mlp_step_np`), so every number above regenerates from this repo alone;
`torch` is still not a dependency. To redo the braking measurements: for each model sweep
`vf` with `u_m = -1` and the rest of the state at zero to get the full-brake
one-step map `F(v)`. Then accumulate `D(v) = dt * v + D(F(v))` down to a small speed
threshold (converges in under 50 steps over `[0, 2.2]`; insensitive to the threshold to five
digits), invert it on a `d` grid to get `V(d)`, and fit the power law `c d^q` (or
`sqrt(2 a d)` for the historical CT constant) over `d ∈ [0.1, 3] m`. The shipped constants
are the tightest `c d^q` under the pointwise minimum of the two models' pair envelopes. The `dt * v` term per step is the right one for this plant: the pose is
integrated with the velocity held over the step. The episode table comes from two runs of
`run_closed_loop` differing only in `--plant`.

## Provenance and adaptations

Upstream: [`simonebaratto/bumper_car_simulator`](https://github.com/simonebaratto/bumper_car_simulator)
at commit `2a822c6fbab5a35fd8e1b9eceda7b16a8471083e` (2026-08-06, *"Refactor braking
profile methods across models for consistency and enhanced functionality"*). The
files that matter are `control/algorithms.py` (`ControllerCBF.gradient_HCBF`,
`ControllerCBF._world_velocity_and_partials`, `CentralizedCBF._compute_safe_input_dt`),
`sysid/models/mlp_model.py` (`MLPModel.braking_profile`) and `config.py`.

Every deliberate divergence, so a future upstream bump can be diffed against this list:

| # | Upstream | Here | Why |
|---|---|---|---|
| 1 | `gradient_HCBF` returns `(h, h_dot, grad_h_dot_i_j, grad_h_dot_j_i)` — two pages of hand-written gradients | only the barrier value (upstream's `h_dot`; its `h` is identically 0) | CasADi and Alloy differentiate it. This is the single largest simplification and the reason the port is short. |
| 2 | `MLPModel.braking_profile`: tabulates `D(v)` on a 2048-point grid from the discrete full-brake recursion, then inverts it | `V(d) = c d^q`, one conservative fit under both models' exact recursions | The table has no symbolic counterpart — the stated reason `CentralizedCBF` refuses HCBF under `time_domain="DT"`. |
| 3 | Envelope saturated at `2 * v_cap` (`saturate=True`) | no saturation | Capping the envelope *tightens* the constraint at long range, where nothing is at risk. Ours is only queried where the row is live. |
| 4 | Constraint imposed in continuous time through Lie derivatives (HCBF and velocity walls are CT-only upstream) | DTCBF decrease condition on the filter model's one-step prediction | Both barriers have relative degree 1, so the discrete model suffices. |
| 5 | `gamma = clip(wn * dt, 0, 1)` = 0.1 at the sweep's `wn = 1.0` | `pair_gamma = wall_gamma = 0.35` | Retuned for our arena and step; 0.35 converged with zero collisions and zero solver failures. |
| 6 | `_world_velocity_and_partials` uses `vy_b = vf (lf sin bf + lr cos bf tan br) / (lf + lr)` | the pose rows of our own ODE, `vy_b = vf sin bf - lf omega` | **Numerically different**: expanded, ours is `vf (lr sin bf + lf cos bf tan br) / (lf + lr)` — the same expression with `lf` and `lr` swapped, i.e. a different body reference point. We take our model's own velocity so the barrier constrains exactly what the plant propagates. |
| 7 | `eps = 1e-4` in the smooth sign `s = d / sqrt(d^2 + eps^2)` | `eps = 0.05` m for pairs; `wall_eps = 1e-4` m for walls | Pair contact is common and the larger smoothing keeps the NLP well conditioned. Wall contact should remain sharp and is rare because the boundary is inset by 1 m. |
| 8 | One slack per pair plus one per car, squared penalty, weight 1e6 (DT path); a single shared slack (CT path) | one slack per constraint row, L1, weight 1e3 | L-infinity lets one hard row relax all the others. |
| 9 | `gradient_walls_velocity` uses four axis walls plus four corner cuts | the same velocity barrier on the four axis-aligned inset walls, without corner cuts | Retains this benchmark's rectangular arena and fixed four-rows-per-car scaling while gaining the reference barrier's control authority. |
| 10 | `cooperative` flag toggles a factor 2 on the braking envelope | folded into the pair envelope constant | A centralized filter commands both cars, so cooperative braking is the correct assumption; there is no second case to switch on. |
| 11 | `gradient_HCBF_deflated` (`kq = 0.8`, `kp = 1.0`), `active_cars` masking, `input_time_constant` steering lag | not ported | Not needed to exercise the oracle path this benchmark measures. |

Scenario constants also differ, since ours were inherited from the previous version of
this problem: 15x15 m arena and 8 cars here versus upstream's `[-3.5, 3.5] x [-4, 6.5]`
and 3 cars; `collision_radius` 1.9 versus 2.0. The `safety_factor` of 1.2 is upstream's.

Upstream's HCBF sweep runs on `MLPModel`, a natively discrete network. That is the default
model here too, for both the plant and the filter's prediction (`ROADMAP.md` §2.6).

## Running

From the repository root:

```bash
uv run python benchmarks/run.py closed-loop --problem unbumpercars --smoke
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver ipopt --oracle casadi
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver sqp --oracle alloy
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver sqp --oracle casadi
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver none
uv run python -m benchmarks.problems.unbumpercars.run_closed_loop --solver ipopt --oracle both --dump-alloy-c
```

Runs write under `benchmarks/results/closed-loop/unbumpercars/<solver>+<oracle>/`
(`none/` for open loop); import this directory's hand-authored
`foxglove-layout.json` in Foxglove Desktop to view it. The direct module remains
useful for side-by-side Alloy/CasADi runs and advanced filter options.

Common options:

```bash
--ncars 4
--steps 80
--no-walls
--plant ct           # step the plant with the RK4 map of the continuous-time model
--filter-model dt    # predict with the discrete MLP instead; see the top of this file
--out-dir benchmarks/results/closed-loop
--show
```

Outputs are written under `<out-dir>/<solver>+<oracle>/`:

```text
rollout.npz                  state/input trajectories
stats.csv                    per-step solver + oracle instrumentation
summary.json                 aggregate metrics and implementation metadata
episode.mcap                 Foxglove telemetry and planar car scene (see below)
representative_fe_inputs.npz closed-loop oracle input for FE benchmarks
config/provenance/metadata   reproducibility data
trajectories.png             matplotlib trajectory plot
performance.png              solve/evaluation timing plot
alloy_c/                     generated C kernels when --dump-alloy-c is used with Alloy
```

Generated outputs belong under `benchmarks/results/`.

### Scene geometry

The 3D scene draws what the filter actually constrains, at the scale the filter
implies. Each car gets its own colour and shows:

- a body box `lf + lr + 0.33` m long and 0.9 m wide, centred `(lf - lr) / 2` ahead
  of the state position (the state tracks the centre of gravity, not the geometric
  centre);
- the body disc of radius `collision_radius / 2 = 0.95` m, which circumscribes the
  body box; two of these touching is what counts as a collision;
- a fainter ring of radius `safety_radius / 2 = 1.14` m — the separation the pair
  barrier actually enforces, 0.19 m of margin per car around the body disc;
- two arrows from the car, pointing along `theta + u[1] * max_delta` with length
  scaled by the throttle `u[0]`: the desired input in the car's colour and the
  applied input in black. A failed solve brakes (`u = [-1, 0]`), which shows up as a
  black arrow flipped to point backwards.

The arena is drawn twice: the walls, and the rectangle inset by `wall_margin`, which
is the region the wall barriers keep the car centres inside.

### Why it is called hyperbolic

Setting `b = 0` with `s = 1` and rearranging:

```text
v_x + (a_env^4 + (c_env v_y)^4)^(1/4) = 0   <=>   v_x^4 - c_env^4 v_y^4 = a_env^4
```

which is one branch of a quartic hyperbola in the plane of the pair's *relative velocity* —
vertex at a closing speed of `a_env`, i.e. brake and stop in time, and asymptotes of slope
`c_env`, i.e. pass wide enough to miss. With the exponent 2 instead of 4 it is literally
`v_x^2 - c_env^2 v_y^2 = a_env^2`, a textbook hyperbola; the 4 only sharpens the corner
between the two regimes toward a hard maximum of them. Nothing in position space
corresponds to this, which is why the scene draws only the two discs.

## Implementations

### CasADi

`CasadiDTCBFSafetyFilter` builds one MX NLP with IPOPT, with `expand=True` and an exact
Lagrangian Hessian (plus a separate `ca.Function` for the same Hessian, for
instrumentation). `--limited-memory-hessian` swaps both IPOPT oracle providers to
`ipopt.hessian_approximation = limited-memory` instead.

Instrumentation recorded per step:

- IPOPT wall time,
- IPOPT status and iteration count,
- objective, min constraint value, the L1 slack total (largest single slack under `max_slack`),
- explicit `ca.Function` timings for `f`, `g`, `grad_f`, `jac_g`, and optionally
  `hess_lag`,
- CasADi's reported `n_call_*` counters.

Important caveat: the explicit instrumentation functions are close to, but not
necessarily identical to, the exact internal oracle functions CasADi wires into
IPOPT.

### Alloy

`AlloyDTCBFSafetyFilter` builds an Alloy oracle with outputs:

```text
cost(z, bar_x, u_des, weights, physics, dt)
g(z, bar_x, u_des, weights, physics, dt)
```

The RK4 neural dynamics are evaluated with `al.map_` over the car axis, so the
prototype exercises the mapped neural dynamics path we care about. The filter
then creates Alloy factories for:

- `grad:cost:z`,
- `spjac:g:z`,
- `sphess:gamma:z:z` with `gamma = lam:cost * cost + dot(lam:g, g)`, unless
  `--limited-memory-hessian` is passed.

The whole solve is one `al.nlp(...)` SolverFunction: the derivative factories
above are built inside `al.nlp`, and the filter runs through a generated C
solver wrapper with no Python callbacks in the loop. With `--solver ipopt`, the
wrapper drives `IpStdCInterface.h`; with `--solver sqp`, `alloy-sqp` assembles
sparse PIQP subproblems and can use either Alloy- or CasADi-generated oracles.
The SQP controller tries its default filter globalization first and retries a
strict failure from the same warm start through l1/watchdog-five; reported time
and evaluation counts include both attempts. Warm starts carry the primal
iterate plus the constraint and box multipliers between steps
(`lam_ineq0`/`lam_box0`). With `--dump-alloy-c`, the full solver module
(wrapper + kernels) is rendered to inspectable C files.

Instrumentation recorded per step (from the `alloy_solver_stats` struct):

- total solve time with the FE / solver / QP / globalization / glue split,
  status (alloy + native), and iteration count,
- objective, min constraint value, the L1 slack total (largest single slack under `max_slack`),
- per-oracle-function evaluation counts,
- Alloy build/JIT-compile timings,
- sparse Jacobian nnz and lower-triangular Hessian nnz.

## Interpreting the IPOPT Alloy-vs-CasADi timings

Both implementations use IPOPT and warm-start primal variables plus constraint
and box multipliers. Alloy runs entirely through the generated C solver wrapper:
IPOPT callbacks call generated kernels in the same `.so`, with no Python or
ctypes callback in the solve loop. Its stable stats ABI separates FE, native
solver, and wrapper/glue time and records evaluation counts. CasADi reports its
own solver and callback counters, so the categories are close but not guaranteed
to have identical accounting boundaries.

The default comparison uses exact Lagrangian Hessians on both sides;
`--limited-memory-hessian` selects IPOPT's approximation instead. Both paths receive the same
symbolic model constants and `dt`, and CasADi expansion is enabled unless
`--no-casadi-expand` is passed. Strict comparisons should retain raw IPOPT
status in addition to the benchmark's strict solver-success and feasibility
rule.

## HCBF migration results (2026-08-11)

> These performance measurements predate the 2026-08-12 velocity-wall correction above.
> They describe the same pair HCBF and model, but the older, cheaper position wall graph;
> regenerate the scalability sweep before quoting them for the current formulation.

Position DTCBF → HCBF at `C=4`, 80 steps, seed 42, IPOPT with Alloy oracles — the operating point
before the canonical one grew, so the two columns are directly comparable:

| | position DTCBF | HCBF |
|---|---|---|
| min pair distance | 1.552 m | 2.280 m |
| steps inside the 1.9 m collision radius | 40 / 81 | 0 / 81 |
| average tracking cost | 14.72 | 1.68 |
| average IPOPT time | 18.7 ms | 7.5 ms |
| solver failures | 0 | 0 |

Across `C ∈ {4, 8}` × seeds `{42, 1, 2, 3, 7}`: no collisions, no solver failures,
minimum pair distance 2.279–2.280 m in every episode, largest slack anywhere 0.18.

### Alloy vs CasADi

Same NLP, same IPOPT, same options — only the oracle provider differs.
[`docs/scalability.md`](../../../docs/scalability.md#discrete-time-hcbf-safety-filter-unbumpercars)
carries the current numbers across `C ∈ {2, 4, 8}` and both filter models, and is the place to
update: **the short version is 2.8–3.9x on the continuous-time model and 4.5–9.0x on the
discrete MLP**, the advantage growing with both car count and network size, with identical IPOPT
iteration counts in every cell.

What follows is the original per-solve breakdown at the canonical point (`C=8`, 200 steps) in
both Hessian modes, kept for the Hessian-mode comparison the scalability doc does not repeat.
**Measured with the CT plant** (`--plant ct`) and the CT filter model, where both oracle providers walk a
bit-identical state sequence:

| | | total | p95 | IPOPT iters | FE | native solver | glue |
|---|---|---|---|---|---|---|---|
| exact Hessian (default) | alloy | 19.27 ms | 25.87 ms | 9.5 | 17.72 ms | 1.33 ms | 0.21 ms |
| | casadi | 72.60 ms | 97.21 ms | 9.5 | — | — | — |
| `--limited-memory-hessian` | alloy | 17.60 ms | 22.08 ms | 19.2 | 11.08 ms | 6.32 ms | 0.20 ms |
| | casadi | 56.07 ms | 69.94 ms | 19.3 | — | — | — |

Alloy is **3.8x faster with exact Hessians, 3.2x with limited-memory**, and the matching
iteration counts say IPOPT walks the same path in both columns — so the gap is oracle
evaluation and call overhead, not a different solve.

The `—` cells are what the table was written with; CasADi's FE share is now measured rather
than estimated. `nlpsol` accumulates per-callback wall time in `stats()` under
`t_wall_nlp_*`, and the filter reports their sum as `eval_ms["fe_total"]`, the same quantity
Alloy's `stats.t_fe` carries — so `stats.csv` and the Foxglove `fe_time_ms` channel are now
populated for both oracle providers. On a spot check (`C=4`, 30 steps, exact Hessian, DT plant) FE is
**90% of CasADi's solve wall time**. The per-function columns beside it are still single
re-evaluations at the solution, so they do not sum to `fe_total`; at `C=8` exact they were
`f` 0.199, `g` 0.448, `grad_f` 0.185, `jac_g` 0.813 and `hess_lag` **3.711** ms — the Hessian
alone is most of the difference.

Exact Hessians halve the iteration count (19.2 → 9.5) and nearly eliminate IPOPT's own
time (6.32 → 1.33 ms for Alloy), at the cost of evaluating `sphess:gamma:z:z` every
iteration (FE 11.08 → 17.72 ms). Net wall clock is a wash for Alloy and clearly worse for
CasADi. They are the default anyway, because the sparse-Hessian-through-`Ops.MAP` path is
what this problem exists to exercise.

With exact Hessians the two oracle providers stay bit-for-bit together over the whole episode
(average tracking cost 4.254 both, minimum distance 2.274 m both). Under limited-memory
they drift slightly apart by the end (3.99 vs 3.91) — the closed loop amplifies last-bit
differences in the iterate over 20 s. Neither has a collision or a solver failure.

## Alloy features closed by this prototype

The original prototype exposed the following gaps; all are now closed on the
Alloy path.

### 1. Exact sparse Hessian through `Ops.MAP` (closed)

IPOPT's exact Hessian path would require the sparse Hessian of the Lagrangian
with respect to `z` through the mapped RK4 neural dynamics:

```text
sphess:lagrangian:z:z
```

The oracle deliberately uses `al.map_` to evaluate the per-car neural RK4 model.
Alloy now propagates reverse and sparse second-order AD through `Ops.MAP` while
preserving the compact mapped representation. The filter builds
`sphess:gamma:z:z`, passes its lower-triangular sparsity to IPOPT, and evaluates
it from IPOPT's objective factor and constraint multipliers.

### 2. Native generated-C solver path (closed)

The filter is an `al.nlp(...)` SolverFunction. Its generated C wrapper calls
`IpStdCInterface.h` directly, routes IPOPT callbacks to generated oracle kernels,
and fills Alloy's stable stats struct with FE/solver/glue timings.

### 3. IPOPT warm-start/status parity (closed)

The low-level Alloy IPOPT wrapper now exposes and accepts the same data we use
from CasADi:

- previous `lam_x`,
- previous `lam_g`,
- final multipliers,
- iteration count,
- value-callback counts.

The closed-loop Alloy filter reuses these multipliers after successful solves
and reports iteration and callback statistics alongside CasADi's measurements.

### 4. Callback accounting (closed)

The generated wrapper avoids Python callbacks entirely and reports FE, native
solver, and wrapper/glue timing through `alloy_solver_stats`.

### 5. Parameterized model constants (closed)

The Alloy and CasADi ODE/RK4 paths take physical constants and `dt` symbolically.
`CarPhysics` and `ClosedLoopConfig` defaults fill those values for normal runs.

## Next comparison matrix

Suggested next benchmarking pass:

1. CasADi MX baseline, limited-memory Hessian.
2. CasADi MX with `expand=False`, limited-memory Hessian.
3. CasADi exact Hessian.
4. Alloy generated-C solver path, limited-memory Hessian.
5. Alloy generated-C solver path with exact sparse Hessian.

For each row, record:

- build time,
- JIT/compile time,
- IPOPT status and iterations,
- solver wall time,
- function/Jacobian/Hessian evaluation time,
- callback counts,
- final objective, slack, min constraint,
- trajectory-level min distance and tracking cost.
