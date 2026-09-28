# Case study E5: differentiable MPC in a learning loop (DiffMPC)

The CPU benchmark of DiffMPC (Table 3 of the DiffMPC paper, `benchmarking/reinforcement-learning/` in
ToyotaResearchInstitute/diffmpc), run with the authors' scripts for mpc.pytorch, DiffMPC and trajax,
and with Scaly generating the whole episode, and its gradient, as one C function.

```bash
examples/case_studies/diffmpc/baseline/setup.sh    # diffmpc, its mpc.pytorch fork and trajax, pinned
uv run examples/case_studies/diffmpc/baseline/run_baselines.py --out examples/case_studies/diffmpc/results/baselines.json
uv run examples/case_studies/diffmpc/compare.py --out examples/case_studies/diffmpc/results/scaly.json
uv run --with jupyterlab jupyter lab examples/case_studies/diffmpc/diffmpc.ipynb
```

`diffmpc.ipynb` explains the study, builds and checks Scaly's episode and gradient without any baseline
environment, shows the hoisting and the gradient question below, and plots the recorded grid and
learning curves. Its `RUN_BASELINES` flag reruns the baselines.

## The benchmark

A batch of initial states runs a 50-step episode. Every step solves an MPC problem at the current state,
applies the first control to `x+ = A x + B u + b` and adds `|x+|^2 + |u|^2` to the cost; the
"backward" is the gradient of the batch's summed cost with respect to the diagonal of the MPC's state
weight `Q`. Forward and backward are each one call, timed after a warm-up, once per seed (a new random
`A`, `B`, `b`, batch of states). The problems:

| Problem | nx | nu | T (knots) | Batch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 4 | 40 | 64 |
| 2 | 8 | 4 | 30 | 16 |
| 3 | 8 | 4 | 30 | 64 |
| 4 | 8 | 4 | 30 | 256 |
| 5 | 16 | 8 | 30 | 16 |
| 6 | 16 | 8 | 30 | 64 |

All of them are unconstrained, with no linear cost and zero references, so each MPC is an LQ problem
that one solver iteration solves exactly: mpc.pytorch's LQR, DiffMPC's SQP step (a PCG solve of the
KKT system to 1e-12) and trajax's iLQR step.

## What Scaly generates

`scaly_impl.py` writes the MPC as a Riccati `scan` backward over the horizon (`Q_uu` by the generated
Cholesky) returning `u_0 = K_0 x_0 + k_0`, `vmap`s it over the batch inside a `scan` over the episode,
and differentiates the episode in reverse mode. The MPC's reverse derivative is attached with
`sc.custom_derivative` as the implicit rule of the LQ problem (Amos et al., "Differentiable MPC",
NeurIPS 2018): the adjoint problem has the same Riccati gains and one linear term, so the rule is one
rollout of the solution and its adjoint, and every input's cotangent is a sum over the horizon
(`implicit_rule`, checked against AD through the recursion to 1e-15 for all six inputs).

Two things make the comparison need care.

- **Scaly hoists the benchmark's MPC.** With the problem data broadcast to every MPC, as the
  benchmark's code has it, the Riccati recursion depends on nothing that changes, and Scaly's `scan`
  hoists that loop-invariant part out of the episode: the forward pass becomes one recursion and 3200
  affine feedbacks (0.19 ms for Problem 1). That is a legitimate compilation of the program, and it
  says nothing about solving MPC problems. The comparison therefore uses `hoist=False`, where every
  batch element carries its own weights through the episode, so every step of every element solves
  its own MPC, the work the baselines do. The hoisted variant is reported beside it.
- **The baselines do not compute the same gradient.** mpc.pytorch returns the gradient of the episode
  cost. DiffMPC's and trajax's rules return no cotangent for the MPC's initial state, so their
  "gradient" drops `du_0/dx_0`, the path by which a change of `Q` moves later states and so later
  controls. For Problem 1, seed 0, the true gradient's largest entry is 1.1 and the truncated one's is
  118, with different signs. Scaly computes both (`rule="implicit_no_x0"` is the truncated one), so
  each baseline's gradient can be checked.

## Results

Apple M3 Max, macOS 26, Python 3.13, PyTorch 2.14, JAX 0.5.3, Apple clang 21 at `-O2 -mcpu=native
-fno-math-errno`. This is not the reference machine of [fairness](../../../docs/results/fairness.md);
none of this belongs on the results pages. Every library pinned to one thread (Scaly's C is
single-threaded). Median over five seeds of one timed call each, after a warm-up, as the benchmark
scripts time them; the one-minute load at each process start was 2.5 to 5.9. Scaly's times are the
per-solve variant with the implicit rule.

| Problem | nx, nu, T, batch | mpc.pytorch fwd / bwd, ms | DiffMPC | trajax | Scaly | Scaly faster than the fastest baseline, fwd / bwd |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8, 4, 40, 64 | 612 / 1341 | 518 / 1056 | 444 / 884 | **38.7 / 161** | 11.5x / 5.5x |
| 2 | 8, 4, 30, 16 | 267 / 598 | 91 / 181 | 87 / 170 | **7.2 / 30.1** | 12.1x / 5.6x |
| 3 | 8, 4, 30, 64 | 461 / 1019 | 338 / 681 | 329 / 665 | **29.3 / 121** | 11.2x / 5.5x |
| 4 | 8, 4, 30, 256 | 1334 / 2894 | 1167 / 2426 | 1333 / 2705 | **118 / 483** | 9.9x / 5.0x |
| 5 | 16, 8, 30, 16 | 448 / 1003 | 352 / 710 | 246 / 505 | **61.0 / 235** | 4.0x / 2.1x |
| 6 | 16, 8, 30, 64 | 1210 / 2750 | 1382 / 2825 | 1022 / 2053 | **245 / 937** | 4.2x / 2.2x |

The other Scaly variants, forward / backward in ms:

| Problem | AD through the Riccati recursion | Riccati hoisted (the benchmark's code as written), implicit rule |
| ---: | ---: | ---: |
| 1 | 38.3 / 227 | 0.19 / 84.5 |
| 2 | 7.3 / 42.4 | 0.06 / 15.8 |
| 3 | 28.9 / 169 | 0.16 / 63.2 |
| 4 | 114 / 677 | 0.53 / 252 |
| 5 | 61.2 / 313 | 0.17 / 122 |
| 6 | 246 / 1246 | 0.47 / 528 |

Gradients, largest difference relative to the reference's largest entry, every seed:

| Baseline | Against Scaly's full gradient | Against Scaly's truncated gradient |
| --- | ---: | ---: |
| mpc.pytorch | 1e-12 to 6e-12 | (a different quantity) |
| trajax | (a different quantity) | 5e-14 to 1e-12 (against the truncated gradient of its one-knot-longer problem: its script's `U` has `T` rows) |
| DiffMPC | (a different quantity) | 2e-5 to 6e-5 |

Scaly's full gradient matches central differences of a NumPy rollout to 3e-9 and its truncated one a
NumPy reverse sweep with the policy's state path cut to 1e-9. DiffMPC's is accurate to 2e-5 to 6e-5,
likely the limit of its PCG solves (an absolute tolerance of 1e-12 on `r' Phi^-1 r`); the plan's "match
DiffMPC to 1e-8" is not possible on DiffMPC's side.

- **nx = 8 (Problems 1 to 4): 9.9 to 12.1x on the forward pass and 5.0 to 5.6x on the gradient**
  against the fastest baseline, which is the plan's success criterion (5x on both) met, barely at
  Problem 4's gradient. The baselines' time here is mostly per-call overhead: Problem 5's work is 8x
  Problem 2's (16 states against 8) and trajax takes 2.8x as long, Scaly 8.4x.
- **nx = 16 (Problems 5, 6): 4.0 to 4.2x forward, 2.1 to 2.2x on the gradient; the criterion is not
  met.** There the arithmetic dominates, and XLA runs each operation across the batch at once while
  Scaly's `vmap` runs element by element (CS-9).
- **The implicit rule is worth 1.3 to 1.4x on the gradient** over AD through the recursion. The
  gradient still costs 3.8 to 4.2x the forward pass: Scaly recomputes the MPC's output for the rule (CS-10),
  and the rule recomputes the Riccati recursion from its inputs.
- **Hoisting**: the benchmark's code as written runs 120 to 520x faster forward in Scaly than per
  solve; the gradient only about 1.9x, because hoisting does not reach into the rule (CS-14).
- **Compile**: Scaly's build and first calls take 2.0 to 6.3 s once per problem. The JAX scripts
  recompile for every seed (the matrices are closure constants); their processes took 11 to 53 s for
  five seeds, against 10 to 44 s for mpc.pytorch, which compiles nothing.

### Learning curves

`learning_curves.py` descends on Problem 1's state weights (seed 0) from `Q = 3 I`, 20 steps of
`qd <- max(qd - 0.1 g / |g|_inf, 0.01)`, with each implementation's own gradient (mpc.pytorch and
DiffMPC through `baseline/learning_curve.py`, which reuses their scripts' rollouts). From `Q = I` there
is nothing to learn: with `R = I` the MPC's objective is the episode's own cost and 40 knots are close
to an infinite horizon, so the true gradient there is 4e-5 of the cost.

| Curve | Episode cost, start | after 20 steps | Agreement |
| --- | ---: | ---: | --- |
| Scaly, full gradient | 27293.14 | 27227.82 | |
| mpc.pytorch | 27293.14 | 27227.82 | with Scaly's full: costs to 3e-14, weights to 2e-11 |
| Scaly, truncated gradient | 27293.14 | 27281.37 | |
| DiffMPC | 27293.14 | 27281.37 | with Scaly's truncated: costs to 2e-9, weights to 4e-6 |

The plan asked for identical learning curves; they are identical in pairs, and the two pairs differ: the
truncated gradient gains less than a fifth as much (12 against 65 in 20 steps).

With each library's default thread pools (`--threads default`, in `results/baselines.json`), no
baseline gets faster: DiffMPC's times stay within 3%, mpc.pytorch's grow 1.2 to 1.5x and trajax's 1.1
to 1.5x. The one-thread columns above are each baseline's best on this machine.

Not done: Theseus (manylinux x86_64 wheels only), GPU columns (out of scope for single-threaded
generated C).

## Files

| File | What it is |
| --- | --- |
| `baseline/setup.sh` | pins diffmpc, its mpc.pytorch fork and trajax in `baseline/third_party` |
| `baseline/run_baselines.py` | the benchmark's own scripts, patched only as its docstring lists, in fresh processes |
| `baseline/learning_curve.py` | descent through mpc.pytorch's or DiffMPC's gradient, with their scripts' rollouts |
| `scaly_impl.py` | the data, the MPC and its implicit rule, the episode (hoisted or per solve), its gradient, the learning curve |
| `run_scaly.py`, `compare.py` | the Scaly side of the grid, one fresh process per problem, and the table |
| `learning_curves.py` | the four learning curves |
| `diffmpc.ipynb` | the study as a notebook |

`tests/integration/test_case_study_diffmpc.py` checks, on a small problem, the rule against AD for every
input and the episode's gradient (with and without the rule, hoisted and per solve) against finite
differences.
