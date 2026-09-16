# Chain of masses

A chain of `M` point masses hanging between a wall and an actuated end mass, connected
by nonlinear springs under gravity. The controller drives the end mass to a reference
position while damping the swing of the masses in between. It is the suite's
state-dimension scaling axis: `NX = 3·(2·(M−2)+1)` grows with `M` while the structure of
the problem stays the same.

- Formulation, plant, and CasADi mirror: `__init__.py`
- Receding-horizon episode: `closed_loop.py`
- Formulation gates: `checks.py` (run by `benchmarks/run.py smoke`)
- Foxglove layout: `foxglove-layout.json`

```bash
uv run python benchmarks/run.py closed-loop --problem chain          # canonical episode
uv run python benchmarks/run.py closed-loop --problem chain --smoke  # short toolchain check
uv run python benchmarks/run.py closed-loop --problem chain --solver sqp --oracle scaly
uv run python benchmarks/run.py closed-loop --problem chain --solver sqp --oracle casadi
```

The SQP columns share one solver and differ only in whether Scaly or CasADi
generates the C-ABI objective and derivative oracles.

## Model

One mass is fixed at the wall origin. Of the remaining `M−1`, the last is the actuated
end mass, whose velocity **is** the control; the `M−2` in between carry velocity states:

```text
x = [positions of the M-1 free masses (3 each); velocities of the M-2 intermediate masses (3 each)]
u = velocity of the end mass,   u ∈ [-1, 1]^3
```

The spring between neighbours `i-1` and `i` (or the wall) pulls with
`F = (D/m)·(1 − L/‖d‖)·d` for `d = x_i − x_{i-1}`, applied with opposite signs to the two
masses it connects, and gravity acts on `z` of each intermediate mass:

```text
xdot = [velocities of the intermediate masses; u; spring forces + gravity]
```

Constants are symbolic parameters (per the roadmap's L4 requirement), defaulting to
`m = 0.033`, `D = 1.0`, `L = 0.033`, `gravity = -9.81`, `dt = 0.2`. **There is no
damping.** Discretization is ERK4, and the closed loop's plant is an independent NumPy
RK4 (`rk4_step_np`) so a controller-side codegen bug cannot hide by also being in the
plant; `check_dims_and_rk4` holds the two to `1e-10`.

The episode starts with the free masses spread along the x axis at `x = 7·i/(M−1)`, so
each spring begins stretched to 1.75 m against a rest length of 0.033 m — 53× — with
spring force 1.72 N per link against 0.324 N of weight per mass. That initial condition,
plus the absent damping, is why the first second of the episode swings hard: with
`u ≡ 0` the chain oscillates between `z = 0` and `z = -1.3` indefinitely, while over the
same window the closed loop takes the velocity RMS from 1.0 to 0.001. The recorded
open-loop plan shows the controller predicting the swing it is riding out.

## Cost, and the one deliberate divergence from laopt

The cost tracks a single end-mass reference and drives the intermediate velocities and
the control to zero. Stage terms enter on normalized time (`h = 1/N`, not `dt`, matching
laopt's `MultipleShooting`), the terminal term unscaled:

```text
minimize  Σ_k h·½·( Q_END·‖p_end,k − END_REF‖² + Q_VEL·‖v_k‖² + R_U·‖u_k‖² )
                 + ½·Q_END_TERMINAL·‖p_end,N − END_REF‖²

Q_END = 2.5   Q_VEL = 25.0   R_U = 0.1   Q_END_TERMINAL = 10.0   END_REF = (0.75, 0, 0)
```

`END_REF` is defined once, in `__init__.py`, and read by both the Scaly objective and its
CasADi mirror.

**This is where we knowingly differ from laopt.** laopt writes each end-mass term
expanded, as `-q·x + ½·w·‖p‖²`, which by completing the square is minimal at `x = q/w`.
It then uses the same `q = -7.5` for the stage and the terminal term against `w = 2.5`
and `w = 10` respectively, so its stage cost pulls the end mass towards `x = 3.0` while
its terminal cost pulls towards `x = 0.75` (`examples/chain_mass/chain_mass_ocp.hpp`
lines 38-52 in the laopt source tree, which is unpublished, so there is no link to
give). The reference implementations of this problem track one position throughout:

- **acados** `chain_mass` uses a `LINEAR_LS` cost with `yref = [xrest; 0]` and
  `yref_e = xrest` under `W_e = Q` — the same reference *and* the same weight matrix at
  the terminal node. `xrest` is a computed steady state with the end mass at
  `xEndRef = (L·(M_int+1)·6, 0, 0)`, i.e. each spring stretched to 6× its rest length;
  with `M_int` the intermediate masses (3 for our `M = 5`) that is 0.79 m
  ([`main.py`][acados-main], [`utils.py`][acados-utils]).
- The **ACADO hanging-chain** QP benchmark uses `α‖x_actuator − x_end‖² + β Σ‖ẋ_i‖² +
  γ‖u‖²` with one actuator target `x_end = (1, 0, 0)` in every stage
  ([repository][acado-chain], [description][alpaqa-paper]).

So the split is an oversight in laopt rather than an instance worth reproducing, and we
keep its terminal target: `0.75` is within 5% of the 0.79 m that acados asks this size of
chain to stretch to, whereas `3.0` is over 4× further out. Two consequences
worth knowing:

- The objective **value** is no longer comparable to laopt's, because writing the same
  minimizer as a squared norm adds a constant per term. Gradients, Hessians, sparsity,
  and the argmin are unchanged, so nothing the sweeps measure is affected.
- If laopt numbers are ever needed for a direct comparison, fix the example there
  (`state_cost.q(3*(M-2)) = -1.875`, which is `−Q_END · 0.75`) rather than re-splitting
  the reference here. The `one_reference` gate exists to stop that regression: it
  requires the objective's gradient to vanish in **every** end-mass block when the whole
  chain sits on `END_REF` at rest.

Only `u ∈ [-1, 1]^3` is constrained; there are no state constraints. The initial state
is pinned with an explicit equality row, which is equivalent at the optimum to laopt's
fixed box on `X_0`.

## Canonical point

`M = 5`, controller horizon `N = 12`, 90 plant steps at `dt = 0.2` (18 simulated
seconds). The horizon is deliberately shorter than laopt's `N = 40`, because laopt solves
this once open-loop over `tf = 8.0` while we run it as a receding-horizon loop. 90 steps
is what it takes to settle: the end mass ends within 0.06 of `END_REF`, moving under
3 mm per step, with `|u|` under 0.02. The FE sweep harvests the midpoint step's oracle
input, rolled into the `N = 40` transcription so the benchmark cell keeps laopt's
horizon.

[acados-main]: https://github.com/acados/acados/blob/master/examples/acados_python/chain_mass/main.py
[acados-utils]: https://github.com/acados/acados/blob/master/examples/acados_python/chain_mass/utils.py
[acado-chain]: https://github.com/dkouzoup/hanging-chain-acado
[alpaqa-paper]: https://arxiv.org/pdf/2112.02370
