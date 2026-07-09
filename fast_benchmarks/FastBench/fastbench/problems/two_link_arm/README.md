# two_link_arm — manipulator setpoint tracking

A two-link planar manipulator `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ`, regulated to a
joint-space setpoint with gravity feed-forward. Fully actuated and stable —
isolates dense nonlinear-model handling (configuration-dependent inertia, trig
coupling) from the difficulty of underactuation.

State `x = [q1, q2, q̇1, q̇2]`, input `u = [τ1, τ2]`. Params `m=1, l=1,
lc=0.5, I=0.083`. `dt = 0.05 s`, `N = 25`, `n_sim = 80`. `|τ| ≤ 20`,
`|q̇| ≤ 10`. Target `q = [π/2, −0.5]`, `uref =` gravity torque at target.
Plant has a 10 % payload mismatch + noise.

Supported by nonlinear solvers (IPOPT, acados, do-mpc, GRAMPC, **fastsqp**).

## Sources
- Spong, Hutchinson, Vidyasagar, *Robot Modeling and Control*, 2006.
- Crocoddyl manipulator examples — https://github.com/loco-3d/crocoddyl
