# planar_quadrotor — 2-D quadrotor stabilization

A quadrotor in the vertical plane with two rotor thrusts, stabilized to hover
at the origin from an offset. Nonlinear through the attitude; non-negative
thrust limits and a hover-thrust feed-forward reference.

State `x = [pₓ, p_z, φ, vₓ, v_z, ω]`, input `u = [T_L, T_R]`.
Params `m = 0.5, l = 0.25, I = 0.0125`. `dt = 0.05 s`, `N = 30`, `n_sim = 100`.
`0 ≤ Tᵢ ≤ 2mg`, `|φ| ≤ 1.2`. `Q = diag(8,8,4,1,1,0.5)`, `R = 0.05·I`,
`Qf = 5·Q`-ish. Plant is 10 % heavier + noise. Start `[-1, -0.5, 0.2, 0, 0, 0]`.

Supported by nonlinear solvers (IPOPT, acados, do-mpc, GRAMPC, **fastsqp**).

## Sources
- Tedrake, *Underactuated Robotics* — http://underactuated.mit.edu
- Sabatino, *Quadrotor control*, KTH MSc thesis 2015.
