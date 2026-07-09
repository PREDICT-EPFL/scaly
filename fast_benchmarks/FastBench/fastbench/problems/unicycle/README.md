# unicycle — nonholonomic parking

A unicycle `ẋ = v cosθ, ẏ = v sinθ, θ̇ = ω` driven from an offset pose to the
origin. Nonholonomic, so the linearization at the target is uncontrollable —
a sharp NMPC-vs-linear-MPC discriminator.

State `x = [X, Y, θ]`, input `u = [v, ω]`. `dt = 0.1 s`, `N = 25`,
`n_sim = 70`. `|v| ≤ 1`, `|ω| ≤ 2`. `Q = diag(5,5,1)`, `R = 0.1·I`,
`Qf = diag(20,20,4)`. Plant has +5 % actuator gain + noise. Start `[2, 2, 0]`.

Supported by nonlinear solvers (IPOPT, acados, do-mpc, GRAMPC, **fastsqp**).

## Sources
- Aguiar & Hespanha, nonholonomic tracking/stabilization.
- CommonRoad mobile-robot benchmarks — https://commonroad.in.tum.de
