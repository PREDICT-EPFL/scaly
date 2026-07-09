# van_der_pol — nonlinear regulation

The Van der Pol oscillator `ẋ1 = x2, ẋ2 = μ(1−x1²)x2 − x1 + u`, stabilized to
the origin against its limit cycle. A standard small nonlinear OCP test.

`dt = 0.1 s`, `N = 25`, `n_sim = 60`. `|u| ≤ 1`, `|xᵢ| ≤ 5`.
`Q = I`, `R = 0.1`, `Qf = 5·I`. Model `μ = 1`; plant `μ = 1.2` + noise.
Start `[2, 0]`.

Supported by nonlinear solvers (IPOPT, acados, do-mpc, GRAMPC, **fastsqp**).

## Sources
- COPS benchmark — https://www.mcs.anl.gov/~more/cops/
- CasADi examples — https://github.com/casadi/casadi
