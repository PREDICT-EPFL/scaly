# double_integrator — LTI/QP regulation

The canonical double integrator `ẍ = u`, regulated to the origin under input
and state bounds. The smallest non-trivial MPC-QP; a fast unit test and QP
baseline.

State `x = [p, v]`, input `u = [a]`. `dt = 0.1 s`, `N = 20`, `n_sim = 60`.
`|a| ≤ 1`, `|p| ≤ 5`, `|v| ≤ 3`. `Q = diag(10,1)`, `R = 0.5`, `Qf` from DARE.
Plant has a 10 % actuator-gain loss + noise. Start `[3, 0]`.

Supported by **all** solvers (LTI/QP); a cross-validation case.

## Sources
- Boyd & Vandenberghe, *Convex Optimization*, 2004.
- OSQP MPC example — https://osqp.org/docs/examples/mpc.html
