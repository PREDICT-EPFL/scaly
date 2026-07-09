# chain_mass — linear-spring chain (LTI / QP)

A 1-D chain of `nm = 4` point masses connected by linear springs with light
damping; the left end is anchored to a wall and a force actuates the
right-most mass. In **deviation coordinates** (displacement from the spring
equilibrium) the dynamics are linear and the cost quadratic, so the OCP is a
convex QP — the right target for OSQP and PIQP, while still a valid NMPC
instance for IPOPT and acados. It is the linear-spring relative of the
nonlinear chain-mass problem used in the acados benchmarks.

## Model

State `x = [e₁..e₄, ė₁..ė₄]` (deviations), input `u = [F]` on the last mass.
Continuous `ẋ = A_c x + B_c u` from a tridiagonal stiffness matrix; the
prediction model uses the exact discretization `x⁺ = A_d x + B_d u`.

| param | model | plant |
|------|-------|-------|
| spring k | 10 N/m | **10.5 N/m (+5 %)** |
| mass m | 1.0 kg | 1.0 kg |
| damping c | 0.4 | 0.4 |

`dt = 0.05 s`, `N = 30`, `n_sim = 80`. Plant noise `~N(0, 5e-4)`.
QP size: `nz = nx·(N+1) + nu·N = 8·31 + 30 = 278`.

## OCP

`Q = diag(10·𝟙₄, 1·𝟙₄)`, `R = 0.1`, terminal `Qf` = solution of the discrete
algebraic Riccati equation. Constraints: `|F| ≤ 10 N`, `|eᵢ| ≤ 0.6 m`.
Regulate a perturbed chain back to equilibrium (`x = 0`). Success: `‖x_N‖ < 0.15`
(the tolerance covers the steady-state offset from the plant mismatch, since the
regulator has no integral action — identical across all solvers).

## Supported solvers

LTI/QP → **all**: `osqp`, `piqp`, `casadi_ipopt`, `acados`, `do_mpc`, `grampc`.
This is the cross-validation problem: every solver must return the same cost.

## Sources
- acados chain-mass benchmark — https://github.com/acados/acados/tree/main/examples/acados_python/chain
- Wirsching, Bock, Diehl, *Fast NMPC of a chain of masses*, IEEE CCA 2006 — https://ieeexplore.ieee.org/document/4064780
