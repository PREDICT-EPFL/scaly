# kinematic_vehicle — bicycle trajectory tracking

A kinematic bicycle tracks a time-parameterized sine "lane-change" reference
at a target speed — a compact stand-in for autonomous-driving / racing
tracking MPC. Nonlinear in heading and speed, and a **time-varying reference**
(exercises the per-stage reference plumbing).

## Model

State `x = [X, Y, ψ, v]`, input `u = [a, δ]`.

```
Ẋ = v cosψ   Ẏ = v sinψ   ψ̇ = (v/L) tanδ   v̇ = a
```

| param | model | plant |
|------|-------|-------|
| wheelbase L | 2.7 m | **2.97 m (+10 %)** |
| target speed v₀ | 5 m/s | — |
| lane amplitude | 2.0 m | — |
| lane wavelength | 30 m | — |

`dt = 0.05 s`, `N = 25`, `n_sim = 120` (6 s). Measurement noise added per step.

## OCP

Reference `X_ref = v₀ t`, `Y_ref = A sin(2π X_ref / 30)`,
`ψ_ref = atan(dY/dX)`, `v_ref = v₀`. Cost `Q = diag(2, 8, 4, 1)`,
`R = diag(0.1, 1)`, terminal `Qf = diag(4, 16, 8, 2)`.
Constraints: `|a| ≤ 3 m/s²`, `|δ| ≤ 0.5 rad`, `0 ≤ v ≤ 25 m/s`.
Start 1 m off the lane. Success: final position within 1.5 m of the reference.

## Supported solvers

Nonlinear → `casadi_ipopt`, `acados`, `do_mpc`, `grampc`. (Not `osqp`/`piqp`.)

## Sources
- R. Rajamani, *Vehicle Dynamics and Control*, Springer 2012 (kinematic bicycle).
- acados `race_cars` example — https://github.com/acados/acados/tree/main/examples/acados_python/race_cars
- CommonRoad motion-planning benchmarks — https://commonroad.in.tum.de
