# pendulum_swingup — cart-pole swing-up

A pole hinged on a cart actuated by a single horizontal force. The angle
`theta` is measured from the **upright** (`theta = 0` unstable, `theta = pi`
hanging). The force limit is set so the upright cannot be reached by a single
push, so the controller must pump energy — a standard underactuated,
nonconvex benchmark.

## Model

State `x = [p, theta, dp, dtheta]`, input `u = [F]`.

```
den   = M + m − m cos²θ
p̈     = (−m l sinθ θ̇² + m g cosθ sinθ + F) / den
θ̈     = (−m l cosθ sinθ θ̇² + F cosθ + (M+m) g sinθ) / (l·den)
```

| param | model | plant (simulation) |
|------|-------|--------------------|
| cart mass M | 1.0 kg | 1.0 kg |
| pole mass m | 0.10 kg | **0.115 kg (+15 % mismatch)** |
| length l | 0.8 m | 0.8 m |
| g | 9.81 | 9.81 |

`dt = 0.02 s`, horizon `N = 50`, simulation `n_sim = 150` steps (3 s).
Process noise `~N(0, 1e-3)` is added to the plant each step.

## OCP

Quadratic cost `Q = diag(5, 50, 0.5, 0.5)`, `R = 0.01`, terminal
`Qf = diag(50, 500, 5, 5)`, regulating to `x = 0`.
Constraints: `|F| ≤ 25 N`, `|p| ≤ 2.5 m`.
Start: hanging at `x0 = [0, π, 0, 0]`. Success: `|θ| < 0.15`, `|θ̇| < 0.5`,
`|p| < 0.5` at the final time.

## Supported solvers

Nonlinear → `casadi_ipopt`, `acados`, `do_mpc`, `grampc`. (Not `osqp`/`piqp`.)

## Sources
- acados `pendulum_on_cart` example — https://github.com/acados/acados/tree/main/examples/acados_python/pendulum_on_cart
- M. Kelly, *An Introduction to Trajectory Optimization*, SIAM Review 59(4), 2017 — https://epubs.siam.org/doi/10.1137/16M1062569
- OptimTraj cart-pole demo — https://github.com/MatthewPeterKelly/OptimTraj
