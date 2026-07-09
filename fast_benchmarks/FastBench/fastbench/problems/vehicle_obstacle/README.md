# vehicle_obstacle — lane keeping with obstacle avoidance

A kinematic bicycle keeps a straight lane at a target speed while avoiding a
circular obstacle on the lane. The obstacle is a genuine **nonlinear inequality
constraint** `(X−xo)² + (Y−yo)² ≥ (r+margin)²`, so this problem is solved only
by backends that handle general path constraints (IPOPT here); box-only
backends (acados-box, fastsqp, QP solvers) report it as unsupported.

State `x = [X, Y, ψ, v]`, input `u = [a, δ]`. `dt = 0.05 s`, `N = 30`,
`n_sim = 120`. Obstacle at `(15, 0)`, `r = 2`, margin `0.6`. `|a| ≤ 3`,
`|δ| ≤ 0.5`, `|Y| ≤ 6`, `0 ≤ v ≤ 8`. Success = never hits the obstacle, gets
past it, and returns to the lane.

## Sources
- acados race_cars / obstacle examples — https://github.com/acados/acados
- CommonRoad — https://commonroad.in.tum.de
