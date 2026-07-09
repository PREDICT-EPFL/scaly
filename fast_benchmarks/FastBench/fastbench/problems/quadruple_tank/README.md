# quadruple_tank — MIMO level tracking (Johansson 2000)

Four interconnected tanks fed by two pumps through split valves; the
square-root outflow is nonlinear and the cross-coupling makes it a classic MIMO
process benchmark. Track level setpoints in the two lower tanks.

State `x = [h1, h2, h3, h4]` (cm), input `u = [v1, v2]` (V). Johansson
parameters (`A`, outlet areas `a`, pump gains `k`, valve splits γ₁=0.7,
γ₂=0.6). `dt = 2 s`, `N = 25`, `n_sim = 70`. `0 ≤ v ≤ 12`, `0 ≤ hᵢ ≤ 30`.
Targets `h1=h2=14`. Plant has +5 % outlet areas + noise.

Supported by nonlinear solvers (IPOPT, acados, do-mpc, GRAMPC, **fastsqp**).
`sqrt` is guarded by `fmax(h, 1e-6)` so the model stays defined off-equilibrium.

## Sources
- K. H. Johansson, *The quadruple-tank process*, IEEE TCST 2000 — https://doi.org/10.1109/87.845876
