# oscillating_masses — scalable LTI/QP

A row of `nm = 6` masses linked to each other and to two end walls by springs
and dampers; forces actuate the first and last masses. Lightly damped (open
loop oscillates). The state scales as `2·nm` (default `nx = 12`), giving a
sizeable banded QP (`nz = nx·(N+1) + nu·N = 12·31 + 2·30 = 432`) for
stress-testing QP solvers.

State `x = [p(6), ṗ(6)]`, input `u = [F₁, F₆]`. Params `k=1, m=1, d=0.1`.
`dt = 0.2 s`, `N = 30`, `n_sim = 80`. `|F| ≤ 0.5`, `|pᵢ| ≤ 4`.
`Q = diag(1·𝟙₆, 0.1·𝟙₆)`, `R = 0.1·I₂`, `Qf` from DARE. Plant: +5 % stiffness.

Supported by **all** solvers (LTI/QP). Change `nm` to scale the QP.

## Sources
- Stellato et al., *OSQP*, Math. Prog. Comp. 2020 — https://osqp.org
- Kvasnica et al., *Multi-Parametric Toolbox 3* — https://www.mpt3.org
