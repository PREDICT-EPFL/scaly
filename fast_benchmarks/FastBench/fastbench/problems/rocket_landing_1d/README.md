# rocket_landing_1d — affine-LTI soft landing (QP)

Vertical rocket soft landing: descend under gravity using non-negative thrust,
reach the ground with near-zero velocity, never go below the ground. Constant
gravity makes the dynamics **affine** (carried through the discrete model and
the QP `c` term) — a convex landing QP and the 1-D cousin of powered-descent
guidance.

State `x = [h, v]`, input `u = [a]` with `a ≥ 0`. `dt = 0.1 s`, `N = 30`,
`n_sim = 60`. `0 ≤ a ≤ 2g`, `0 ≤ h ≤ 200`, `-20 ≤ v ≤ 5`.
`Q = diag(5,5)`, `R = 0.1`, `Qf` from DARE, hover-thrust reference `uref = g`.
Plant uses a 3 % gravity error + noise and a hard ground floor. Start `[50, -2]`.

Supported by **all** solvers (affine LTI/QP).

## Sources
- Açıkmeşe & Ploen, *Convex programming approach to powered descent guidance*, JGCD 2007.
- Malyuta et al., *Convex Optimization for Trajectory Generation*, IEEE CSM 2022 — https://arxiv.org/abs/2106.09125
