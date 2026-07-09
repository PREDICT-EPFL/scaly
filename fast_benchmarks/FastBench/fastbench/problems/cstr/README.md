# cstr — exothermic reactor stabilization

A dimensionless exothermic CSTR (Uppal–Ray form) held at an operating point
against perturbations. The Arrhenius term `exp(x2)` makes it stiff and
genuinely nonlinear.

```
ẋ1 = −x1 + Da(1−x1)exp(x2)
ẋ2 = −x2 + B·Da(1−x1)exp(x2) − β(x2 − u)
```

State `x = [conversion, temperature]`, input `u = [cooling]`.
`Da = 0.072, B = 8, β = 0.3`. `dt = 0.2`, `N = 25`, `n_sim = 70`.
`|u| ≤ 2`, `x1 ∈ [0,1]`, `x2 ∈ [0,6]`. Target = steady state (computed at
build). Plant has +5 % reaction rate + noise.

Supported by nonlinear solvers (IPOPT, acados, do-mpc, GRAMPC, **fastsqp**).

## Sources
- Uppal, Ray, Poore, *On the dynamic behavior of CSTRs*, Chem. Eng. Sci. 1974.
- Seborg et al., *Process Dynamics and Control*; do-mpc CSTR — https://www.do-mpc.com
