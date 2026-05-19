# Continuous-time CBF safety filter with neural dynamics

Problem statement for the Alloy driving application (see [`roadmap.md`](roadmap.md), Workload C). This is a continuous-time variant of the discrete-time `unbumpercars` filter described in [`examples/unbumpercars/README.md`](../../examples/unbumpercars/README.md), with two structural changes:

1. **Continuous-time dynamics and barriers.** The model is an ODE $\dot x = F(x, u)$; CBF constraints are imposed via Lie derivatives, not one-step prediction.
2. **Position-based barrier (relative degree 2).** The original filter used the velocity-augmented C3BF (relative degree 1) to sidestep the controllability issue of position-only barriers. The new filter uses the natural position barrier $h(x) = \|p_i - p_j\|^2 - (R + d_{\text{margin}})^2$ together with a higher-order CBF (HOCBF) construction. The HOCBF expansion is what brings Jacobians of the dynamics into the constraint.

Two model variants matter:

- **Input-affine** $\dot v = f_{\text{nn}}(x) + g_{\text{nn}}(x)\,u$, with $f_{\text{nn}}, g_{\text{nn}}$ two neural networks. The HOCBF constraint is affine in $u$; the safety filter is a **QP**.
- **Fully nonlinear** $\dot v = f_{\text{nn}}(x, u)$, with $f_{\text{nn}}$ a single neural network. The constraint is nonlinear in $u$; the safety filter is an **NLP**.

The model only learns the velocity prediction block; pose kinematics and steering actuator stay analytic.

## Vehicle model

### State and input

Per car ($i = 1, \dots, N$):

$$x = (\pi,\; \theta,\; v,\; \delta) \in \mathbb{R}^7,
\qquad
u = (u_{\text{tr}},\; u_{\text{st}}) \in [-1, 1]^2$$

- $\pi = (p_x, p_y) \in \mathbb{R}^2$: planar position,
- $\theta \in \mathbb{R}$: heading,
- $v = (v_f, \beta_f, \beta_r) \in \mathbb{R}^3$: forward speed and front/rear slip angles,
- $\delta \in \mathbb{R}$: steering angle.

### Dynamics blocks

$$\dot \pi = \kappa_\pi(v, \theta), \qquad
\dot \theta = \kappa_\theta(v, \delta), \qquad
\dot v = \nu(x, u), \qquad
\dot \delta = (k_\delta\, u_{\text{st}} - \delta)\,/\,\tau$$

- **Pose kinematics** $\kappa_\pi : \mathbb{R}^3 \times \mathbb{R} \to \mathbb{R}^2$, $\kappa_\theta : \mathbb{R}^3 \times \mathbb{R} \to \mathbb{R}$ are the analytic kinematic-bicycle equations with separate front/rear slip angles. They depend on state only, never on $u$.
- **Velocity prediction** $\nu(x, u) \in \mathbb{R}^3$ is the learned block — the only place where the neural network appears.
- **Steering actuator** is first-order linear. The rate saturation present in the discrete-time filter is dropped at the model level and recovered indirectly through the box constraint $u_{\text{st}} \in [-1, 1]$ plus tuning of $\tau$.

Two specializations of the velocity block:

$$\boxed{\nu(x, u) = f_{\text{nn}}(x) + g_{\text{nn}}(x)\, u} \quad\text{(input-affine)}$$

$$\boxed{\nu(x, u) = f_{\text{nn}}(x, u)} \quad\text{(fully nonlinear)}$$

In the input-affine variant, $f_{\text{nn}} : \mathbb{R}^7 \to \mathbb{R}^3$ and $g_{\text{nn}} : \mathbb{R}^7 \to \mathbb{R}^{3 \times 2}$ share the same MLP body (`256 -> 128 -> {3 or 6}`) with softplus activations for $C^2$ smoothness; the heads split into drift and control-influence channels.

### Combined per-car ODE

Stacking the four blocks, the per-car dynamics in input-affine form are

$$\dot x = F(x) + G(x)\, u,
\qquad
F(x) = \begin{pmatrix} \kappa_\pi(v, \theta) \\ \kappa_\theta(v, \delta) \\ f_{\text{nn}}(x) \\ -\delta\,/\,\tau \end{pmatrix},
\qquad
G(x) = \begin{pmatrix} 0_{2 \times 2} \\ 0_{1 \times 2} \\ g_{\text{nn}}(x) \\ \begin{pmatrix} 0 & k_\delta/\tau \end{pmatrix} \end{pmatrix}.$$

The position rows of $G$ are zero — this is exactly what makes a position barrier relative degree 2.

In the fully nonlinear variant the same kinematic and steering blocks are kept and only the velocity rows of $F$ become $f_{\text{nn}}(x, u)$; we write the resulting combined dynamics $\dot x = F_{\text{full}}(x, u)$.

## Pair barrier and HOCBF

### Barrier and derivatives

For an unordered pair of cars $(i, j)$ define

$$h_{ij}(x) = \|\Delta \pi\|^2 - (R + d_{\text{margin}})^2,
\qquad
\Delta \pi = \pi_i - \pi_j.$$

Safety is $h_{ij}(x) \ge 0$. Differentiating along the dynamics,

$$\dot h_{ij}(x) = 2\,\Delta \pi^\top\,\Delta \dot \pi, \qquad
\Delta \dot \pi = \kappa_\pi(v_i, \theta_i) - \kappa_\pi(v_j, \theta_j).$$

$\dot h_{ij}$ depends on $(v_i, \theta_i, v_j, \theta_j)$ but not on $u$, so the barrier has relative degree (at least) 2. The second derivative is

$$\ddot h_{ij} = 2\,\|\Delta \dot \pi\|^2 + 2\,\Delta \pi^\top\,(\ddot \pi_i - \ddot \pi_j),$$

and each position acceleration is

$$\ddot \pi_i \;=\; \frac{\partial \kappa_\pi}{\partial x}(x_i)\,\dot x_i,
\qquad
\frac{\partial \kappa_\pi}{\partial x}(x_i) \in \mathbb{R}^{2 \times 7}.$$

The pose-kinematics Jacobian $\partial \kappa_\pi / \partial x$ is analytic and only has nonzero columns on the $(v, \theta)$ slots of the state ($\partial \kappa_\pi / \partial \pi = 0$ and $\partial \kappa_\pi / \partial \delta = 0$).

### Input-affine expansion

Substituting $\dot x_i = F(x_i) + G(x_i)\,u_i$ into $\ddot \pi_i$ gives

$$\ddot \pi_i \;=\; \underbrace{\frac{\partial \kappa_\pi}{\partial x}(x_i)\,F(x_i)}_{\text{drift}} \;+\; \underbrace{\frac{\partial \kappa_\pi}{\partial x}(x_i)\,G(x_i)}_{\in\;\mathbb{R}^{2 \times 2}}\,u_i.$$

Because the only nonzero rows of $G$ are the velocity and steering rows, and $\partial \kappa_\pi / \partial \delta = 0$,

$$\frac{\partial \kappa_\pi}{\partial x}(x_i)\,G(x_i) \;=\; \frac{\partial \kappa_\pi}{\partial v}(v_i, \theta_i)\;g_{\text{nn}}(x_i).$$

This is the only place where a neural-network output multiplies a Jacobian to form a $u$-coefficient. The dynamics Jacobian that appears explicitly is $\partial \kappa_\pi / \partial v$ (analytic). The NN Jacobians $\partial f_{\text{nn}} / \partial x$ and $\partial g_{\text{nn}} / \partial x$ do **not** appear in $\ddot h_{ij}$, because the barrier does not depend on $v$.

Splitting $\ddot h_{ij}$ as drift plus control terms,

$$\ddot h_{ij}(x, u_i, u_j) \;=\; a_{ij}(x) \;+\; b_{ij}^{i}(x)\,u_i \;+\; b_{ij}^{j}(x)\,u_j,$$

with

$$a_{ij}(x) \;=\; 2\,\|\Delta \dot \pi\|^2 \;+\; 2\,\Delta \pi^\top\!\left[\,\frac{\partial \kappa_\pi}{\partial x}(x_i)\,F(x_i) \;-\; \frac{\partial \kappa_\pi}{\partial x}(x_j)\,F(x_j)\,\right],$$

$$b_{ij}^{i}(x) \;=\;\phantom{-} 2\,\Delta \pi^\top\,\frac{\partial \kappa_\pi}{\partial v}(v_i, \theta_i)\,g_{\text{nn}}(x_i) \;\in\; \mathbb{R}^{1 \times 2},$$

$$b_{ij}^{j}(x) \;=\; -2\,\Delta \pi^\top\,\frac{\partial \kappa_\pi}{\partial v}(v_j, \theta_j)\,g_{\text{nn}}(x_j) \;\in\; \mathbb{R}^{1 \times 2}.$$

### Fully-nonlinear expansion

For the NLP variant, freeze the current state $\bar x$ as a parameter and let $u$ be the only decision. Substituting $\dot v = f_{\text{nn}}(x, u)$ into the chain rule for $\ddot \pi_i$ gives

$$\ddot \pi_i(\bar x_i, u_i) \;=\; \frac{\partial \kappa_\pi}{\partial v}(\bar v_i, \bar \theta_i)\;f_{\text{nn}}(\bar x_i, u_i) \;+\; \frac{\partial \kappa_\pi}{\partial \theta}(\bar v_i, \bar \theta_i)\;\kappa_\theta(\bar v_i, \bar \delta_i).$$

The second term is fixed by $\bar x$ alone; only the first term moves with $u_i$. Define the per-car constants

$$A_i(\bar x) \;:=\; \frac{\partial \kappa_\pi}{\partial v}(\bar v_i, \bar \theta_i) \;\in\; \mathbb{R}^{2 \times 3},
\qquad
c_i(\bar x) \;:=\; \frac{\partial \kappa_\pi}{\partial \theta}(\bar v_i, \bar \theta_i)\,\kappa_\theta(\bar v_i, \bar \delta_i) \;\in\; \mathbb{R}^{2},$$

so that $\ddot \pi_i = A_i(\bar x)\,f_{\text{nn}}(\bar x_i, u_i) + c_i(\bar x)$. Substituting into $\ddot h_{ij} = 2\,\|\Delta \dot \pi\|^2 + 2\,\Delta \pi^\top (\ddot \pi_i - \ddot \pi_j)$ and grouping by $u$-dependence:

$$\ddot h_{ij}(\bar x, u_i, u_j) \;=\;
\underbrace{2\,\|\Delta \dot \pi(\bar x)\|^2 + 2\,\Delta \pi(\bar x)^\top\bigl(c_i(\bar x) - c_j(\bar x)\bigr)}_{\displaystyle c_{ij}(\bar x)\,\in\,\mathbb{R}\;\;\text{— const}}
\;+\;
\underbrace{2\,\Delta \pi(\bar x)^\top A_i(\bar x)}_{\displaystyle w_{ij}^{i}(\bar x)\,\in\,\mathbb{R}^{1 \times 3}\;\;\text{— const}}
\,\underbrace{f_{\text{nn}}(\bar x_i, u_i)}_{u_i\text{-dependent}}
\;+\;
\underbrace{\bigl(-2\,\Delta \pi(\bar x)^\top A_j(\bar x)\bigr)}_{\displaystyle w_{ij}^{j}(\bar x)\,\in\,\mathbb{R}^{1 \times 3}\;\;\text{— const}}
\,\underbrace{f_{\text{nn}}(\bar x_j, u_j)}_{u_j\text{-dependent}}.$$

Folding in the HOCBF terms — also constant during the solve, since $\dot h_{ij}(\bar x)$ and $h_{ij}(\bar x)$ depend on $\bar x$ only — the full pair-constraint expression splits cleanly into a state-only constant plus two NN evaluations:

$$\boxed{\;
g_{ij}(\bar x, u_i, u_j) \;=\;
\underbrace{\bigl[\,c_{ij}(\bar x) + (\gamma_1+\gamma_2)\,\dot h_{ij}(\bar x) + \gamma_1 \gamma_2\,h_{ij}(\bar x)\,\bigr]}_{\displaystyle C_{ij}(\bar x)\;\;\text{— precomputed once per filter call}}
\;+\; w_{ij}^{i}(\bar x)\,f_{\text{nn}}(\bar x_i, u_i)
\;+\; w_{ij}^{j}(\bar x)\,f_{\text{nn}}(\bar x_j, u_j).
\;}$$

Per filter call, the NLP oracle reduces to: (i) compute the constants $C_{ij}(\bar x)$, $w_{ij}^{i}(\bar x)$, $w_{ij}^{j}(\bar x)$ once for every pair (and analogues for walls), and (ii) at each IPOPT iteration evaluate $f_{\text{nn}}(\bar x_i, u_i)$ once per car and contract with the precomputed row vectors. Pair constraints share the per-car NN forward pass — there are $\binom{N}{2}$ pair rows but only $N$ NN evaluations.

The constraint Jacobian wrt $u_i$ contains the NN Jacobian exactly once:

$$\frac{\partial g_{ij}}{\partial u_i}(\bar x, u_i, u_j) \;=\; w_{ij}^{i}(\bar x)\,\frac{\partial f_{\text{nn}}}{\partial u}(\bar x_i, u_i) \;\in\; \mathbb{R}^{1 \times 2},
\qquad
\frac{\partial g_{ij}}{\partial u_j}(\bar x, u_i, u_j) \;=\; w_{ij}^{j}(\bar x)\,\frac{\partial f_{\text{nn}}}{\partial u}(\bar x_j, u_j).$$

Alloy's factory produces $\partial f_{\text{nn}}/\partial u$ through `jac:f_nn:u`; the full constraint Jacobian is a sparse contraction of these per-car NN Jacobians with the constant rows $w_{ij}^{i}, w_{ij}^{j}$.

### HOCBF condition

Use linear class-$\mathcal{K}$ functions $\alpha_k(s) = \gamma_k s$ and define $\psi_0 = h_{ij}$, $\psi_1 = \dot \psi_0 + \gamma_1 \psi_0$. The HOCBF condition $\dot \psi_1 + \gamma_2 \psi_1 \ge 0$ expands to

$$\boxed{\;\ddot h_{ij}(x, u_i, u_j) \;+\; (\gamma_1 + \gamma_2)\,\dot h_{ij}(x) \;+\; \gamma_1 \gamma_2\,h_{ij}(x) \;\ge\; 0\;}$$

with the input-affine or nonlinear $\ddot h_{ij}$ plugged in.

## Wall barriers

For the left wall, $h_L(x_i) = e_1^\top \pi_i - x_{\min} - d_{\text{margin}}$ (other walls analogous). Both derivatives follow the same structure as the pair barrier with $\Delta \pi$ replaced by $e_1$ and a single car:

$$\dot h_L(x_i) \;=\; e_1^\top\,\kappa_\pi(v_i, \theta_i),
\qquad
\ddot h_L(x_i, u_i) \;=\; e_1^\top\,\frac{\partial \kappa_\pi}{\partial x}(x_i)\,\dot x_i.$$

Input-affine form:

$$\ddot h_L(x_i, u_i) \;=\; a_L(x_i) \;+\; b_L(x_i)\,u_i,
\qquad
b_L(x_i) \;=\; e_1^\top\,\frac{\partial \kappa_\pi}{\partial v}(v_i, \theta_i)\,g_{\text{nn}}(x_i) \in \mathbb{R}^{1 \times 2}.$$

HOCBF condition with wall-specific coefficients $(\gamma_1^w, \gamma_2^w)$:

$$\ddot h_w(x_i, u_i) \;+\; (\gamma_1^w + \gamma_2^w)\,\dot h_w(x_i) \;+\; \gamma_1^w \gamma_2^w\,h_w(x_i) \;\ge\; 0
\qquad\forall\; (i, w).$$

## Safety filter optimization

Let $u_i^{\text{des}}$ be the user-requested input for car $i$ at the current time. The decision variables are $u = (u_1, \dots, u_N) \in \mathbb{R}^{2N}$ and a single nonnegative slack $s \ge 0$:

$$\min_{u,\, s}\;\; \sum_{i=1}^{N} (u_i - u_i^{\text{des}})^\top Q\,(u_i - u_i^{\text{des}}) \;+\; M\,s^2$$

subject to, for the current state $\bar x$ (parameter):

- **Pair constraints** $\forall\; i < j$:
  $$\ddot h_{ij}(\bar x, u_i, u_j) + (\gamma_1 + \gamma_2)\,\dot h_{ij}(\bar x) + \gamma_1 \gamma_2\,h_{ij}(\bar x) \;\ge\; -s,$$
- **Wall constraints** $\forall\; i,\, w$:
  $$\ddot h_w(\bar x_i, u_i) + (\gamma_1^w + \gamma_2^w)\,\dot h_w(\bar x_i) + \gamma_1^w \gamma_2^w\,h_w(\bar x_i) \;\ge\; -s,$$
- **Box** $u_i \in [-1, 1]^2,\; s \ge 0.$

$Q \succ 0$, $M \gg 0$ (start with $M = 1000$ as in the discrete-time filter). The single shared slack guarantees feasibility under tight configurations.

### QP variant (input-affine model)

Substituting the input-affine expansion, every constraint is affine in the decision $w = (u_1, \dots, u_N, s) \in \mathbb{R}^{2N + 1}$. The constraint row for pair $(i, j)$ places $b_{ij}^{i}(\bar x)$ at the $u_i$ slot, $b_{ij}^{j}(\bar x)$ at the $u_j$ slot, and $1$ at the slack slot, with right-hand side

$$l_{ij}(\bar x) \;=\; -\bigl[\,a_{ij}(\bar x) + (\gamma_1 + \gamma_2)\,\dot h_{ij}(\bar x) + \gamma_1 \gamma_2\,h_{ij}(\bar x)\,\bigr].$$

Wall rows are analogous with a single $b_L(\bar x_i)$ block per row. The safety filter is then the convex QP

$$\min_w\;\; \tfrac{1}{2}\,w^\top H\,w \;+\; q(\bar x)^\top w
\quad\text{s.t.}\quad
A(\bar x)\,w \ge l(\bar x),\quad w_{\text{lb}} \le w \le w_{\text{ub}}.$$

$H$ is the constant block-diagonal $\operatorname{diag}(2Q, \dots, 2Q, 2M)$. $q$, $A$, $l$ are assembled from $\bar x$ and $u_i^{\text{des}}$ once per filter call and handed to PIQP.

### NLP variant (fully nonlinear model)

Each $\ddot h_{ij}(\bar x, u_i, u_j)$ and $\ddot h_w(\bar x_i, u_i)$ is kept symbolic in $(u_i, u_j)$. The cost and slack structure are unchanged. The resulting NLP

$$\min_{u, s}\;\;(\text{same quadratic cost})
\quad\text{s.t.}\quad
g(\bar x,\, u,\, s) \ge 0,\quad u \in [-1, 1]^{2N},\;\; s \ge 0$$

is handed to IPOPT. Alloy's factory produces sparse $\partial g / \partial(u, s)$ and the Lagrangian Hessian $\partial^2 \mathcal{L} / \partial(u, s)^2$ from the symbolic $g$; the chain rule through $f_{\text{nn}}(x, u)$ is handled by AD.

## Functions Alloy needs to evaluate

For each filter call we need to evaluate, at the current state $\bar x = (\bar x_1, \dots, \bar x_N)$:

| Quantity | Variant | Source |
| --- | --- | --- |
| $f_{\text{nn}}(\bar x_i)$, $g_{\text{nn}}(\bar x_i)$ | input-affine | MLP forward, per car |
| $f_{\text{nn}}(\bar x_i, u_i)$ | fully nonlinear | MLP forward inside the NLP oracle, per car, per IPOPT iteration |
| $\partial f_{\text{nn}} / \partial u (\bar x_i, u_i)$ | fully nonlinear | Alloy `jac:f_nn:u` inside the NLP oracle |
| $\kappa_\pi(\bar x_i)$, $\kappa_\theta(\bar x_i)$ | both | analytic, per car |
| $\partial \kappa_\pi / \partial v (\bar x_i)$, $\partial \kappa_\pi / \partial \theta (\bar x_i)$ | both | analytic, per car |
| $h_{ij}(\bar x)$, $\dot h_{ij}(\bar x)$ | both | composition of position + $\kappa_\pi$, per pair |
| $a_{ij}(\bar x)$, $b_{ij}^{i}(\bar x)$, $b_{ij}^{j}(\bar x)$ | input-affine | composition of the above, per pair |
| $h_w(\bar x_i)$, $\dot h_w(\bar x_i)$, $a_w(\bar x_i)$, $b_w(\bar x_i)$ | both | analytic plus NN blocks, per (car, wall) |
| QP data $(H,\, q(\bar x),\, A(\bar x),\, l(\bar x))$ | input-affine | scatter of pair and wall coefficients into the decision vector |
| Symbolic constraint vector $g(\bar x, u, s)$ | fully nonlinear | substitute $f_{\text{nn}}(x, u)$ into the HOCBF expressions |

The natural Alloy layout is one named `Function` per block — `f_nn`, `g_nn`, `pose_kin`, `pose_jac`, `pair_constraint`, `wall_constraint` — composed into a top-level `safety_filter` `Function` whose output is either `al.qp(...)` or `al.nlp(...)`. Per-car structure is the natural target for `Ops.MAP` over the car axis; per-pair structure becomes a `Ops.MAP` over the upper-triangular pair list once that indexing is materialized.

## Design notes

- **Smoothness.** ReLU activations inside $f_{\text{nn}}, g_{\text{nn}}$ are replaced with softplus. IPOPT needs $C^2$-smooth constraints, and the same network is reused by the QP path for consistency.
- **Steering saturation.** Kept linear at the model level; box constraint $u_{\text{st}} \in [-1, 1]$ and the time constant $\tau$ stand in for the discrete-time rate limiter. If a hard rate limit matters later, a smooth $\tanh$ envelope on $\delta_{\text{ref}} - \delta$ preserves the input-affine structure.
- **Class-$\mathcal{K}$ tuning.** Linear class-$\mathcal{K}$ functions are kept for both pair and wall HOCBFs so the QP variant stays linear in $u$. Higher-order $\alpha$'s can be substituted in the NLP variant if needed.
- **Slack penalty.** Start with $M = 1000$ and a single shared slack, matching the existing filter.
- **Network sizing.** The current `model_kinematic_mlp.pth` body (`256 -> 128`) is reused. For input-affine, the head splits `3` (drift) and `6` (flattened $3 \times 2$ control influence). For fully nonlinear, the head is `3`.
