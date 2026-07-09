# dc_motor — LTI position tracking (QP)

Armature-controlled DC motor, third-order LTI. Track a shaft-angle setpoint
under a voltage limit.

State `x = [θ, ω, i]`, input `u = [V]`. Params `J=0.01, b=0.1, Kt=Kb=0.01,
R=1, L=0.5`. `dt = 0.05 s`, `N = 25`, `n_sim = 80`. `|V| ≤ 12`,
`|ω| ≤ 50`, `|i| ≤ 10`. `Q = diag(50,1,0.1)`, `R = 0.01`, `Qf` from DARE.
Setpoint `θ = 1 rad`. Plant has +10 % winding resistance + noise.

Supported by **all** solvers (LTI/QP).

## Sources
- Franklin, Powell, Emami-Naeini, *Feedback Control of Dynamic Systems*.
- UMich CTMS DC-motor example — https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition
