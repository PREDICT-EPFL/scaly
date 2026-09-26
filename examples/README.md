# Examples

Each script runs on its own (`uv run examples/<name>.py`) and prints what it found.
`tests/integration/test_examples.py` and `tests/linalg/test_examples.py` check each one against a
NumPy or SciPy reference.

| Example | Problem | What it uses |
| --- | --- | --- |
| `mdp_value_iteration.py` | Planning for a robot on a slippery floor: value iteration for a stochastic shortest path | `segment_sum`, `segment_min`, `where` and comparisons, `while_loop` stopped on `norm_inf`, `cast` to an `int64` policy |
| `lasso_admm.py` | Compressed sensing: a sparse signal from few measurements, the LASSO by ADMM | `while_loop`, soft thresholding with `copysign` and `maximum`, `logical_or` of residual tests, dense `cholesky` and `cho_solve`, a `bool` output |
| `lqr_tuning.py` | Tuning LQR weights for a quadrotor in a gust, on a heavier vehicle than modelled | two `scan`s (the Riccati recursion backwards, the closed loop forwards with a negative stride), the step number (`index=True`), reverse mode through both scans and the small solves in them |
| `truss_sizing.py` | Minimum-compliance truss design by optimality criteria | `take` and `put_add` with run-time `int64` indices (the bar table is an input), dense `cholesky`, gradients through the solve and the scatter |
| `heat_control.py` | Optimal heating of a plate made of two materials | `SparseMatrix` assembly, `SparseLDL` implicit time stepping, `sc.S` passing the sparse system matrix between `Function`s, adjoint gradients from the solves' implicit rules |
| `hanging_chain.py` | Calibrating a hanging chain's stiffness and mass from two photographs | Newton in a `while_loop` with AD inside the body, a step safeguarded by `isfinite`, `sc.custom_derivative` with the implicit-function rule |
| `sqp_newton_sparse.py` | Pendulum swing-up: SQP Newton steps on a sparse KKT system | `sparse_hessian`, `sparse_jacobian`, `SparseLDL` with `inertia()` and `health()` |
| `kalman_update.py` | A Kalman update of a spatial field with a sparse information prior | a quasi-definite `SparseLDL` solve and its Jacobians (the Kalman gain) |
| `multiple_shooting.py`, `simple.py` | Small introductions to `Function`s and derivatives | |
