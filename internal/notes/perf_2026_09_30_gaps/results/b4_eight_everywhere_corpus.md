| kernel | t7base/jit | b4l8/jit |
| --- | --- | --- |
| rosen_hess | 1.3 ns | 1.000 |
| cartpole_jac | 58.8 ns | 1.000 |
| mlp_small_jac | 494.3 ns | 0.997 |
| quat_jac | 6.2 ns | 1.000 |
| chol_solve_16 | 217.7 ns | 0.999 |
| chol_solve_64 | 5.906 µs | 1.007 |
| chol_solve_200 | 91.875 µs | 0.992 |
| matvec_256 | 6.521 µs | 0.998 |
| matmul_48 | 4.008 µs | 1.046 |
| mlp_big_fwd | 8.490 µs | 1.001 |
| elementwise_1e5 | 719.709 µs | 0.984 |
| rollout_grad_100 | 11.448 µs | 1.000 |
| riccati_50 | 7.239 µs | 1.001 |
| sparse_ldl_grid30 | 41.042 µs | 1.019 |
| sparse_ldl_mpc50 | 23.334 µs | 1.027 |
| race_cars_40 | 8.844 µs | 0.999 |
| race_cars_jac_40 | 5.781 µs | 1.000 |
| race_cars_200 | 47.833 µs | 0.996 |
| chain_5 | 91.042 µs | 1.002 |
| chain_9 | 285.834 µs | 1.000 |
| npmpc_12 | 15.489 µs | 0.999 |
| unbumpercars_8 | 472.083 µs | 1.003 |
| ipm_hs118_dense | 14.625 µs | 1.001 |
| ipm_qafiro_sparse | 16.927 µs | 0.997 |
| ipm_cvxqp1_dense | 313.458 µs | 0.961 |
| ipm_mpc_12_4_20_sparse | 571.000 µs | 1.021 |
geomean b4l8/jit / t7base/jit: 1.002
