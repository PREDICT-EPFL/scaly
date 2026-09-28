# Case study E2: Real-time Neural MPC

The control-frequency benchmark of Real-time Neural MPC (Salzmann et al., RA-L 2023, Table II),
reproduced with the authors' scripts on current acados, and then with Scaly generating the network's
code in two places: inside acados, as the exact model's sensitivities, and in place of PyTorch, as
RTN-MPC's Taylor surrogate.

```bash
examples/case_studies/neural_mpc/baseline/setup.sh    # ml-casadi pinned, acados built; prints ACADOS_SOURCE_DIR
export ACADOS_SOURCE_DIR=.../third_party/acados DYLD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
uv run --with torch --with $ACADOS_SOURCE_DIR/interfaces/acados_template examples/case_studies/neural_mpc/compare.py --out examples/case_studies/neural_mpc/results/grid.json
uv run --with torch --with $ACADOS_SOURCE_DIR/interfaces/acados_template examples/case_studies/neural_mpc/kernel_bench.py --out examples/case_studies/neural_mpc/results/kernels.json
uv run --with jupyterlab jupyter lab examples/case_studies/neural_mpc/neural_mpc.ipynb
```

`neural_mpc.ipynb` explains the study, builds and checks Scaly's network kernels without PyTorch or
acados, times them, shows the generated C, and plots the recorded results. Its `RUN_BASELINES` flag
reruns the acados grid and the kernel comparison.

## The benchmark

Table II is not the quadrotor of the rest of the paper. It is a 1D double integrator, `x = (s, s_dot)`,
`u = s_ddot`, `|u| <= 10`, plus an untrained MLP residual `f_D(x)` (2 inputs, 2 outputs, tanh), under
acados SQP-RTI: N = 10 over 1 s, ERK (RK4, one step), Gauss-Newton, full condensing HPIPM, tracking a
sine on `s`. The timed loop is 50 control steps, each setting the reference and the initial state,
running one RTI and reading the solution. The upstream code is ml-casadi's
`examples/mpc_mlp_naive_example.py` and `examples/mpc_mlp_cnn_example.py` at `44ec47f` (2023-11-01);
`baseline/run_acados.py` ports both with the network size as a flag. There are two columns:

- naive: the network expanded into the CasADi model, which acados code-generates with its sensitivities;
- RTN-MPC: a first-order Taylor surrogate whose point, value and Jacobian are acados parameters,
  computed by PyTorch at the previous solution after every RTI.

## One change to the authors' protocol

The authors zero the network's output layer so the untrained model does not change the dynamics. With
CasADi 3.8 that lets the expanded model drop the network: acados' generated `wr_expl_vde_forw.c`
contains no `tanh` at all, and the naive column runs at about 12 kHz whatever the network (the `naive0`
rows below, 2 x 128 and 5 x 128). Here the output weights are `1e-6 N(0, 1) / sqrt(width)` instead:
the dynamics stay nominal to about 1e-6, and both columns evaluate every layer. `--out-scale 0`
restores the authors' zeroed layer.

## Files

| File | What it is |
| --- | --- |
| `baseline/setup.sh` | pins ml-casadi and acados, builds acados |
| `baseline/run_acados.py` | the paper's two columns, ported to current acados |
| `scaly_impl.py` | the network in Scaly, read from the PyTorch state dict with `load_torch_state_dict`; acados' `expl_ode_fun`, `expl_vde_forw` and `expl_vde_adj` as Scaly Functions; RTN-MPC's surrogate at the 10 nodes as one `vmap` |
| `run_scaly.py` | the two Scaly columns, reusing the baseline's OCP and loop: `dropin` generates acados' code, swaps its model sources for Scaly's `casadi=True` output and builds; `taylor` replaces PyTorch's `approx_params` |
| `compare.py` | the grid, three fresh processes per cell, the fastest kept |
| `kernel_bench.py` | the two kernels that differ between columns, timed alone from C |
| `neural_mpc.ipynb` | the study as a notebook: explanation, the Scaly kernels live (checked and timed), the recorded grid and kernel results plotted |

`tests/integration/test_case_study_neural_mpc.py` checks the drop-in path (CasADi layer, `casadi_int`
as `int`, a zero-length parameter, column-major `Sx`) against NumPy.

## The drop-in, and what it needed

acados' generic external path accepts Scaly's `casadi=True` output with two adjustments, both in
`scaly_impl.py`. Scaly's CasADi layer refuses dense matrices, since Scaly is row-major and CasADi
column-major, so the sensitivity `Sx` travels as a flat column-major vector. acados copies a dense
argument as contiguous values either way. And acados compiles with `casadi_int` as `int`, so each
installed source starts with that definition.

The network is written into each function's body. Called as a model `Function`, the Jacobians
recompute the forward pass beside the call's own value, which doubles the kernel (60 against 31 µs at
5 x 128).

## Results

Apple M3 Max, macOS 26, Apple clang 21, PyTorch 2.14 on one thread, acados `00947dd` (2026-09-25).
This is not the reference machine of [fairness](../../../docs/results/fairness.md); none of this
belongs on the results pages. Frequency is 1 / mean step time over the 50 timed steps, as the paper
reports it; the paper's i7 numbers are context only.

| Network | Parameters | Paper naive, Hz | Paper RTN, Hz | Naive (CasADi), Hz | RTN (PyTorch), Hz | Scaly in acados, Hz | Scaly Taylor, Hz | Linearization naive / Scaly, µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 4262 | 4262 | 7047 | – | – | – | 9 / – |
| 2 x 16 | 354 | 2228 | 1096 | 6974 | 3553 | 10274 | 6188 | 27 / 14 |
| 2 x 128 | 17,154 | 116 | 1071 | 513 | 3289 | 2249 | 2812 | 1861 / 359 |
| 5 x 16 | 1,170 | 1139 | 885 | 4582 | 3048 | 9364 | 4735 | 81 / 27 |
| 5 x 128 | 66,690 | 31 | 784 | 134 | 2812 | 694 | 2103 | 7355 / 1328 |
| 12 x 32 | 11,778 | 168 | 588 | 1201 | 2398 | 1678 | 5022 | 745 / 214 |
| 12 x 512 | 2,891,778 | 11 | 507 | 2 | 945 | 9 | 36 | 518565 / 106498 |
| 2 x 128, output layer zeroed (the authors' protocol) | 17,154 | | | 11915 | | | | 5 / – |
| 5 x 128, output layer zeroed (the authors' protocol) | 66,690 | | | 11984 | | | | 5 / – |

Linearization is acados' `time_lin` per RTI: 40 calls of `expl_vde_forw`, 4 per shooting interval.
Every column's closed loop agrees with the naive column's: the drop-in to 5e-13, the two surrogates to
6e-8 (a Taylor model, and PyTorch's float32).

### The two kernels alone

`expl_vde_forw` is one call as acados makes it (CasADi's code as acados generates it for the naive
model, against Scaly's drop-in; same compiler and flags). The surrogate is RTN-MPC's value and Jacobian
at the 10 nodes: PyTorch in float32 (what ml-casadi runs; it casts the points with `.float()`), PyTorch
in float64 (ml-casadi's `batched_jacobian` on a double network), and Scaly timed from C and through its
Python call. C timings are the best batch of `kernel_bench.py`'s harness; PyTorch's the best of 200 calls.

| Network | `expl_vde_forw` CasADi, µs | Scaly, µs | Ratio | Surrogate PyTorch f32, µs | PyTorch f64, µs | Scaly (C), µs | Scaly (Python), µs | Scaly source, MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 x 16 | 0.3 | 0.2 | 1.7x | 106 | 102 | 2.8 | 16.5 | 0.0 |
| 2 x 128 | 45.7 | 8.5 | 5.4x | 126 | 125 | 91.0 | 94.9 | 0.8 |
| 5 x 16 | 1.0 | 0.5 | 1.9x | 147 | 148 | 5.0 | 8.0 | 0.1 |
| 5 x 128 | 181.0 | 31.7 | 5.7x | 196 | 222 | 319.0 | 329.7 | 3.0 |
| 12 x 32 | 17.3 | 4.9 | 3.5x | 274 | 299 | 51.0 | 95.7 | 0.5 |
| 12 x 512 | 12797.0 | 3904.0 | 3.3x | 1212 | 2658 | 26412.0 | 26961.0 | 130.8 |

- The drop-in's sensitivities are 1.7 to 5.7x faster than CasADi's at every size, with the same result
  to 1e-16. The network costs acados the linearization time above, which is where the naive column's
  collapse comes from.
- The surrogate is a different story. Scaly is 29 to 38x faster than PyTorch for width-16 networks and
  5.4x at 12 x 32, where PyTorch's dispatch is most of its time. It is 0.6 to 0.7x at 5 x 128 and 0.05x
  at 12 x 512. There PyTorch multiplies each layer's weights by all 10 points and tangents at once,
  one matrix product per layer, while Scaly's `vmap` runs, for each point, a matrix-vector product for
  the value and a two-column product for the Jacobian: the 23 MB of float64 weights are read 20 times
  per call. Float64 against float32 accounts for 2.2x of the 22x.
- Scaly's source carries every weight matrix twice, once in each layout its two passes read it in:
  131 MB of C at 12 x 512, against CasADi's 72 MB.
