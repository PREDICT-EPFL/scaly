# PIQP's sparse interior-point method in Scaly

This example writes PIQP 0.7.0's proximal interior-point method for sparse quadratic programs in
Scaly and generates it as C, one solver per sparsity pattern. It uses PIQP's KKT mode
`sparse_ldlt_cond`, which eliminates every constraint and factors one positive definite matrix
over the variables:

```
Psi = P + diag(x_reg) + A^T A / delta + G^T diag(z_reg)^-1 G
```

The solver solves

```
minimize    1/2 x^T P x + c^T x
subject to  A x = b,  h_l <= G x <= h_u,  x_l <= x <= x_u
```

and takes PIQP's decisions: on the problems the tests compare, it ends with PIQP's status after
PIQP's number of iterations, with the same `rho` and `delta` at every iteration, except on a few
problems where PIQP's own KKT systems disagree with each other, described below.

## Using it

### From Python

```python
from piqp_sparse import Settings, SparseSolver

solver = SparseSolver(Settings())
solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u)  # generates the solver for these patterns
status = solver.solve()
solver.result.x, solver.result.info.iter

solver.update(c=c_new, h_u=h_u_new)                 # new values, same patterns
status = solver.solve()
```

| Call | Takes | Returns |
|---|---|---|
| `setup` | `P` (n × n), `c` (n), and optionally `A` (p × n), `b` (p), `G` (m × n), `h_l`, `h_u` (m), `x_l`, `x_u` (n) | nothing |
| `update` | any of the nine, by keyword; the others keep their values | nothing |
| `solve` | nothing | PIQP's `Status` |
| `result` | | `x`, `y`, `z_l`, `z_u`, `z_bl`, `z_bu`, `s_l`, `s_u`, `s_bl`, `s_bu`, and `info`: `status`, `iter`, `rho`, `delta`, ..., PIQP's `Info` without its timings |

Matrices are SciPy sparse arrays. Only the upper triangle of `P` is read, and an absent bound is
`-inf` or `inf`, or any value beyond `1e30` in magnitude, as in PIQP. `update` accepts new values
within the patterns given to `setup` and refuses a different pattern. A new matrix runs the
equilibration again, unless `preconditioner_reuse_on_update`; new vectors reuse the scaling.

The first `solve` for a new pattern traces, lowers and compiles the solver. This takes seconds for
small problems and about a minute at 300 variables. Scaly caches the result, so later runs with the
same patterns and settings start at once.

### The generated Functions

`SparseSolver` is a Python wrapper. The solver itself is three Scaly Functions, which
`build_solver` returns for the patterns of `P`, `A` and `G`, and which `write_module` exports as C:

```python
from pathlib import Path

from piqp_sparse import Settings, build_solver, pack_raw
from scaly.codegen import write_module

setup, solve, update = build_solver(P, A, G, Settings())
raw = pack_raw(P_values, c, A_values, b, G_values, h_l, h_u, x_l, x_u)  # stored entries of each matrix
data, scaling, AtA = setup(raw)
variables, status, info = solve(data, scaling, AtA)
data, AtA = update(pack_raw(...), scaling)           # new values, the same scaling

for fun in (setup, solve, update):
  write_module(fun, Path("generated"))               # piqp_sparse_setup.h and .c, ...
```

```
raw (9) ──setup──▶ data (14) + scaling (5) + AtA ──solve──▶ result (10) + status + info (27)
raw (9) + scaling (5) ──update──▶ data (14) + AtA ─────────▶ the same solve
```

### From C

Each Function becomes one C function, with one argument per array: inputs first, then outputs,
then the workspace. Each argument points to a struct `piqp_sparse_<function>_<name>_t` holding an
aligned `data[size]` array. Below, `n` is the number of variables, `p` of equalities and `m` of
rows of `G`; `nnz(P)` counts the stored entries of `P`'s upper triangle and `nnz(AtA)` those of
`A^T A`'s. An empty array, as `G` when `m = 0`, may be passed as a null pointer; no other may.

```c
void piqp_sparse_setup(/* 9 inputs */, /* 20 outputs */, piqp_sparse_setup_workspace_t* w);
void piqp_sparse_update(/* 14 inputs */, /* 15 outputs */, piqp_sparse_update_workspace_t* w);
void piqp_sparse_solve(/* 20 inputs */, /* 38 outputs */, piqp_sparse_solve_workspace_t* w);
```

**`piqp_sparse_setup`**: the infinite bounds, the Ruiz equilibration and `A^T A`.

| | Arguments | Type | Size |
|---|---|---|---|
| in | `raw_P`, `raw_c`, `raw_A`, `raw_b`, `raw_G` | `double` | `nnz(P)`, `n`, `nnz(A)`, `p`, `nnz(G)` |
| in | `raw_h_l`, `raw_h_u`, `raw_x_l`, `raw_x_u` | `double` | `m`, `m`, `n`, `n` |
| out | `data_P`, `data_c`, `data_A`, `data_b`, `data_G`, `data_h_l`, `data_h_u`, `data_x_l`, `data_x_u` | `double` | as the raw inputs, scaled |
| out | `data_x_b_scaling` | `double` | `n` |
| out | `data_h_l_mask`, `data_h_u_mask`, `data_x_l_mask`, `data_x_u_mask` | `bool` | `m`, `m`, `n`, `n`: which bounds are finite |
| out | `scaling_c`, `scaling_delta_x`, `scaling_delta_y`, `scaling_delta_z`, `scaling_delta_b` | `double` | `1`, `n`, `p`, `m`, `n` |
| out | `AtA` | `double` | `nnz(AtA)` |

**`piqp_sparse_update`**: new values with the scaling of the last setup, without the equilibration.

| | Arguments | Type | Size |
|---|---|---|---|
| in | `raw_P`, ..., `raw_x_u` | `double` | the nine raw inputs of setup |
| in | `scaling_c`, ..., `scaling_delta_b` | `double` | the five scalings of setup |
| out | `data_P`, ..., `data_x_u_mask` | `double`, `bool` | the fourteen data outputs of setup |
| out | `AtA` | `double` | `nnz(AtA)` |

**`piqp_sparse_solve`**: PIQP's whole iteration, from the scaled data to the unscaled result.

| | Arguments | Type | Size |
|---|---|---|---|
| in | `data_P`, ..., `data_x_u_mask` | `double`, `bool` | the fourteen data outputs of setup or update |
| in | `scaling_c`, ..., `scaling_delta_b` | `double` | the five scalings of setup |
| in | `AtA` | `double` | `nnz(AtA)` |
| out | `result_x`, `result_y` | `double` | `n`, `p` |
| out | `result_z_l`, `result_z_u`, `result_s_l`, `result_s_u` | `double` | `m` each |
| out | `result_z_bl`, `result_z_bu`, `result_s_bl`, `result_s_bu` | `double` | `n` each |
| out | `status` | `int32_t` | `1`, a `scaly.ScalySolveStatus` code |
| out | `info_iter`, `info_factor_retires`, `info_no_primal_update`, `info_no_dual_update` | `int32_t` | `1` each |
| out | `info_rho`, `info_delta`, `info_mu`, `info_sigma`, `info_primal_step`, `info_dual_step`, `info_primal_res`, `info_primal_res_rel`, `info_dual_res`, `info_dual_res_rel`, `info_primal_res_reg`, `info_primal_res_reg_rel`, `info_dual_res_reg`, `info_dual_res_reg_rel`, `info_primal_prox_inf`, `info_dual_prox_inf`, `info_prev_primal_res`, `info_prev_dual_res`, `info_primal_obj`, `info_dual_obj`, `info_duality_gap`, `info_duality_gap_rel`, `info_reg_limit` | `double` | `1` each |

`status` is 0 `OK`, 2 `MAX_ITER`, 3 `PRIMAL_INFEASIBLE`, 4 `DUAL_INFEASIBLE` or 5 `NUMERICS`, the
codes every generated solver returns; `SparseSolver` turns it back into PIQP's `Status`. The
generated code keeps nothing between calls, so the caller owns every buffer and passes all nine raw
inputs to `update`, changed or not. Each function also has a pointer-array entry,
`piqp_sparse_<function>_abi(arg, res, w)`, which `examples/case_studies/maros_meszaros/driver.c`
uses.

## Where PIQP's code went

| PIQP 0.7.0 | Here | What it holds |
|---|---|---|
| `settings.hpp` | `settings.py` | `Settings`, PIQP's settings and defaults |
| `results.hpp` | `results.py` | `Status`, `Info`, `Result` |
| `variables.hpp` | `variables.py` | `Variables` |
| `sparse/data.hpp`, `data.tpp` | `data.py` | `Structure` (the patterns), `Data` (the values), `make_data` (`set_h_l`, `disable_inf_constraints`, ...) |
| `sparse/preconditioner.hpp`, `.tpp` | `preconditioner.py` | Ruiz equilibration (`scale_data`) and the `unscale_*` helpers of `Scaling` |
| `sparse/kkt.hpp`, `kkt.tpp`, `kkt_all_eliminated.hpp` | `kkt.py` | `KKT`: the pattern and ordering of `Psi`, its assembly, `update_scalings_and_factor`, `solve`, `eval_P_x`, ... |
| `sparse/ordering.hpp`, `ldlt.hpp` | `scaly.linalg.permutation`, `ldl` and `solve_triangular` | the AMD ordering, the factorization, and QDLDL's `lsolve` and `ltsolve` |
| `kkt_system.hpp`, `.tpp` | `kkt_system.py` | `KKTSystem`: `x_reg` and `z_reg`, iterative refinement, recovery of the eliminated duals and slacks |
| `solver.hpp`, `solver.tpp` | `solver.py` | `SparseSolver` (`setup`, `update`, `solve`), `build_solver` and the iteration: `calculate_mu`, `calculate_step`, `update_residuals_nr`, `check_infeasibility_certificates`, ... |

Everything that depends only on the patterns runs in NumPy when the solver is generated: the
pattern of `Psi`, its fill-reducing ordering, and the tables that place each product of entries.
The generated code computes values only.

Each `setup` runs a generated `setup` function, the infinite bounds, the equilibration and
`A^T A`. Each `solve` runs a generated `solve` function, PIQP's whole iteration in one call.
Inside it, the factorization with its retries and the KKT solve with its refinement are each one
function of the generated C, which every place of the iteration calls, as PIQP calls its methods.

## Checking it against PIQP

`reference.py` calls the PIQP library vendored by the `scaly-piqp` plugin through its C interface,
with `kkt_solver = sparse_ldlt_cond`. `tests/integration/test_piqp_sparse.py` compares the two
solvers on random problems with every kind of constraint, on infeasible and unbounded problems, with
iterative refinement, cost scaling and no equilibration, and on the 48 Maros–Mészáros problems in
`tests/integration/data`. To print the comparison table for those problems:

```bash
uv run examples/case_studies/maros_meszaros/maros_meszaros.py            # all 48
uv run examples/case_studies/maros_meszaros/maros_meszaros.py HS21 QAFIRO
```

Some problems are sensitive to rounding. Late in a solve, with `rho` and `delta` at `1e-10`, `Psi`
holds entries near `1e10` beside a diagonal of `1e-10`, and a direct solve with it, this solver's
or SciPy's, keeps few correct digits. Whether a step still makes progress then depends on the
order of elimination. On QAFIRO, PIQP's full KKT system takes 13 iterations, its condensed one 16,
and this solver 13. On QSC205, PIQP's condensed system solves in 19 iterations while this solver,
with the same AMD algorithm but a different implementation, reaches the iteration limit, and
solves in 22 with reverse Cuthill-McKee. QSHARE1B and QBEACONF behave alike, and PIQP's condensed
system itself fails QBEACONF, which its full system solves. PIQP's two systems disagree with each other on 8 of
the 48 problems. The tests require PIQP's status and iterations on the other 40, and equal
objectives wherever both solvers solve.

## Differences from PIQP

- The settings are constants of the generated code, so changing one generates another solver.
- PIQP's timings are not reported. A Python-level clock around a compiled call would mostly time
  the call.
- `scaly.linalg.ldl` computes `L` and `D` in a different order than PIQP's QDLDL, and
  `scaly.linalg.solve_triangular` sums each row of `ltsolve` before subtracting it, so the two
  agree to rounding, not bit for bit. Success is PIQP's test, no zero pivot, and not
  `scaly.linalg`'s status, which also refuses a pivot that is tiny beside its diagonal entry.
- PIQP stops refining a solution whose residual is not finite and then skips the recovery of the
  eliminated duals, a failure its caller ignores. This solver keeps the last finite solution and
  recovers the duals from it.
