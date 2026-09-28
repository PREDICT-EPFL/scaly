# QPs through the PIQP library, the generated PIQP and IPOPT

Four QP families, each written once as a parametric `sc.opt.problem`, solved by five solvers, each
`sc.opt.solver(problem, method)` with the one signature every opt method has: a warm start and the
problem's parameters in, the solution, its multipliers and an `Info` out.

| Solver | Built by | Behind the generated C |
| --- | --- | --- |
| PIQP library, sparse and dense | `sc.opt.solver(problem, sc.opt.PIQP(sparse=...))` | the vendored PIQP 0.6.2 (`scaly-piqp` plugin) |
| generated PIQP, sparse and dense | `sc.opt.solver(problem, sc.opt.IPM(sparse=...))` | nothing: `scaly.opt.ipm` generates PIQP's algorithm for the problem's structure |
| IPOPT | `sc.opt.solver(problem, sc.opt.IPOPT(...))` | the vendored IPOPT 3.14 with MUMPS (`scaly-ipopt` plugin) |

| File | Content |
| --- | --- |
| `problems.py` | the families: oscillating-masses MPC (any horizon), factor-model portfolio, soft-margin SVM, dense random QP (`sc.opt.QP`) |
| `compare.py` | the measurements, one fresh process and empty JIT cache per (problem, solver): build, generation and compile time, C lines and object size (and the solver libraries loaded), solve time from C, iterations, objective and primal residual; `qp_data(problem)`, the extracted QP every answer is checked against |
| `time_entry.c` | calls a generated entry point from C through the universal ABI, so no Python is timed |
| `compare.ipynb` | runs `compare.py` (or reads its last results from `build/results.json`) and tabulates and plots them |

```bash
uv run examples/qp_solvers/compare.py                    # about 15 minutes; writes build/results.json
uv run --with jupyterlab jupyter lab examples/qp_solvers  # then compare.ipynb
```

The C of every solver lands in `examples/generated/qp_solvers/<problem>/<solver>/` (git-ignored).
`tests/integration/test_qp_solvers_example.py` checks that the generated PIQP takes the library's
iterations and returns its solution on small instances of every family, with both backends.
