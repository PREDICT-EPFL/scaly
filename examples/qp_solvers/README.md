# QPs through the PIQP library, the generated PIQP and IPOPT

Four QP families, each written once as a parametric `sc.problem`, solved by five solvers that all take
the problem's parameters and return its `x`:

| Solver | Built by | Behind the generated C |
| --- | --- | --- |
| PIQP library, sparse and dense | `sc.solver(problem, "piqp", options={"sparse": ...})` | the vendored PIQP 0.6.2 (`scaly-piqp` plugin) |
| generated PIQP, sparse and dense | `generated_piqp.solver(problem, backend)` | nothing: `scaly.solvers.ipm` generates PIQP's algorithm for the problem's structure |
| IPOPT | `sc.solver(problem, "ipopt")` | the vendored IPOPT 3.14 with MUMPS (`scaly-ipopt` plugin) |

| File | Content |
| --- | --- |
| `problems.py` | the families: oscillating-masses MPC (any horizon), factor-model portfolio, soft-margin SVM, dense random QP (`sc.qp_problem`) |
| `generated_piqp.py` | `solver(problem, backend)`: the `sc.solver`-shaped front end for `scaly.solvers.ipm` (the extraction `sc.solver` runs, then a `QPStructure` and `ipm.Solver`); `qp_data(problem)` for checking any solver's answer. This becomes `backend="scaly"` in Tier 4 of the PIQP plan |
| `compare.py` | the measurements, one fresh process and empty JIT cache per (problem, solver): build, generation and compile time, C lines and object size (and the solver libraries loaded), solve time from C, iterations, objective and primal residual |
| `time_entry.c` | calls a generated entry point from C through the universal ABI, so no Python is timed |
| `compare.ipynb` | runs `compare.py` (or reads its last results from `build/results.json`) and tabulates and plots them |

```bash
uv run examples/qp_solvers/compare.py                    # about 15 minutes; writes build/results.json
uv run --with jupyterlab jupyter lab examples/qp_solvers  # then compare.ipynb
```

The C of every solver lands in `examples/generated/qp_solvers/<problem>/<solver>/` (git-ignored).
`tests/integration/test_qp_solvers_example.py` checks that the generated PIQP takes the library's
iterations and returns its solution on small instances of every family, with both backends.
