# #48: sparse derivative outputs as sparse leaves

Design: Expr-level `sparse_jacobian`/`sparse_hessian` return a PACK over the canonical pattern
(internal `_Compact` keeps coloring width and the compressed batch). Function-level builders take
`layout=` ("csc" default, "csr", or a coordinate `SparsityPattern`) and the factory puts a
`SparseLayout` on the output leaf. `output_sparsities`, `SparseJacobian`, the `_val_perm` tables,
the CasADi gather and `casadi_output_sparsities` are gone. Solver oracles use `layout="csr"` and
`_value_order` turns the leaf layout into the descriptor's COO pattern, so plugins are unchanged.
A map of independent stages keeps the 2-D Jacobian; its batched pattern comes from `sc.vmap` of
the stage's derivative Function (roadmap edit needed, rule 6).

- [x] ad/sparse.py, factory, concrete, digest, model, api, ctx, solvers, codegen
- [x] tests ad, function, codegen; examples; benchmarks harness
- [ ] remaining test groups (solvers, passes, viz, typing, benchmarks, integration)
- [ ] rebase onto main (#382, #314 landed)
- [ ] differential test: SciPy CSC/CSR/COO x full/upper/lower, top level and mapped
- [ ] docs (write-docs): sparsity guide, codegen guide, generated_interface, versioning, ir.md, api/ad.md
- [ ] roadmap edit (batched pattern line), regenerate C snapshots + op counts/source size/compile time
- [ ] ty, ruff, full suite via ~/.scaly-suite/run; PR; cross-review; follow-up issue (int64 logical size)
