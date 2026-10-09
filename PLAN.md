# #77 `sc.print`, float64

Design: `internal/notes/core_compiler_roadmap.md#loops-conditionals-and-printing`.

- [ ] Tests: builder and verifier (`tests/ir/test_expr.py`), unreachable print (`tests/function/test_api.py`),
  end to end through the JIT and compiled C (`tests/codegen/test_print.py`)
- [ ] Expression dialect: `ExprOp.PRINT`, `OP_INFO`, builder with the trace recorder, verify rule, assembly text
- [ ] AD: forward, batched forward, reverse, sparsity as identity on the first value
- [ ] Tracer: raise on a print unreachable from the outputs
- [ ] Program dialect: `ProgramOp.PRINT` statement, verify rule, text, lowering rule
- [ ] Passes: `hoist_invariant` and `widen_ranges` leave procedures with a print alone; `prepare_scalar` bounds its arguments
- [ ] C: `SCALY_PRINTF` preamble only when the program prints; statement rendering
- [ ] JIT: flush C `stdout` after a call of a module that prints
- [ ] Public `sc.print`, API entry, guide entry
- [ ] Checks: `wt hook pre-merge --yes`, C snapshots byte-identical
