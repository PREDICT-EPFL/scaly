# #69 lowering-time fusion: plan

Stack (posted on #69): (1) inlining in `LowerCtx.read`, (2) consumer agreement, (3) sparse traversal,
(4) #166 criteria and the KKT fixture gate. Loop-body rules for `LOOP` come after #411 merges.

## Layer 1 (this branch)

Done:
- `passes/lowering/fusion.py`: the plan (deferred nodes, scatter sums), broadcast sets of mapped callees.
- `LowerCtx`: iterative memoized `read`/`read_flat`, `compute`, `_store` on demand, `delinearize`,
  gather tables composed through movement chains, contiguous slices read at an offset.
- `fuse_elementwise.py`, `combine_scatter_sums.py`, `inline_producer` and `fold_arith`'s fill folding deleted.
- Gates measured: chain M=5 66,744 doubles, race-car `SZ_W` 0, npmpc_jac N=100 5,696, unbumpercars C=8 122,848.

Remaining:
- Snapshot regeneration and the reviewed diff with operation counts, source size, compile time.
- Tests: `tests/passes/lowering/test_fusion.py` (NumPy differential per rule, each with a perturbation),
  gates in `tests/benchmarks/test_kernel_gates.py` (chain M=5 < 100k, race-car `SZ_W` == 0),
  carried-range guard (`_merge`, `emit_reduction`, `delinearize` never split a carried range).
- Docs: `docs/how_it_works/lowering.md` (pipeline table, first listing, elementwise fusion,
  scatter sums) with real output; `docs/dev/codebase.md` package map.
- Node-id baseline regeneration; full suite through `~/.scaly-suite/run`.
- Open point for review: known zeros of an inlined intermediate in an auto-lowered procedure that
  `scalarize` later expands are no longer forced to zero against NaN operands.
