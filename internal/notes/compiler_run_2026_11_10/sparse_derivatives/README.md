# Section 3: Sparse derivatives

Packet of section 3 of the [compiler run](../plan.md), parent #293. Built by #333 (#221), #347
(#167), #390 (#332), #419 (#225) and #455 (#48). Open when this was written: #427 and #214. The
section had no simplification pass (#467). AD is automatic differentiation, CSC and CSR are
compressed sparse column and row storage, nnz the number of stored entries. A seed is one
direction of a forward derivative, and a (seed, entry) pair is one entry of one direction.

## Where this section sits

- A user asks for `sc.sparse_jacobian` or `sc.sparse_hessian` of a Function and gets one sparse output whose values are stored in the layout they name (`"csc"` by default, `"csr"`, or their own coordinate order). The JIT returns a SciPy array and the header carries that layout's tables.
- A derivative with many seeds no longer computes the pairs that are known zeros: a call's support follows the support of its arguments, a selected zero tangent stays zero through calls, and array lowering loops over each entry's own seed list.
- The interface is published in `docs/guide/sparsity.md` and stated as stable in `docs/dev/versioning.md`. It breaks 0.1.x and targets 0.2.0.
- Section 8 fuses the seed loops with the Hessian assembly (#69) and computes only demanded entries (#75). Its first layer, #452, already changed this section's headline number, as decision 1 below says.
- Section 2 keeps patterns that differ per batch entry and dense blocks inside a pattern for later (#91).

## Needs the maintainer's decision

1. The chain M=9 Hessian is straight-line code again since #452, with no budget change: 58,541 operations per stage, 2.32 MB of C, 25 s of generation and 101 s of `cc -O2`, where #419's seed-list build gave 67,308 operations, 1.12 MB and about 6 s each. #461 measured the result at 507 µs. This is the build cost #214 declined on 2026-10-27. Recommend measuring the seed-list build (main at `0a618a71`) once before choosing: if it runs in about 1 ms, restore it and keep #214's decision, otherwise accept the straight-line build and close #427 and #214 on #461's numbers.
2. Public interface, a break for 0.2.0 (#455): sparse derivatives are sparse leaves. `Function.output_sparsities`, the `SparseJacobian` class and `casadi_output_sparsities` are gone, the Function forms take `layout=`, and the header loses `rows`, `cols`, `_csr_val_perm` and `_csc_val_perm`. Recommend keeping, with a release note.
3. Part of the same break: `Function.sparsity()` returns a `CanonicalPattern`, which has `to_mask` and no `to_csr`, `to_csc` or `canonicalize`. Recommend keeping, in the same release note.
4. Roadmap departure, applied by #455: `sparse_jacobian` of a map returns the two-dimensional pattern of the whole map, and `sc.vmap` of a stage's sparse derivative gives the batched one. Recommend keeping, since the solver oracles and CasADi expect a matrix.
5. Behaviour change, confirmed by the maintainer on 2026-11-11: a sparse derivative whose logical size exceeds int64 raises when it is built, where main accepted it (#447). Listed so the break is in one place.
6. Re-cut under rule 5 of the plan: #225's target of about 60,000 operations moved to #427, and #419 closed #225 at 67,308. Recommend deciding it with item 1.

## Expected gaps

- The chain M=9 stage against its operation target and its build cost: #427, then #214.
- The mapped-window grouping is written twice, for values and for Jacobian masks: #350.
- A sparse product with a dense matrix operand keeps one seed stack per seed, and batched sparse products raise: #403.
- A sparse tensor whose logical size exceeds int64 cannot be lowered: #447.
- The dense Hessian assembly after the seed loops (transpose, compressed scatter, gather) is not fused, and entries nobody reads are computed: #69, #75. Lanes across stages: #71.
- Patterns that differ per batch entry, dense blocks and supernodes: #91.
- Row colouring and reverse-mode sparse Jacobians: #96. Colouring order: #217. Local Hessians per stage: #215.
- The section's simplification pass did not run: #467.

## Unowned

- Sections 3, 4 and 9 closed without a simplification pass: filed as #467.
- #447 had no `kind:` and `area:` labels. It now carries `kind:feature` and `area:compiler`.
- Nothing else. The evidence behind item 1 is a comment on #427, which owns it.

# Reference

Checked on main at `cd2b6c75` on 2026-11-17. Paths are under `src/scaly/` and line numbers are on
that commit. Beside this file: `demo.py` and `demo_output.txt`.

## Decisions

Hardest to undo first.

1. **A sparse derivative is a sparse leaf in a declared layout** (#455; `function/api.py`,
   `ad/sparse.py:47-92`). `sc.sparse_jacobian(f, of, wrt, layout=)` and `sc.sparse_hessian(f, of,
   wrt, triangle=, layout=)` return a Function whose output type carries the pattern. `layout` is
   `"csc"`, `"csr"` or a `SparsityPattern` that lists the same coordinates in the caller's order,
   and a layout that does not list the derivative's pattern is refused. The expression forms return
   an `Expr` over the canonical pattern and refuse any other layout. Once released, the value order
   is what users index and what generated C writes.
2. **The header publishes the declared layout only** (#455; `tests/baseline/c/spjac.h`). A sparse
   output has `NNZ`, `NDIM`, `STORAGE_SIZE`, `shape` and the tables of its layout (`col_ptr` and
   `row_ind` for CSC). The compact `rows` and `cols` triplets and both value-permutation tables are
   gone. `docs/dev/versioning.md` says the value order follows the declared layout and is part of
   the interface, and the contents of the tables are not. `_JIT_CACHE_VERSION` went from 10 to 11.
3. **`Function.output_sparsities` is gone, and `Function.sparsity()` returns a
   `CanonicalPattern`** (#455). A sparse output has one spelling, its type. CasADi queries read the
   type, and the CasADi layer no longer gathers a compact output into column order. Solver oracles
   keep their computed order and declared values alias the outputs.
4. **Values are written straight in the declared order** (#455;
   `passes/lowering/ctx.py:123-136`). When an output is a `PACK` of gathered values, lowering
   composes the layout's order into the gather, so no reorder loop follows the kernel. In
   `tests/baseline/c/spjac.c` only the scatter table changed.
5. **A map's sparse Jacobian is the matrix of the whole map** (#455, roadmap departure;
   `ad/sparse.py::_sparse_jacobian_vmap`). The local colouring is kept. The roadmap line that asked
   for a batched pattern is replaced and names why.
6. **A logical size above int64 raises at construction** (#455; `ad/sparse.py:80-84`). Before
   #455 the output was a dense vector of stored values and such a derivative worked. #447 keeps
   the question of supporting it.
7. **A call's support is its callee's rules under the arguments' support** (#333;
   `ir/support.py:104-143` and `190-217`). One specialisation per distinct tuple of argument
   supports, cached on the callee, at most 32. Past the bound the call uses the callee's own
   support, which is sound and less precise. Which 32 patterns are kept depends on the order the
   process saw them. Equal stage patterns of a map share one specialisation
   (`_mapped_support_groups`).
8. **Jacobian masks follow the same specialisations, and forward mode skips a known-zero
   selection before it builds a product** (#347; `ad/sparsity.py:206-236`). This is what removes
   the false dependency on the primal in #167's example. A declared custom derivative pattern
   stays authoritative.
9. **Array lowering iterates nonzero (seed, entry) pairs** (#419;
   `passes/lowering/plans.py:225-237` and `328-365`, `contraction.py:41-76`). Only forward
   helpers and callees lowered under an argument support take this path (`LowerCtx.seed_pairs`).
   An elementwise node groups its entries by the length of their seed list and loops once per
   group. A product whose operand has known zeros walks the stored entries of that operand,
   grouped by the contracted coordinate. Primal work stays outside the seed loops. No
   scalarization budget changed.
10. **A callee is lowered once per argument support** (#419; `passes/lowering/calls.py:21-45`).
    The procedure is named `<callee>_support`, at most 32 per callee, and falls back to the
    ordinary procedure past that. Its private buffers are marked `per_call`, which lets
    `pack_workspace` reuse the workspace of mapped temporaries.
11. **Per-stage sparse Jacobians survive declared-layout windows** (#390; `ad/sparse.py`). CSC and
    coordinate windows over the variable keep the local colouring, where they fell back to
    colouring the whole pattern.
12. **#225's operation target moved to #427.** The coordinator's re-cut. #419 lands the seed
    lists and the measurements, and #427 holds the diagnosis: the scalar floor never grew, and the
    October figure of 46,680 came from an assembly fusion that #277's sharper zero simplification
    stopped.

## Demo

`uv run internal/notes/compiler_run_2026_11_10/sparse_derivatives/demo.py`. Four parts: a call's
support under a partly zero argument, #167's example at top level and through a callee, sparse
derivative leaves in two layouts against SciPy and finite differences, and the chain M=9 Hessian
stage with default budgets. It takes no run times. Output on main at `cd2b6c75`, on la015 while
other work ran:

```text
== 1. a call's support follows its arguments (#221) ==
callee alone: every entry of 12 entries may be nonzero
called with a tangent of 4 nonzero entries: support [0, 4, 5, 11]
specialisations cached on the callee: 1, the product inside it has 4 entries

== 2. a selected zero tangent stays zero (#167) ==
top level: d(tangent)/dx has 0 entries before simplification and 0 after
through a callee: d(tangent)/dx has 0 entries before simplification and 0 after
values with run-time seeds [2, -3] at p = 0.7: [ 1.4 -2.1] (NumPy [ 1.4 -2.1])

== 3. a sparse derivative is a sparse leaf in a declared layout (#48) ==
sparse_jacobian layout=csc: output type (8, 10) nnz 24, result csc_array, max |values - SciPy csc.data| 3.9e-11
sparse_jacobian layout=csr: output type (8, 10) nnz 24, result csr_array, max |values - SciPy csr.data| 3.9e-11
sparse_hessian lower, csr: nnz 19 of 100, max |values - finite differences| 3.2e-10
header tables: NDIM, NNZ, STORAGE_SIZE, col_ptr, row_ind, shape
output_sparsities on the Function: False
expression form: an Expr of shape (2, 3) with pattern nnz 3
a 10^5 x 10^5 leaf with 2 stored entries, differentiated by itself: sparse derivative logical size 100000000000000000000 exceeds int64; see #447

== 4. the chain M=9 Hessian stage, 40 stages, default budgets (#225, #214, #427) ==
Hessian (1965, 1965), 32643 stored entries, 42 colors
operations per stage 58541 over 40 stages (253,786 before #225, target about 60,000)
generated C 2320659 bytes, generation 25.0 s, cc -O2 101.0 s (la015, shared: observations, not timings)
```

The fourth part is not what #419 measured. The same script against earlier commits of main
gives:

| Main at | Operations per stage | Generated C | Generation |
| --- | ---: | ---: | ---: |
| launch of #225 (from #419) | 253,786 | 632,057 B | 5.2 s |
| `0a618a71`, after #419, #434, #455 and #454 | 67,308 | 1,121,786 B | 6.1 s |
| `f6010125`, #452 | 58,541 | 2,320,659 B | 25.5 s |
| `cd2b6c75`, today | 58,541 | 2,320,659 B | 25.0 s |

58,541 is the count #427's diagnosis reported for the build with the budgets lifted. #452 changes
neither budget, so one of the procedures that was refused expansion now fits, and I did not trace
which. The published comparison of the run's two ends (`docs/benchmarks/compiler_revisions.md`,
#461) has the same end state with Clang: C compilation from 5.88 to 20.76 s, source from 598 to
2,320 kB and evaluation from 3,633 to 507 µs.

The section issue's first demo item, the support of every seed-axis value in the stage's forward
helper against `jacobian_sparsity`, was not rerun here. `tests/integration/test_call_support.py`
holds it, with 249 and 567 nonzero pairs at the two call sites.

## Reading list

About 290 lines, in reading order.

1. `ir/support.py:104-143`. `specialize_support`, the loop that suspends callee walks, and the
   cache with its bound.
2. `ir/support.py:190-217`. `_call_support`: a call asks for its callee under its argument
   supports, and a map tiles one answer per group of stages.
3. `ir/support.py:269-290`. `_mapped_support_groups`: the windows of each stage and the key they
   share.
4. `ad/sparsity.py:206-236`. `_call_contexts`: the same grouping for Jacobian masks (#350 merges
   the two).
5. `passes/lowering/calls.py:21-45`. `_ensure_callee`: one procedure per argument support.
6. `passes/lowering/plans.py:225-237` and `328-365`. `seed_regions` and
   `lower_supported_elementwise`: entries grouped by seed-list length, one nest per group.
7. `passes/lowering/contraction.py:41-76`. `_lower_supported_matmul`: tables grouped by the
   contracted coordinate.
8. `ad/sparse.py:47-92`. `_Compact`: values in coordinate order, the triangle selection, and
   `leaf`, which canonicalizes and packs.
9. `passes/lowering/ctx.py:123-136`. `_declared_values`: the declared order folded into the
   gather.

Second: `function/api.py::sparse_jacobian` and `_declared_pattern` (the layout argument),
`ad/sparse.py::_sparse_jacobian_vmap`, and the `per_call` test in
`passes/program/pack_workspace.py::_plan_pack`.

## Source lines

Under `src/`, from `git show --numstat` of each squash commit: #333 +140 −54, #347 +95 −41, #390
+28 −24, #419 +240 −45, #455 +258 −360. That is 761 added and 524 removed, a net 237. #455
removes more than it adds, since the compact output, the CasADi gather and the permutation tables
went away. After the simplification pass: not available, the pass did not run (#467).

## Least sure of

- **Which build of chain M=9 is the right one.** The seed-list build was main's default for one
  day, between #419 and #452, and nobody timed it. The only run time on record is the
  straight-line build's 507 µs. The October study's hand-built seed loops ran in 636 µs against
  244 µs for straight-line code, so the seed-list build may also meet #214's 1 ms.
- **What still exercises the seed lists.** With the chain stage expanded again, no benchmark
  kernel is known to take the seed-list path under default budgets. `tests/passes/lowering/test_seed_pairs.py`
  and the `seed_pairs.c` snapshot pin it, and the demo's first part reaches the support it reads.
- **Why #452 changed the count.** I bisected to the commit and read its diff for budget changes.
  I did not find the procedure that now passes `AUTO_SCALAR_OPS_PER_PROC`.
- **The review history of #455.** Its three review threads were interrupted once on 2026-11-11.
  The review that counts read `07e142ac` and accepted the header change. The two rebases after it,
  one of them after the merge queue ejected the pull request, had a check of the difference only.
- **The bound of 32.** It is written in `ir/support.py` and again as a literal in
  `_ensure_callee`. Beyond it analysis and lowering each fall back on their own, and nothing
  tests that they fall back for the same call.
- **Timings.** Generation and compile times here were taken on la015 with other threads running.
  Only #461's numbers are measurements.
