# Section 7: Tensor operations

Packet of section 7 of the [compiler run](../plan.md), parent #296. Built by #401 (#66), #404
(#67), #410 (#68) and #408 (#65). #431 is the simplification pass over the first three. #408
merged after that pass and is not covered by it. IR is the intermediate representation, AD
automatic differentiation.

## Where this section sits

- A user writes N-d code: `x.sum(axis, keepdims=True)`, `x.mean`, `sc.diag`, `sc.reduce_max` and its relatives, `@` on batches of matrices with NumPy's rules, and `sc.einsum`, each with forward and reverse derivatives, sparsity patterns and generated C. `demo.py` shows it.
- The IR has one reduction, `REDUCE(kind, axes)` with the kinds add, max and min, and `MATMUL` takes leading batch axes. `einsum` adds no operation and no derivative rule. `maximum`, `minimum`, `abs`, `floor` and `ceil` have derivative rules.
- Of the Milestone 2 (0.3.0) row of the roadmap's *Milestone checks*, the section completes "einsum against NumPy". The #69 workspace gates and #71's matrix-vector gate belong to section 8.
- Section 8 removes the copies `einsum` puts around each product and fuses reductions with their producers (#69), then gives lanes to nests and to max and min (#71). The heading *For #69* below is written for it.
- Section 2's #403 extends batched products to sparse operands, and section 4's #56 lets an integer result such as `argmax` leave a Function.

## Needs the maintainer's decision

1. Roadmap departure, applied by #408: elementwise `sc.maximum` and `sc.minimum` propagate NaN, as `np.maximum` does, in folding and in generated C. The roadmap had chosen `fmin` and `fmax`, which ignore a NaN, and #408 replaced that line. Recommend keeping: a NaN residual must not pass a convergence test. No other departure was applied, and #401, #404 and #410 did not edit the roadmap.
2. Public interface, published: `Expr.sum(axis=None, *, keepdims=False)`, `Expr.mean` with the same signature, `sc.diag(x, k=0)`, `sc.matmul` and batched `@`, `sc.einsum(spec, *operands)`, `sc.reduce_max`, `sc.reduce_min`, `sc.argmax`, `sc.argmin`, `sc.norm_inf` and `sc.norm_1` are in the user guide and on the API page. None is filtered. Each is finished in the sense `AGENTS.md` asks: what it does not support raises, and the limits are written beside it. Recommend keeping them published.
3. Names to settle before 0.3.0: the parameter is `axis`, as in NumPy, where the roadmap wrote `axes`. `sc.matmul` is a new export beside `@`. The reductions are `reduce_max` and `reduce_min` where NumPy has `max` and `min`. `norm_1` and `norm_inf` run over every entry of a matrix, as `norm_2` does, where `np.linalg.norm(A, 1)` is an induced matrix norm. Recommend keeping the names and saying "over every entry" in the two docstrings.
4. Behaviour that changed for existing code (#408): `maximum`, `minimum`, `floor` and `ceil` have derivatives where they raised, and `abs` has slope zero at zero where it gave NaN. A quadratic-program proof now refuses a nonsmooth operation that depends on the variables. The derivatives guide documents all of it. Recommend keeping.
5. `sc.einsum` accepts less than `np.einsum`: a string spec only, no `optimize`, no operand reordering, floating operands only once two operands are multiplied, and no batched sparse operand (#403). Recommend keeping, since each refusal raises and the docstring names the string form.
6. Two published sentences were wrong since #404: `docs/how_it_works/ir.md` and `docs/how_it_works/lowering.md` said a matrix product takes operands of rank at most two. This packet's pull request corrects both. Recommend keeping the correction.
7. `mean` of an integer or bool tensor raises (#430). #401 deferred the promotion to #55, which closed without it. Recommend NumPy's rule, a float64 result.
8. The roadmap's #67 paragraph says the folded seed-axis product becomes one batched node. That holds for dense operands, and sparse operands keep #367's arrangement until #403. Recommend leaving the roadmap as it is.

## Expected gaps

- A batched product with a sparse operand raises `NotImplementedError`, and forward mode through a sparse product folds the seeds or builds one product per seed: #403.
- The Jacobian sparsity of a batched product enumerates every output entry times the contracted extent: #407.
- The derivative of a max or min reduction makes a sparse operand dense: #413.
- A selected zero tangent becomes NaN through a later singular partial, so forward and reverse mode differ for `sqrt(maximum(x, 0))` below zero: #414.
- `argmax` and `argmin` return int64, which cannot leave a Function yet, so the guide casts them to float64: #56.
- A scalar-lowered sum of negative zeros is `-0.0`, where NumPy, block lowering and the default lowering give `+0.0`: #429, a sub-issue of #296, to be fixed after #69 regenerates the snapshots.
- `einsum` is copy-heavy, since each transpose, diagonal and size-1 expansion is a copy loop, and no reduction is fused with its producer: #69.
- No lanes over nests, and none on max and min accumulators: #71.
- No product reduction: an open question of the roadmap that waits for a user, with no issue.
- No run times were measured. They wait for the Phase 4 run of the Milestone 2 checks.

## Unowned, filed by this section

- #429 a scalar-lowered sum keeps the sign of a negative zero. The coordinator made it a sub-issue of #296 and decided it, so it is listed above.
- #430 `mean` of an integer or bool tensor raises, where NumPy returns float64.
- #403 and #407, filed by #404, had no `kind:` and `area:` labels. They now carry `kind:feature` and `area:compiler`.

# Reference

Line numbers are on main at `16a1ee61`, after #431, and paths are under `src/scaly/`. Beside
this file: `demo.py` and `demo_output.txt`, and `checks.py` and `checks_output.txt` for the last
heading.

## Decisions

Hardest to undo first.

### Public interface

1. **`Expr.sum` and `Expr.mean`** (#401; `ir/expr.py:723-748`). `axis` is `None` for every axis,
   an integer or a tuple of integers. Negative axes count from the end. A list, a boolean, an
   axis out of range and a repeated axis raise. `keepdims` is keyword-only. `sum(axis=())`
   returns every entry unchanged. An empty domain sums to zero and its mean is NaN, without
   NumPy's warning. `mean` counts the structural zeros of a sparse operand. `sum` keeps the
   operand's dtype (#360) and refuses booleans. `mean` of an integer or a boolean raises (#430).
2. **`sc.reduce_max` and `sc.reduce_min`** (#408; `ir/expr.py:1337-1359`). Functions, not
   methods, with the axes and `keepdims` of `sum`. They take floating operands, propagate NaN
   as `np.max` does, and count the structural zeros of a sparse operand. An empty domain raises
   `ValueError` when the graph is built, where NumPy raises when it runs.
3. **`sc.argmax` and `sc.argmin`** (#408; `ir/expr.py:1362-1400`). One integer axis or `None`,
   which indexes the flattened array. The result is the first extremum, and a NaN comes before
   any number, as in NumPy. It is int64 and has no derivative. A sparse operand is made dense.
   An int64 result cannot leave a Function until #56, so the guide returns it through a cast.
4. **`sc.norm_inf` and `sc.norm_1`** (#408; `ir/expr.py:1403-1410`). The largest absolute entry
   and the sum of absolute entries, over every entry whatever the rank, as `norm_2`. For a
   matrix these are not NumPy's induced norms. `norm_inf` of an empty tensor raises, and
   `norm_1` of one is zero.
5. **`sc.maximum` and `sc.minimum` propagate NaN** (#408; `ad/rules.py:179-192`). They fold with
   `np.maximum` and `np.minimum`, and the C calls a helper with the same rule. Before #408 the
   C called `fmax` and `fmin`, which return the other operand. This is the roadmap departure:
   #408 replaced the roadmap's line that chose `fmin` and `fmax` by one that names what it
   replaced.
6. **The nonsmooth derivative conventions** (#408; `ad/rules.py:110-127`, `152-160`). At a tie,
   `maximum` and `minimum` give each operand one half, and a reduction gives each of the `n`
   tied entries `1/n`, the absent entries of a sparse operand included. `abs` has the slope of
   the sign, zero at zero. `floor` and `ceil` have slope zero and an empty derivative pattern.
   A branch that is not chosen contributes an exact zero, even where its partial is NaN or
   infinite. The derivatives guide states all of this under *Non-smooth conventions*.
7. **`sc.diag(x, k=0)`** (#401; `ir/expr.py:1413-1433`). A vector gives a square matrix by
   `SCATTER`, a matrix gives its diagonal by `GATHER`, and any other rank raises, as in
   `numpy.diag`. `k` outside the matrix gives an empty diagonal.
8. **Batched `@` and `sc.matmul`** (#404; `ir/expr.py:1290-1312`). NumPy's `matmul` rules: the
   last two axes multiply, leading axes broadcast, a vector is one row on the left and one
   column on the right. Scalars and non-floating operands raise. `sc.matmul` is exported.
9. **`sc.einsum(spec, *operands)`** (#410; `ir/einsum.py`). It takes NumPy's string form:
   ASCII letters, one `...` per term, an optional `->`, spaces ignored. A label repeated in one
   operand is a diagonal, and a label of extent one broadcasts against the same label in
   another operand. It rejects: a spec that is not a string, two arrows, an operand count that
   differs from the number of terms, any other character, a rank that does not match its term,
   an output label that repeats or occurs in no input, an output without `...` when an input's
   `...` stands for axes, a repeated label with unequal extents in one operand, and extents
   that differ without one of them being one. Through the operations it builds, it also
   refuses a product of non-floating operands and a batched sparse product (#403). It has no
   `optimize` argument and never reorders operands.
10. **A quadratic-program proof refuses nonsmooth operations that depend on the variables**
    (#408; `solvers/qp.py:74-102`). `maximum`, `minimum`, `abs`, `floor`, `ceil` and a max or
    min reduction raise `NotQuadratic`, through calls and maps too. Their Hessian is zero, so
    the derivative test alone would have accepted them.

### IR

11. **`ExprOp.SUM` became `ExprOp.REDUCE`** (#401; `ir/expr.py:1315-1334`,
    `ir/expr_spec.py:165-176`). The attributes are `kind`, one of `"add"`, `"max"`, `"min"`,
    and `axes`, a sorted tuple of distinct non-negative integers. The result drops the reduced
    axes, and `keepdims` is a `RESHAPE` of it. An empty `axes` is a valid node and is kept. The
    rename is mechanical across `ad/`, `passes/expr.py`, lowering and `viz/graph.py`, and no
    `SUM` is left in `src/`, `tests/` or `docs/`.
12. **The cut between #66 and #65**, fixed on #296. #66 brought the three kinds into the IR,
    the builders and derivatives of `add`, and the lowering of all three. #65 brought the
    public extrema, NaN propagation and the nonsmooth derivative rules. Between the two,
    derivatives of max and min raised and named #65.
13. **Sparse max and min count absent entries as zero** (#401; `passes/lowering/reduce.py:61-92`,
    `ir/support.py:374-380`). A group with an absent entry starts from zero and a complete group
    from the identity, minus or plus infinity. The internal node over an empty domain gives the
    identity, so its support is unknown and its stored pattern is dense. The public builders
    refuse an empty domain.
14. **The stored pattern of a reduction** (#401; `ir/expr.py:348-353` and `311-312`). A result
    of rank below two is dense. Reducing only non-empty batch axes keeps the operand's pattern.
    Otherwise the pattern is the projection of the stored entries onto the last two kept axes.
15. **A batched sparse product is refused when the graph is built** (#404;
    `ir/expr.py:1299-1300`). `matmul` raises `NotImplementedError` and names #403 when an
    operand has rank above two and either operand is sparse. Nothing densifies, as decision 4 of
    the roadmap asks.
16. **The support of a batched product** (#404; `ir/support.py:414-446`). With both operands
    known it is a Boolean product of block-diagonal patterns, one block per batch entry. With
    one known it is that operand's non-zero rows or columns per batch entry.
17. **`argmin` and `argmax` are built from existing operations** (#408;
    `ir/expr.py:1379-1400`). The entries equal to the extremum, or NaN when the extremum is NaN,
    select their index, and an int64 `REDUCE` of kind min takes the first. Its identity is the
    largest int64, set in `passes/lowering/schedule.py:32-36`.

### Differentiation

18. **A dense seed batch is one batched `MATMUL`, and sparse operands keep #367's
    arrangement** (#404; `ad/forward.py:185-237`). For dense operands the seed axis is a
    leading batch axis of one product per active operand. When the other operand is sparse the
    seeds are folded into the matrix dimension, and the tangent of a sparse operand is one
    product per seed. Both wait for #403.
19. **The reverse rule of a sum is a broadcast, and `_unbroadcast` is an axis reduction**
    (#401; `ad/reverse.py:107-113`, `ad/rules.py:411-418`). The cotangent is reshaped to the
    `keepdims` shape and multiplied by ones. `_unbroadcast` sums over the broadcast axes and
    reshapes, with no `SCATTER`, which replaces #26's interim rule.
20. **The partial of a max or min reduction is a mask with a weight** (#408;
    `ad/rules.py:18-26`, `116-127`). `Mask` gained a `weight` beside its condition. Both modes
    multiply the seed by it under a select, on the dense operand (#413).
21. **The reverse rule of a product keeps its four rank-two cases** (#404;
    `ad/reverse.py:253-273`). The general batched rule below them covers every rank.
    `test_matrix_vector_vjp_graph_uses_tensor_ops` pins the graph of the matrix-vector case, and
    no C snapshot notices when the four cases are deleted.
22. **`A @ ones` and `ones @ A` fold to reductions** (#401, #404; `passes/expr.py:280-287`), and
    the fold of a transposed matrix times a vector reads the last two axes
    (`passes/expr.py:265-277`).

### Lowering and generated C

23. **Reductions go through one scheduler** (#401, #404; `passes/lowering/schedule.py:15-76`),
    described under *For #69*. Each output keeps the summation order of the reference schedule.
24. **A max or min accumulates through a select** (#408; `passes/lowering/schedule.py:79-86`).
    `combine_reduction` keeps the accumulator when it is NaN or when the term does not beat it,
    and takes the term otherwise, so a NaN on either side stays. #401 had used the program's
    `MAXIMUM` and `MINIMUM`.
25. **Elementwise extrema call two scalar helpers in the C** (#408; `codegen/c.py:74` and
    `109-118`). `scaly_minimum` and `scaly_maximum`, with an `f` suffix for float32, are static
    inline functions that return a NaN operand and otherwise compare once. Each operand's text
    appears once, so a nest of extrema grows linearly in source size
    (`tests/codegen/test_extrema.py`). The names are reserved in `utils/names.py`, and lane
    wrappers call them per lane. `tests/baseline/c/wide.c` changed for it, by 378 bytes.
26. **Generated C that relies on NaN refuses fast math** (#408; `codegen/c.py:76-80` and
    `92-106`). A source with an extremum helper or a NaN self-comparison carries an `#error`
    under `-ffast-math` and `-ffinite-math-only`, as `isfinite` already did.
27. **Scalar-lowered sums keep a negative zero, unlike NumPy and block lowering; fix deferred to
    #429 after #69 regenerates the snapshots.** Decided by the coordinator on #429. The fix
    changes every scalar-lowered sum in the emitted C, and the default lowering already matches
    NumPy.
28. **A dense operand with known support is reduced over its support** (#401;
    `passes/lowering/reduce.py:42-58`). This is what keeps the work of a reverse-mode
    broadcast within the entries that can be non-zero. `tests/baseline/c/jac.c` changed for it:
    eight terms became two, 5,456 bytes became 5,688, and ten loops became eleven.
29. **A product lowers with its batch axes as outer output axes** (#404;
    `passes/lowering/contraction.py:16-35`). A batch axis of extent one reads coordinate zero,
    so broadcasting copies nothing.

### `einsum`

30. **Pairwise, left to right, with no search for an order** (#410; `ir/einsum.py:88-113`).
    The roadmap decides it. The guide says the order of the operands decides the size of the
    intermediates.
31. **A label that only one remaining operand carries is summed before the product**
    (`ir/einsum.py:84-86`, `93-94`), and a single operand is a diagonal `GATHER`, a `REDUCE`
    and a `TRANSPOSE`.
32. **A label shared by both operands and still needed later is a batch axis of the product**
    (`ir/einsum.py:95-97`). This is where `bij,bjk->bik` becomes one batched `MATMUL`.

### Left as it is by the simplification pass

Judgement calls that #431 did not take. #431 covers #401, #404 and #410. #408 landed after it
and has had no such pass.

- `_matmul_vjp` keeps four rank-two cases that the batched rule below them covers. Deleting them changes the derivative graph of every product. One test pins that graph and no C snapshot covers it.
- `_jvp_many_matmul_left` and `_jvp_many_matmul_right` keep a folded branch for a sparse operand next to the batched one. It goes with #403.
- Both build `x @ y` only to read its shape.
- The verifier's `_matmul_shape` derives the result shape again instead of sharing the builder's lines. Sharing them would make the verifier repeat a builder's mistake.
- `_reduce` checks its axes by hand. NumPy's `normalize_axis_tuple` is shorter and accepts booleans, which the tests reject on purpose.
- The `keepdims` shape is spelled six times: in `sum`, `mean`, `_extreme`, the reverse rule, `reduction_partial` and `reduce_entries`. One helper could own it. It was left because #408 owned three of those lines while the pass ran.
- The identity of each kind is spelled three times in `passes/lowering/`, for the same reason.
- `mean` could call `sum`. A boolean operand would then raise where it does not today.
- `_lower_reduce` tests "is this a full reduction" three times. Reordering its branches is easy and was left alone because section 8 is editing `passes/lowering/`.
- The support rule of a product returns early for a full and for a known-zero operand. Both returns are implied by the general rule and were removed, then restored: the review showed that without them a full or empty result allocates an index per output entry, 512 MiB for `(8192, 1) @ (1, 8192)`.
- `_matmul_mask` has a batched branch beside the rank-two one, which starts from the active dependencies. Merging them is #407.
- `_unbroadcast` was not touched: `ad/rules.py` belonged to #408 while the pass ran.

## For #69 (section 8)

How reductions, products and `einsum` lower today. Fusion has to keep, replace or remove each
of these.

**The scheduler, `emit_reduction`** (`passes/lowering/schedule.py:15-76`). Every reduction
except the supported form below hands it the output ranges, the reduction ranges, the target
and the scalar value. It reads the coefficient of each reduction variable in every load of the
value (lines 41-43) and whether a reduction bound is itself loaded (line 49), then picks one of
three forms.

- *Destination form* (lines 50-53), when a load is strided in a reduction variable and no
  bound is loaded. The outputs are set to the identity in one nest. A second nest puts the
  reduction ranges outside the output ranges and updates the destination in place, so the inner
  loop is contiguous. Summing a matrix over its first axis takes this form, and so does a
  matrix product, whose right operand is strided in the contracted index.
- *Accumulator form* (lines 55-63), otherwise. Inside the output nest a scalar starts at the
  identity, the reduction loops update it, and one store writes it. Summing over the last axis
  takes this form.
- *Four output lanes* (lines 65-74), on top of the accumulator form when the innermost output
  range has a static trip count of four or more and no bound is loaded. The innermost output
  axis is cut into blocks of four with four named accumulators, and a tail loop takes the
  remainder. This happens during lowering, so fusion sees four copies of the value with
  substituted coordinates, not one.

**Max and min** (`passes/lowering/schedule.py:79-86`). The three forms are the same, and only
the update differs. For a max it is `acc = (acc != acc || term < acc) ? acc : term`, built as
a `SELECT` over an `OR` of a NaN test and a comparison, and the identity is minus infinity, or
the largest int64 for the index reduction of `argmin` and `argmax`. Fusion must not break
three things here.

- The term occurs twice in the update, once compared and once chosen. Today it is bound to a
  scalar first, so an inlined producer such as `sin` runs once. A producer substituted into
  both places would run twice, and a nest of extrema would double with each level. The
  elementwise helpers exist for the same reason (`tests/codegen/test_extrema.py`).
- The order of the operands and of the two tests carries the NaN rule. Rewriting the select
  to `fmax`, or to a comparison alone, loses a NaN in one of the two operands. `checks.py`
  compares NaN results with NumPy.
- The source keeps its guard against fast math whenever such an update is present.

**The loaded-bound contract from #46** (line 49, and `passes/lowering/reduce.py:83-92`). A
reduction over stored entries loops over groups with a `REDUCE` range whose bounds are loaded
from a table of row pointers. `emit_reduction` then keeps the output coordinates outside,
uses one accumulator and expands no lanes. #66 kept this for sparse max and min: they run two
group loops, one over the groups where every entry is stored, starting from the identity, and
one over the groups with an absent entry, starting from zero. The roadmap says a kernel with
loaded bounds is not fused before #69 says so.
`test_sparse_reductions_keep_loaded_bounds_and_linear_tables` pins it.

**The compact traversal #66 added** (`passes/lowering/reduce.py:26-28` and `42-58`). A dense
operand whose known support is smaller than its size, under a reduction that keeps an axis,
does not go through `emit_reduction`. One loop stores the identity, then one `REDUCE` range
over the supported entries combines `src[entries[k]]` into `out[groups[k]]` through two index
tables. The destination index is loaded, so this is a scatter-add and not an accumulator.
Reverse mode produces this shape: the cotangent of a broadcast is `_unbroadcast`, a `REDUCE`
over the broadcast axes, and its operand often has known support. The producer is read only at
`entries`, so a producer inlined at those coordinates does work in proportion to the support.

**A batched product** (`passes/lowering/contraction.py:16-35`). One output nest over the whole
result shape, batch axes first, and one `REDUCE` range over the contracted axis. `batch_axes`
(line 34) tells the destination form to keep the non-unit batch ranges outside the reduction
(`schedule.py:52`), so each batch entry runs the rank-two schedule. A sparse operand goes to
`_lower_stored_matmul` and never has a batch.

**What `einsum` emits around each product** (`ir/einsum.py:88-113`). Each operand is
transposed to batch, free and contracted axes and reshaped to a matrix per batch entry. The
product is reshaped back, and the result is transposed to the output order. A `RESHAPE` is an
alias and costs nothing (`passes/lowering/movement.py:23-33`). A `TRANSPOSE` that is not the
identity is a permuted copy into a new buffer (`movement.py:128-137`). The diagonal of a
repeated label and the expansion of a label of extent one are `GATHER` copies
(`ir/einsum.py:69-74` and `77-82`, `passes/lowering/gather.py:14-23`). The roadmap's #68
paragraph expects #69 to turn the transposes into index remaps.

The demo's last table gives cases to test against. A transpose that changes nothing is
simplified away, so `bij,bjk->bik`, `ij,kj->ik` and `ij,jk,kl->il` lower to the same loops and
buffers as the product written with `@`. `...ij,...j->...i` with batches `(2, 5)` and `(5,)`
lowers to 11 loops and 5 buffers where `@` takes 5 and 4: the ellipsis axis both operands share
becomes the batch label and moves to the front, which permutes the left operand, a copy of 90
doubles, and then the result, a copy of 30.

**`fuse_ranges` and `widen_ranges`.**

- `fuse_ranges._loops` (`passes/program/fuse_ranges.py:76-89`) does not accept a nest with a
  `REDUCE` range. A reduction is therefore never fused there, and when a reduction consumes a
  buffer of a fusion region the pass returns the whole procedure unchanged (lines 229-230).
- `widen_ranges._ordered_reduction` (`passes/program/widen_ranges.py:299-323`) gives lanes to a
  `REDUCE` range only when its one statement adds into an accumulator or into its destination.
  A max or min is a select, so it gets no lanes: a full sum of 64 entries calls a lane helper
  and a full max of 64 entries is a scalar loop. In the demo the C of `reduce_max` is 1,498
  bytes and that of the `sum` 3,910, the difference being the lane helpers. #71 owns lanes on
  any reduction.
- `fold_arith` does not split the index of a `REDUCE` range
  (`passes/program/fold_arith.py:144`).

## Demo

`uv run internal/notes/compiler_run_2026_11_10/tensor_operations/demo.py` from a checkout. One
Python function builds ten results from four inputs, once with scaly expressions and once
with NumPy arrays: a `sum` with `keepdims`, a `mean` over two axes, `reduce_max`, `norm_inf`,
`a @ b` with leading axes `(2, 1)` and `(5,)`, `sc.diag`, and `einsum` specs for a batched
contraction, an ellipsis, a trace, a diagonal and a chain of three operands. The script
compares the compiled values with NumPy, compares the Jacobian of all ten results by each
input with central finite differences, prints operation counts, source size and compile time
of the generated C, and sets four `einsum` specs beside the same contraction written with `@`.
It takes no run times. `demo_output.txt` is the whole output on la015, and this is the same
output without the table of counts:

```text
== values against NumPy (np.sum, np.mean, np.max, np.matmul, np.einsum, np.diag) ==
inputs: a (2, 1, 3, 4), b (5, 4, 3), v (5, 3), s (3, 3)
a.sum(axis=-1, keepdims=True)              shape (2, 1, 3, 1)   max |scaly - numpy| 0.0e+00
a - a.mean(axis=(0, 3), keepdims=True)     shape (2, 1, 3, 4)   max |scaly - numpy| 2.2e-16
reduce_max(a, axis=-1)                     shape (2, 1, 3)      max |scaly - numpy| 0.0e+00
norm_inf(s)                                shape ()             max |scaly - numpy| 0.0e+00
a @ b                                      shape (2, 5, 3, 3)   max |scaly - numpy| 0.0e+00
einsum bij,bjk->bik                        shape (5, 4, 3)      max |scaly - numpy| 1.8e-15
einsum ...ij,...j->...i                    shape (2, 5, 3)      max |scaly - numpy| 8.9e-16
einsum ii->                                shape ()             max |scaly - numpy| 0.0e+00
diag(einsum ii->i)                         shape (3, 3)         max |scaly - numpy| 0.0e+00
einsum ij,jk,kl->il                        shape (3, 3)         max |scaly - numpy| 1.1e-16

== Jacobians of all ten results against central finite differences ==
d everything / d a: shape (236, 24)  nonzeros  1422 max |scaly - finite difference| 1.1e-09
d everything / d b: shape (236, 60)  nonzeros  1080 max |scaly - finite difference| 1.8e-09
d everything / d v: shape (236, 15)  nonzeros   117 max |scaly - finite difference| 7.5e-10
d everything / d s: shape (236, 9)   nonzeros    52 max |scaly - finite difference| 2.6e-10

== einsum beside the same contraction written with @ ==
spec              written with @                       loops         buffers         C bytes
bij,bjk->bik      x @ y                              5 and 5         3 and 3   3261 and 3261
...ij,...j->...i  (p @ v[..., None])[..., 0]        11 and 5         5 and 4   6297 and 5103
ij,kj->ik         s @ t.T                            6 and 6         4 and 4   5061 and 5061
ij,jk,kl->il      (s @ d) @ t                        8 and 8         5 and 5   4019 and 4019
```

## Reading list

About 290 lines, in reading order, on main at `16a1ee61`.

1. `ir/expr.py:1315-1334`. `_reduce`: the axes it accepts and the node it builds.
2. `ir/expr.py:1297-1304`. `matmul`: the refusals and the result shape.
3. `ir/expr.py:1351-1359`. `_extreme`: what the public max and min add to `_reduce`.
4. `ad/reverse.py:107-113` and `ad/rules.py:411-418`. The reverse rule of a reduction, and `_unbroadcast` as a sum.
5. `ad/rules.py:110-127`. The partials of the elementwise and the reduced extrema: ties, shares and the mask.
6. `ad/forward.py:214-237`. The seed axis as a batch axis, and the folded branch a sparse operand keeps.
7. `passes/lowering/schedule.py:15-86`. `emit_reduction`, the three forms, and the select of a max or min.
8. `passes/lowering/reduce.py:18-92`. The dense nest, the traversal over known support, and the groups of stored entries with max and min.
9. `passes/lowering/contraction.py:16-35`. The batched product.
10. `ir/einsum.py:69-74` and `88-113`. The diagonal, then the pairwise loop.

## Source lines

Lines under `src/`, from `git show --numstat`.

- #401 added 262 and removed 64 in 12 files, #404 added 124 and removed 69 in 10 files, and #410 added 115 in 2 files, of which `ir/einsum.py` is 113. #408 added 242 and removed 49 in 12 files.
- Before the simplification pass the three pull requests it covers had added 501 lines and removed 133, a net 368.
- #431, the simplification pass, added 14 and removed 31 in 6 files, which leaves a net 351.
- #408 landed after the pass and was not simplified. With its net 193 the section's net addition is 544 lines.

## What the coordinator was least sure of

`checks.py` runs the four checks below, and `checks_output.txt` is its output on la015.

**That `sum` over an empty axis tuple keeps NumPy's sign of zero.** It does in block lowering
and in the default lowering, and it does not in scalar lowering. NumPy's sum starts from `0.0`,
so `np.sum([-0.0, 0.0, 1.0], axis=())` is `[0.0, 0.0, 1.0]`. #66 kept an empty `axes` as a
`REDUCE` node, and block lowering gives each entry an accumulator that starts at `0.0`, which
matches. `test_no_axis_sum_keeps_numpy_signed_zero` pins that form only. Scalar lowering emits
the terms without the identity, so `sum(axis=())` is a copy and returns `-0.0`. The same holds
for full sums, axis sums and `mean`. Full sums already behaved this way before the section.
#429 owns it, and the coordinator deferred the fix until #69 has regenerated the snapshots.

**That a seed batch cannot reach the wrong forward rule.** It cannot. The rule
(`ad/forward.py:185-208`) sends a product with a sparse operand to its own branch first. There
the tangent of a sparse operand is one product per seed, and the tangent of the dense operand
goes to a helper that tests the other operand's pattern and folds. Two dense operands reach the
helpers with no pattern and take the batched branch. `matmul` refuses a batched sparse operand
when the graph is built, so a wrong turn would raise, not give a wrong number. The check
differentiates five dense shape pairs, three mixed ones and one sparse pair by each operand,
compares with finite differences, and finds only batched products for dense operands and only
operands of rank at most two beside a sparse one.

**That `einsum` matches `np.einsum` on rank three and above.** It does on everything tried.
3,000 random specs of one to three operands, with repeated labels, labels and ellipsis axes of
extent one, and ellipses in any position, were built on constants and folded: 2,974 that NumPy
accepts give NumPy's values, the 26 that NumPy rejects are rejected, and 40 were also compiled
to C. 1,889 had an operand of rank three or more and 1,661 a diagonal. Folding uses the same
NumPy evaluation as the simplifier, so it checks the builder. The 40 compiled specs and the 20
of `tests/ir/test_einsum.py` check the lowering.

**That the extrema follow NumPy on NaN and the conventions at ties.** They do on what the check
tries: `reduce_max`, `reduce_min`, `argmax`, `maximum` and `norm_inf` on data with a NaN give
NumPy's results, a tie splits the gradient in halves in both the reduction and the elementwise
form, and `abs` and `floor` have the documented slopes. The check is small. #408's own tests
are the evidence for sparse operands, float32 and both AD modes, and I did not repeat them.

**That the simplification pass changed nothing.** The C snapshots are byte-identical, and a
comparison of supports, stored patterns and Jacobian masks on 1,036 random graphs before and
after found no difference. Neither saw what the review found: two of my deletions kept every
result and allocated hundreds of megabytes on large products. They are restored. No test in the
suite bounds the memory of the support and sparsity analyses. The snapshots also say little
about products: #404 noted that no snapshot covers the derivative code of a product.

**What a batched seed product costs at run time.** #404 replaced one large folded product by a
product with the seed axis as a batch axis, which changes the loop order of every dense
derivative through a product. Nothing was timed, by the run's rule, so the Phase 4 measurements
are the first evidence. The same holds for max and min, which lost `fmax` and have no lanes.

**What I did not expect.** #55 closed with the integer mean still open, although three records
said #55 owned it. Two published pages still described the rank-two product. An ellipsis that
only one operand broadcasts costs two copies in `einsum`. A reduction that consumes a mapped
result stops `fuse_ranges` for the whole procedure. And #431 was removed from the merge queue
once without being tested, when two entries ahead of it failed.
