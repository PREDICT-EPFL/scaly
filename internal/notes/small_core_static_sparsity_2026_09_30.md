> Frozen design note, 2026-09-30 and 2026-10-01. It proposed the revisions that
> `core_compiler_roadmap.md` adopted on 2026-10-01; the maintainer settled every decision in
> [Decisions to take](#decisions-to-take). The roadmap is the maintained text; this note keeps the
> reasoning and the alternatives.

# A small core and static sparsity

Abbreviations: AD is automatic differentiation, ABI the generated C calling convention, CSC and CSR
compressed sparse column and row storage, KKT the Karush–Kuhn–Tucker system of an interior-point
method (IPM), TACO the Tensor Algebra Compiler, BLASFEO the embedded dense linear-algebra library.

## Why revisit the roadmap

The roadmap of 2026-09-29 followed devrush closely: it planned about 67 expression ops by its end,
including `TRISOLVE`, `CHOLESKY`, `LDL`, `LU`, `SPARSE_LDL`, `SPARSE_LDL_SOLVE`, `TAKE`, `PUT`, four
sparse conversion ops and a pattern in `TensorType` with sparse rules for every op. Since then
devrush was restructured (`internal/notes/restructure_plan_2026_09_28.md` on that branch): its core
went down to 57 builtin ops and its linear algebra left the core, but as eight ops registered from
`scaly.linalg` through `ir/expr.py::register_op`, each with verify, forward, multi-seed, reverse,
sparsity, fold and lower rules plus traits. The ops left the core; the op count did not go down.

Three questions from the maintainer drive this note: how small the core opset can be, and what its
ops should be called; whether every node should carry a sparsity pattern; and what a TACO-style
sparse compiler, specialized to patterns known at build time, would add beside the tinygrad-style
loop compiler already planned.

## Verdict

1. **A closed language of about 50 ops, most of them one table.** Two leaves, one elementwise
   table (math, arithmetic, predicates, `SELECT`, `CAST`), four shape ops, `REDUCE`, `MATMUL`,
   `GATHER`, `SCATTER`, `PACK`, `VALUES`, the callee ops `CALL`, `LOOP` (maps included) and `COND`,
   and `PRINT` and `EXTERN`. No op registry.
2. **Every algorithm is a library `Function`.** Factorizations, triangular and sparse solves,
   segment sums, KKT assembly, interpolation search. What devrush's registered ops needed from the
   compiler reduces to four core capabilities, all already half-planned: a run-time trip count on
   `LOOP`, in-place loop carries, unrolling decided in lowering, and vector lanes on reductions with
   run-time bounds. External C is its own construct, `sc.extern` over an `sc.CLibrary`, always
   concrete.
3. **Sparsity is two facts, both on every node.** The type may carry a *stored* pattern, a contract
   about storage fixed by deterministic rules like shape inference. Separately, every node has a
   derived, cached *support*: the entries that may be nonzero, whatever the storage. Derivative
   sparsity stays a third, separate analysis that consults support.
4. **TACO's methods apply inside lowering, not as a new dialect.** With the pattern known at build
   time, everything TACO does at run time (merging index lists, assembling output patterns) happens
   in Python, and lowering chooses per region among straight-line code, loops over constant tables,
   affine loops and dense blocks. Dense tensors are the all-dense case of the same iteration plans,
   so there is one loop compiler, not two.
5. **The order of work becomes three usable milestones**: differentiable loops, static sparse
   functions end to end, and competitive kernels with TinyMPC and the generated IPM as the gates.

Four independent investigations and one adversarial review produced this; see
[How this note was made](#how-this-note-was-made). They agreed on points 1, 2, 4 and 5. They split
on where the stored pattern lives, which is the first decision below.

## The core language

The test for an op: a pass needs a rule for it that composing other core ops cannot supply without
losing one of derivative or structural sparsity, a loop staying a loop, work proportional to the
real data, a lowering decision that cannot be recovered afterwards, or cost. Everything else is a
builder (Python producing core ops) or a library `Function`, with `sc.custom_derivative` where
tracing the derivative is wrong or slow. From tinygrad we take the rule that passes match families,
not members, so an elementwise op is one table row read by the verifier, the constant folder, the
partials of [#20] and the renderer, not the seven files `docs/dev/codebase.md` lists today.

| Family | Ops | Notes |
|---|---|---|
| leaves | `INPUT`, `CONST` | `CONST` gains a uniform form, so `zeros_like` of a 10⁵×10⁵ matrix is O(1) |
| elementwise table | 17 unary math ops as on main; `ADD SUB MUL DIV POW ATAN2 MINIMUM MAXIMUM`; `FLOORDIV MOD`; `LT LE EQ NE AND OR NOT ISFINITE`; `SELECT`; `CAST` | `SELECT` and `CAST` have their own AD rules; libm functions stay, since removing them costs accuracy and saves nothing |
| shape | `RESHAPE`, `TRANSPOSE`, `SLICE` (with steps), `CONCAT` | `stack`, `pad`, `split`, `broadcast_to`, `diag` are builders; broadcasting stays implicit |
| reduction, contraction | `REDUCE(x; kind=add\|max\|min, axes)`, `MATMUL(a, b)` batched | `einsum`, `argmax`, `norm_inf` are builders; see "Why `MATMUL` and not `einsum`" |
| indexing | `GATHER(x, idx; axis, mode)`, `SCATTER(base, idx, values; axis, combine, mode)` | see below |
| sparse boundary | `PACK(values; pattern)`, `VALUES(x)` | the stored pattern is in the type (decision A) |
| callees | `CALL`, `LOOP`, `COND` | `VMAP` is a `LOOP` without carries |
| effects and external code | `PRINT`, `EXTERN` | see "External code"; `EXTERN` absorbs `SOLVER_CALL` |

`COPYSIGN` is dropped: devrush's own AD rules were its main user, and `sc.copysign` becomes a
builder over `where`, `abs` and a comparison, which differs only in the sign of zeros and NaNs.

That is about 50 ops, against 38 on main today, 65 on devrush (57 builtin and 8 registered) and
about 67 at the end of the current roadmap, while adding loops, branches, runtime indexing,
predicates and sparse tensors.

**Indexing.** `GATHER` and `SCATTER` are what XLA, StableHLO, MLIR, PyTorch, tinygrad and main
already call them. The index becomes an int64 operand of any shape, so a constant index is simply
the static case and every rule that reads `attrs["indices"]` today reads the constant operand.
`combine` is `set`, `add`, `max` or `min`, resolved in index order (the last write wins for `set`,
`add` sums in order). `mode` uses JAX's names: `fill` reads a fill value out of range and drops an
out-of-range write through a scratch slot, and `promise_in_bounds` skips the check. Main's
`scatter(values, idx, shape)` is `SCATTER(zeros, idx, values, add)`. `segment_sum`, `segment_max`,
`take`, `dynamic_slice`, `index_add` and `index_set` are builders. `TAKE`, `PUT`, `PUT_ADD`,
`SEGMENT_REDUCE`, `INDEX_ADD` and `INDEX_SET` never exist on main.

**Callees.** `LOOP` is [#31]'s design with one change: the trip count becomes a user-facing operand
with a static upper bound, which is what devrush's `ragged_add` and `ragged_dot` were for. A
`VMAP` is a `LOOP` with no carry and no condition, as `lax.map` is a carry-less `scan`. The merge
lands after `LOOP` exists, as a rename that keeps `vmap.c` and every snapshot byte for byte, because
the mapped path is the most measured one. `COND` takes a boolean or a clamped integer selector;
there is no separate `SWITCH`.

**`PRINT` comes early.** `sc.print` is the first tool that makes generated code debuggable, before
the graph viewer is made useful, so it lands right after wave 0, in float64 only, with integer and
boolean formats added by [#30]. [#77]'s semantics stand (a print runs whenever generated code
computes its value; identical prints intern to one; the AD rule keeps the primal), and the guide
states them next to the feature.

**Why `MATMUL` and not `einsum`.** Mathematically neither is needed: tinygrad writes a matrix
product as a broadcast multiply followed by a sum, `(a[:, :, None] * b[None, :, :]).sum(1)`. That
works there because the only analyses that walk tinygrad's graph node by node are shapes and dtypes,
and the scheduler fuses the `(m, k, n)` product away. Even so, tinygrad recognizes a multiply under
a reduction again when it chooses tensor-core kernels, so the matrix product is an op in disguise.
In Scaly every node is analysed one by one: derivative sparsity keeps a relation per node, support
is cached per node, AD builds tangents per node, and scalarize counts operations per node. Each of
them would see an `(m, k, n)` intermediate, larger than both operands and the result, before any
fusion runs. The criterion is therefore: one node for a pairwise contraction, because its
intermediate is larger than its operands. `einsum` is not one op, because a contraction of several
operands is a sequence of pairwise contractions whose order is a cost decision that should be made
once, by the builder, rather than redone by every pass. A pairwise `einsum` is a batched `MATMUL`
up to transposes and reshapes, which cost nothing once [#69] reads them as index maps. If those
transposes ever stop being free, the fix is in lowering, or in turning `MATMUL` into a pairwise
contraction with labelled axes (XLA's `dot_general`); `einsum` still stays a builder.

**Not in the core.** No `register_op`, no traits, no public pass slots or option namespaces. The
extension contract is `Function`, `sc.custom_derivative`, `LOOP`, and external bodies. What each of
devrush's registered ops needed maps onto a core capability:

| Devrush op (`linalg/ops/`) | What it needed | Core capability |
|---|---|---|
| `ragged_add`, `ragged_dot` | inner runs `lo[g] <= p < hi[g]` of run-time length | `LOOP` with a run-time trip count under a static bound |
| `ragged_add` on the work column, row swaps in `lu`, updates in the sweeps | updating a large carry without copying it | buffer reuse in lowering, see "In-place updates" |
| `cholesky`, `ldl`, `trisolve` with `Unroll` | straight-line code on small sizes | lowering unrolls a static-length loop under the target's straight-line budget, honouring the node's lowering hint |
| `trisolve`'s hand-written four partial sums, `blocked_sum` | lanes on a reduction whose bounds are loaded | [#71]'s lane split with a remainder loop on run-time bounds |

The acceptance test is explicit: the library sparse LDL^T must produce the loop structure of devrush's
direct lowering (`linalg/ops/sparse_ldl.py::_lower_sparse_ldl`) and come within its measured times.
If it needs a registered op to get there, this part of the design has failed and is reopened.

**In-place updates.** The semantics stay functional: `SCATTER` returns a new value, and a library
`Function` never says "update in place". Writing into the old buffer is a lowering decision, as in
XLA's buffer assignment and Futhark's in-place updates, and it needs no op. Three rules, cheapest
first:

1. *Last use.* A `SCATTER` writes into its base's buffer when nothing reads the base after it in the
   schedule. Inside a loop step the same holds for a carry: when every read of the old carry is
   scheduled before the update, the next carry reuses its slot. This alone covers a left-looking
   LDL^T: step `j` reads the columns of `L` before `j`, then writes column `j`, and nothing reads the
   old `L` afterwards.
2. *Same index.* `SCATTER(base, idx, f(GATHER(base, idx)))` reads only what it writes, so its read
   and write loops may be fused, for any index, including a pivot chosen at run time.
3. *Disjoint indices.* When the old value is read after an earlier write in the same step, devrush's
   proof applies ([#36]): every index is evaluated at build time and the entries read are disjoint
   from those written.

When none applies, lowering copies the carry, which costs O(size) per step and can turn a factor
quadratic. So a loop may state the contract, `in_place=("L", "w")`, and lowering raises when it
cannot honour it, as the no-densification contract does for sparse storage. What can still separate
a library LDL^T from devrush's direct kernel is the quality of the loops (accumulators in registers,
chunks of columns, lanes), which is lowering work and not a missing op.

**Custom rules on library Functions.** `sc.custom_derivative(fn, jvp=, vjp=, sparsity=)` ([#22])
sets forward and reverse rules and the derivative sparsity of any Function, library ones included,
at construction. The rules are Functions themselves, so higher derivatives differentiate the rules.
Every call (`CALL`, `LOOP`, the map case) goes through the same funnel, so a rule holds wherever the
Function is used. The output's stored pattern needs no rule: it is part of the declared output type.
The support of an output is read from the body unless the declaration gives a pattern.

A solve's derivatives must also reuse the primal's factorization; today a call's value and its
derivative run the callee's forward pass twice. So the reverse rule takes JAX's `custom_vjp` form:
a forward rule returns the outputs and private residuals (a factorization, a loop trajectory), and
the backward rule receives the residuals with the cotangents. The residuals are results of the same
invocation ([#28]), computed once, and they keep their dependence on the inputs so second
derivatives stay right.

**External code.** Calling C through an ordinary `Function` does not fit: a Function may have holes,
its body is a graph, and compile and link flags have no place on it. External code gets its own two
values:

- `sc.CLibrary(name, sources=, headers=, include_dirs=, defines=, compile_flags=, link=, versions=)`
  is a frozen description of how to build and link some C, shared by every symbol taken from it. Its
  content (source text, flags, linked library identities and versions) enters the JIT key ([#64]).
- `sc.extern(library, symbol, inputs=..., outputs=..., workspace=0)` declares one C function with a
  concrete signature: every leaf has a shape, dtype and pattern, so an extern is concrete by
  construction. Its calling convention is the generated one, one typed pointer per leaf and a
  workspace ([#56]). Calling it builds `EXTERN` nodes, one per output, grouped by invocation ([#28]).

A shape-generic C kernel is a Python function that builds one `extern` per shape, generating the
source text if needed; the template machinery is not involved. An extern has no derivative unless a
Function wrapping it gets `sc.custom_derivative`, and its derivative sparsity is declared or dense.
The first contract allows only deterministic, reentrant calls: inputs read only, outputs fully
written, no retained pointers. The JIT compiles the reachable libraries into the same shared object
or links them; ahead-of-time export writes the flags into a build manifest beside the generated C,
since Scaly does not own the user's build. Solver plugins (PIQP, IPOPT) become externs with the
plugin's library, which removes `SOLVER_CALL` and devrush's `function.extern = callee`, set after
construction.

**Dense kernels written in Python.** Scaly will not tile like BLASFEO by itself soon, and does not
need to: a user, or a `scaly.linalg` module, can write a register-blocked kernel as ordinary
expressions, choosing the tile sizes in Python (4×4 blocks of slices, a `LOOP` over the `k` panels
carrying the block accumulator). The core owes that user predictable lowering:

- a small block scalarizes into straight-line code over C locals (today's `scalarize`);
- a small loop carry, such as a 4×4 accumulator, lives in C locals, not in workspace slots
  (a requirement on [#31]'s lowering);
- affine slices are read in place, never copied ([#69]'s rule that movement ops are not materialized);
- a packed layout the user asks for, a transpose or reshape, is materialized once and hoisted out of
  the loops (`hoist_invariant`);
- lowering never undoes structure the user wrote: a `Function` boundary, a loop and a lowering hint
  (`.scalar()`, `.block()`) are kept;
- generated buffer parameters become `restrict` (they are not today), so the C compiler vectorizes
  the straight-line blocks.

Each of these gets a test asserting the generated C keeps the structure.

## Sparsity

Three facts must not be confused:

- **Stored pattern**: which entries a value physically stores, in the ABI and in buffers.
- **Support**: which entries may be nonzero. This is a fact about the value, meaningful for any
  storage: `concat([A, 0])`, a constant with zero blocks, `H + rho*I`, `d[:, None] * G`.
- **Derivative sparsity**: a relation between the entries of an output and of a `wrt`.

A stored pattern bounds the support; the converse fails. Derivative sparsity cannot be computed from
the support of single nodes once calls are involved, because a callee's dependency relation is
what can be computed once per callee (`ad/sparsity.py`), but it must consult support: today
`_matmul_mask` ignores zeros in operand values, so a dense-typed constant with zero blocks gets a
dense Jacobian.

**What devrush does.** `linalg/sparse.py::SparseMatrix` is a library value holding CSC `indptr`,
`indices` and a values `Expr`. Every operation computes its pattern in NumPy and writes the values
with static gathers and segment sums: the matrix-vector product is
`segment_sum(values * gather(x, cols), rows, m)`, and `_spgemm` lists every scalar product pair.
Inside the graph the pattern is invisible, as the maintainer suspected; outside it the pattern has
four homes (the wrapper, the `S` leaf and template key, `Function.output_sparsities`, and the LDL
op's table attributes). Devrush's `TensorType.sparsity` still exists but nothing sets it. Devrush's
own notes record the cost: `perf_gaps_proposal_2026_09_30.html` item A7 (the dense backend cannot use
matrix-vector kernels because every product goes through index tables) and its QP Hessian investigation
(the parameter-dependent Hessian is built dense, then gathered, every solve).

**Support on every node.** `Expr.support` is computed lazily, cached per node, and never part of
the intern key, since it is a pure function of the node. `None` means any entry may be nonzero.
Rules, one per op, default `None`, so a missing rule is sound but imprecise: image under the static
coordinate map for movement ops, `GATHER`, `SCATTER` and `CONCAT`; union for `ADD`, `SUB`,
`MAXIMUM`, `MINIMUM` and `SELECT`; intersection for `MUL`; the numerator for `DIV`; unchanged for
zero-preserving unary ops; the Boolean product for `MATMUL`, computed with SciPy; the callee's
support tiled over stages for `VMAP`; a fixed point over carries for `LOOP`. A randomized harness
checks soundness: random values on random patterns, and every entry outside the support is zero.
Support feeds simplification (`_is_zero` becomes "support is empty", disjoint products fold),
derivative sparsity, scalar lowering, and, later, demanded-entry analysis.

**Where the stored pattern lives: two designs.** Both give every node a pattern; they differ in
whether storage is a contract.

- **Hybrid (chosen).** `TensorType(shape, dtype, diff, pattern=None)`. The pattern is canonical,
  interned, and compared by identity. It covers the last two axes, and leading axes are batch axes
  sharing it. It is computed by fixed rules when the graph is built, exactly like shape inference,
  and it uses the same rule table as support restricted to sparse operands. An op whose rule gives
  no pattern raises on a sparse operand instead of densifying. `PACK` and `VALUES` convert between
  a values vector and a sparse tensor; `to_dense` and `from_dense` are builders. The physical order
  of stored values belongs to lowering, and CSC order is fixed only at the ABI.
- **Derived only.** No pattern in the type. A sparse matrix is an ordinary `(m, n)` expression
  produced by `SCATTER(values, P)`, and lowering chooses per node between a dense buffer and compact
  storage of its support. `sc.S` leaves carry the pattern in the signature. No sparse op exists and
  AD needs no sparse rule for the primal formulas.

The derived design is the more minimal, and it exploits zeros in dense-declared code automatically.
The adversarial review found four problems with it that the hybrid does not have:

1. **No contract.** A user cannot require that a 10⁵×10⁵ matrix with 10⁶ entries is never
   densified (80 GB dense against 8 MB compact). A density threshold and a lowering report find the
   problem after simplification may already have allocated it: today `passes/expr.py::_evaluate`
   folds `scatter(CONST(values), P)` by allocating the full shape.
2. **Unstable output.** If storage follows inferred support, a sharper rule can cross a threshold
   and change buffers, loops and the generated C, and turn a `0 * inf` from NaN into zero. Storage
   must follow declared structure and a fixed policy.
3. **Signatures carry patterns anyway.** `sc.S` leaves, template keys, derivative outputs ([#48])
   and headers all need the matrix identity, so the derived design moves the pattern into a second
   descriptor beside the type rather than removing it.
4. **"No sparse AD rules" is true only for the primal.** Reverse mode through `y = A @ x` builds
   the outer product `ybar[:, None] * x[None, :]` (`ad/reverse.py::_matmul_vjp`) and then gathers
   the stored entries: 10¹⁰ entries computed for 10⁶ used. Both designs need a sampled adjoint (the
   cotangent of stored values evaluated only on the pattern) in the first differentiated sparse
   milestone. The hybrid states it as one `MATMUL` reverse rule on a sparse operand.

What the hybrid gives up is automatic compaction of dense-declared values. That stays possible
later as an explicit, opt-in conversion (`sc.sparsify(x)` is `PACK(GATHER(x, support(x)))`) with its
own reproducibility rule. Dense-declared zeros are still exploited without it by scalar lowering,
by derivative sparsity and by demanded-entry analysis, which addresses the QP Hessian investigation: compute only the
entries a gather reads.

**Layouts at boundaries.** Inside the graph the type fixes which entries are stored, never their
order: lowering picks the traversal and the order of internal buffers. Order is fixed where values
cross the ABI. A sparse leaf declares its layout: CSC by default, CSR, or a coordinate list in an
order the caller gives, for solvers that read triplets in their own order. The layout is a field of
the leaf in the signature, not of the pattern: two layouts of one pattern are the same mathematical
type, with the same algebra. It enters the instance key and the header, which publishes the
pointer and index tables of that layout. Which triangle of a symmetric matrix is stored is a
pattern choice (`sc.triu(H)`), since QDLDL, OSQP and PIQP read the upper triangle and others the
lower. Lowering writes output values straight into the requested order, with the permutation
folded into the stores' position tables, so a CSR Hessian for one solver and a CSC one for another
cost the same. The derivative wrappers take the same choice, as in
`sc.sparse_hessian(f, of, wrt, layout="csr", triangle="upper")`. This replaces [#48]'s single CSC
order and the header's two permutation tables.

Common to both designs, and a gate of its own: no analysis, constant or loop may do work
proportional to a matrix's full size. On main, `zeros_like` builds a full array, `ad/sparsity.py`
allocates CSR row pointers of length `expr.size`, transpose and slice rules build `arange(size)`,
and `_matmul_mask` enumerates the dense contraction domain. A test that builds, analyses,
differentiates and lowers a 10⁵×10⁵ matrix with 10⁶ entries under fixed time and memory limits
comes first. The 89 places on main that assume `size` equals buffer length need a storage
descriptor under either design; a missed one is a wrong address, not lost speed.

## Sparse lowering, from TACO, specialized to static patterns

What TACO contributes, and what survives when every pattern is known at build time:

- **Algebra separate from storage separate from schedule.** Kept. The expression dialect stays
  algebra; storage and schedule are decided in lowering.
- **Per-axis level formats** (dense, compressed, singleton) instead of whole-matrix formats. Kept as
  lowering vocabulary: an execution plan for a mapped stage Jacobian is a dense stage level, dense
  rows and compressed columns, with one table shared by all stages. Not in the type, since the
  compiler can choose the format itself.
- **Merge lattices** for sums and products of different patterns. Resolved at build time: the
  union and intersection regions are computed in NumPy, and each region gets its own loop body.
  There are no run-time index comparisons.
- **Workspaces** (a dense accumulator column in a sparse-sparse product). Kept, with sizes and the
  entries to clear known statically.
- **Output pattern assembly.** Done in Python when the graph is built; nothing is assembled at run
  time.
- **Position-space versus coordinate-space scheduling.** Needed internally, not as a public
  scheduling language.

From Sympiler: symbolic inspection in Python (orderings, elimination tree, fill, supernodes)
produces immutable tables, and numerical code is generated against them. This is how the library
LDL^T should be built, and devrush's `linalg/symbolic.py` already is.

**Where it sits.** After AD and simplification, lowering builds a private iteration plan per
kernel: iteration domains, access maps (affine where `LowerCtx.index_at` recovers them, position
tables otherwise), the scalar body, reduction order, workspace scope and output writes. The dense
loop-nest emitter of [#27] is the all-dense case of these plans, so one emitter serves both. The
first sparse kernels are:

- one generic **compact map**: a loop over the output's stored entries that reads each operand
  through a static position table, with a zero slot for absent entries. It covers movement ops,
  `CONCAT`, `GATHER`, `SCATTER`, elementwise ops and `SELECT`, and it is the whole of KKT assembly;
- **`MATMUL` with a sparse operand**: CSR row loops with loaded bounds for a sparse matrix times a
  dense one, and a column loop over a dense accumulator for sparse times sparse. Devrush's
  enumeration of every product pair goes away;
- **`REDUCE` over stored entries**, including absent entries for `max` and `min`.

Each region then takes one of four forms: straight-line code when small, a loop over constant
tables when irregular, an affine loop for regular runs and repeated stage blocks, or a dense block
with dense kernels. Detecting dense blocks, bands and supernodes inside a pattern becomes more
valuable than any run-time machinery, but it comes after the first milestone.

Fusion must know about sparse traversals: whether a consumer shares a producer's traversal, inlines
it, keeps a local workspace or materializes it. The current [#69] decides from consumer counts alone,
and it also needs the consumer-agreement case tinygrad's `run_rangeify` has, which devrush's many
short IPM loops now justify. Sparse kernels still come before fusion (decision 5 stands), but the
iteration plan is designed once, for both, before [#27] fixes its interface.

## What is new and what is not

Combining dense and sparse compilation is not new: TACO and MLIR's sparse dialect already compile
mixed dense and sparse expressions, SparseTIR combines sparse formats with dense scheduling,
Sympiler moves factorization analysis to compile time, SpComp generates structure-specific code
from fixed patterns, and CasADi combines structural sparsity, differentiation, maps and generated
code. The defensible distinction is narrower: one compiler keeps derivative sparsity and the
repeated stage structure of optimal control intact from the oracles through KKT assembly to the
linear solves, and generates dependency-free C with a fixed memory footprint. That has to be shown
by measurements that isolate what keeping those boundaries buys.

## Effect on the roadmap

| Decision | Change |
|---|---|
| 1. One AD engine | Keep. Add the sampled `MATMUL` adjoint on a sparse operand. |
| 2. N-d lowering and loop compiler are one project | Keep, widened: dense nests are the all-dense case of the iteration plans. |
| 3. Keep `MATMUL`, `einsum` as sugar | Keep. |
| 4. Sparsity in `TensorType` | Change: stored pattern in the type, plus support on every node; physical order in lowering; two boundary ops instead of four. |
| 5. Sparse before the loop compiler | Keep, with the iteration plan designed first. |
| 6. One loop op | Keep, widened to maps: `VMAP` folds into `LOOP` as a byte-identical rename after `LOOP` lands; the trip count becomes user-facing; loops may state an in-place contract. |
| 7. Invocation key | Keep; add residuals that a forward rule returns and its backward rule consumes. |
| 8. No global options | Keep. |
| 9. Two indexing ops | Change: `GATHER` and `SCATTER` with the index as an operand from the start; `TAKE`/`PUT` never exist. |
| 10. Custom behaviour at Function boundaries | Keep, strengthened: no op registry; `custom_derivative` takes the residual form; external code is `sc.extern` over an `sc.CLibrary`. |
| 11. Entry signature by leaf dtype | Keep. |
| 12. Template API open | Settled: the typing playground's surface, extended from shapes to dtypes and patterns; templates land before the sparse boundary. |

Items:

- **Keep** wave 0 as written ([#9], [#10], [#60], [#61], [#16], [#62], [#38], [#26], [#64], [#63]), with [#26] as the first step of the new `SCATTER`.
  **Keep** [#15], [#17], [#18], [#19], [#20], [#21], [#22], [#25], [#30], [#65], [#55], [#56], [#66], [#67], [#28], [#29], [#31], [#32], [#33], [#78], [#43],
  [#48], [#76].
- **Merge** [#35] and the former C-137 task into one early item: the index becomes an operand of `GATHER` and
  `SCATTER`, with `combine` and `mode`.
- **Rewrite** [#39] and [#42] around the hybrid, with `PACK`/`VALUES` only.
  **Rewrite** [#46]: support rules and harness, the compact map, sparse `MATMUL` and `REDUCE`
  kernels, and the sampled adjoint. **Shrink** [#47] to sparse-sparse products; assembly is
  composition.
- **Rewrite** [#51], [#52], [#54] as library `Function`s over `LOOP`, gated on devrush's measurements;
  their ops are deleted. [#54]'s symbolic analysis module stays as specified.
- **Move earlier** [#36] (in-place carries, with the syntactic case), [#73] and [#90] (keeping call
  boundaries and sharing the forward pass).
- **Change** [#83] into `sc.CLibrary` and `sc.extern` with the `EXTERN` op, absorbing `SOLVER_CALL`.
  [#53] is one `COND`. [#30] and [#65] lose their own ops, which become table rows and builders.
- **Move** [#77] (`sc.print`) to right after wave 0, float64 first. [#68] (`einsum`) no longer comes
  before fusion. Drop `COPYSIGN` from [#30].
- **Rewrite** the "Signatures and templates" section from the typing playground, and order its
  items ([#7], [#8], [#11]) before [#39] and [#42]; [#44] (pattern holes) lands with the
  sparse boundary.
- **Add**: support analysis and its soundness harness; the full-size-work gate; iteration plans for
  dense and sparse; a run-time trip count on `LOOP`; unrolling in `LOOP` lowering by the target's
  budget; lanes with a remainder on loaded-bound reductions; private results for derivative
  reuse; demanded-entry analysis; later, dense-block and supernode detection.

## Milestones

1. **Differentiable generated loops.** Wave 0, then `sc.print`, the AD engine through [#22], predicates, select
   and cast, integer leaves, `GATHER`/`SCATTER` with index operands, `LOOP` with forward and reverse
   AD, in-place carries. Gate: an RK4 rollout with its gradient at N = 200, with build time and C
   size constant in N; a Newton iteration inside a while loop, with early exit, zero-trip loops and
   a second derivative. In parallel, the templates ([#7], [#8], [#11]), which must land before
   milestone 2 starts its sparse boundary.
2. **Static sparse functions end to end.** Stored patterns and support, sparse leaves and headers,
   pattern holes in templates ([#44]), the iteration plan with the compact map and sparse
   `MATMUL`, the sampled adjoint, the library
   LDL^T with an implicit solve and factor reuse. Gates: the full-size-work test; KKT assembly and
   matrix-vector products on devrush's Maros–Meszaros subset with no dense `(m, n)` buffer; the
   library LDL^T against devrush's direct kernel. This milestone decides whether factorization ops
   are needed.
3. **Competitive kernels and generated solvers.** Lowering-time fusion, lanes, float32, dense
   scheduling (register blocks, batching a mapped matrix-vector product with one constant matrix
   into a matrix product), setup and solve separation. Gates: [#72] TinyMPC and [#70] the
   generated sparse IPM, against devrush's reported timings.

## Decisions to take

Settled on 2026-10-01:

- **A. Where the stored pattern lives: the hybrid.** A stored pattern in the type, set by fixed
  rules when the graph is built, plus support on every node. `PACK` and `VALUES` are the only
  sparse ops.
- **B. A closed language, gated.** No op registry. Factorizations and solves are library
  `Function`s, judged against devrush's direct kernels; if the library LDL^T cannot match them, the
  question is reopened.
- **C. A reference schedule, not bits fixed by definition.** Each kernel has one reproducible
  reference schedule. Target schedules (lanes, blocking) may round differently and are checked by
  numerical and solver-robustness tests. This replaces the roadmap's rule that a factorization's
  summation order is part of its definition (the "Contract for every solve" in its "Linear solves"
  section).
- **D. Templates first, with the playground's surface.** The design in `typing_playground/` is the
  template API, and its items come before the sparse boundary, so sparse leaves key on patterns
  from their first commit.
- **E. Smaller cuts.** `VMAP` folds into `LOOP`; `COPYSIGN` is dropped; `sc.print` moves to right
  after wave 0.

Settled on 2026-10-01, after the first round:

- **F. External code** as `sc.CLibrary` plus `sc.extern`, always concrete, with the `EXTERN` op.
- **G. `custom_derivative` in the residual form**, so a solve's derivatives reuse its factorization.
- **H. A layout on sparse leaves** (CSC, CSR or a given coordinate order), outside the pattern.

**Extending the playground to dtypes and patterns.** The extension is as direct as hoped where the
key is concerned: `LeafKey(shape)` in `trees.py` becomes the bound `TensorType`, a leaf declaration
gains optional dtype and pattern fields that may each be holes, and `Function.instances` keys on
the bound types. Nothing in `lift`, `group` or the parameter lists changes. Five points remain:

1. A sparse leaf's numerical value is a SciPy sparse array. The maintainer's answer: the numerical
   leaf type becomes `np.ndarray | scipy.sparse.sparray`, so sparseness need not be declared
   statically, a pattern hole binds from either kind of argument, and a numerical sparse output
   returns the union.
2. A numerical call with a SciPy argument must canonicalize it and digest its pattern to find the
   instance, which is O(nnz) per call. A concrete sparse leaf should also accept its values vector
   directly, as the fast path.
3. The seeds and multipliers of derived functions copy only the shape (`_decl` in `function.py`);
   they must copy the whole leaf type, so a forward seed of a float32 input is float32 and a
   multiplier of a sparse output knows its pattern.
4. `_mangle` still joins tokens with `__`, which the README's decisions forbid, and does not encode
   the nesting. The dtype tokens (`f32`, `i64`) and the pattern token (`p` plus a digest) go there.
5. Typing `vmap` by its callee's trees is still the README's open item, and batched sparse leaves
   (one local pattern for `N` stages) meet it there.

## Risks

1. The library LDL^T misses devrush's direct kernel. Expressing it is not the problem; the risk is in
   lowering: buffer reuse failing on nested loops (made visible by the in-place contract), and loop
   quality (register accumulators, column chunks, lanes).
2. Lowering grows into a second compiler. Keep one generic compact map, and dedicated kernels only
   for `MATMUL` and `REDUCE`.
3. Hidden work proportional to the full size of large matrices, in analyses and constant folding.
4. Private results shared between a primal and its derivatives give wrong higher derivatives if
   their dependence on the inputs is dropped.
5. Table and code growth: exact patterns do not justify enumerating every multiply-add; each kernel
   form needs a size budget.
6. Dense kernel quality: all-dense plans are a representation, not BLASFEO-class performance.

## How this note was made

Four read-only investigations ran in parallel from one brief, each leading one question and
commenting on the others: the opset (against main, devrush's `ir/expr.py`, `linalg/ops/`,
tinygrad's `uop/` and JAX's `lax`); the sparse representation (devrush's `linalg/sparse.py`,
`symbolic.py`, `sparse_factor.py`, main's `ad/sparsity.py` and `ad/sparse.py`); TACO and its
successors (the TACO, formats, workspaces, scheduling and conversion papers, Sympiler, SparseTIR,
MLIR's `SparseTensorAttrDefs.td` and `Sparsification.cpp`); and the whole system, as an adversary
of the roadmap. The sparse investigation proposed the derived-only design. An adversarial review of
it against main's code produced the four problems listed under [Sparsity](#sparsity), and the
hybrid keeps the derived design's support analysis, boundary rules and zero sparse arithmetic ops.
Devrush was read at its tip of 2026-09-30, after its restructure and the codegen and speed-gap work.

[#20]: https://github.com/PREDICT-EPFL/scaly/issues/20
[#31]: https://github.com/PREDICT-EPFL/scaly/issues/31
[#30]: https://github.com/PREDICT-EPFL/scaly/issues/30
[#77]: https://github.com/PREDICT-EPFL/scaly/issues/77
[#69]: https://github.com/PREDICT-EPFL/scaly/issues/69
[#71]: https://github.com/PREDICT-EPFL/scaly/issues/71
[#36]: https://github.com/PREDICT-EPFL/scaly/issues/36
[#22]: https://github.com/PREDICT-EPFL/scaly/issues/22
[#28]: https://github.com/PREDICT-EPFL/scaly/issues/28
[#64]: https://github.com/PREDICT-EPFL/scaly/issues/64
[#56]: https://github.com/PREDICT-EPFL/scaly/issues/56
[#48]: https://github.com/PREDICT-EPFL/scaly/issues/48
[#27]: https://github.com/PREDICT-EPFL/scaly/issues/27
[#9]: https://github.com/PREDICT-EPFL/scaly/issues/9
[#10]: https://github.com/PREDICT-EPFL/scaly/issues/10
[#60]: https://github.com/PREDICT-EPFL/scaly/issues/60
[#61]: https://github.com/PREDICT-EPFL/scaly/issues/61
[#16]: https://github.com/PREDICT-EPFL/scaly/issues/16
[#62]: https://github.com/PREDICT-EPFL/scaly/issues/62
[#38]: https://github.com/PREDICT-EPFL/scaly/issues/38
[#26]: https://github.com/PREDICT-EPFL/scaly/issues/26
[#63]: https://github.com/PREDICT-EPFL/scaly/issues/63
[#15]: https://github.com/PREDICT-EPFL/scaly/issues/15
[#17]: https://github.com/PREDICT-EPFL/scaly/issues/17
[#18]: https://github.com/PREDICT-EPFL/scaly/issues/18
[#19]: https://github.com/PREDICT-EPFL/scaly/issues/19
[#21]: https://github.com/PREDICT-EPFL/scaly/issues/21
[#25]: https://github.com/PREDICT-EPFL/scaly/issues/25
[#65]: https://github.com/PREDICT-EPFL/scaly/issues/65
[#55]: https://github.com/PREDICT-EPFL/scaly/issues/55
[#66]: https://github.com/PREDICT-EPFL/scaly/issues/66
[#67]: https://github.com/PREDICT-EPFL/scaly/issues/67
[#29]: https://github.com/PREDICT-EPFL/scaly/issues/29
[#32]: https://github.com/PREDICT-EPFL/scaly/issues/32
[#33]: https://github.com/PREDICT-EPFL/scaly/issues/33
[#78]: https://github.com/PREDICT-EPFL/scaly/issues/78
[#43]: https://github.com/PREDICT-EPFL/scaly/issues/43
[#76]: https://github.com/PREDICT-EPFL/scaly/issues/76
[#35]: https://github.com/PREDICT-EPFL/scaly/issues/35
[#39]: https://github.com/PREDICT-EPFL/scaly/issues/39
[#42]: https://github.com/PREDICT-EPFL/scaly/issues/42
[#46]: https://github.com/PREDICT-EPFL/scaly/issues/46
[#47]: https://github.com/PREDICT-EPFL/scaly/issues/47
[#51]: https://github.com/PREDICT-EPFL/scaly/issues/51
[#52]: https://github.com/PREDICT-EPFL/scaly/issues/52
[#54]: https://github.com/PREDICT-EPFL/scaly/issues/54
[#73]: https://github.com/PREDICT-EPFL/scaly/issues/73
[#90]: https://github.com/PREDICT-EPFL/scaly/issues/90
[#83]: https://github.com/PREDICT-EPFL/scaly/issues/83
[#53]: https://github.com/PREDICT-EPFL/scaly/issues/53
[#68]: https://github.com/PREDICT-EPFL/scaly/issues/68
[#7]: https://github.com/PREDICT-EPFL/scaly/issues/7
[#8]: https://github.com/PREDICT-EPFL/scaly/issues/8
[#11]: https://github.com/PREDICT-EPFL/scaly/issues/11
[#44]: https://github.com/PREDICT-EPFL/scaly/issues/44
[#72]: https://github.com/PREDICT-EPFL/scaly/issues/72
[#70]: https://github.com/PREDICT-EPFL/scaly/issues/70
