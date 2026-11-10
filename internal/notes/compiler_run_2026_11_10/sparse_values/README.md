# Section 1: Sparse values into generated C

Packet of section 1 of the [compiler run](../plan.md), parent #37. Built by stack #278: #274
(#39), #277 (#40 and #276), #286 (#41), #287 (#42), #289 (#43). CSC and CSR are compressed sparse
column and row storage, COO a coordinate list, nnz the number of stored entries, ABI the binary
interface of the generated C, IR the intermediate representation, AD automatic differentiation.

## Where this section sits

- A user can pass a sparse matrix (SciPy, or values in a declared CSC, CSR or coordinate layout) through a `Function`, get SciPy arrays back, and export C, C++ or CasADi code whose buffers hold only stored entries and whose headers publish the layout tables.
- Every expression node carries its stored pattern in its type and knows which entries may be nonzero; simplification, derivative sparsity and scalar lowering use this.
- Section 2 (Sparse kernels, #292) replaces the temporary dense paths below with planned sparse kernels, the sampled adjoint and sparse-sparse products.
- Section 3 (Sparse derivatives, #293) makes call support argument-aware and turns derivative outputs into sparse leaves with a declared layout.
- None of this is in the guide or on the API page until #46 and #47 land.

## Needs the maintainer's decision

1. Roadmap departure, applied: decision 4 now says buffers and the ABI never change when an analysis sharpens, but kernel code may (zero skipping, cotangent compaction); `[0, 1] / [0, 0]` gives `[0, inf]` expanded and `[nan, inf]` as a loop. Recommend keeping.
2. Roadmap departure, applied: downstream of a custom derivative, derivative sparsity is not bounded by a node's own support, because the custom derivative may be nonzero where the value is zero. Recommend keeping.
3. Roadmap departure, applied: stored values are in canonical row-major order inside the IR, with permutation copies at boundaries. Recommend keeping until #46's plans choose the order.
4. Public ABI: each sparse leaf publishes only its layout's tables (`<fn>_<leaf>_csc_col_ptr`, `_csc_row_ind`, …); C omits empty tables; an empty sparse leaf accepts a null pointer. Recommend keeping.
5. Public interface: CasADi export accepts only unbatched CSC sparse leaves. Recommend keeping; conversions can be added later.
6. Public interface, unpublished: `sc.arg(name, TensorType(shape, pattern=P), layout=...)`, numerical values in the declared layout, batched results as N-dimensional COO, derived functions inheriting layouts. Recommend keeping.
7. Public convention: `sc.window` buffers of sparse mapped inputs use canonical order, direct arguments the declared order. Recommend the declared order everywhere (#300).
8. Proposal, not applied: `design_text.diff` names the stored pattern and the support analysis in `docs/how_it_works/architecture.md`. Recommend applying it with #46 and #47.

## Expected gaps

- Derivatives of a product with a sparse operand go through a dense copy, n² buffers in the demo: #46.
- Mixed products use build-time contribution tables rather than iteration plans: #46.
- Sparse-sparse products are refused: #47.
- Derivative outputs keep arbitrary value order and the legacy permutation tables: #48.
- No pattern holes: #44.
- Call support comes from the callee alone: #221.
- No support rules for select, axis reductions or loops, whose ops do not exist yet: #30, #66, #31.
- Static types of sparse numerical arguments and results: #288.
- Zero-size dense leaves fail through CasADi: #290. Dense matrices in the CasADi layer: #103.

## Unowned, filed by this section

- #300 window buffers use canonical order. #301 constant digests have no byte comparison.
- #302 listings and type errors hide the stored pattern. #303 sparse arithmetic skips simplification.
- #304 `A @ I` does not fold. #305 canonical patterns cannot be copied or pickled.

# Reference

Line numbers are on the #43 head (`a3f39a32`; its last commit changed tests only, so `src/` is as
at `248b2159`), paths under `src/scaly/`. Files: `design_text.diff`,
`banded_quadratic.py` and `banded_quadratic_output.txt`.

## Decisions

Choices the roadmap's Sparse tensors section did not settle, or that depart from it, hardest to
undo first. "Rejected" is the alternative the layer considered and why it lost.

### Public interface and documented behaviour

1. **Header format for sparse leaves** (#43; `codegen/abi.py:56-62`, `codegen/aot.py:251-264`,
   `codegen/cpp.py:165-171`). Each leaf publishes `<fn>_<leaf>_NROW`, `_NCOL`, `_NNZ`, `_NDIM`,
   `_STORAGE_SIZE`, `_shape`, and only its own layout's tables: `csc_col_ptr` then `csc_row_ind`,
   `csr_row_ptr` then `csr_col_ind`, or `rows` then `cols` for a caller's coordinate order.
   Batched leaves publish the local pattern once, with the full mathematical shape. C omits empty
   tables (consumers must guard), C++ publishes `std::array<int, 0>`, and an empty sparse leaf
   accepts a null data pointer. Rejected: publishing every format plus permutations (redundant)
   and one-element placeholder arrays (fabricated entries). Once released, this is ABI.
2. **Downstream of a custom derivative, derivative relations are not bounded by a node's own
   value support** (#40, kept by #41 and
   #42; `ad/sparsity.py`, `_callee_mask`). A registered custom derivative replaces the body's
   derivative and may be nonzero where the value is identically zero, so its declared (or dense)
   rows survive through everything downstream. Rejected: the blanket filter by value support
   (gave dense and sparse Jacobians that disagree, and corrupted compressed values), a
   `CALL`-only exemption (loses downstream rows), rejecting such rules (changes the documented
   custom-derivative contract). Fable 5.2, asked independently, reached the same conclusion. This
   reads against the roadmap's "relations only over each node's nonzero entries"; the roadmap now
   says so, with a "Replaced" line. Reversing it means revisiting the custom-derivative contract.
3. **Generated C changes when the support analysis sharpens** (#40/#276, #41). Decision 4 says
   generated C does not change when an analysis gets sharper. Two places now do: scalar expansion
   writes zero for entries the support proves zero (`passes/program/scalarize.py:89-94`), and
   reverse differentiation compacts slice and broadcast cotangents to their known support
   (`ad/reverse.py`, `ad/rules.py::_unbroadcast`), which can change small programs outside the
   snapshot corpus. Buffer sizes and the ABI do not change. Consequence, documented on the lowering
   page: `sc.const([0, 1]) / sc.const([0, 0])` gives `[0, inf]` when expanded and `[nan, inf]` as
   a loop, and under `auto` that depends on placement and size limits. Rejected: evaluating known
   zeros anyway, or skipping them in loop code too. The roadmap's decision 4 now says "buffer sizes
   and the ABI".
4. **Mixed sparse and dense operands** (#39, extended by #42; `ir/expr.py:218-248`). A dense
   operand counts as every coordinate, so storage never depends on values: `x + 0` for sparse `x`
   has a full explicit pattern, `x * 0` keeps `x`'s pattern, a singleton matrix axis broadcasts by
   expanding coordinates. #39 refused results with fewer than two axes; #42 admits a sparse matrix
   times a dense vector with a dense result. Rejected: refusing mixed operations, or compacting by
   value. This is IR type semantics that #46's kernels build on.
5. **Python spelling of the boundary** (#42). `sc.arg(name, TensorType(shape, pattern=P),
   layout="csc" | "csr" | SparsityPattern)`; `pack(values, P, shape=)` infers batch axes;
   `Expr.values()`, `to_dense()`; `values(dense)` is the identity. A numerical sparse argument is a
   SciPy array or a values vector in the declared layout (a dense matrix is refused). Two-axis
   results come back as SciPy arrays in the declared format, batched results as N-dimensional COO.
   Forward results inherit the output's layout, adjoints and gradients the input's; explicit dense
   `jacobian`/`hessian` stay dense. All unpublished, so cheap until 0.2.0. Static typing only sees
   the sparse numerical union when `layout=` is explicit (#288).
6. **`sc.window` buffers use canonical order, direct arguments use the declared layout** (#42;
   `function/sugar.py:139-207`, `247-260`). Raw window vectors for sparse mapped inputs hold values
   in row-major canonical order, while a direct numerical argument holds them in the leaf's
   declared order. Rejected: per-callee declared order inside windows (a second convention for
   internal mapped windows). I think this inconsistency should be settled before 0.2.0 publishes
   windows over sparse leaves.
7. **CasADi accepts only unbatched CSC leaves** (#43; `codegen/casadi.py:71-79`). CasADi's
   external ABI reads CSC values; CSR, coordinate-order and batched sparse leaves are refused with
   an error naming the leaf and layout. Rejected: workspace conversions (an ABI commitment). I
   answered this for the thread; both reviewers agreed. It can be relaxed later by adding
   conversions.

### IR representation

8. **Sparse constants are `PACK` over a dense `CONST` of the values** (#42;
   `ir/expr.py:423-439`). A sparse-typed `CONST` stays invalid. Rejected: a sparse `CONST`, which
   would need its own folding, hashing and lowering. Changing this touches several passes.
9. **Canonical order is row-major, and values inside the graph use it** (#39, #42;
   `ir/types.py::SparsityPattern.canonicalize`, `passes/lowering/ctx.py:251-259`). Leaves, outputs
   and calls convert with a permutation copy at the boundary; private AD helpers use CSR order.
   The roadmap said the internal order "belongs to lowering"; this fixes it in the IR for now.
   Rejected: per-buffer layout propagation, left to #46's iteration plans. The `CanonicalPattern`
   docstring (`ir/types.py:247`) still says physical order belongs to lowering; the section's
   simplification pass corrects it.
10. **Separate `CanonicalPattern` type, weakly interned under one lock, SHA-256 digest of the
    coordinate tuples** (#39; `ir/types.py:238-267`). `TensorType.pattern` accepts only the
    canonical form; callers canonicalize explicitly and keep the permutation. Weak interning means
    "digest once" holds per live object. The lock came from Sol reproducing two identities for
    one pattern under threads. Rejected: a canonical mode on `SparsityPattern`.
11. **Uniform constants and their folding** (#41; `ir/expr.py:366-370`, `passes/expr.py:121-140`).
    Every bitwise-uniform constant becomes a scalar-backed zero-stride view before interning. A
    uniform `SUM` over more than 1,024 entries folds to value times count, which rounds
    differently from NumPy (2,048 × 0.1 gives 204.8, NumPy 204.80000000000004). A scatter of
    constants folds only up to `max(1024, len(values))` entries. Lowering a large uniform constant
    emits a one-element table and a fill loop. Rejected: dense folding (full-size work).
12. **Constants are keyed by a 32-byte BLAKE2b digest of their compacted bytes, with no byte
    comparison on a digest match** (#41; `ir/expr.py:685-695`). Two distinct constants with equal
    digests would intern as one node. Rejected: full-byte keys (serialize logical arrays). No
    recorded reason for omitting the comparison.

### Internal

13. **Simplification skips all sparse arithmetic** (#39; `passes/expr.py:163-164`, `276`).
    Pattern-losing rewrites were a Sol blocker; the final fix bypasses every sparse arithmetic
    rewrite, so sparse `x - x`, `x * 0`, `x * 1` and `-(-x)` stay as written. Rejected:
    per-rule pattern checks (failed on lowering hints).
14. **`CALL` support is the callee's own support, tiled over maps** (#40; `ir/support.py`).
    Argument specialization is #221. The roadmap text merged the two; it now separates them.
15. **Support representation** (#40, #41; `ir/support.py:22-25`, `64-73`). `frozenset` of flat
    coordinates below 1,024 entries, sorted read-only int64 arrays above, large full supports as
    `None`; the cache encodes uncomputed versus unknown. Support reaches lowering as a
    `BUFFER.attrs["support"]` annotation valid only until scalar expansion. `OpInfo.support_rule`
    is a string tag also read by derivative sparsity.
16. **Derivative relations are Boolean COO matrices with logical shape** (#41;
    `ad/sparsity.py:132-141`); only occupied rows and columns are compressed for composition and
    coloring. Indexed call and map consumers still convert to CSR over the logical domain.
17. **A `PRINT` whose value only feeds a product folded to zero disappears** (#40). Observable,
    accepted by review as consistent with "prints run when their value is computed".
18. **Temporary paths left for #46 and #47** (#42). Stored-entry elementwise maps with one padded
    zero slot, `SUM` over stored entries, mixed products through build-time contribution tables
    (`passes/lowering/contraction.py:142-172`, work up to nnz × dense output width), sparse-sparse
    products refused. **The derivative of a product with a sparse operand projects that operand to
    a dense matrix** (`ad/forward.py:164-168`, `ad/reverse.py:98-101`), so it allocates and works
    over the logical size. The demo below shows the effect.

## Design text

The pull request that lands this packet also edits `internal/notes/core_compiler_roadmap.md`:
decision 4 and the #39 to #43 paragraphs now record decisions 2 to 9, 11, 12, 14 and 18 above, and
the three departures (items 2, 3 and 9) each keep a "Replaced in the compiler run" line with the
old wording and the reason. The rest are internal.

`design_text.diff` is a proposal, not applied: two edits to `docs/how_it_works/architecture.md`
(the static-type paragraph names the stored pattern; the pass table gains the support analysis).
That page is published, so under the 0.2.0 limits it should land with #46 and #47. It names no
API, only the idea of a stored pattern.

## Demo: f(x) = x'Qx with a banded Q

`banded_quadratic.py` declares Q as a 5×5 tridiagonal CSC input (4 on the diagonal, −1 beside
it, 13 stored entries) and x as a dense 5-vector, then builds `f`, `sc.gradient`, `sc.hessian`
and `sc.sparse_hessian`. For each it prints every node's support, the traced and the simplified
expression, the program, the generated C and header, and a numerical call; the last stage lowers
the same functions at n = 100 and 400 and reports the largest work buffer. Run it from a checkout
of the #43 head with `uv run python banded_quadratic.py`. The full output is
`banded_quadratic_output.txt` (1,413 lines).

**It runs end to end.** All four functions lower, render C and a header, compile and match NumPy:
f = 140, gradient `[4, 8, 12, 16, 32]`, Hessian the tridiagonal 8/−2 matrix, sparse Hessian its
13 values.

Type and pattern of Q:

```
TensorType(shape=(5, 5), dtype=float64, diff=True, pattern=CanonicalPattern(shape=(5, 5),
  rows=(0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4), cols=(0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4), digest='e18432b2…'))
SparseLayout(format='csc', order=(0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12))
```

`f`, support of each node, simplified expression, program (attributes trimmed and the program
abridged; the full text is in the output file):

```
INPUT      shape=(5,)    support=unknown (any entry)
INPUT      shape=(5, 5)  pattern nnz=13  support=13 of 25: [0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24]
MATMUL     shape=(5,)    support=5 of 5
MATMUL     shape=()      support=1 of 1

%0 = expr.input {name="x"} : tensor<5xfloat64 diff>
%1 = expr.input {name="Q"} : tensor<5x5xfloat64 diff>
%2 = expr.matmul(%1, %0) : tensor<5xfloat64 diff>
%3 = expr.matmul(%0, %2) : tensor<float64 diff>

prog.proc @f(%Q: memref<13xfloat64>, %x: memref<5xfloat64>, %f_2: memref<1xfloat64>) {
  %k1 = prog.buffer : memref<13xint64, constant> = dense<[0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12]>
  %s0 = prog.buffer : memref<5xfloat64, private>      // zeroed
  %k3 = … rows, %k4 = … cols
  prog.for %p = 0 to 13 {kind=reduce} {
    %r = prog.assign prog.load %k3[%p]
    prog.store %s0[%r] + %Q[%k1[%p]] * %x[%k4[%p]], %s0[%r]
  }
  … f = x · s0
}
```

The generated C is the same loop over 13 stored entries (`arg[0][k1[p]] * arg[1][k4[p]]`). The
header adds:

```
typedef struct { SCALY_ALIGNAS(16) double data[13]; } f_Q_t;
#define f_Q_NROW 5
#define f_Q_NCOL 5
#define f_Q_NNZ 13
#define f_Q_STORAGE_SIZE 13
static const int f_Q_csc_col_ptr[6] = {0, 2, 5, 8, 11, 13};
static const int f_Q_csc_row_ind[13] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
```

Q's input buffer holds 13 values, and the generated code never touches the 12 absent entries.

**Where it stops being sparse: every derivative rebuilds Q densely.** The simplified gradient is

```
%2 = expr.matmul(%0, %1)                       // Q @ x, sparse
%3 = expr.values(%0) : tensor<13xfloat64>
%4 = expr.scatter(%3) {indices=[0, 1, 5, …, 24]} : tensor<25xfloat64>
%5 = expr.reshape(%4) {shape=[5, 5]}            // dense copy of Q
%6 = expr.matmul(%1, %5)                       // x @ Q, dense
%7 = expr.add(%2, %6)
```

The Hessian multiplies that dense Q by a 5×5 identity constant, and the sparse Hessian multiplies
it by a dense 5×3 coloring matrix, then gathers 13 values. Support is right throughout (13 of 25
entries on every Q-shaped node), but the program allocates the logical size anyway. Section 7
of the output:

```
n=100  nnz(Q)=298   f: largest private buffer 100      gradient: 10000     sparse Hessian: 10000
n=400  nnz(Q)=1198  f: largest private buffer 400      gradient: 160000    sparse Hessian: 160000
```

This is decision 18's temporary dense projection, which #46 replaces with sparse tangents and
the sampled adjoint. #41's scale gate passes because it builds its matrix by scattering into a
dense-typed graph, never as a sparse leaf, so it does not cover this path.

Two smaller observations from the listings. The expression text prints Q as
`tensor<5x5xfloat64 diff>`, with no sign of its pattern (deferred in #39). And the dense Hessian
keeps `matmul` by an identity constant and four transposes after simplification.

## Reading list

309 lines on the #43 head, in reading order.

| Range | Lines | What to judge |
| --- | ---: | --- |
| `ir/types.py:238-267` | 30 | Immutable coordinates, identity equality and the locked weak intern table. |
| `ir/types.py:317-321` | 5 | Pattern rank and agreement with the last two axes. |
| `ir/types.py:332-340` | 9 | Shared batch pattern and the stored buffer length. |
| `ir/expr.py:218-221` | 4 | A dense operand expanded to every coordinate. |
| `ir/expr.py:235-257` | 23 | The fixed pattern rules, with #42's vector-result, mixed elementwise and sparse-times-dense `MATMUL` cases. |
| `ir/expr.py:280-282` | 3 | Exact result-pattern check, below #42's `PACK`, `VALUES` and callee exceptions. |
| `passes/expr.py:163-164` | 2 | The blanket bypass of sparse arithmetic simplification. |
| `passes/expr.py:275-276` | 2 | Support-driven zero folding, dense results only. |
| `ir/support.py:64` | 1 | Unknown support and the two carriers. |
| `ir/support.py:76-92` | 17 | Iterative cached traversal through operands and callees. |
| `ir/support.py:93-108` | 16 | The stored-pattern bound, and uncomputed versus unknown in the cache. |
| `ir/support.py:135-137` | 3 | Union and intersection with an unknown side. |
| `ad/sparsity.py:207-211` | 5 | Pruning by coefficient support for `MUL` and `DIV`. |
| `passes/program/scalarize.py:89-94` | 6 | Writing zero outside known support without evaluating the arithmetic. |
| `ir/expr.py:366-370` | 5 | Bitwise uniform compression before freezing and hashing. |
| `ir/expr.py:685-695` | 11 | Constant keys from compacted data, no logical serialization, no collision check. |
| `ir/support.py:22-25` | 4 | The sorted int64 carrier for large supports. |
| `ir/support.py:156-160` | 5 | Constant support without enumerating large uniform tensors. |
| `passes/expr.py:121-123` | 3 | The scatter-folding threshold. |
| `ad/sparsity.py:132-141` | 10 | Composition over occupied coordinates, no logical-length row pointers. |
| `ir/layout.py:22-30` | 9 | Layout as a separate record from the canonical pattern. |
| `function/tree.py:160-166` | 7 | Layout bound to a declared sparse leaf. |
| `function/model.py:118-120` | 3 | Layout in the instance key. |
| `ir/layout.py:96-107` | 12 | Copying and canonicalizing SciPy input; exact pattern match. |
| `ir/layout.py:109-116` | 8 | Output arrays sharing cached read-only indices. |
| `ir/expr.py:1111-1123` | 13 | `PACK`'s type, `VALUES`'s storage shape, `VALUES(PACK(v))` = `v`. |
| `passes/lowering/ctx.py:251-259` | 9 | Storage-sized inputs and the declared-to-canonical copy. |
| `passes/lowering/ctx.py:352-369` | 18 | Logical-to-stored slot map with a padded zero slot. |
| `passes/lowering/contraction.py:154-172` | 19 | Mixed-product tables and accumulation, pending #46. |
| `codegen/abi.py:56-62` | 7 | One layout's tables per leaf, no permutations. |
| `codegen/aot.py:251-264` | 14 | Input and output metadata, shared batch pattern, no empty C arrays. |
| `codegen/aot.py:231-236` | 6 | Legacy derivative permutation tables kept until #48. |
| `codegen/cpp.py:118-119` | 2 | C++ buffer aliases sized by storage. |
| `codegen/cpp.py:165-171` | 7 | C++ metadata per declared layout. |
| `codegen/casadi.py:71-75` | 5 | The CSC-only, unbatched CasADi restriction. |
| `codegen/casadi.py:95-96` | 2 | Layout tables passed to CasADi as compressed columns. |
| `codegen/c.py:118-121` | 4 | Null data pointers accepted for empty sparse leaves. |

The two places I would read beyond this list: `ad/forward.py:164-168` and `ad/reverse.py:98-101`,
the dense projection behind the demo's finding.

## Size

Lines added and removed, excluding blank and comment-only lines; docstrings count.

| Layer | `src/` added | `src/` removed | `tests/` added | Largest `src/` files |
| --- | ---: | ---: | ---: | --- |
| #39 | 176 | 29 | 307 | `ir/types.py` +78, `ir/expr.py` +77 |
| #40 | 201 | 41 | 351 | `ir/support.py` +114 (new) |
| #41 | 442 | 165 | 395 | `ir/support.py` +188, `ad/sparsity.py` +162 |
| #42 | 877 | 169 | 800 | `ir/expr.py` +127, `ir/layout.py` +125 (new), `passes/lowering/ctx.py` +75 |
| #43 | 118 | 31 | 319 | `codegen/cpp.py` +43, `codegen/aot.py` +36 |
| Sum of layers | 1,814 | 435 | 2,172 | net +1,379 in `src/` |
| Main to #43 head | 1,728 | 349 | 2,127 | net +1,379; later layers rewrote earlier lines |

After the section's simplification pass: not yet run (Phase 3 of the plan); this table gains a
second column then.

What could go, each checked for callers:

- **Now:** `CanonicalPattern.to_mask` (`ir/types.py:273-275`), no caller. The `b_slot is not None`
  test in `passes/lowering/contraction.py:161-165`, always true once sparse-sparse is refused at
  140-141.
- **Now, as a small refactor:** one support carrier instead of `frozenset` plus int64 arrays
  (`ir/support.py:64-73`, `141-142`). Pattern-preserving sparse simplification instead of the
  blanket bypass (`passes/expr.py:163-164`, `276`; `passes/arith.py:127-137`): removing the bypass
  alone makes `x - x` and `x * 0` lose their pattern, but `PACK` now exists to rebuild a zero.
- **After #46:** the dense projections in `ad/forward.py:164-168` and `ad/reverse.py:98-101`, the
  dense round-trip at `ad/reverse.py:55-56` once every rule returns the exact pattern, and the
  mixed-product tables in `passes/lowering/contraction.py:142-172`.
- **After #47:** the sparse-sparse refusal at `passes/lowering/contraction.py:140-141`.
- **After #48:** the legacy derivative tables and value permutations (`codegen/aot.py:219-244`,
  `269-272`; `codegen/cpp.py:57-74`; `codegen/casadi.py:101-151`) and the derivative half of
  `SparsityPattern` (it stays for coordinate-order layouts).
- Already gone: #39's `ConcreteFunction` guard (removed by #42) and #42's header refusal (removed
  by #43).

## What I am least sure of

- **Whether the stack meets #41's goal for derivatives.** It meets the gate as written, but the
  gate never differentiates through a sparse leaf. The demo shows the gradient and the sparse
  Hessian of x'Qx allocate n² entries. #46 owns the fix, and a comment there asks its gate to
  include a sparse-leaf derivative case.
- **Decision 4's wording versus items 2 and 3.** The roadmap now narrows decision 4 rather than the
  code changing. If you meant generated code to be fully independent of the support analysis, the
  zero skipping and cotangent compaction would have to move behind an explicit option.
- **The `sc.window` order (item 6, #300).** One convention for raw values would be simpler, and this
  one will become public with mapped sparse leaves.
- **Constant keys without a collision check (item 12, #301).** A collision is very unlikely at 32
  bytes, but it would silently merge two constants. A byte comparison on a digest match costs
  little for small constants.
- **The blanket simplification bypass (item 13, #303).** It is correct but leaves `x - x` unsimplified
  for sparse `x`.

Where the reviewers disagreed:

- **#39:** Sol treated pattern-losing simplification as a blocker; Opus treated it as follow-up
  work, because the `ConcreteFunction` guard kept such graphs from running. It was fixed in #39
  (item 13), and both confirmed the fix.
- **#40:** Sol approved the scalar zero skipping; Opus withheld approval until the lowering page
  explained the `[0, inf]` versus `[nan, inf]` difference. Only documentation changed.
- **#42:** Opus noted that a fixed `Function` declaration cannot vary its layout, so the layout in
  the instance key never distinguishes anything today. The thread kept it because #42's task
  requires it.
- **#41 and #43:** the reviewers never disagreed. For #43, both endorsed the CasADi restriction.
