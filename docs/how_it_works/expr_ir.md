# The expression dialect

The expression dialect is what you build when you write alloy. It records *what* to compute and
nothing about how: a graph of values and the operations between them, with no loops, no buffers
and no memory. That is what makes it differentiable and rewritable. Choosing an implementation
happens later, in [the program dialect](program_ir.md).

## `Expr`

An `Expr` is one node in a directed acyclic graph. It is frozen, it carries a type, and it knows
its arguments:

```python
import alloy as al

x = al.sym("x", 3)          # an INPUT node, shape (3,)
y = x.sin() + x * x         # ADD(SIN(x), MUL(x, x))
z = y.sum()                 # SUM(...)  -> shape ()
```

Every node has:

| Field | Meaning |
| --- | --- |
| `op` | an `ExprOp` — the operation this node performs |
| `args` | the operand nodes, in order |
| `type` | a `TensorType`: shape, dtype, sparsity, and the differentiability flag |
| `attrs` | per-op data that is not an operand — the callee of a `CALL`, the index table of a `GATHER` |
| `name` | set on inputs, otherwise `None` |
| `id` | the node's Python object identity. Because nodes are interned, structural equality *is* object identity, so this doubles as a structural key. Printing does not use it — the stable `%0`, `%1` names come from a topological walk. |

Nodes are **interned**: building the same operation on the same arguments with the same attributes
returns the object you already have. Two structurally identical subgraphs are therefore one
subgraph, shared by both consumers, and common subexpressions collapse as you build rather than in
a later pass. Interning is by value in a weak-reference table, so nodes nothing refers to are
collected.

This is why an IR class must never be importable under two module paths: two copies of the class
mean two intern tables, and identity silently stops meaning equality. See
[the architecture](architecture.md#the-rules-that-keep-it-this-way).

## Types

`TensorType` is the type of every expression.

```python
al.TensorType(shape=(3, 4), dtype=al.dtypes.float64, sparsity=None, diff=True)
```

**Shape** is a tuple of non-negative integers; `()` is a scalar. Shapes are static — there are no
symbolic dimensions.

**`DType`** is a small interned descriptor with a name, a bit width and a C spelling. The canonical
instances live in `al.dtypes`:

These are the types *inside* the graph. They are not the ABI: a generated function always exchanges
`double` buffers with its caller, and the Python call path converts to and from `float64` at the
boundary. See [the C ABI](c_abi.md#calling-convention).

| Instance | C type | Bytes |
| --- | --- | --- |
| `al.dtypes.bool_` | `uint8_t` | 1 |
| `al.dtypes.int32` | `int32_t` | 4 |
| `al.dtypes.int64` | `int64_t` | 8 |
| `al.dtypes.float32` | `float` | 4 |
| `al.dtypes.float64` | `double` | 8 (the default) |

Mixed-dtype arithmetic is refused rather than promoted. There is no implicit widening: an
expression combining `float32` and `float64` raises at construction. The rule is conservative on
purpose — it keeps today's `float64` workloads exactly as they are, and it means that when an
explicit `Expr.cast` op lands, every mixed-precision decision will be visible at the call site
where it was made.

**`diff`** marks whether a value depends differentiably on symbolic inputs. Constants are not
differentiable; structural operations pass the flag through; arithmetic propagates it from its
operands; and the non-smooth operations (`floor`, `ceil`, `minimum`, `maximum`, `solver_call`)
clear it. AD reads this flag to decide where a derivative is zero by construction.

**`SparsityType`** attaches a structural pattern to a rank-2 value — see
[Sparsity](../guide/sparsity.md). When present, its shape must match the tensor shape exactly.

**The lowering hint** is a per-node preference for how the value should eventually be computed:
`auto` (let the lowerer choose), `scalar` (SX-like unrolled instructions), `block` (MX-like loops),
or `opaque` (keep this a named boundary and do not look inside).

```python
sparse_model = (x.sin() + x * x).scalar()
dense_layer = (A @ x).block()
```

Today these are metadata: they are preserved, printed and inspectable, but the lowerer does not
yet partition on them. The one that already has teeth is `opaque`, which is how a solver stays a
call instead of being expanded.

## Operations

Thirty-eight operations, grouped by what they do. `arity` is the operand count; a dash means
variadic. `diff` is whether AD can pass through the op at all.

### Arithmetic and elementwise

| Op | Arity | Diff | Notes |
| --- | --- | --- | --- |
| `neg` | 1 | yes | |
| `add` `sub` `mul` `div` | 2 | yes | NumPy broadcasting |
| `pow` | 2 | yes | |
| `sin` `cos` `tan` | 1 | yes | |
| `asin` `acos` `atan` | 1 | yes | no multi-seed forward rule yet |
| `atan2` | 2 | yes | no multi-seed forward rule yet |
| `sinh` `cosh` `tanh` | 1 | yes | |
| `exp` `log` `sqrt` | 1 | yes | |
| `abs` | 1 | yes | no multi-seed forward rule yet |
| `floor` `ceil` | 1 | **no** | result is marked non-differentiable |
| `minimum` `maximum` | 2 | **no** | result is marked non-differentiable |

### Structural

| Op | Arity | Notes |
| --- | --- | --- |
| `input` | 0 | a named symbol |
| `const` | 0 | a materialized array; never differentiable |
| `sum` | 1 | full reduction to a scalar; there is no axis argument |
| `reshape` | 1 | size-preserving |
| `transpose` | 1 | axes must be a permutation |
| `slice` | 1 | integer, multi-dimensional and strided indexing |
| `gather` `scatter` | 1 | flat index tables |
| `stack` `concat` | – | along any axis |
| `vec` | 1 | flatten to rank 1 |
| `matmul` | 2 | rank ≤ 2 |

### Boundaries

| Op | Arity | Diff | Notes |
| --- | --- | --- | --- |
| `call` | – | yes | a named `Function` applied to arguments |
| `map` | – | yes | one callee applied across slices of its arguments |
| `solver_call` | – | **no** | an opaque solve; see [Solvers](solvers.md) |

`dot`, `sumsqr` and `norm_2` are not operations — they are builders that expand into the ops above.

Deliberately absent for now: `expm1` and `log1p` (useful, not load-bearing); splines and
interpolants, which need their own design for knots, extrapolation, derivative behaviour and table
codegen before they can be an op; and matrix decompositions and control flow.

## Functions

A `Function` is a named graph boundary: named inputs, named outputs, and optional sparsity metadata
per output.

```python
f = al.Function("f", [x], [y], ["x"], ["y"])
```

It is the unit of three different things at once — composition (`f.call(...)` puts a first-class
`call` node in a bigger graph), differentiation (`f.factory(...)` derives a new `Function`), and
compilation (calling it produces C).

`Function.call` normalizes constant-like arguments to `Expr` and checks every argument shape
against the corresponding formal. Because a call is a real node rather than an inlining, the
callee's structure survives into the generated C as a real C function, which is what keeps
generated code small when the same block appears a hundred times.

## Verification

Construction checks the cheap invariants inline so the common path stays fast. Everything else is
an explicit pass:

```python
al.verify_expr(y)                     # silent on success
al.verify_expr(y, spec=al.spec_expr)  # naming the spec explicitly
```

The verifier walks the graph in topological order and raises `VerifyError` at the *first* invalid
node, naming the node, its op and the rule it failed. Two specs are exported:

- `spec_expr_shared` — what every node must satisfy: non-negative shape, a real `DType`, arity
  matching `OP_INFO`, sparsity shape agreeing with tensor shape.
- `spec_expr` — the above plus per-op rules: `reshape` preserves size, `transpose` axes are a
  permutation, `matmul` contracting dimensions agree, `call` argument shapes match the callee,
  `map` outer tensors are rank 1 with a consistent slice size, `const` value shape and dtype match
  the declared type.

Run it after a non-trivial rewrite or an AD transform, and write negative tests against it. It is
the same `Rule` and `Spec` machinery the [program dialect](program_ir.md) verifies with.

## Structural equality and rewrites

`Expr.id` is construction identity. For structure, use `structural_key()`, `structural_hash()` and
`structurally_equal()` — they compare the shape of the graph without disturbing node identity,
which is what lets a pass recognize an equivalent subgraph without rewriting anything.

The rewrite layer is small and deliberately so:

```python
y_cse = al.cse(y)
y_clean = al.simplify(y_cse)
```

`simplify` covers constant folding and algebraic identities — `x + 0`, `x * 1`, `x * 0`, identity
reshape and transpose, slice-of-slice, slice-of-stack. `cse` merges structurally equal subgraphs.
Both go through an op-indexed `PatternMatcher` with a topological replacement cache, so DAG sharing
survives a rewrite instead of being expanded into a tree. This is far smaller than tinygrad's
`UPat`/`PatternMatcher`, and it is enough for AD cleanup and canonicalization.

## Reading a graph

```python
y.debug()                     # topological dump with stable %0, %1, ... names
al.format_expr(y)             # the same thing as a string
al.render_expr_assembly(f)    # SSA-like assembly, expr.* prefix
al.expr_graph(y)              # nodes and edges as JSON, for tooling
```

The assembly form is the stable one: it is meant to be diffed, pasted into a bug report and
asserted on in tests. A `Function` renders as an `expr.module` that includes the bodies of every
function it transitively calls — the same set that will appear in the generated C.

For watching a graph move through the compiler instead of reading one snapshot, see
[Visualization](../guide/visualization.md).

## Device placement

Every `Function` carries a `DeviceSpec`:

```python
fn_gpu = fn.with_device("cuda:0")
assert fn_gpu.device == al.DeviceSpec("cuda", 0)
```

Each backend registers a `BackendSupport` capability table, so placing a `float64` graph on a
backend that does not advertise `float64` fails at construction with a diagnostic naming the
offending input — not at runtime, and never as a silent fall back to the host. Today only `host`
lowers; other placements are accepted, tracked and printed, and raise `JitError` when called.
