# The intermediate representations

Scaly has two intermediate representations (IRs), called dialects. The expression dialect records
what to compute and is what you build, differentiate and rewrite. The program dialect records how,
with explicit loops, buffers and procedures, and is what lowering produces and the C renderer
consumes. [Architecture](architecture.md#the-two-dialects) says why there are two; this page is
the vocabulary of each.

## The expression dialect

The expression dialect is what you build when you write scaly. It records what to compute and
nothing about how: a graph of values and the operations between them, with no loops, no buffers
and no memory. That is what makes it differentiable and rewritable. Choosing an implementation
happens later, in [the program dialect](#the-program-dialect).

### `Expr`

An `Expr` is one node in a directed acyclic graph. It is frozen, it carries a type, and it knows
its arguments:

```python
import scaly as sc

x = sc.sym("x", 3)          # an INPUT node, shape (3,)
y = x.sin() + x * x         # ADD(SIN(x), MUL(x, x))
z = y.sum()                 # SUM(...)  -> shape ()
```

Every node has:

| Field | Meaning |
| --- | --- |
| `op` | an `ExprOp`, the operation this node performs |
| `args` | the operand nodes, in order |
| `type` | a `TensorType`: shape, dtype, sparsity, and the differentiability flag |
| `attrs` | per-op data that is not an operand, such as the callee of a `CALL` or the index table of a `GATHER` |
| `name` | set on inputs, otherwise `None` |
| `lowering` | `auto`, `scalar`, `block`, or `opaque`. Controls procedure scalarization as described in [Lowering and optimization](lowering.md#the-optimization-pipeline). |
| `id` | the node's Python object identity. Because nodes are interned, structural equality is object identity, so this doubles as a structural key. Printing does not use it; the stable `%0`, `%1` names come from a topological walk. |

Nodes are interned: building the same operation on the same arguments with the same attributes
returns the object you already have. Two structurally identical subgraphs are therefore one
subgraph, shared by both consumers, and common subexpressions collapse as you build instead of in
a later pass. Interning is by value in a weak-reference table, so nodes nothing refers to are
collected.

This is why an IR class must never be importable under two module paths: two copies of the class
mean two intern tables, and identity silently stops meaning equality. See
[the codebase rules](../dev/codebase.md#the-rules-that-keep-it-this-way).

### Types

`TensorType` is the type of every expression.

```python
sc.TensorType(shape=(3, 4), dtype=sc.dtypes.float64, sparsity=None, diff=True)
```

`shape` is a tuple of non-negative integers; `()` is a scalar. Shapes are static; there are no
symbolic dimensions.

`DType` is a small interned descriptor with a name, a bit width and a C spelling. The canonical
instances live in `sc.dtypes`:

| Instance | C type | Bytes |
| --- | --- | --- |
| `sc.dtypes.bool_` | `uint8_t` | 1 |
| `sc.dtypes.int32` | `int32_t` | 4 |
| `sc.dtypes.int64` | `int64_t` | 8 |
| `sc.dtypes.float32` | `float` | 4 |
| `sc.dtypes.float64` | `double` | 8 (the default) |

These are the types inside the graph. They are not the ABI: a generated function always exchanges
`double` buffers with its caller, and the Python call path converts to and from `float64` at the
boundary. See [the C ABI](c_abi.md#calling-convention).

Mixed-dtype arithmetic is refused. There is no implicit widening: an expression combining
`float32` and `float64` raises at construction. This keeps today's `float64` workloads exactly as
they are, and when an explicit `Expr.cast` op lands, every mixed-precision decision will be
visible at the call site where it was made.

`diff` marks whether a value depends differentiably on symbolic inputs. Constants are not
differentiable; structural operations pass the flag through; arithmetic propagates it from its
operands; and the non-smooth operations (`floor`, `ceil`, `minimum`, `maximum`, `solver_call`)
clear it. AD (automatic differentiation) reads this flag to decide where a derivative is zero by
construction.

`SparsityType` attaches a structural pattern to a rank-2 value; see
[Sparsity](../guide/sparsity.md). When present, its shape must match the tensor shape exactly.

### Operations

Thirty-nine operations, grouped by what they do. `arity` is the operand count; `n` means
variadic. `diff` is whether AD can pass through the op at all.

#### Arithmetic and elementwise

| Op | Arity | Diff | Notes |
| --- | --- | --- | --- |
| `neg` | 1 | yes | |
| `add` `sub` `mul` `div` | 2 | yes | NumPy broadcasting |
| `pow` | 2 | yes | |
| `sin` `cos` `tan` | 1 | yes | |
| `asin` `acos` `atan` | 1 | yes | no multi-seed forward rule yet |
| `atan2` | 2 | yes | no multi-seed forward rule yet |
| `sinh` `cosh` `tanh` | 1 | yes | |
| `erf` | 1 | yes | |
| `exp` `log` `sqrt` | 1 | yes | |
| `abs` | 1 | yes | no multi-seed forward rule yet |
| `floor` `ceil` | 1 | no | result is marked non-differentiable |
| `minimum` `maximum` | 2 | no | result is marked non-differentiable |

#### Structural

| Op | Arity | Notes |
| --- | --- | --- |
| `input` | 0 | a named symbol |
| `const` | 0 | a materialized array; never differentiable |
| `sum` | 1 | full reduction to a scalar; there is no axis argument |
| `reshape` | 1 | size-preserving |
| `transpose` | 1 | axes must be a permutation |
| `slice` | 1 | integer, multi-dimensional and strided indexing |
| `gather` `scatter` | 1 | flat index tables |
| `stack` `concat` | n | along any axis |
| `matmul` | 2 | rank at most 2 |

#### Boundaries

| Op | Arity | Diff | Notes |
| --- | --- | --- | --- |
| `call` | n | yes | a named `Function` applied to arguments |
| `VMAP` | n | yes | one callee applied across slices of its arguments |
| `solver_call` | n | no | an opaque solve; see [Solvers](solvers.md) |

`dot`, `sumsqr`, `norm_2` and `vec` are not operations. They are builders that expand into the
ops above; `vec` emits a `reshape` to rank 1.

Absent for now: `expm1` and `log1p`; splines and interpolants, which need their own design for
knots, extrapolation, derivative behaviour and table codegen before they can be an op; matrix
decompositions; and control flow.

### Functions

A `Function` is a named graph boundary: named inputs, named outputs, and optional sparsity metadata
per output.

```python
f = sc.Function("f", [x], [y], ["x"], ["y"])
```

It is the unit of composition (`f.call(...)` puts a `call` node in a bigger graph), of
differentiation (`f.factory(...)` derives a new `Function`), and of compilation (calling it
produces C).

`Function.call` normalizes constant-like arguments to `Expr` and checks every argument shape
against the corresponding formal. Because a call is a node and not an inlining, the callee's
structure survives into the generated C as a C function, which keeps generated code small when
the same block appears a hundred times.

### Verification

Construction checks the cheap invariants inline so the common path stays fast. Everything else is
an explicit pass:

```python
sc.verify_expr(y)                     # silent on success
sc.verify_expr(y, spec=sc.spec_expr)  # naming the spec explicitly
```

The verifier walks the graph in topological order and raises `VerifyError` at the first invalid
node, naming the node, its op and the rule it failed. Two specs are exported:

- `spec_expr_shared` is what every node must satisfy: non-negative shape, a real `DType`, arity
  matching `OP_INFO`, sparsity shape agreeing with tensor shape.
- `spec_expr` adds the per-op rules: `reshape` preserves size, `transpose` axes are a permutation,
  `matmul` contracting dimensions agree, `call` argument shapes match the callee, `VMAP` outer
  tensors are rank 1 with a consistent slice size, `const` value shape and dtype match the declared
  type.

Run it after a non-trivial rewrite or an AD transform, and write negative tests against it. It is
the same `Rule` and `Spec` machinery the [program dialect](#the-program-dialect) verifies with.

### Structural equality and rewrites

`Expr.id` is construction identity. For structure, use the `Expr` methods `y.structural_key()`,
`y.structural_hash()` and `y.structurally_equal(other)`. They compare the shape of the graph
without disturbing node identity, which lets a pass recognize an equivalent subgraph without
rewriting anything.

The rewrite layer is small:

```python
y_cse = sc.cse(y)
y_clean = sc.simplify(y_cse)
```

`simplify` covers constant folding and algebraic identities (`x + 0`, `x * 1`, `x * 0`, identity
reshape and transpose, slice-of-slice, slice-of-stack). `cse` merges structurally equal subgraphs.
Both go through an op-indexed `PatternMatcher` with a topological replacement cache, so DAG sharing
survives a rewrite instead of being expanded into a tree. This is far smaller than tinygrad's
`UPat`/`PatternMatcher`, and it is enough for AD cleanup and canonicalization.

### Reading a graph

```python
y.debug()                     # topological dump with stable %0, %1, ... names
sc.format_expr(y)             # the same thing as a string
sc.render_expr_assembly(f)    # SSA-like assembly, expr.* prefix
sc.expr_graph(y)              # nodes and edges as JSON, for tooling
```

The assembly form is the stable one. It is meant to be diffed, pasted into a bug report and
asserted on in tests. A `Function` renders as an `expr.module` that includes the bodies of every
function it transitively calls, the same set that will appear in the generated C.

To watch a graph move through the compiler instead of reading one snapshot, see
[Visualization](../guide/visualization.md).

### Device placement

Every `Function` carries a `DeviceSpec`:

```python
fn_gpu = fn.with_device("cuda:0")
assert fn_gpu.device == sc.DeviceSpec("cuda", 0)
```

Each backend registers a `BackendSupport` capability table, so placing a `float64` graph on a
backend that does not advertise `float64` fails at construction with a diagnostic naming the
offending input. There is no silent fall back to the host. Today only `host` lowers; other
placements are accepted, tracked and printed, and raise `LoweringError` when called.

## The program dialect

The program dialect is the lower of scaly's two representations. Where an `Expr` says what a value
is, a `ProgramNode` says how it gets computed: which loops run, which buffer holds which value,
which procedure runs on which device. It is close enough to code that rendering it to C is
mechanical, and far enough from C that the same program could be rendered to something else.

You do not normally build it by hand. [Lowering](lowering.md) produces it from a `Function`, and
the [C renderer](c_abi.md) consumes it. This page is the vocabulary, for when you need to read a
dump or add an operation.

### One node class

There is a single `ProgramNode`, frozen and hash-consed by op, arguments and attributes, exactly
like `Expr`. Its op tag decides whether a node is a statement, a declaration or a scalar
expression. This is the tinygrad `UOp` style, and it means one pattern matcher, one verifier and
one printer work across the whole dialect.

A small program built directly, so the pieces are visible:

```python
from scaly.ir import program as p
from scaly.ir.program import RangeKind
from scaly.ir.program_spec import verify_program
from scaly.ir.text import format_program
from scaly.ir.types import dtypes

in_buf = p.buffer("in_", dtypes.float64, (16,))
out_buf = p.buffer("out_", dtypes.float64, (16,))
i = p.var("i", dtype=dtypes.int64)

k = p.kernel(
    "k_neg",
    [in_buf, out_buf],
    [
        p.for_(
            p.range_("i", 0, 16, kind=RangeKind.GLOBAL),
            [p.store(p.view(out_buf, [i]), p.neg(p.load(p.view(in_buf, [i]))))],
        ),
    ],
    grid_dims=1,
    device="cuda:0",
)

verify_program(k)
print(format_program(k))
```

```text
kernel k_neg(in_:float64(16,), out_:float64(16,)) device=cuda:0:
  for i in [0, 16) step 1 kind=global:
    out_[i] <- (-in_[i])
```

### Program operations

Operations fall into two families.

#### Structure: statements and declarations

| Op | Role |
| --- | --- |
| `program` | the whole module: a list of procedures and kernels |
| `proc` | a host procedure |
| `kernel` | a device procedure, with grid dimensions and a placement |
| `param` | a formal parameter of a `proc` or `kernel` |
| `buffer` | a storage declaration: dtype, shape, address space, device |
| `view` | an indexed reference into a buffer; the addressing, separate from the access |
| `block` | a statement sequence |
| `range` | a loop domain: start, stop, step, and a `RangeKind` |
| `for` | a loop binding a `range` over a body |
| `store` | write a scalar to a view |
| `store_pair` | evaluate two float64 values, then write consecutive lanes starting at a view |
| `assign` | write a scalar to a variable; `declare=True` also declares the typed local |
| `call` | invoke another `proc` |
| `launch` | start a `kernel`; host only |
| `barrier` | synchronize within a kernel; device only |

#### Scalars: the expression sublanguage inside a statement

`const_int`, `const_float`, `var`, `load`, then `add`, `sub`, `mul`, `div`, `mod`, `neg`, the
transcendentals `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `erf`, `exp`, `log`,
`sqrt`, `abs`, `floor`, `ceil`, and the binary `pow`, `atan2`, `minimum`, `maximum`.

These are the leaves of a `store`: everything a generated C statement can say on its right-hand
side. Membership in `SCALAR_OPS`, `UNARY_FN_OPS` and `BINARY_FN_OPS` tells the verifier, the
printer and the renderer how to treat an op, so a new scalar operation goes in the right set as
well as in the enum.

### Loops carry their intent

A `range` has a `RangeKind`, borrowed from tinygrad's `AxisType`:

`serial`, `vector`, `global`, `thread`, `local`, `warp`, `reduce`, `group_reduce`, `unroll`.

The kind is a verified attribute, so every pass sees it and none has to infer it. A backend uses
it to bind a loop to a launch axis, choose vectorized against unrolled emission, or recognize a
reduction. In the lowered example below, the lowerer marks the accumulation loop `reduce`, so the
renderer does not have to guess from the shape of the body.

### Memory is explicit

Buffers declare an address space, following the OpenCL and CUDA naming: `global`, `local`,
`private`, `constant`. Placement lives on the buffer as a `DeviceSpec`.

Separating `view` from `load` and `store` makes aliasing expressible: a zero-copy reshape or a
contiguous slice is a view onto another buffer's storage, owning nothing itself. The workspace
packer relies on this. An alias owns no slot but extends the lifetime of the buffer it points
into, so a reused slot is never overwritten while something downstream can still read it.

### Verifying a program

Four specs, composed from the same `Rule`/`Spec` machinery the expression dialect uses:

| Spec | Checks |
| --- | --- |
| `spec_program_shared` | what every node must satisfy: buffer attributes, view arguments are scalars, load and store shapes, valid range kind |
| `spec_host_program` | a `proc` contains no device-only op (no `barrier`) |
| `spec_kernel_program` | a `kernel` contains no host-only op (no `launch`) |
| `spec_program_full` | both; the whole-program check |

Whole modules also require unique procedure names. The paired-store rule checks its target and
value types; the pairing pass establishes alias safety.

`lower_function` runs `verify_program` on its output before returning, so a malformed program is
caught at the boundary that produced it and not as strange C much later.

### Reading a lowered program

```python
import scaly as sc
from scaly.passes.lowering import lower_function

@sc.function(sc.L("x", 3), sc.L("y", ...))
def f(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x * x).sum()

print(sc.render_program_assembly(lower_function(f)))
```

```text
prog.module {
  prog.proc @f(%x: memref<3xfloat64>, %y: memref<1xfloat64>) {device=host, input_count=1, lowering="auto", scalarize_mode="disabled", sz_w=0, w_self=0} {
    prog.store 0, %y[0] : float64
    prog.for %i_y = 0 to 3 step 1 {kind=reduce} {
      %v0 = prog.assign prog.load %x[%i_y] : float64 {declare=True}
      prog.store prog.add(prog.load %y[0], prog.add(prog.sin(%v0), prog.mul(%v0, %v0))), %y[0] : float64
    }
  }
}
```

Everything the C will do is already here: the zero-initialization, the reduction loop, the fused
right-hand side, and the fact that no scratch space is needed (`sz_w=0`). `format_program(prog)`
gives the same content in a friendlier layout; `render_program_assembly` is the stable one to
assert on.

The printer is backend-neutral. It shows host and device boundaries, loop kinds and launch
specifications, so a scheduling decision can be read before any renderer has run.
