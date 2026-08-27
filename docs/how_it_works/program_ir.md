# The program dialect

The program dialect is the lower of alloy's two representations. Where an `Expr` says what a value
*is*, a `ProgramNode` says how it gets computed: which loops run, which buffer holds which value,
which procedure runs on which device. It is close enough to code that rendering it to C is
mechanical, and far enough from C that the same program could be rendered to something else.

You do not normally build it by hand. [Lowering](lowering.md) produces it from a `Function`, and
the [C renderer](c_abi.md) consumes it. This page is the vocabulary, for when you need to read a
dump or add an operation.

## One node class

There is a single `ProgramNode` — frozen, hash-consed by op, arguments and attributes, exactly
like `Expr`. Whether a node is a statement, a declaration or a scalar expression is decided by its
op tag, not by its class. This is the tinygrad `UOp` style, and the payoff is that one pattern
matcher, one verifier and one printer work across the whole dialect.

Here is a small program built directly, so the pieces are visible:

```python
from alloy.ir import program as p
from alloy.ir.program import RangeKind
from alloy.ir.program_spec import verify_program
from alloy.ir.text import format_program
from alloy.ir.types import dtypes

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

## Operations

Forty-three operations in two families.

### Structure — statements and declarations

| Op | Role |
| --- | --- |
| `program` | the whole module: a list of procedures and kernels |
| `proc` | a host procedure |
| `kernel` | a device procedure, with grid dimensions and a placement |
| `param` | a formal parameter of a `proc` or `kernel` |
| `buffer` | a storage declaration: dtype, shape, address space, device |
| `view` | an indexed reference into a buffer — the addressing, separate from the access |
| `block` | a statement sequence |
| `range` | a loop domain: start, stop, step, and a `RangeKind` |
| `for` | a loop binding a `range` over a body |
| `store` | write a scalar to a view |
| `assign` | write a scalar to a variable |
| `call` | invoke another `proc` |
| `launch` | start a `kernel` — host only |
| `barrier` | synchronize within a kernel — device only |

### Scalars — the expression sublanguage inside a statement

`const_int`, `const_float`, `var`, `load`, then `add`, `sub`, `mul`, `div`, `mod`, `neg`, the
transcendentals `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `erf`, `exp`, `log`,
`sqrt`, `abs`, `floor`, `ceil`, and the binary `pow`, `atan2`, `minimum`, `maximum`.

These are the leaves of a `store`: everything a generated C statement can say on its right-hand
side. Membership in `SCALAR_OPS`, `UNARY_FN_OPS` and `BINARY_FN_OPS` is what tells the verifier,
the printer and the renderer how to treat a new op, so a new scalar operation goes in the right
set as well as in the enum.

## Loops carry their intent

A `range` has a `RangeKind`, borrowed from tinygrad's `AxisType`:

`serial`, `vector`, `global`, `thread`, `local`, `warp`, `reduce`, `group_reduce`, `unroll`.

The kind is a verified attribute rather than a separate op, so every pass sees it and none has to
infer it. A backend uses it to bind a loop to a launch axis, choose vectorized against unrolled
emission, or recognize a reduction. In the lowered example below, the accumulation loop is marked
`reduce` — the lowerer knew it was a reduction and said so, rather than leaving the renderer to
guess from the shape of the body.

## Memory is explicit

Buffers declare an address space, following the OpenCL and CUDA naming: `global`, `local`,
`private`, `constant`. Placement lives on the buffer as a `DeviceSpec`.

Separating `view` from `load` and `store` is what makes aliasing expressible: a zero-copy reshape
or a contiguous slice is a view onto another buffer's storage, owning nothing itself. The
workspace packer relies on this — an alias owns no slot but extends the lifetime of the buffer it
points into, so a reused slot is never overwritten while something downstream can still read it.

## Verification

Four specs, composed from the same `Rule`/`Spec` machinery the expression dialect uses:

| Spec | Checks |
| --- | --- |
| `spec_program_shared` | what every node must satisfy — buffer attributes, view arguments are scalars, load and store shapes, valid range kind |
| `spec_host_program` | a `proc` contains no device-only op (no `barrier`) |
| `spec_kernel_program` | a `kernel` contains no host-only op (no `launch`) |
| `spec_program_full` | both — the whole-program check |

`lower_function` runs `verify_program` on its output before returning, so a malformed program is
caught at the boundary that produced it rather than as strange C much later.

## Reading a lowered program

```python
import alloy as al
from alloy.passes.lowering import lower_function

@al.function(al.L("x", 3), al.L("y", ...))
def f(x: al.Expr) -> al.Expr:
    return (x.sin() + x * x).sum()

print(al.render_program_assembly(lower_function(f)))
```

```text
prog.module {
  prog.proc @f(%x: memref<3xfloat64>, %y: memref<1xfloat64>) {device=host, input_count=1, sz_w=0, w_self=0} {
    prog.store 0, %y[0] : float64
    prog.for %i_y = 0 to 3 step 1 {kind=reduce} {
      prog.store prog.add(prog.load %y[0], prog.add(prog.sin(prog.load %x[%i_y]), prog.mul(prog.load %x[%i_y], prog.load %x[%i_y]))), %y[0] : float64
    }
  }
}
```

Everything the C will do is already here: the zero-initialization, the reduction loop, the fused
right-hand side, and the fact that no scratch space is needed (`sz_w=0`). `format_program(prog)`
gives the same content in a friendlier layout; `render_program_assembly` is the stable one to
assert on.

The printer is backend-neutral by design. It shows host and device boundaries, loop kinds and
launch specifications so that a scheduling decision can be read and argued with before any
renderer has run.
