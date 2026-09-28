# Intermediate representations

This page is the reference for Scaly's two intermediate representations (IRs),
the expression dialect and the program dialect. The
[architecture overview](architecture.md#why-two-dialects) explains why there are
two and what an IR node is. Each section below lists what a dialect contains
and shows how its constructs look when printed.

## The expression dialect

### Expression nodes

An `Expr` has these fields:

| Field      | Meaning                                                       |
| ---------- | ------------------------------------------------------------- |
| `op`       | The `ExprOp` operation tag                                    |
| `args`     | Ordered operand nodes                                         |
| `type`     | A `TensorType`: shape, data type and the `diff` flag          |
| `attrs`    | Operation-specific data, such as a callee or gather indices   |
| `name`     | The input name, or `None` for other nodes                     |
| `value`    | The NumPy array of a constant, or `None` for other nodes      |
| `lowering` | The `auto`, `scalar` or `block` hint                          |

`sc.format_expr(expr)` prints the graph below one node, and
`sc.render_expr_assembly(fn)` prints a `Function` and every function it calls.
The assembly numbers nodes in traversal order, so two runs print the same text.

### Types

A `TensorType` holds a fixed shape of non-negative integers, a data type and a
`diff` flag. Shape `()` is a scalar.

| Data type           | C type     | Bytes |
| ------------------- | ---------- | ----- |
| `sc.dtypes.bool_`   | `uint8_t`  | 1     |
| `sc.dtypes.int32`   | `int32_t`  | 4     |
| `sc.dtypes.int64`   | `int64_t`  | 8     |
| `sc.dtypes.float64` | `double`   | 8     |

The default is `float64`. There is no cast operation, so an operation on two
data types fails when you build it:

```python
x = sc.sym("x", 3)
x + sc.sym("n", 3, dtype=sc.dtypes.int64)
# TypeError: mixed-dtype operation not supported: float64 vs int64; give all operands the same dtype
```

Data types matter inside the graph and the generated procedures. The exported
C entry still reads and writes every input and output as `double` values, as
described in [the pointer ABI](generated_interface.md#the-pointer-abi).

The `diff` flag says whether a value can carry a derivative. Inputs have it
unless created with `diff=False`. Constants and non-differentiable operations
never have it, and any other operation has it when one of its operands does:

```python
x.type.diff                   # True
sc.const(2.0).type.diff       # False
x.floor().type.diff           # False
(x * x.floor()).type.diff     # True
```

The flag is metadata. Whether a derivative exists is decided by the
differentiation rules, and differentiating through `floor`, `ceil`, `minimum` or
`maximum` raises `NotImplementedError`.

Expression types are dense. A sparse derivative is a one-dimensional `Expr` of nonzero values, and
its pattern lives on the owning `Function` in `output_sparsities`, as the
[sparsity guide](../guide/sparsity.md) shows.

### Operations

Arity is the number of operands.

| Family          | Operations                                    | Arity    | Notes                                     |
| --------------- | --------------------------------------------- | -------- | ----------------------------------------- |
| Sources         | `input`, `const`                              | 0        | A named input or a constant array         |
| Arithmetic      | `neg`, `add`, `sub`, `mul`, `div`, `pow`      | 1 or 2   | Binary operations broadcast like NumPy    |
| Trigonometric   | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | 1 or 2 | Elementwise                         |
| Hyperbolic      | `sinh`, `cosh`, `tanh`                        | 1        | Elementwise                               |
| Other math      | `erf`, `exp`, `log`, `sqrt`, `abs`            | 1        | Elementwise                               |
| Non-smooth      | `floor`, `ceil`, `minimum`, `maximum`         | 1 or 2   | No derivative rule                        |
| Reduction       | `sum`                                         | 1        | All elements into a scalar, no axis       |
| Shape           | `reshape`, `transpose`                        | 1        | Size-preserving reshape, axis permutation |
| Indexing        | `slice`, `gather`, `scatter`                  | 1        | Gather and scatter use flat index tables  |
| Assembly        | `stack`, `concat`                             | Variable | Along one axis                            |
| Linear algebra  | `matmul`                                      | 2        | Operands of rank at most two              |
| Function calls  | `call`, `vmap`                                | Variable | An ordinary or a mapped call              |
| Solvers         | `solver_call`                                 | Variable | A nested solve, not differentiable        |

`dot`, `sumsqr`, `norm_2` and `vec` are builders that expand into these
operations. There are no matrix decompositions, interpolation tables or
data-dependent control flow yet, and no `expm1` or `log1p`.

### Calls and mapped calls

A symbolic call to a `Function` adds one `call` node, and `sc.vmap` adds one
`vmap` node, whatever the size of the callee. Here a stage model, the one from
[Getting started](../guide/getting_started.md), is mapped over ten stages and
the last state goes through a terminal cost:

```python
@sc.function(sc.G(sc.L("z", 2), sc.L("u", 1)), sc.L("znext", ...))
def model(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, u = inputs
    return z + 0.1 * sc.concat([z[1:], u])


@sc.function(sc.L("z", 2), sc.L("cost", ...))
def terminal(z: sc.Expr) -> sc.Expr:
    return sc.sumsqr(z)


@sc.function(sc.G(sc.L("zs", 20), sc.L("us", 10)), sc.L("J", ...))
def stages(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    zs, us = inputs
    znexts = sc.vmap(model, 10, [zs, us])
    return terminal(znexts[18:])


print(sc.render_expr_assembly(stages))
```

The module lists `model` and `terminal` first. This is the part for `stages`:

```
expr.func @stages(%zs: tensor<20xfloat64 diff>, %us: tensor<10xfloat64 diff>) -> (%J: tensor<float64 diff>) {
  %0 = expr.input {lowering="auto", name="zs"} : tensor<20xfloat64 diff>
  %1 = expr.input {lowering="auto", name="us"} : tensor<10xfloat64 diff>
  %2 = expr.vmap(%0, %1) {callee="model", length=10, output=0, slice_size=2, starts=[0, 0], strides=[2, 1]} : tensor<20xfloat64 diff>
  %3 = expr.slice(%2) {index=[slice(18, None, None)]} : tensor<2xfloat64 diff>
  %4 = expr.call(%3) {callee="terminal", output=0} : tensor<float64 diff>
  expr.return %4
}
```

Both nodes keep the callee as an attribute and select one of its outputs with
`output`. The `vmap` node also records the slicing that `sc.vmap` inferred.
Iteration `i` reads `zs[2i : 2i+2]` and `us[i : i+1]`, from `starts` and
`strides`, and writes `slice_size` values of the flat output. Outer operands of
a `vmap` are always one-dimensional. The
[functions guide](../guide/functions.md#regular-repetition-vmap) covers the
slicing rules. The [program dialect](#calls-and-loops-after-lowering) shows what
these two nodes become.

### Verification

Builders reject most invalid nodes as you write them. `x.reshape(4)` on a
three-element `x` raises a `ValueError` before any node exists. `sc.verify_expr`
checks a whole graph against the rule table in `ir/expr_spec.py`, which matters
after a pass has rebuilt it. To see a failure, build a bad node directly with
the `Expr` constructor, which skips the builder's checks:

```python
x = sc.sym("x", 3)
bad = sc.Expr(sc.ExprOp.RESHAPE, (x,), sc.TensorType((4,)), attrs={"shape": (4,)})
sc.verify_expr((bad * 2.0).sum())
# VerifyError: verify_expr: node %128283723843408 op=reshape failed rule 'reshape-size': RESHAPE source size 3 != target size 4
```

The message names the node, its operation, the rule and the reason. An unnamed
node is labelled with its Python object id, which changes from run to run.
Rules that apply to every node, such as the arity check, are in
`spec_expr_shared`. `spec_expr` adds the rules for each operation.

## The program dialect

A `ProgramNode` is a statement, a declaration or a scalar calculation. You do
not build programs yourself. Lowering produces them when Scaly generates C, and
the listings below show the optimized program for each example.

### Program operations

| Operation                    | Meaning                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| `program`                    | A module of procedures                                     |
| `proc`                       | A procedure                                                |
| `buffer`                     | Storage with a shape, data type and address space          |
| `view`                       | An indexed location in a buffer                            |
| `load`, `store`              | Read or write one scalar at a view                         |
| `store_pair`                 | Write two values to adjacent `float64` elements            |
| `assign`                     | Assign a scalar variable, optionally declaring it          |
| `block`                      | A sequence of statements                                   |
| `range`, `for`               | A loop domain and a loop over it                           |
| `call`                       | Call another procedure                                     |
| `const_int`, `const_float`, `var` | Scalar constants and variables                        |

Scalar arithmetic reuses the expression dialect's elementwise operations, plus
`mod` for index arithmetic.

### Calls and loops after lowering

Lowering `stages` from the [expression example](#calls-and-mapped-calls) turns
the `vmap` into a loop around one call and keeps the `call`:

```
prog.module {
  prog.proc @model(%z: memref<2xfloat64>, %u: memref<1xfloat64>, %znext: memref<2xfloat64>) {device=host, input_count=2, lowering="auto", scalarize_mode="procedure", scalarized=True, sz_w=0, w_self=0} {
    %v0 = prog.assign prog.load %z[1] : float64 {declare=True}
    prog.store_pair prog.add(prog.load %z[0], prog.mul(0.1, %v0)), prog.add(%v0, prog.mul(0.1, prog.load %u[0])), %znext[0] : float64
  }
  prog.proc @terminal(%z: memref<2xfloat64>, %cost: memref<1xfloat64>) {device=host, input_count=1, lowering="auto", scalarize_mode="procedure", scalarized=True, sz_w=0, w_self=0} {
    %v0 = prog.assign prog.load %z[0] : float64 {declare=True}
    %v1 = prog.assign prog.load %z[1] : float64 {declare=True}
    prog.store prog.add(prog.mul(%v0, %v0), prog.mul(%v1, %v1)), %cost[0] : float64
  }
  prog.proc @stages(%zs: memref<20xfloat64>, %us: memref<10xfloat64>, %J: memref<1xfloat64>) {device=host, input_count=2, lowering="auto", scalarize_mode="disabled", sz_w=0, w_self=0} {
    %s0 = prog.buffer : memref<20xfloat64, private>
    prog.for %it_t0 = 0 to 10 step 1 {kind=global} {
      prog.call @model(%zs[prog.mul(2, %it_t0)], %us[%it_t0], %s0[prog.mul(%it_t0, 2)]) {callee_needs_w=False, n_in=2, n_out=1, w_self=0}
    }
    %t1 = prog.buffer : memref<2xfloat64, private> = prog.alias %s0 offset 18
    %s1 = prog.buffer : memref<1xfloat64, private>
    prog.call @terminal(%t1, %s1) {callee_needs_w=False, n_in=1, n_out=1, w_self=0}
    prog.store prog.load %s1[0], %J[0] : float64
  }
}
```

The `starts` and `strides` of the `vmap` node became the views
`%zs[2 * it]` and `%us[it]`. A view passed to a call is an address, not a load,
and the C for `stages` shows it as pointer arithmetic (trimmed to the loop and
the call):

```c
  double s0[20];
  const double* t1 = s0 + 18;
  double s1[1];
  for (long long it_t0 = 0; it_t0 < 10; ++it_t0) {
    model_raw((arg[0] + (2 * it_t0)), (arg[1] + it_t0), (s0 + (it_t0 * 2)), NULL);
  }
  terminal_raw(t1, s1, NULL);
```

The two small callees were expanded into scalar statements, which the
`scalarized=True` attribute records. The optimizer may also expand a callee into
its caller. If `stages` returns `znexts` directly, the loop body contains the
arithmetic of `model` and no call remains. A `call` node therefore does not
promise a separate C function. [Lowering and optimization](lowering.md) says
when each happens.

### Reading a lowered program

The attributes on a `prog.proc` line are these:

| Attribute        | Meaning                                                                |
| ---------------- | ---------------------------------------------------------------------- |
| `input_count`    | How many leading parameters are inputs, rendered as `const double*`    |
| `lowering`       | The combined hint of the function's nodes                              |
| `scalarize_mode` | Whether the procedure may be expanded into scalar statements           |
| `scalarized`     | Present once it has been                                               |
| `sz_w`           | Workspace size in doubles the caller must provide, callees included    |
| `w_self`         | The part of that workspace the procedure uses itself                   |

### Views and aliases

A `view` names a location without reading it, and a `buffer` can alias another
buffer at an offset. Together they let a slice or a reshape of a computed value
reuse its storage. Here a slice and a reshape feed a callee that takes a matrix:

```python
@sc.function(sc.L("M", (2, 3)), sc.L("s", ...))
def trace2(M: sc.Expr) -> sc.Expr:
    return M[0, 0] + M[1, 1]


@sc.function(sc.L("x", 8), sc.L("t", ...))
def f(x: sc.Expr) -> sc.Expr:
    y = x.sin()
    return trace2(y[2:].reshape((2, 3)))
```

```
prog.module {
  prog.proc @trace2(%M: memref<2x3xfloat64>, %s: memref<1xfloat64>) {device=host, input_count=1, lowering="auto", scalarize_mode="procedure", scalarized=True, sz_w=0, w_self=0} {
    prog.store prog.add(prog.load %M[0], prog.load %M[4]), %s[0] : float64
  }
  prog.proc @f(%x: memref<8xfloat64>, %t: memref<1xfloat64>) {device=host, input_count=1, lowering="auto", scalarize_mode="disabled", sz_w=0, w_self=0} {
    %s0 = prog.buffer : memref<8xfloat64, private>
    prog.for %i_t0 = 0 to 8 step 1 {kind=global} {
      prog.store prog.sin(prog.load %x[%i_t0]), %s0[%i_t0] : float64
    }
    %t1 = prog.buffer : memref<6xfloat64, private> = prog.alias %s0 offset 2
    %s1 = prog.buffer : memref<1xfloat64, private>
    prog.call @trace2(%t1, %s1) {callee_needs_w=False, n_in=1, n_out=1, w_self=0}
    prog.store prog.load %s1[0], %t[0] : float64
  }
}
```

`%t1` owns no storage. It is `%s0` from element 2, and the C is a single pointer
(trimmed):

```c
double s0[8];
const double* t1 = s0 + 2;
```

The reshape costs nothing at all. Buffers are flat and row-major, so the
six-element alias is passed as the `2x3` parameter unchanged, and `M[1, 1]`
becomes the flat index 4 inside `trace2`. Because an alias reads its source's
storage, the [workspace packer](lowering.md#temporary-storage) treats a read of `%t1` as a read of `%s0` when it
decides which temporaries can share memory.

Every buffer also has an address space. Parameters are `global`, which the
listing leaves out, temporaries are `private`, and constant tables are
`constant`.

### Loop kinds

Each `range` carries a `RangeKind`, printed as `kind=` on a `prog.for`. The set
is modelled on tinygrad's `AxisType`[^tinygrad-axis]. Lowering and lane
widening produce these kinds:

| Kind                  | Produced by                  | Meaning                                                |
| --------------------- | ---------------------------- | ------------------------------------------------------ |
| `global`              | Lowering                     | Independent iterations, as in elementwise loops and `vmap` |
| `reduce`              | Lowering                     | Iterations accumulating into one location, as in `sum` and `matmul` |
| `serial`, `vector`    | Lane widening                | An outer loop over chunks and an inner loop over vector lanes |

The C renderer emits a `global` or `reduce` loop as a plain `for`. The kind
matters to the passes. Loop fusion leaves `reduce` loops alone, and lane
widening splits an independent loop, or a reduction whose order it can keep,
into `serial` chunks of `vector` lanes. With
`lanes=4`, the loop of `x.sin() * x` over 64 elements becomes this (trimmed to
the loop headers):

```
prog.for %i_y_chunk = 0 to prog.div(prog.add(64, prog.sub(%SCALY_WIDTH_f_lanes_1, 1)), %SCALY_WIDTH_f_lanes_1) step 1 {kind=serial} {
  prog.for %i_y_lane = 0 to prog.minimum(%SCALY_WIDTH_f_lanes_1, prog.sub(64, prog.mul(%i_y_chunk, %SCALY_WIDTH_f_lanes_1))) step 1 {kind=vector} {
```

The width is a C macro rather than a number. It is the smaller of the requested
lanes and a cap derived from the target's vector registers, so the C
preprocessor settles it for each target.
[Lowering and optimization](lowering.md) describes lane widening.

### Program verification

`verify_program` applies the same kind of rule table to programs, from
`ir/program_spec.py`. It checks buffer attributes, that views index with
scalars, that loads and stores take views, and that loop kinds are valid. A
whole module must also have unique procedure names. Lowering verifies the
program before and after optimization, so a malformed program fails before any
C exists.

[^tinygrad-axis]: tinygrad, `AxisType` in `tinygrad/uop/ops.py`,
    <https://github.com/tinygrad/tinygrad>. The two sets have since diverged.
