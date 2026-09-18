# Influences

Scaly borrows from four projects: CasADi for symbolic optimal control, tinygrad for a small
compiler, MLIR for staged intermediate representations, and JAX for composable differentiation.
This page lists, for each, what scaly took and what it does differently. Read it if you already
know one of them and want to place scaly.

| | Taken | Changed |
| --- | --- | --- |
| CasADi | `Function` as the unit of composition, differentiation and compilation; the universal C ABI; sparsity as metadata | typed derivative requests instead of strings; repetition preserved by an explicit construct; pure Python |
| tinygrad | one hash-consed node class per dialect; op-indexed pattern rewriting; loops that carry their intent; the sizing discipline | an ahead-of-time compiler with no runtime; explicit verifiers; a second dialect with its own operations |
| MLIR | the dialect discipline; stable, diffable assembly text; progressive lowering | two dialects; a printer with no parser; no C++, no TableGen |
| JAX | `jvp`/`vjp` as the base operations; derivatives as ordinary graphs; batching structure preserved | explicit construction instead of tracing; `vmap` survives into generated code; structural sparsity as a compiler concern |

## CasADi

CasADi is the closest relative and the tool scaly is measured against in
[Benchmark results](../results/index.md).

### Taken

`Function` as the unit of composition, differentiation and compilation is CasADi's idea. It gives
derivatives a name, generated symbols a name, and a horizon of identical stages one C function
instead of a hundred. The [generated interface](generated_interface.md) follows CasADi's, so generated
functions can call each other, a generated solver can drive generated oracles, and an existing C++
consumer has nothing new to learn. Sparsity as structural metadata carried alongside a value, with
compressed sparse row (CSR) and compressed sparse column (CSC) views on it, is also CasADi's model.

`Function.factory` remains a shorthand for building several named derivatives. It is not the unit
of composition and it is not the graph-merging mechanism.

### Changed

Derivative requests are typed objects. CasADi asks for a derivative with a factory string such as
`"jac:eq:z"`. Scaly asks with `sc.factory.Jac("eq", "z")`. The string grammar is compact, but an
editor cannot complete it and it needs a parser. The typed form expresses the same requests and
puts their structure in the type system. The output and input names are still checked when the
request is resolved, because only the function knows them.

Repetition survives differentiation, and whether it does is decided locally. In CasADi, whether a
repeated region reaches generated code as a loop follows from which symbolic type the graph was
built with, and that choice propagates beyond the repeated region. `SX` flattens a `Function.map`
at construction, so the mapped and unrolled forms emit identical C. `MX` keeps the loop and pays a
per-node cost. Scaly writes repetition with [`sc.vmap`](../guide/functions.md), and it survives
colored sparse differentiation to second order and lowering, emitted as a `for` loop around one
function body. The benchmarks measure every supported CasADi encoding instead of choosing one
global baseline; the [benchmark results](../results/index.md) hold the numbers and the
[fairness audit](../results/fairness.md) states their limits.

Pure Python. CasADi is a C++ library with Python bindings. Scaly is Python with NumPy for array
values and SciPy as an internal structural-sparsity dependency. The compiler proper, both dialects,
automatic differentiation (AD), lowering, the passes and the renderer, is about ten thousand lines
and needs no build step. The cost is paid at compile time, since building a large graph in Python
is slower than building it in C++. It is not paid at run time, because the output is C either way.

## tinygrad

### Taken

The single-node-class IR is tinygrad's `UOp` design. It is why scaly's two dialects can share one
verifier, one pattern matcher and one printer: a frozen, hash-consed class with a `StrEnum` op tag
and everything else in arguments and attributes, where the tag says whether a node is a statement
or an expression. Rewrites are op-indexed pattern matching over that graph. `RangeKind` is
tinygrad's `AxisType`, loops carrying their own intent so a backend binds them without re-deriving
what they were for. Scaly also follows tinygrad's sizing discipline: every line has to earn its
place, and a speculative abstraction is a defect.

### Changed

tinygrad is a tensor runtime that lazily schedules and executes. Scaly is an ahead-of-time compiler
for named functions with no runtime. Scaly verifies explicitly, with per-dialect rule tables and a
`VerifyError` naming the first bad node, because a compiler that emits C has to fail at the
boundary. Scaly's second dialect is a separate language with its own operations and its own
verifier. And scaly's graph has named function boundaries in it, which a tensor runtime has no
reason to want.

## MLIR

### Taken

The word "dialect" and the discipline behind it: a named IR with its own operations, its own
verifier and a stable textual form, with lowering as an explicit staged transition between them.
Scaly's assembly text borrows MLIR's spelling closely enough to be readable by anyone who has seen
MLIR, with `expr.*` and `prog.*` prefixes, `tensor<3xfloat64 diff>` and
`memref<16xfloat64, private>`. The text can be diffed, pasted into a bug report and asserted on in
tests.

### Changed

There is no MLIR dependency, no TableGen and no C++. A dialect is a Python module. There are two
of them. A tower of progressive dialects fits a general compiler infrastructure; a library with a
single target needs two. Scaly's text is a printer with no parser, so the format is optimized for a
human reading it.

## JAX

### Taken

The AD model. `jvp` and `vjp` are the base operations, whole derivatives are composed from them,
and every transformation maps a graph to a graph. There is no tape, no recording and no runtime. A
derivative is the same kind of object as the thing it came from and gets the same treatment
downstream, which is what makes `hessian` be `jacobian` of `gradient`. Scaly also takes JAX's term
`vmap` for independent vectorized mapping, with a different abstraction boundary.

### Changed

Construction is explicit. JAX traces Python functions, so the traced object is a shadow of code
written for the tracer. Scaly builds the graph directly through operator overloading and keeps your
names on it. There is nothing to retrace and no gap between what you wrote and what the compiler
holds.

"JIT" means something narrower here. Scaly's just-in-time path is not `jax.jit`. There is no
decorator and nothing to opt into. A `Function` is already a graph, so the only deferred step is
the compile, which happens on the first call and is cached on disk. Nothing is specialized on the
values you pass, so there is no retracing, no recompilation when a shape changes (a shape is a
different function), and no first-call penalty on a warm cache, even in a fresh process. It is the
same compiler the ahead-of-time path runs, invoked on demand, so what you test from Python is what
you ship as C. See [Code generation](../guide/codegen.md).

`vmap` is explicit and survives into the output. JAX's `vmap` is a function-level transformation
driven by batching rules that returns a batched function. Scaly's `vmap` builds an
expression-level `VMAP` node around an already named `Function`, taking explicit outer expressions
and slice rules. That node stays through AD and lowering and becomes a loop in the generated C, so
a hundred-stage horizon produces code of roughly constant size. `sc.vmap` is not a drop-in
version of `jax.vmap`. See [Scalability](../results/scalability.md).

Structural sparsity is a compiler concern. JAX has no equivalent and does not need one, since
dense batched arithmetic on accelerators is its workload. In optimal control a constraint Jacobian
is mostly zeros with exploitable structure, so scaly carries structural patterns, colors them and
emits compact nonzero values.

## What is scaly's own

Colored sparse derivatives through preserved structure. Computing a sparse Jacobian by coloring a
pattern is standard. Doing it on the callee of a `VMAP`, coloring a small local tile, pushing
constant seeds through one derivative function and mapping the result, is what keeps generated
derivative code from growing with the horizon.

Solvers as graph nodes. `sc.problem(...)` returns a backend-free `Problem`; `sc.solver(...)` turns
it into a `Function` whose body is a `solver_call`, so a solve nests inside a larger graph like any
other operation, and the oracles, wrapper and host function compile into a single shared library
with no Python in the loop. See [Solvers](solvers.md).

The tests enforce import layers. Which packages may import which is a table in a test, and the one
sanctioned upward import is named and checked. This keeps the compiler legible as it grows. See
[the import-layer rules](../dev/codebase.md#import-layers).
