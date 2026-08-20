# Alloy next to its neighbours


In the process of developing alloy, we have looked at a multitude of existing libraries that offer similar features, but not always with the same focus.
This page mentions a few projects that have shaped what alloy is today, what we took inspiration from, and what we deliberately changed.

<!--Alloy sits at an intersection that already has good tools in it. CasADi owns symbolic optimal
control. tinygrad showed how small a real compiler can be. MLIR set the vocabulary for staged
intermediate representations. JAX settled what composable differentiation looks like.-->

<!--Alloy borrows from all four, and departs from each of them somewhere specific. This page is the
accounting: what was taken, what was changed, and why. It is written to be argued with — if a
departure below is wrong, that is worth knowing.-->

| | Taken | Changed |
| --- | --- | --- |
| **CasADi** | `Function` as the unit of everything; the universal C ABI; derivative factories; sparsity as metadata | typed derivative requests instead of strings; one graph instead of SX *or* MX; pure Python |
| **tinygrad** | one hash-consed node class per dialect; op-indexed pattern rewriting; loops that carry their intent; the sizing discipline | an ahead-of-time compiler, not a runtime; explicit verifiers; a second dialect that is a real language |
| **MLIR** | the dialect discipline; stable, diffable assembly text; progressive lowering | two dialects, not twenty; a printer with no parser; no C++, no TableGen |
| **JAX** | `jvp`/`vjp` as primitives; derivatives as ordinary graphs; batching structure preserved rather than unrolled | explicit construction instead of tracing; `map` survives into generated code; sparsity is first-class |

## CasADi

CasADi is the closest relative and the tool alloy is measured against, in
[Benchmark results](../results/index.md).

**What alloy keeps.** `Function` as the unit of composition, differentiation and compilation is
CasADi's idea and it is the right one — it gives derivatives a name, generated symbols a name, and
a horizon of identical stages one C function instead of a hundred. The generated
[C ABI](c_abi.md) is CasADi-compatible in spirit for a practical reason: a shared calling
convention means generated functions can call each other, a generated solver can drive generated
oracles, and an existing C++ consumer does not need to learn anything new. Sparsity as structural
metadata carried alongside a value, with CSR and CSC views on it, is also CasADi's model.

**Where alloy departs.**

*Derivative requests are typed objects, not strings.* CasADi asks for a derivative with a factory
string — `"jac:eq:z"`. Alloy asks with `al.jac("eq", "z")`. The string grammar is expressive and
compact, and it cannot be completed by an editor, and it needs a parser that becomes a small
language of its own. The typed form gives up nothing, deletes the parser, and puts the request's
structure in the type system — the output and input names are still checked when the request is
resolved, because only the function knows them. This is a deliberate break, not a compatibility
gap.

*One graph, with a per-node hint.* CasADi makes you choose SX or MX up front, and mixing them is
awkward enough that most codebases pick one. Alloy has a single `Expr` type carrying a lowering
hint — `scalar`, `block`, `opaque` or `auto` — so the choice is local to a subexpression rather
than global to a type. The honest caveat is that today the hints are recorded but only `opaque`
changes what the lowerer does; region formation on the others is open work.

*Pure Python.* CasADi is a C++ library with Python bindings. Alloy is Python with NumPy for
array values and SciPy as an internal structural-sparsity dependency. The entire compiler — both dialects, automatic differentiation (AD),
lowering,
the passes and the renderer — is about ten thousand lines, editable without a build step. The
cost is real and is paid at compile time: building a large graph in Python is slower than building
it in C++. It is not paid at run time, because the output is C either way.

## tinygrad

**What alloy keeps.** The single-node-class IR is tinygrad's `UOp` design and it is why alloy's
two dialects can share one verifier, one pattern matcher and one printer between them: a frozen,
hash-consed class with a `StrEnum` op tag and everything else in arguments and attributes, where
"is this a statement or an expression" is answered by the tag rather than by the class. Rewrites
are op-indexed pattern matching over that graph. `RangeKind` is tinygrad's `AxisType` — loops
carrying their own intent, so a backend binds them without re-deriving what they were for. So is
the sizing discipline: every line earns its place, and a speculative abstraction is a defect.

**Where alloy departs.** tinygrad is a tensor runtime that lazily schedules and executes; alloy is
an ahead-of-time compiler for named functions with no runtime at all. That difference shows up in
three places. Alloy verifies explicitly, with per-dialect rule tables and a `VerifyError` naming
the first bad node, because a compiler that emits C has to fail at the boundary rather than
downstream. Alloy's second dialect is a separate language with its own operations and its own
verifier, not a linearized form of the first. And alloy's graph has named function boundaries in
it, which a tensor runtime has no reason to want.

## MLIR

**What alloy keeps.** The word "dialect" and the discipline behind it: a named IR with its own
operations, its own verifier, and a stable textual form, with lowering as an explicit staged
transition between them rather than as one function that emits code. Alloy's assembly text borrows
MLIR's spelling closely enough to be readable by anyone who has seen MLIR — `expr.*` and `prog.*`
prefixes, `tensor<3xfloat64 diff>`, `memref<16xfloat64, private>` — because the value of that text
is that it can be diffed, pasted into a bug report and asserted on in tests.

**Where alloy departs.** There is no MLIR here — no dependency, no TableGen, no C++. A dialect is a
Python module. There are two of them and there will not be twenty; the tower of progressive
dialects is the right structure for a general compiler infrastructure and the wrong one for a
library with a single well-understood target. And alloy's text is a printer with no parser: round
tripping is not a goal, which frees the format to be optimized for a human reading it.

## JAX

**What alloy keeps.** The AD model. `jvp` and `vjp` are the primitives, whole derivatives are
composed from them, and every transformation maps a graph to a graph — no tape, no recording, no
runtime. A derivative is the same kind of object as the thing it came from and gets the same
treatment downstream, which is what makes `hessian` simply be `jacobian` of `gradient`. The
thinking behind `map` is `vmap`'s: one callee applied across slices of its arguments, expressed
once.

**Where alloy departs.**

*Construction is explicit.* JAX traces Python functions, which is ergonomic and means the traced
object is a shadow of code you did not write for the tracer. Alloy builds the graph directly
through operator overloading and keeps your names on it. There is no tracing abstraction, nothing
to retrace, and no gap between what you wrote and what the compiler holds.

*The word "JIT" means something narrower here.* Alloy's just-in-time path is not `jax.jit`. There is
no decorator and nothing to opt into: a `Function` is already a graph, so the only thing deferred is
the compile, which happens on the first call and is cached on disk. Nothing is specialized on the
values you pass, so there is no retracing, no recompilation when a shape changes — a shape *is* a
different function — and no first-call penalty on a warm cache, even in a fresh process. It is the
same compiler the ahead-of-time path runs, invoked on demand; that is why what you test from Python
is what you ship as C. See [Code generation](../guide/codegen.md).

*`map` survives into the output.* A `vmap` is a batching rule that rewrites a computation into a
wider one. Alloy's `map` is preserved through lowering into a real loop in the generated C, and
through AD into a mapped derivative, so a hundred-stage horizon produces code of roughly constant
size rather than a hundred unrolled stages. Keeping that structure intact is most of why the
generated sources are small; see [the numbers](../results/scalability.md).

*Sparsity is first-class.* JAX has no real equivalent, and does not need one — dense batched
arithmetic on accelerators is the workload it was built for. Optimal control is the opposite: a
constraint Jacobian is mostly zeros with structure worth exploiting, so alloy carries structural
patterns, colors them, and emits compact nonzero values. That machinery is a large part of what
alloy is *for*.

## What is alloy's own

Three things are not borrowed from anywhere above.

**Colored sparse derivatives through preserved structure.** Computing a sparse Jacobian by coloring
a pattern is standard. Doing it on the *callee* of a `map` — coloring a small local tile, pushing
constant seeds through one derivative function, and mapping the result — is what keeps generated
derivative code from growing with the horizon.

**Solvers as graph nodes.** `al.qp(...)` and `al.nlp(...)` return real `Function`s whose body is a
`solver_call`, so a solve nests inside a larger graph like any other operation, and the whole
thing — oracles, wrapper, host function — compiles into a single shared library with no Python in
the loop. See [Solvers](solvers.md).

**A layering the tests enforce.** Which packages may import which is a table in a test rather than
a convention in a document, and the two sanctioned exceptions to it are named and checked. It is
not a compiler technique; it is the thing that keeps the compiler legible as it grows. See
[the architecture](architecture.md#layers).
