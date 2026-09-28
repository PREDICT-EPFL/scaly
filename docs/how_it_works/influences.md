# Influences

This page outlines the main similarities and differences between Scaly and the
projects that influenced it most. Early prototypes were built directly on
top of some of them. We eventually built our own stack from scratch because
none offered the combination of features and flexibility we needed.

## CasADi

[CasADi](https://web.casadi.org)[^casadi] is probably the most widely used
library in the optimal-control community, and the closest to Scaly in scope.
Scaly borrows several of its ideas:

- **`Function` as the central object.** The unit of composition,
  differentiation and code generation is not the expression graph itself but a
  named wrapper around it. The wrapper fixes the names and shapes of the inputs
  and outputs, which is what lets the generated code be fully specialized.
- **Sparse derivatives by coloring.** Structural sparsity patterns are
  propagated through the graph and colored to compress the evaluation of
  Jacobians and Hessians, as [Differentiation](autodiff.md) describes.
- **The C calling convention.** The generated entry point has the signature of
  CasADi's generated C, and it can carry CasADi's metadata queries too, as the
  [code generation guide](../guide/codegen.md#casadi-compatible-exports) shows.
  Tools built around CasADi's generated code, such as
  [acados](https://docs.acados.org/), can then call a Scaly function.

It differs on two core points:

- **No global `SX`/`MX` split.** CasADi asks you to build a model from `SX`, a
  graph of scalar operations, or `MX`, a graph of matrix operations and calls,
  and that choice holds for the whole model. Scaly did not borrow it. It has
  only `Expr`, which keeps arrays and calls like `MX`, and each node carries a
  local lowering hint in `Expr.lowering`. The compiler decides per function
  whether to expand it into scalar statements, which is what `SX` gives you,
  and `.scalar()` or `.block()` on an expression overrides that decision where
  it is used. [Lowering and optimization](lowering.md#loops-or-scalar-code)
  describes the policy.
- **Every numerical evaluation is compiled.** By default CasADi evaluates a
  function with its own virtual machine, and compiled C is an option. Scaly
  has no interpreter. A numerical call always generates and compiles C.

## Triton, Warp and Numba

[Triton](https://github.com/triton-lang/triton)[^triton],
[NVIDIA Warp](https://github.com/NVIDIA/warp) and
[Numba](https://numba.pydata.org)[^numba] are also embedded domain-specific
languages (eDSLs) that compile decorated Python functions. Triton and Warp
write GPU kernels. Numba compiles numerical Python for the CPU through LLVM,
and for GPUs as well. The differences with Scaly are:

- **They translate the function's code, Scaly traces it.** Triton walks the
  function's syntax tree and emits Triton IR, a set of MLIR dialects. Warp
  parses the source and generates C++ or CUDA, together with an adjoint kernel
  for reverse-mode differentiation. Numba compiles the function's bytecode.
  Python control flow in these functions becomes control flow in the output,
  so a `for` loop generally stays a loop. Warp unrolls loops whose bound is a
  constant of at most 16 iterations, and Triton unrolls `tl.static_range`.
  Scaly runs the body once with symbolic inputs, as
  [Compiler architecture](architecture.md#tracing) describes, so a Python loop
  is always unrolled. A loop in Scaly's output comes from `sc.vmap`, a
  reduction or an array operation.
- **The unit of work.** A Triton or Warp kernel describes what one program
  instance or thread does, and you choose the launch grid. A Numba function is
  ordinary imperative code with its own loops. A Scaly `Function` describes a
  whole calculation on fixed-shape arrays, and the compiler chooses the loops.

## JAX

[JAX](https://github.com/jax-ml/jax)[^jax] and Scaly both trace a Python
function into a graph[^jax-tracing], both name their derivative building blocks
`jvp` and `vjp`, and in both a derivative is a function you can call, compose or
differentiate again. The main differences are:

- **The output is C, not an XLA program.** Scaly compiles to C source with a
  pointer interface that other C code can call without any runtime.
- **Sparsity is structural and first-class.** JAX's `jacfwd` and `jacrev`
  return dense arrays. Scaly finds which entries of a Jacobian or Hessian can
  be nonzero and computes only those, which matters for the mostly zero
  constraint Jacobian of a long horizon. The
  [sparsity guide](../guide/sparsity.md) shows the compact outputs.

## tinygrad

[tinygrad](https://github.com/tinygrad/tinygrad) is a small deep-learning
framework whose compiler is written the same way as its tensor library, as
rewrites over one kind of node. Scaly's IR machinery follows it closely:

- **One node class per dialect.** tinygrad represents every operation as a
  `UOp` tagged with an `Ops` value and interns equal nodes. `Expr` with
  `ExprOp` and `ProgramNode` with `ProgramOp` follow the same pattern.
- **Op-indexed pattern matching and graph rewrite.** `PatternMatcher` and
  `rewrite` in `ir/match.py` follow tinygrad's `PatternMatcher` and
  `graph_rewrite`. Patterns are grouped by operation, so a node is only tested
  against the patterns for its own operation, and one bottom-up walk rebuilds
  the graph while keeping shared nodes shared. Scaly's patterns are an
  operation and a Python predicate rather than tinygrad's structural `UPat`
  trees.
- **Verifiers as rule tables.** `Spec` in `ir/spec.py` indexes verification
  rules by operation, like tinygrad's `spec.py`, down to the split into a
  shared table (`spec_expr_shared`) and one table per dialect.
- **Small registries.** The `sc.dtypes` registry and the program dialect's loop
  kinds, modelled on tinygrad's `AxisType`, come from the same source.

## MLIR

[MLIR](https://mlir.llvm.org)[^mlir] is the LLVM project's framework for
compilers built from several intermediate representations. Scaly takes two
ideas from it:

- **Dialects and progressive lowering.** A program moves through separately
  defined sets of operations, each with its own verifier, and each lowering
  step goes one level down.
- **The textual assembly format.** The listings printed by
  `sc.render_expr_assembly` and `sc.render_program_assembly` follow MLIR's
  shape, `%n = dialect.op(operands) {attributes} : type`, with `tensor<...>`
  and `memref<...>` types. Scaly only prints this format and cannot parse it,
  and the listings are not valid MLIR.

[^triton]:
    Philippe Tillet, H. T. Kung and David Cox, "Triton: an intermediate
    language and compiler for tiled neural network computations", MAPL 2019,
    <https://doi.org/10.1145/3315508.3329973>.

[^numba]:
    Siu Kwan Lam, Antoine Pitrou and Stanley Seibert, "Numba: a LLVM-based
    Python JIT compiler", LLVM-HPC 2015, pp. 1-6,
    <https://doi.org/10.1145/2833157.2833162>.

[^casadi]:
    Joel A. E. Andersson, Joris Gillis, Greg Horn, James B. Rawlings and
    Moritz Diehl, "CasADi: a software framework for nonlinear optimization and
    optimal control", Mathematical Programming Computation 11, 2019,
    <https://doi.org/10.1007/s12532-018-0139-4>.

[^jax]:
    James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash
    Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake
    VanderPlas, Skye Wanderman-Milne and Qiao Zhang, "JAX: composable
    transformations of Python+NumPy programs", 2018.

[^jax-tracing]:
    Roy Frostig, Matthew James Johnson and Chris Leary, "Compiling
    machine learning programs via high-level tracing", SysML 2018.

[^mlir]:
    Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis,
    Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache and
    Oleksandr Zinenko, "MLIR: scaling compiler infrastructure for domain
    specific computation", CGO 2021,
    <https://doi.org/10.1109/CGO51591.2021.9370308>.
