# Compiler maintenance context

These implementation details were removed from the public compiler explanations during the
user-facing documentation rewrite. They are useful when changing the compiler, but are not
prerequisites for understanding or using Scaly. Check the named implementation before relying on
these notes; they describe the compiler at the time of the documentation rewrite.

## Rendering and recording

`codegen/aot.py` lowers a function once into a render context. `CModule` derives its header,
source, workspace size, and link flags from that context. Link flags remain lazy so rendering a
solver module to text does not itself require the native solver libraries.

`viz/recording.py` installs an observer factory in `codegen/aot.py`. Running the implementation
module directly with `python -m scaly.codegen.aot` can create a second module instance and observer
registry. The supported command is `scaly_codegen`, with `python -m scaly.codegen` retained as a
compatibility entry point.

## Mapped derivative construction

Forward-mode helper cache keys include the active formal indices, seed count, and specialized
constant seed values. Helpers omit primal and seed arguments that their expressions do not use.
For mapped constant seeds, `ad/forward.py` can specialize periodic seed tiles with periods up to
eight. Compatible specialized results can share a concatenated mapped output, with gathers
recovering the requested layouts.

The structured sparse Jacobian path in `ad/sparse.py` first tries mapped pieces whose outer
inputs match the differentiated expression. If per-formal contributions partition the nonzeros,
it concatenates their compact values and adjusts coordinate order. Overlapping contributions use
scatters and addition. Sparse Hessians instead use global star coloring and a recovery table.

`SCALY_STRICT_JVP_MANY=1` makes unsupported batched forward rules raise instead of falling back to
individual seeds. It is useful when checking that a new batched rule actually runs.

## Program optimization

Hoisted procedure names encode the invariant formal positions. An accumulator written by both
invariant and per-iteration statements cannot move wholly into the invariant prologue.

Automatic scalarization currently uses separate limits for procedure arithmetic, total program
size, and attempted expansion work. Explicit scalar requests bypass automatic limits. The
implementation in `passes/program/scalarize.py` owns those values.

Expression rewrites use the iterative driver in `ir/match.py`. Statement nesting and callee
recursion are separate from expression depth. `prepare_scalar` bounds emitted scalar expression
depth, while range expressions retain their original evaluation frequency.

The C renderer contains a targeted no-inline workaround in `_force_noinline_raw`. Read that
implementation before changing the raw-procedure declaration policy.
