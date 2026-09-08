# Algebraic simplification across Alloy's dialects

Design investigation, 2026-09-08. This note records source evidence, the scalarization closeout,
and proposed shared compiler rewrites. The current arithmetic contract is documented in
[lowering](../../docs/how_it_works/lowering.md). The program-pass pipeline now has an explicit
order. Shared rules and matcher work remain planned. No scoped math-mode API is proposed.

## Scalarization closeout

Review found that the original 2,048-element limit measured tensor width rather than generated
arithmetic. A local probe with
20 nested sine operations over 2,048 elements passed the automatic guards and expanded from
1.3 KB to 247 KB of C. A single Clang compilation rose from 0.053 seconds to 1.79 seconds.
Fable's independent review also reported an eligible body of about 200,000 scalar operations
taking 405 seconds in Clang. These are diagnostic measurements, not frozen benchmark results.

Automatic selection now limits each procedure to 4,096 unique scalar arithmetic operations after
folding and sharing. A separate preflight guard caps expansion work at 65,536 buffer elements and
executed stores, including callees. The program-wide growth budget is 16,384 arithmetic nodes,
assignments, and stores across selected procedures. Explicit `.scalar()` bypasses these budgets
but still requires supported control flow and eligible callees. Earlier explicit selections also
consume the growth budget for later automatic selections.

These conservative limits admit small race-car stages and leave the expensive chain stage for
explicit `.scalar()`. Boundary and stress fixtures pin operation count, aggregate growth, and
expansion work separately. The [closeout measurements](perf_2026_09_07/README.md#c-44-closeout)
record the minimal runtime and cold-compilation comparison. A full benchmark rerun follows more
Track C work, as requested. Measurements follow the
[existing protocol](../../docs/results/fairness.md#the-measurement-protocol).

C-44 owns scalarization and disclosure of the current algebraic contract. C-52 owns the completed
package split with fixed pass order. Shared rules and matcher work follow as C-12/C-53. Task status
lives in `internal/todo.md`.

## Library evidence

Two Luna agents independently inspected CasADi and tinygrad. CasADi probes used the installed
3.8.0 release. Tinygrad inspection and probes used the clean local checkout at upstream revision
`69915d61c233182deb34eab3680be93729c2eff5`, dated 2026-09-07. The tables map numerical
rule families relevant to Alloy and identify additional structural passes. They are not an
enumeration of every backend instruction-selection pattern in either library.

### CasADi 3.8.0

Both SX and MX use shared scalar simplification functions in
[`calculus.hpp`](https://github.com/casadi/casadi/blob/3.8.0/casadi/core/calculus.hpp#L1698-L1982).
The following expressions were also checked with isolated Python probes. Here `x` is symbolic,
and construction-time simplification is enabled.

| Input | Constructed result in SX and MX | Assumptions or consequence |
| --- | --- | --- |
| `x-x` | `0` | Erases NaN and infinity subtraction. |
| `x/x` | `1` | No proof that `x` is nonzero or finite. |
| `0*x`, `x*0`, `0/x` | `0` | Suppresses invalid runtime arithmetic, including zero denominator. |
| `x+0`, `x*1`, `x**1` | `x` | Signed-zero behavior is not preserved generally. |
| `x**0` | `1` | Does not evaluate the base at runtime when no other use remains. |
| `x/(2*x)`, `x/(-x)` | `0.5`, `-1` | Syntactic quotient cancellation without a nonzero proof. |
| `sqrt(x*x)` | `fabs(x)` | Can avoid overflow in the intermediate square. |
| `log(exp(x))` | `x` | Can avoid overflow or underflow in the exponential. |
| `sin(x)**2 + cos(x)**2` | `1` | Uses the real trigonometric identity rather than rounded evaluations. |
| `x+x`, `x*x` | Twice `x`, square of `x` | Specialized operation construction. |

Source inspection additionally found double-negation and double-inverse elimination, cosine
evenness, square/absolute-value normalization, local additive cancellation and coefficient
combination, and expansion of constant integer powers up to magnitude 100. These are local
patterns with structural matching; the source does not implement unrestricted polynomial
normalization. Some transformations change ordinary finite rounding as well as exceptional
values. See the unary, additive, multiplicative, and power sections of `calculus.hpp` above.

The process-global
[`setSimplificationOnTheFly`](https://github.com/casadi/casadi/blob/3.8.0/casadi/core/global_options.hpp)
switch defaults to true. Disabling it preserved `x-x`, `x/x`, `0*x`, `0/x`, `x**0`,
`sqrt(x*x)`, and `x/(2*x)` in both types. However, MX still reduced `log(exp(x))` in the
probe while SX retained it. This switch is not a comprehensive floating-point semantics mode.

Literal invalid constants expose different construction paths. A second independent probe
confirmed these default results:

| Constant expression | SX | MX |
| --- | --- | --- |
| `T(0)/T(0)` | Positive infinity | NaN |
| `T(inf)*T(0)` | Zero | NaN |
| `T(0)/T(nan)` | Zero | NaN |

Here `T` is the type in the column. These are observed implementation details, not behavior
recommended for Alloy. SX also canonicalizes a literal negative zero to its cached zero node.
[`sx_elem.cpp`](https://github.com/casadi/casadi/blob/3.8.0/casadi/core/sx_elem.cpp).

Sparse structure participates in simplification. MX combines input sparsities using operation
properties describing whether a zero operand produces a zero result. An empty structural matrix
multiplied by or divided by a symbolic matrix can remain structurally empty; addition to a
diagonal matrix retains the diagonal structure. Structural absence is distinct from a stored
numeric zero. The implementation also contains zero and identity matrix-product rules.
[`mx_node.cpp`](https://github.com/casadi/casadi/blob/3.8.0/casadi/core/mx_node.cpp#L734-L813),
[`mx.cpp`](https://github.com/casadi/casadi/blob/3.8.0/casadi/core/mx.cpp#L658-L677).

Explicit `simplify` and function expansion are separate graph transformations. Function
construction supports `cse=True`; its default is false. In the agent's SX probe with repeated
`sin(x)*cos(x)` and construction simplification disabled, enabling CSE reduced instruction count
from seven to six. `live_variables` controls temporary storage reuse, not the algebraic rules.
Function compilation traverses the graph reachable from outputs. Code generation then emits the
resulting function algorithm. These mechanisms are distinct from construction-time identities.
[`sx_function.cpp`](https://github.com/casadi/casadi/blob/3.8.0/casadi/core/sx_function.cpp#L323-L476).

CasADi 3.8.0 also has an ordered `Function.transform` pipeline. The source defines `simplify`,
`expand`, and `external` verbs. Simplification tasks include `empty_inputs`, `combine_terms`,
`cse`, `ref_count`, and `const_folding`. The dictionary shorthand enables all of these except
`combine_terms` by default. This CSE default belongs to explicit transformation, unlike the
function-construction option above. `combine_terms` performs broader term collection, so the
description of local construction rules must not be generalized to all explicit transformations.
These details are source-backed, rather than inferred from the construction probes.
[`function.cpp`](https://github.com/casadi/casadi/blob/3.8.0/casadi/core/function.cpp#L326-L447).

### Tinygrad at the pinned revision

The main rule sets are `symbolic_simple`, `symbolic`, and `sym` in
[`uop/symbolic.py`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/uop/symbolic.py).
They are applied at multiple compilation stages. Their ordinary algebraic identities are not
guarded by a general floating fast-math switch in the inspected implementation.

| Family | Concrete rule or behavior | Conditions |
| --- | --- | --- |
| Neutral elements | `x+0`, `x*1`, integer quotient by one, shifts by zero | Patterns distinguish operations and dtypes where needed. |
| Cancellation | Floating `x/x -> 1`, `(x*y)/y -> x` | Source explicitly acknowledges failure when the cancelled operand is zero. |
| Zero multiplication | Symbolic `x*0 -> 0` | Known bare NaN or infinity constants instead produce NaN. Loaded exceptional values are not protected. |
| Zero division | Symbolic `0/x -> 0` in the probe | Constant evaluation keeps `0/0 -> NaN` and `0/nan -> NaN`. |
| Subtraction | Symbolic `x-x -> 0` in the probe | Subtraction uses addition and negation, rather than a separate subtraction pattern here. |
| Coefficient combination | `x*a+x*b -> x*(a+b)`, `x+x -> x*2` | Local floating rewrites can change rounding and overflow. |
| Reassociation | `(x+c1)+c2 -> x+(c1+c2)` and analogous associative operations; `(x/y)/z -> x/(y*z)` | Quotient pattern excludes identical `y` and `z`; not generally IEEE-equivalent. |
| Distribution | Some product-over-sum transformations | Several patterns are restricted to weak integers, rather than general floats. |
| Self comparison | `x<x -> false`; `x!=x -> false` | Inequality-to-self folding is restricted to integers and booleans, preserving the float NaN test. |
| Selection | Equal branches, constant conditions, nested compatible conditions | Invalid-index handling precedes ordinary zero folding. |
| Powers | Exponent zero, reciprocal for negative exponents, integer powers by squaring, half-integer powers through square root | Positive constant bases can become `exp2(x*log2(base))`. |
| Casts | Remove identical casts; collapse lossless cast chains; fold constants and compatible bitcasts | Dtype, representability, and bit-width conditions apply. |
| Bounds | Constant result when lower and upper bounds agree; eliminate dominated maximum operands | Requires interval facts. |

The agent's probes confirmed symbolic `x-x`, `x/x`, `0*x`, `0/x`, and `x**0` becoming
`0`, `1`, `0`, `0`, and `1`. Constant probes produced NaN for `nan*0`, `inf*0`, and
`0/0`, and preserved negative zero for `0/(-inf)`. Thus symbolic identities and evaluation of
known constants deliberately have different behavior at exceptional values.

The symbolic probe used `UOp.variable("x", 0, 10, dtypes.float32)` and `UOp.simplify()`,
which applies `symbolic` plus constant-cast folding. The interval includes zero. The claim about
unknown loaded NaN values comes from the unconditional pattern and its source comment, not from
evaluating an infinite value outside that variable's declared interval. Integer `/` promotes to
floating division and did not produce the same simplified graph as integer `//` in the probe.

Additional source mappings explain which mechanisms generalize:

| Mechanism | Source and limits |
| --- | --- |
| Integer index algebra | [`uop/divandmod.py`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/uop/divandmod.py) uses sign, divisibility, congruence, and interval conditions for quotient/remainder rewrites. Floor division is not interchangeable with Alloy's C-style truncating integer division. |
| Constant evaluation and interning | [`uop/ops.py`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/uop/ops.py) provides `exec_alu`, structural interning, and graph traversal. The symbolic folder retains mathematical integer constants until a later width commitment; this is tinygrad's representation choice. |
| Loop simplification | [`codegen/simplify.py`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/codegen/simplify.py) merges or splits ranges and collapses range-independent reductions. These transformations need range and use information. |
| Transcendental evaluation | [`codegen/decomp/transcendental.py`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/codegen/decomp/transcendental.py) has explicit exceptional-value cases and approximation algorithms. A bounded-input sine shortcut is separate from the normal path. |
| Target decomposition | [`codegen/decomp/op.py`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/codegen/decomp/op.py) and [`codegen/__init__.py`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/codegen/__init__.py) choose native operations or decompositions. `TRANSCENDENTAL` and `DISABLE_FAST_IDIV` control particular lowering choices, not a blanket floating algebra policy. |
| Dead graph and effects | Traversal starts from required roots. `AFTER` dependencies retain stores, calls, barriers, and control-related nodes; invalid-index stores can become no-ops. Effect dependencies make graph reachability meaningful. |

Tinygrad's unified `UOp` makes the same rewrite machinery usable before and after lowering.
It still has separate movement, index, range, buffer, and target-specific rules. Its reuse comes
from shared representation plus explicit dependencies, not from every operation having identical
semantics at every stage.

## Proposed Alloy arithmetic policy

Always-on symbolic identities are a defensible default for this compiler. Both libraries provide
precedent, but neither is a specification for Alloy. Document the selected rule families and test
them directly. Do not describe them as preserving IEEE 754 exceptional values or signed zero.
Some identities can also avoid overflow, underflow, or intermediate rounding for finite inputs.

Start with neutral elements, zero annihilation, syntactic self-cancellation, negation normalization,
and bounded constant-power simplification. Consider coefficient combination separately because
it changes rounding on ordinary inputs. Inverse transcendental identities and changes to reduction
order are separate decisions; the existence of those rules upstream does not require adopting them.

Prefer evaluating all-known constants with defined dtype semantics before applying symbolic
identities. In particular, do not copy SX's observed literal `0/0 -> inf`. Preserve a known invalid
operation as a runtime operation if the folder cannot faithfully evaluate or represent its result.
For a symbolic denominator, an explicitly documented `0/x -> 0` rule can still apply.

Algebraic simplification permits particular value substitutions. It does not authorize deleting
required call effects, ignoring aliasing, changing index-division semantics, or removing precision
boundaries. No graph hint or global `-ffast-math` flag is needed to express this chosen default.

## The architectural decision

Alloy can share arithmetic rules and rewrite machinery across its expression and program dialects
without adopting a general compiler framework. It cannot make every optimization independent of
types, shape, memory, and execution order. Those facts determine whether a rewrite is valid.

Use **expression dialect**, the established project term, for the mathematical tensor graph.
**Loopy code**, or **loop form**, retains buffers and loops. **Scalarized code**, or **scalar form**,
describes expanded scalar calculations. These terms are defined in the lowering documentation
and the lowering-mode docstrings. They describe the existing program dialect, not new dialects.
No identifier rename is part of this investigation.

## What already exists

- `src/alloy/ir/match.py` provides an operation-indexed pattern matcher and a bottom-up,
  memoized graph rewrite. Its implementation currently depends on `Expr` and `ExprOp`.
- `src/alloy/ir/expr.py` and `src/alloy/ir/program.py` both intern structurally identical nodes.
  This shares graph representation. Sharing execution additionally requires valid value lifetimes.
- `src/alloy/passes/expr.py` contains algebraic identities, tensor constant evaluation, and
  common-subexpression elimination (CSE), including normalization of commutative operands.
- `src/alloy/passes/program/scalarize.py` separately implements scalar arithmetic folding, substitutes
  buffer contents, and schedules the reachable output graph into scalar declarations and stores.
- `src/alloy/passes/program/_common.py` contains another graph walker and rebuilder.
  `_prune_dead_buffers` in `program/fuse_elementwise.py` removes unused declarations;
  it is not general dead-code elimination (DCE).
- `LowerCtx.emit_elementwise` in `src/alloy/passes/lowering.py` already places ordinary
  `ProgramNode` arithmetic inside loop bodies. Scalar expansion is unnecessary for rewriting
  these arithmetic trees.

The duplication to remove first is arithmetic folding and traversal. Constant-buffer propagation,
alias analysis, and execution scheduling are distinct responsibilities.

## A small common implementation

### Share local arithmetic rules

Express each shared identity once, with two small dialect adapters for inspecting operands and
constructing results. The adapter must expose operation, dtype, constant values, and result shape
where relevant. An identity returning an operand must preserve the result type and broadcasting.
Scalar constants and tensor constants need different materialization; their arithmetic policy
should agree.

Keep tensor-specific rules, such as composing slices and reshapes, with expression passes.
Keep loop- and buffer-specific rules with program passes. Do not introduce a third node language
merely to implement common arithmetic rules.

Constant evaluation must use the operation's dtype semantics. Python integer arithmetic and
host double-precision evaluation are not universal implementations of integer and float32
operations. Algebraic permissions do not authorize removing rounding or truncation boundaries.

### Share the rewrite driver

Generalize the existing matcher and iterative traversal just enough to rebuild either node type.
Preserve graph sharing, use deterministic rule ordering, and bound repeated rewrites. Keep
whole-array constant evaluation bounded so simplification cannot materialize enormous constants.
The generic driver owns traversal; each dialect owns its node reconstruction and structural rules.

Tinygrad provides a concrete reference for this change. At the pinned revision,
[`PatternMatcher`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/uop/ops.py#L1513-L1540)
indexes patterns by root operation, rejects impossible child-operation matches early, combines
matchers, and passes optional context to rewrite functions. `UPat` supports nested operands,
named captures, repeated captures, dtype restrictions, constants, and operand permutations.
Alloy's current `Pattern` instead delegates the match to an arbitrary predicate on an `Expr`.

[`RewriteContext`](https://github.com/tinygrad/tinygrad/blob/69915d61c233182deb34eab3680be93729c2eff5/tinygrad/uop/ops.py#L1690-L1802)
uses an explicit work stack and a replacement map to preserve graph sharing. Its full driver
visits newly created replacement subgraphs. A separate walk mode deliberately does not revisit
them. It also controls traversal into call bodies and includes cycle/stack checks. These are
specific capabilities to assess for Alloy, rather than a proposal to make `match.py` abstract
without a use case. Start with nested matching, dialect rebuilding, shared replacements, and
termination checks. Compiled pattern matching and detailed rewrite tracing can wait for measured
need. Tinygrad's matcher itself still assumes `UOp`; copying it unchanged would not support both
Alloy node types.

New affine and integer-index rewrite families remain with C-8 and C-9. The common arithmetic
machinery must preserve existing index semantics, but does not need those new analyses first.

Run the arithmetic rules on expression graphs and again on program arithmetic, including loop
bodies. Expansion, fusion, and index substitution expose new opportunities, so targeted cleanup
after these transformations is useful. Avoid running every expensive analysis after every pass.

The same rules do not imply the same opportunities. A tensor containing some zero entries may
become individual zero constants only after indexing or expansion. A symbolic index into that
tensor cannot always reveal the same facts without retaining a conditional or splitting a loop.

### Separate value cleanup from memory optimization

For a pure value graph, DCE follows reachability from required outputs. CSE can reuse equal
operations on equal immutable operands. Operations with required effects must also be roots.

This is easiest before scalar scheduling introduces named `ASSIGN` and `VAR` nodes. After that,
even scalar form needs a definition-to-use map to trace a variable back to the assignment that
produces it. A common node interface cannot substitute for those missing value dependencies.

Program statements need additional treatment. For example:

```text
a = load(p[0])
store(p[0], 7)
b = load(p[0])
```

The two load nodes can have identical syntax and object identity in Alloy today. Their values
are different. A variable name reused in a loop has the same problem. A shared arithmetic matcher
does not establish that a load or variable denotes the same value across statements.

This is a constraint on new optimization passes, not a demonstrated stale-load bug in the current
scalarizer. `_Frame.run` evaluates each store with a fresh memo and resolves loads against current
buffer contents. The existing repeated-store/alias test now uses the same interned load before
and after a write and checks that the two captured outputs are 16 and 99. A diagnostic mutation
that reused the memo across stores made the test fail. The renderer also emits a load at each
statement occurrence; interning the description does not eliminate either execution.

Start program CSE within straight-line scopes, invalidate memory-dependent candidates at writes,
and treat calls conservatively. Reuse existing buffer-reference and alias helpers. Retain output
stores and required calls. Delete a private-buffer producer only when all of its effects are
proved unobservable. Removing an unused scalar declaration is easier than removing a store loop.

A small pure-operation classification plus read/write-buffer information is enough for this
conservative starting point. Moving loads across loops, forwarding stores through aliases, or
sharing arbitrary loop-carried calculations requires stronger dependence analysis. Converting
those calculations to single static assignment (SSA) values could help later, but is not a
prerequisite for shared arithmetic simplification now.

## What to borrow from MLIR

MLIR's canonicalizer combines a common bounded rewrite driver with operation-specific patterns
and constant-folding hooks. That separation is useful here. Its CSE pass separately consults
memory-effect information. These are two distinct reusable mechanisms, not one universal
algebraic rewrite. Alloy can adopt that separation while retaining its two node types and explicit
pipeline. [Canonicalization](https://mlir.llvm.org/docs/Canonicalization/),
[CSE](https://mlir.llvm.org/docs/Passes/#-cse).

The proposed first implementation does not need dialect registration, a generated operation
definition language, arbitrary control-flow regions, an analysis-invalidation framework, or a
configurable pass-manager language. The substantial work is testing semantic consistency and
respecting memory effects, rather than building a framework.

## A usable first implementation

The `alloy.passes.program` package split preserves the existing pass order in an explicit pipeline.
Shared arithmetic folding can then serve both existing dialects, with cleanup
applied to loop form as well as scalar form. Pure-value cleanup and conservative program liveness
are subsequent, separately reviewable changes. Stronger memory optimization should follow an
observed workload need.

Validation needs the same arithmetic cases in expression graphs, loop bodies, and expanded
procedures; mixed constant tensors; different dtypes; repeated stores through aliases; loop-carried
values; and pure versus opaque calls. Exceptional-value cases should pin the chosen algebraic
contract instead of assuming strict IEEE 754 propagation. Benchmark measurements must separately
check graph construction, lowering, C compilation, and execution time.

## Implementation status, 2026-09-08

C-53 and C-10 landed the same day, after C-12. `passes/arith.py` holds the shared rules over an
`Arith` adapter (operation kind, constant value, shape/dtype fit, constant materialization, node
construction) with one instance per dialect. The expression pass, `scalarize`, and the new
`fold_arith` loop-body pass call the same `fold`. Integer constant evaluation truncates toward zero
and refuses results outside the dtype; float evaluation refuses division by zero, invalid
operations, and overflow but accepts underflow. `0 / x -> 0` is applied as documented above, so a
signed zero from a negative denominator is not preserved. The matrix forms of matmul-with-ones were
reverted after review showed them slower in loop form; see C-10 in the todo. Independent reviews
(one Fable, one GPT through `codex`) found no ordering error in the residue-class seed assembly or
in the rewrite driver's memo.
