> Planned work, not a frozen record. One `#` section per refactoring; add new ones alongside
> rather than editing settled ones, and remove a section once it has landed.

# Function templates

Todo API-1, following the solver API that landed on dev. The interface sketch and its design
rationale live in [the typing playground](../../typing_playground/README.md#templates). The sketch
does not lower, compile, or evaluate expressions; its tests prove the proposed interface only.

`FunctionTemplate` owns a declaration with shape holes and produces concrete `Function` instances.
The compiler continues to consume concrete Functions. This keeps unresolved shapes out of the
expression graph and lets the existing call, differentiation, and compilation machinery remain
responsible for each instance.

The production implementation still needs decisions and verification in these places:

- Specialization keys and generated C names must distinguish every input property that changes
  the graph, including nested structure and relevant `TensorType` metadata. The sketch only
  mangles flat shapes. Decide how holes obtain dtype and differentiability before copying it.
- Trace each instance once, including the bare mode that infers declarations from a call.
  The sketch currently traces bare bodies twice.
- Lift the existing derivative wrappers over concrete instances. Seed and multiplier declarations
  must use the generated input types while preserving the source tree structure; a constant
  output's differentiability flag is not the multiplier's flag.
- Keep symbolic and numerical calls typed, preserve eager checks for fully declared templates,
  and cover actual compilation, cache reuse, and composition in addition to static assertions.
- Settle typed `vmap` separately. The sketch adds a leading axis to each leaf, while the real
  builder uses flattened slices, starts, and strides. Calling a template inside a traced body
  can instantiate it there; a consumer that inspects a callee needs a concrete instance.

The existing implementation is in `src/scaly/function/{tree,model,api}.py`; the template sketch
is `typing_playground/templates.py`. Naming and declaration shorthand remain the playground's
open questions. Overload sets and constraints on shape holes remain deferred.

# Shared compiler rewrites

The 2026-09-08 [investigation](algebraic_simplification_2026_09_08.md) updates C-12's proposed scope
and supersedes the size-only acceptance condition in "One matcher" above. The useful comparison
is now shared arithmetic across both dialects and both program forms. Combining the verifier's
tables with the matcher is optional. The investigation owns the source comparison and design;
C-52 owns the fixed pipeline package, C-12 the matcher, C-53 the arithmetic rules, and C-54 later
memory-aware cleanup. Their actionable status lives only in `internal/todo.md`.

Landed 2026-09-08 (C-12, C-13, C-53): `ir/match.py` carries the one iterative driver for both
dialects, `passes/program/_common.py` the program adapter, `passes/arith.py` the shared identities,
and `passes/program/fold_arith.py` applies them to loop bodies. The "One matcher" section that
preceded this one asked for the merge to land only if the code shrank; it did not (the driver grew
by about 40 lines for the generic protocol, replacement revisiting, and the step bound), and the
condition was superseded by this section's aim. `Spec` in `ir/spec.py` was left as is.

# Affine index maps for gathers and scatters

Landed 2026-09-09; kept because C-9 and C-56 link here for what was read and what was rejected.

Todo C-9. The AD rules build `GATHER` and `SCATTER` index arrays whose contents are affine in the
trip index, and `passes/lowering.py` emits each one as a `static const int64_t` table. On
unbumpercars C=32 those tables are 53.9 of the 54.8 MB of generated source, twelve of them
444,416 entries each, which is why the cell does not compile under the 50 MiB cap.

What was read in tinygrad, and what was taken. `uop/divandmod.py` (109 lines) and the div/mod part
of `uop/symbolic.py` are the affine folder the todo points at: gcd factoring, congruence folding
(`rem.vmin // c == rem.vmax // c` means the mod is affine on this range), `nest_by_factor`, and the
recombination `(x % c) + (x // c) * c -> x` that reverses reshape peeling. `codegen/simplify.py`
adds `simplify_merge_adjacent` and `pm_split_ranges`, which trade one loop range for `hi * c + lo`
and keep the result only when `count_divmod` did not increase.

Almost none of it is needed here as a *pass*. tinygrad needs a folder because its indices arrive as
already-built symbolic trees from reshape and permute, so the div/mod structure has to be recovered
after the fact. Scaly's tables arrive as concrete integer arrays. Recognising the affine structure
once, at lowering, and emitting the minimal expression is strictly cheaper than emitting `k` through
a chain of views and folding it back. Rejected, therefore: the `UPat`/`PatternMatcher` port
(`ir/match.py` already carries the one driver, and there is no tree to match), the div/mod folder as
a rewrite over expressions, congruence folding under range bounds (the array's own bounds are
exact), and the loop-merge/split pair, which is C-8's problem and needs a cost model Scaly does not
have.

The one rule that *is* needed is the recombination `(x % c) + (x // c) * c -> x`, and it is applied
at emission rather than as a pass. A coordinate written literally as `(k // stride) % dim` costs two
divisions, and emitting it that way cost race_cars about 5%. But `k // stride[i] // dims[i]` is
`k // stride[i - 1]`, so the coordinates telescope: the index is a combination of the plain
quotients `k // stride[i]` with coefficients `c[i] - c[i+1] * dims[i+1]`, and no level needs a
modulo at all. That is exact integer algebra for a non-negative trip index, it is nine lines inside
`index_at`, and it brings the runtime back to the table's. Only the residual table's own index
`k % len(residual)` survives, and there is no quotient to fold it against.

What was taken is the *shape* of the answer: an index is a base plus a sum of terms, one per range,
each a coefficient times a coordinate of the trip index. So the representation is one
`AffineIndexMap(dims, coeffs, residual)` in `passes/affine.py`, recovered from the array by greedy
factoring: at each level pick the smallest inner block length `m` dividing the remaining length such
that the reshaped rows differ by one constant offset, record `(d, delta)`, and recurse into the
first row. Whatever is left when no divisor works stays a table, indexed by `k % m`. A `residual` of
length one is the fully affine case and emits no table at all; `dims == ()` is the previous
behaviour unchanged. One representation covers the strided window, the multi-level `start + it * stride + j`
form, and the mixed case where an inner tile (`unique_j` in `ad/forward.py`) is genuinely arbitrary
but small and stage-invariant.

A size threshold was tried and reverted. Keeping tables below some length materialized protects a
small hot gather, but it makes the emitted shape depend on N, which
`test_vmap_sparse_hessian_c_source_is_constant_in_length` rejects and rightly so. With the
recombination in place there is nothing to protect.

The consequence is that no AD rule changes. `ad/forward.py` and `ad/reverse.py` keep building
concrete index arrays, which stay easy to read and to test against NumPy, and the structure is
recovered where it is needed. Views-as-index-expressions (C-8) is the case this does not cover,
because there the index is not a materialized array to factor.

# Open problems

These remaining limitations have follow-up entries in `internal/todo.md`. Their implementation
choices are not settled. Move an entry to its own `#` section once its design is decided.
Automatic differentiation is abbreviated AD below.

## Zero-input `Function`s, and the flat call seam that survives because of them

Found while landing the typed call surface, by trying to ban zero-input `Function`s and watching
the suite break. Two independent producers, both legitimate:

- **Differentiation.** `_call_jvp_function`, `_call_jvp_many_function` and
  `_call_jvp_many_const_function` in `ad/forward.py` each keep only the callee inputs the
  derivative actually depends on. A constant derivative depends on none, so the synthesized callee
  has no inputs at all — `duplicate_fw_second_x`, `scale_add_fw_y_x`,
  `bicycle_stage_interstage_fw_eq_znext` among others.
- **Solvers.** The oracle and bounds functions of any problem declared without parameters:
  `standalone_qp_oracle`, `settings_qp_bounds`.

Both lower to real C procedures taking no arguments, so neither is a mistake to delete.

What we are living with. `Function.__call__` dispatches on leaf kind, and an empty tree has no
`Expr` leaf, so it cannot be routed symbolically — it reads as an evaluation. That is the reason
`ad/forward.py` is the one module outside `function/` permitted to use `_flat_symbolic_call`,
pinned by `test_flat_call_seams_stay_inside_their_sanctioned_modules` in `tests/test_import_layering.py`.

What a fix looks like, and why it did not ride along with the API change. A `CALL` to a
no-argument procedure returning a constant is pure overhead; inlining the constant at the call site
would remove the AD producer, let the ban stand, and shrink the generated source. But it changes
what AD emits, so it regenerates the byte-for-byte C corpus (`tests/test_c_snapshot.py`,
`tests/baseline/c/`) — an AD and codegen change wearing an API cleanup's clothes. The solver side
needs its own answer either way: a parameterless oracle has nothing to take, so either a zero-input
signature stays legal or the descriptor stops building one.

## `vmap` and the AD entry points erase the callee's declared trees

`Function` is meant to be the unit of composition, and after the typed call surface it nearly is:
`fn(tree)` is typed on both the symbolic and the numeric side, and the derivative wrappers keep the
source's input tree, so `sc.gradient(fn, "f", "x")` is a `Function[SI, NI, Expr, np.ndarray]`.
Three holes remain, all on the paths that matter most for composing:

- **`sc.vmap` is untyped end to end.** `vmap(callee: Any, length: int, inputs: Any, output: int = 0)
  -> Expr` in `function/sugar.py`. The callee's declared trees are never read, the result is a bare
  flat `Expr` rather than the callee's output tree repeated, and an output is selected by integer
  index — the addressing-by-position that the declared trees removed everywhere else.
- **`sc.jvp`, `sc.jvp_many`, `sc.vjp` and `sc.vjp_many` are expression-level.** They take
  `Sequence[Expr]` and return `tuple[Expr, ...]`: typed, but tree-blind, with no Function-level
  spelling that preserves structure.
- **The callees AD synthesizes are `Any`-typed.** Every builder in `ad/forward.py` and
  `ad/reverse.py` goes through `Function._from_exprs`, whose trees come from `flat_tree`, so each
  result is a `Function[Any, Any, Any, Any]`. This already leaks into the call surface: both
  `__call__` overloads match `Any`, which is why their order in `function/model.py` is load-bearing
  and had to be commented.

A composition through `vmap` or the low-level derivative builders loses its declared tree types.
The public Function-level derivative wrappers preserve the source input tree. Extending that
property to mapped structure remains open (todo API-2). The playground README records this
limitation, and
`typing_playground/function.py` holds a candidate interface.

Whatever lands needs both kinds of test the tree work uses, because they catch different things:
runtime structure tests beside `tests/function/test_tree.py`, and static ones in
`tests/typing/test_arity.py`, which runs under `ty check --error-on-warning` where every
`ty: ignore` marks an expected error and an unused one fails the check.

## Compiled CasADi artifacts across worktrees

The API merge review reproduced four CasADi IPOPT test failures from cached libraries whose
runtime search paths pointed into a deleted worktree. All five tests in
`tests/benchmarks/test_casadi_ipopt.py` passed with a fresh `SCALY_CASADI_IPOPT_CACHE`.

`benchmarks/harness/casadi_ipopt.py` keys its cache on the serialized problem, options, CasADi
version, solver library content, compiler, and optimization flag. The key omits the library search
paths supplied by `backend_compile_flags`. Identical solver libraries in a new worktree therefore
reuse an artifact whose runtime search paths still name the old one.

A fix must either include those paths in the cache identity or remove the artifact's dependency
on worktree paths. Verify relocation after the original directory disappears; a successful load
while both worktrees exist does not exercise the failure. A separate cache is the temporary
workaround, not the intended behavior.

# Joint derivative callees and scalar rules

C-46 and C-49, 2026-09-09. The [operation audit](perf_2026_09_07/c49_ad_op_audit.md)
compares the same chain stage with CasADi SX's explicit forward-over-reverse construction. It also
isolates the cost of separate tangents for each formal input. That comparison supports joint
propagation and smaller scalar rules. Automatic expansion before differentiation remains deferred.

`ad/forward.py` seeds every active formal in one sweep for ordinary calls and generic mapped
calls. Constant seeds enter the same helper for ordinary calls. Mapped constant tiles and local
coloring retain their specialized seed counts, because replacing them with the global seed count
would increase work. Their derivative bodies share one concatenated output when the iteration
count and shared input slices agree. Gathers recover the separate results in their original order.

Packing is deliberate. The current VMAP lowering executes each selected output separately and
uses scratch storage for the other outputs. A helper with several outputs would therefore still
repeat the primal and adjoint work. One concatenated output shares that work through the existing
VMAP representation without adding a lowering pass or changing the public function outputs.
The helper cache remains weakly keyed by the source callee, with active formal indices and seed
values in the specialization keys. Derived bodies retain the source callee's lowering hint.

The scalar changes reuse the primal quotient in division derivatives, share the square-root
reciprocal across seeds, recognize elementwise squares and vector self-dots, and move negation
outside products and quotients. Matrix self-products retain both product-rule terms. Reverse
multiplication already returns the same expression for both operands of a square, so it needs no
special case. These rules follow the existing [arithmetic semantics](../../docs/how_it_works/lowering.md#arithmetic-semantics),
including their rounding and exceptional-value limits. Tests also cover finite inputs at scales
of `1e-200` and `1e200`, where squaring the denominator in the old rule loses the derivative.

# Mapped scalar ranges and derivative assembly

C-77 starts the range propagation part of C-8 in the Program dialect. Sources read are
`perf_2026_09_07/tinygrad_rangeify.md`, particularly the consumer agreement rule in sections 1
and 2, and the current scalarizer, scalar scheduler, mapped lowering, and elementwise fusion.
Scalarized call bodies expand back into a shared expression graph at the mapped call site.
Their pointer views retain the caller's offsets. Only the surviving output expressions are
scheduled, so discarded compressed derivative entries no longer keep arithmetic alive.

Assembly views already have integer index expressions in Program IR. Static index maps can
associate each demanded scalar with the mapped range that produces it. Consumers agreeing on
that range share the producer; incompatible demands retain storage. Writes and pointer aliases
bound propagation, and hoisted calls keep their original execution frequency.

Rejected are changes to derivative coloring, benchmark-specific output maps, generated-C edits,
and extending the single-store elementwise matcher with stage-specific cases. Those approaches
either change a different compiler layer or lose the producer's shared scalar expressions.
The full tinygrad scheduler and reduction machinery remain outside this first step.

### Periodic constants and invariant reciprocals

The September 22 performance study, section 2, identifies repeated constant tiles and repeated
loop-invariant divisions. The earlier tinygrad rangeify study, section 1, places constant-buffer
folding before materialization. Scaly will shrink load-only constant tables to their shortest
complete period and index them modulo that period. A period of one becomes a scalar. Comparison
uses the declared dtype's bytes, preserving signed zeros and NaN payloads. Aliases, calls and
non-load uses retain the original storage. This runs before fusion so constant masks are visible.

Reciprocal rewriting is explicitly opt-in. Multiplication by a reciprocal can round differently,
and the reciprocal itself can overflow or underflow when direct division would remain finite. A scalar pass hoists one reciprocal per divisor outside a statically
nonempty loop. It rejects divisors depending on loop variables, scalar assignments in the loop,
or buffer storage the loop can change. Nested loops are processed within their own scope so an
empty inner loop cannot introduce a speculative division.

## CPU build recipes for C79

The C79 task and the September 22 compiler survey specify one generated program with lane and
math-library options. `codegen/toolchain.py` will describe CPU flags, link flags, and the deployment
baseline in an immutable `BuildRecipe`. The existing AOT render context will retain that recipe so
headers, source comments, and link flags describe the same render. Compiler selection will not
change the generated program. Lowering receives only the lane and reciprocal options, preserving
the import layers. JIT build and invalidation will resolve the same native recipe.

The two render dialects remain independent of the C or C++ header language. Scalar libm stays the
AOT and headline-study default. Native JIT may select glibc vector libm only on a compatible host.
The recipe uses existing compiler discovery; replacing it with zig is the separate R71 task.

### Explicit mapped lanes

The September 22 performance study, sections 3.1 through 3.6, supplies the lane split and target
policy. The tinygrad rangeify study describes preserving independent axes through scheduling.
The widening pass will represent chunks and lanes as nested ranges. It retains a single scheduled
body, clamps the substituted stage index for inactive lanes, and bounds stores by the number of
valid lanes. Scalar C traverses valid lanes; GNU C evaluates arithmetic with vector values and
uses per-lane loads, stores and scalar library calls where a vector operation is unavailable.

Each widened range becomes one helper shared by full chunks and the remainder. Its width is a
fixed render choice or a compiler-selected macro, capped by peak simultaneous scalar liveness.
The budget uses 64-bit scalar slots, selected from the compiler target macros. AVX-512 has 256
slots, AVX and AArch64 have 64, SSE2 has 32, and a generic scalar target has 16. The cap permits a
fourfold allowance for spills. Local lane storage reserves eight elements, so changing the macro cannot change caller
workspace. Optional glibc vector calls require matching instruction-set and library-version guards.
Plain C emits scalar paired stores and omits GNU attributes and vector declarations.

Standalone reduction widening stages independent loads, then evaluates the original scalar
recurrence in lane order. It preserves summation order and scalar expression grouping, including
multiply-add contraction. Reductions nested under mapped lanes keep their original serial loops.

### Per-query sparsity masks across calls

The chain M9 study profile spends its first 45 seconds in structural Jacobian analysis before
rendering. Reviewed `ad/sparsity.py` and the unchanged baseline implementation: CALL and VMAP
restart the mask memo for every formal argument, repeating nested callee analysis. Reuse the
query's memo with `(expression id, differentiation-variable id)` keys so distinct formals retain
separate masks. Reject a global cache because query-local reuse fixes the repeated work without
retaining expression graphs or sparse matrices between queries. NumPy mask comparisons and a
count of uncached node/variable pairs will cover correctness and construction complexity.

The next profile reaches `fuse_ranges._rewrite`: its second rewrite walks every inserted scalar
subgraph again for each assembly output. Keep substitution unchanged and memoize the following
bottom-up arithmetic fold for the lifetime of one procedure fusion. The existing generic rewrite
driver has a per-call memo, so using it separately on each output cannot share this work. An optional caller memo on the shared rewrite driver avoids another traversal implementation.
It retains original nodes and results, and is valid only for unchanged patterns, rebuilding and
options across calls. Caller memoization rejects expression nodes because their lowering hints
depend on traversal provenance; the optimization serves Program nodes only.

### Active-width private lane storage

Private lane buffers reserve eight lanes in the workspace but index elements with the helper's
selected width. This keeps smaller widths contiguous without changing workspace upper bounds.
Resolve private alias chains to owner-buffer offsets before inserting the lane stride, and remove
the resulting unused alias declarations. Aliases of external buffers retain their layout by leaving
that candidate loop scalar. The stride uses the helper width even in a partial final chunk.
