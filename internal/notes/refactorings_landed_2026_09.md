# Landed refactorings, September 2026

Moved out of `refactorings.md` on 2026-09-29 once each had landed. Frozen: the text is as it was
when the work was planned or closed, and it records what was read and what was rejected. How the
code works today is in `docs/how_it_works/` and the module docstrings.

- Shared compiler rewrites: C-12, C-13, C-52, C-53.
- Affine index maps for gathers and scatters: C-9.
- Joint derivative callees and scalar rules: C-46, C-49.
- Mapped scalar ranges and derivative assembly: C-77, C-78, C-79.

## Shared compiler rewrites

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

## Affine index maps for gathers and scatters

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

## Joint derivative callees and scalar rules

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

## Mapped scalar ranges and derivative assembly

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

#### Periodic constants and invariant reciprocals

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

### CPU build recipes for C79

The C79 task and the September 22 compiler survey specify one generated program with lane and
math-library options. `codegen/toolchain.py` will describe CPU flags, link flags, and the deployment
baseline in an immutable `BuildRecipe`. The existing AOT render context will retain that recipe so
headers, source comments, and link flags describe the same render. Compiler selection will not
change the generated program. Lowering receives only the lane and reciprocal options, preserving
the import layers. JIT build and invalidation will resolve the same native recipe.

The two render dialects remain independent of the C or C++ header language. Scalar libm stays the
AOT and headline-study default. Native JIT may select glibc vector libm only on a compatible host.
The recipe uses existing compiler discovery; replacing it with zig is the separate R71 task.

#### Explicit mapped lanes

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

#### Per-query sparsity masks across calls

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

#### Active-width private lane storage

Private lane buffers reserve eight lanes in the workspace but index elements with the helper's
selected width. This keeps smaller widths contiguous without changing workspace upper bounds.
Resolve private alias chains to owner-buffer offsets before inserting the lane stride, and remove
the resulting unused alias declarations. Aliases of external buffers retain their layout by leaving
that candidate loop scalar. The stride uses the helper width even in a partial final chunk.
