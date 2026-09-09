# Where the Hessian kernels lose time, 2026-09-07

An exploration note, not a plan. It takes the 2026-09-05 sweep cells apart, measures where each
Alloy kernel spends its time against the fastest CasADi encoding, and tests candidate fixes by
hand-editing the generated C and by small throwaway edits to the compiler. The tasks that follow
from it live in `internal/todo.md` (C-43 to C-49, BH-48). This note owns the evidence.

Everything was measured on the reference machine, idle, `performance` governor, boost off, with the
sweep's own compile line (`clang++ -O3 -std=c++17`, no `-march`). Numbers are Google Benchmark
process means or best-of-five loops from a small driver that includes the generated `.c` directly
and calls the static stage functions. Every rewritten kernel that reports a time passed the cell's
dense-reference correctness check, or was compared bit-for-bit against the original.

Scripts in this directory:

- `run_cell.py <workload> <size> <backend> <tag>` regenerates, compiles, checks and times one cell
  from the current worktree into `/tmp/perf/cells/`. The whole edit-regenerate-measure loop is
  about ten seconds for race-car and npmpc.
- `bench_cell.sh <celldir> <kernel.c> [cflags]` recompiles one cell's kernel with extra flags and
  reruns the cell's own benchmark binary.
- `scalarize_stage.py <kernel.c> <outname> <seedarg|NOSEED> [function]` interprets one generated
  stage function symbolically, inlining nested calls and unrolling loops, folds `0*x`, `1*x`,
  `x+0`, constant arithmetic and optionally the constant seed tile, hash-conses the result and
  emits straight-line scalar C. It is the prototype of the scalarization pass proposed below.
- `experiments.patch` is the throwaway compiler diff whose numbers are quoted here, including the
  2026-09-08 division rules. It
  is left applied in the worktree that produced this note and is not meant to merge as is.

## Summary

The C track in `todo.md` attributed the runtime gap to the glue loops between stage calls, the
materialized `N × width` intermediates and the index tables (C-8, C-9). Measured, the glue is 22%
of race-car, 3% of npmpc and 26% of chain. The stage kernels themselves carry the loss, for three
different reasons on the three problems:

| Problem, size | Alloy | Best CasADi | Where Alloy's time goes | Main cause |
|---|---:|---:|---|---|
| race-car N=50 | 32.7 µs | SX 21.4 | eq stage 23.0, glue 7.3, cost+corridor 2.4 | tensor-shaped code for a scalar body: no scalar CSE, 0/1 seeds multiplied at runtime; libm is 7.7 of the 23.0 |
| npmpc N=12 | 82.8 µs | MX 51.1 | two MLP stage kernels 79.8, glue 2.8 | 204 matrix-vector products emitted as serial dot-product reductions: 61 of the 83 µs |
| unbumpercars C=8 | 3485 µs | MX 9950 | matvec nests with inner trip counts 128 and 256 | same as npmpc |
| chain M=5 | 2581 µs | SX 96 | eq stage 1920, glue 660 | 6340-line stage body with 1624 loops over 24 dense seed vectors; the glue zero-fills 43 buffers of 23,544 doubles per call |

What the throwaway compiler edits achieved, correctness checks passing:

| Cell | Before | After | Change |
|---|---:|---:|---|
| npmpc N=12 | 83.2 | 45.7 | matmul loop order by layout + `A.T @ v -> v @ A` fold |
| unbumpercars C=8 | 3485 | 827 | same |
| race-car N=50 | 32.6 | 31.4 | constant seed tile baked into the VMAP callee |
| race-car N=500 | 329 | 309 | same |
| chain M=5 | 2600 | 2458 | same; the real fix is scalarization, see below |

What hand-rewriting the generated C showed is still on the table:

| Kernel piece | Generated | Hand-rewritten | How |
|---|---:|---:|---|
| race-car eq stage, 50 calls | 23.0 µs | 18.1 µs | scalar SSA with hash-consing and folded seeds (20.2 with seeds kept symbolic) |
| chain eq stage, 40 calls | 1910 µs | 247 µs | same rewrite, bit-identical output (298 with seeds symbolic) |
| npmpc, whole kernel | 82.7 µs | 47.2 µs (38.6 with `-march=native`) | every matvec in contiguous column-sweep order using the transposes the kernel already builds |

The full suite with the patch applied: 405 passed, 1 xfailed, 2 failed, both
`tests/test_c_snapshot.py` pins of the generated C text.

## Method

Each cell directory under `benchmarks/results/followup/2026-09-05/sweep/` and the pilot's
`benchmarks/results/pilot/2026-09-05/` keeps the generated `.c`, the sample inputs, the expected
dense Hessian, `benchmark.cpp` and `compile.log`. Copying a cell and rerunning its compile line
reproduces the recorded timing to within 1% (race-car N=50: 32.6 recorded, 32.6 to 33.4 rerun; SX
21.4 both). From there:

1. A driver that `#include`s the kernel `.c` times the entry point and each `static noinline`
   stage function in isolation with the real per-stage arguments. The difference is the glue.
2. Ablation: delete one class of loops from the `.c` (stage calls, matvec nests, transposes,
   transcendental calls replaced by identity) and time what remains, without the correctness check.
3. Instruction histograms of the compiled object with `objdump -d` and microbenchmarks of the
   isolated loop shapes.
4. `scalarize_stage.py` to see how the same stage looks as straight-line scalar code.
5. Compiler edits, regenerated through `run_cell.py`, checked by the harness.

`perf` could not open events from this session (`perf_event_paranoid = 4`, see the follow-up), which is why everything is
ablation and drivers.

## Race-car, N=50: a scalar body emitted as tensor code

Budget, in µs, from the driver decomposition:

| Piece | Alloy | SX |
|---|---:|---:|
| eq-interstage stage kernel, 50 calls | 23.0 | |
| glue loops in the entry point | 7.3 | |
| corridor and cost stage kernels, 50 + 51 + 51 calls | 2.4 | |
| total | 32.7 | 21.4 |
| of which libm (`sin`, `cos`, `tanh` replaced by identity) | 7.7 in the eq stage | 10.7 |

The eq stage kernel alone costs more than SX's entire Hessian. Both spend a third to a half of their
time in libm: on this machine a dependent `sin` or `cos` costs 12 ns and `tanh` 32 ns, Alloy makes
14 calls per stage (5 `sin`, 5 `cos`, 4 `tanh`, `pow(x, 2)` folds to a multiply) and SX 18.

Per-stage instruction counts from the object files, SX divided by 50:

| | mul | add | sub | div | calls |
|---|---:|---:|---:|---:|---:|
| Alloy eq stage | 977 | 596 | 53 | 63 | 14 |
| SX per stage, everything | 600 | 230 | 84 | 39 | 19 |

Alloy does 1.6× the multiplies and 2.6× the adds for the same result. Two causes, separated by the
scalarizer:

- **No scalar-level CSE and a loop-and-array code shape.** The stage body declares 109 stack
  arrays and 83 loops of trip count 1 to 4; hash-consing the unrolled scalar graph with the seeds
  kept symbolic gives 1618 operations and 20.2 µs against the original 23.0.
- **Seeds are runtime data.** The Hessian is computed by `sparse_hessian` as `jvp_many` over the
  gradient with a global star-coloring seed matrix, `Expr.const` of shape `(4, 306)`. The VMAP
  forward rule slices that constant into a per-stage table (`k10[1200]` in the entry point) and
  passes it as a mapped *argument*, so the callee is generic in `fwd_z` and multiplies by 0 and 1
  at runtime; 32 of its 48 outputs per stage are structurally zero. Folding the actual seed tile
  `[1,0,1,0,0,0; 0,1,0,1,0,0; 0,0,0,0,1,0; 0,0,0,0,0,1]` into the scalar graph gives 1190
  operations and 18.1 µs, bit-identical.

The forward rule already has the mechanism, `_call_jvp_many_const_function`, and uses it on the
corridor and cost stages (their names carry `fwd1c<hash>`), but only when local coloring gives
fewer colors than the global seed count. The 23-line experiment in `experiments.patch` adds one
more case: a constant tangent whose per-iteration tile is the same at every iteration is baked
into the callee. This removed the `k10` table and the seed gather and produced a `fwd4c<hash>`
callee, but the tensor-level simplifier only folds a multiply when the constant is all zeros or
all ones, so the mixed vectors `[1,0,1,0]` stay in the body (49 of 873 multiplies in the N=5
expression assembly are by a mixed 0/1 vector). The compiled stage went from 23.0 to 21.0 µs and
the cell from 32.6 to 31.4. The remaining 21.0 to 18.1 is the scalarization.

Glue, by deleting one top-level loop at a time from a glue-only variant (7.2 µs total):

| µs | Loop | What it is |
|---:|---|---|
| 1.38 | `i_t38` | cost Hessian: `s4 * (mask30 * gather(s5) + mask34 * gather(s6))` over 1224 elements, index arithmetic `/ 6 % 51 * 6 + % 6` per element |
| 1.35 | `i_t51` | cost Hessian: scatter through three index tables with the same arithmetic |
| 1.03 | `i_t7` | corridor contribution: `ones[k6[i]] * s0[f(k6[i])]`, an indirect gather with `/ 6 % 50 % 6` |
| 0.69 | `d0_t54` | sum of the four contribution buffers plus transpose to `(306, 4)` |
| 0.57 + 0.56 | `i_t15`, `i_t20` | the two halves of the eq output scattered into full `(4, 306)` buffers |
| 0.28 | `d0_t12` | transpose of the eq output from `(50, 4, 12)` to `(4, 50, 12)` |
| 0.20 | final | gather of the 657 nonzeros |
| rest | | zero fills, broadcast copies of the diagonal cost |

Every one of these is either seed handling that a folded seed removes, or a layout change between
`(nseed, length*slice)` and `(length, nseed, slice)`, or the final sum-and-gather. With folded seeds
the cost Hessian is a diagonal of 306 values that the code currently expands to four masked copies
of 1224.

Compiler flags, same kernels:

| Flags | Alloy | SX |
|---|---:|---:|
| `-O3` (protocol) | 33.4 | 21.3 |
| `-O3 -fno-math-errno` | 31.8 | 20.6 |
| `-O3 -march=native` | 26.5 | 20.5 |
| `-O3 -ffast-math` | 24.9 | 16.4 |

The baseline `x86-64` target compiles both kernels to SSE2 scalar code. Alloy's loop-shaped code
gains 21% from `-march=native` and SX's straight-line code 4%. The sweep protocol therefore
handicaps the loop-preserving encoding more than the unrolled one, and Alloy's actual use is a JIT
on the target machine. Whether the fair comparison is with or without `-march=native` is a protocol
decision for `fairness.md`; the numbers here say it is worth one column.

## npmpc, N=12 and unbumpercars, C=8: the matrix-vector product

Budget for npmpc, µs:

| Piece | Alloy |
|---|---:|
| `h32x32 ... fwd3c... adj_eq_x`, 12 calls | 51.5 |
| `h32x32 ... fwd1c... adj_eq_u`, 12 calls | 28.3 |
| three cost stage kernels | 0.1 |
| glue | 2.8 |
| total | 82.8 |
| with all 27 matvec nests deleted | 21.9 |
| with `exp` replaced by identity | 76.9 |

The two stage kernels contain 17 matrix-vector products per stage, 12 of them 32×32: primal,
adjoint and forward-over-adjoint for three seeds on `x` plus one on `u`. They are lowered as

```c
for (i < 32) { s[i] = 0; for (k < 32) s[i] = s[i] + W[i*32 + k] * v[k]; }
```

which is a 32-long dependent add chain per output. Without reassociation the compiler cannot
break it, and out-of-order execution overlaps only about three rows. MX suffers from the same
shape: `casadi_mtimes_dense` loops `k` innermost too, and stubbing it out takes the MX kernel from
51.1 to 14.4 µs.

Isolated 32×32 matvec, `-O3` and `-O3 -march=native`, ns per product:

| Form | `-O3` | native |
|---|---:|---:|
| Alloy nest as generated (dot form) | 350 | 328 |
| `casadi_mtimes_dense`, either `tr` | 388 | 438 |
| column sweep on a contiguous transpose, `for k: for i: s[i] += WT[k*32+i] * v[k]` | 173 | 43 |
| same sweep on the row-major matrix (stride 32) | 197 | 364 |
| dot form, four output rows per pass, four explicit accumulators | 184 | 186 |

Two facts decide the emission: the column sweep is bit-identical to the dot form because each
output still sums its terms in the same order, so it is a pure win; and it needs the reduction axis
to be the *slow* axis of the matrix, otherwise the compiler emits gathers under `-march=native` and
loses. Row-major `W @ v` therefore wants the blocked dot form and `W.T @ v` wants the column sweep
with no transpose at all.

Hand-rewritten kernel, correctness check passing:

| Variant | `-O3` | native |
|---|---:|---:|
| generated | 82.7 | |
| every nest interchanged to `k` outer, row-major layout kept | 54.5 | 89.1 |
| every nest as a contiguous column sweep using the transposes already in the function | 47.2 | 38.6 |
| the above minus the per-call transposes of the weights | 43.8 | |
| MX, for reference | 51.1 | 41.9 |

Compiler edits, `run_cell.py`:

| Step | npmpc N=12 | native | unbumpercars C=8 | native |
|---|---:|---:|---:|---:|
| baseline | 83.2 | | 3485 | |
| reduction loop outermost for every matmul | 55.6 | | 4614 | |
| plus `A.T @ v -> v @ A`, `v @ A.T -> A @ v` in `passes/expr.py` | 44.4 | 52.9 | | |
| plus layout choice: blocked dot for a contiguous reduction axis, sweep otherwise | 45.1 | 59.6 | 830 | 1215 |
| plus the four accumulators as four unrolled statements | 45.7 | 47.2 | 827 | 974 |

The unbumpercars regression at step one is the stride problem at inner trip counts 128 and 256; the
layout rule fixes it. The unrolled statements matter because clang vectorizes a four-iteration
inner loop with gathers under `-march=native`.

What is left in npmpc after the patch, per stage: the `u` kernel recomputes the primal and adjoint
MLP passes that the `x` kernel already did (12 rather than 10 products of 32×32 per stage), because
the VMAP forward rule emits one mapped callee per formal; each call transposes the three weight
matrices, which are parameters and identical at every stage (3.5 µs of 47); and row-major `W @ v`
runs at 184 ns where a hoisted transpose would give 43 with native flags. Batching the three `x`
seeds into one `W @ [v1 v2 v3]` product is the same idea from the other side.

## Chain, M=5: dense tangents for a sparse body, and a workspace of a million doubles

Budget, µs, from the ablation:

| Piece | Alloy | SX | mapped SX |
|---|---:|---:|---:|
| eq stage kernel, 40 calls | 1920 | | |
| glue | 660 | | |
| total | 2581 | 96 | 711 |

The stage kernel is `chain_eq_stage_M5_adj0_0_1_fwd24_adj_eq_z`: 6340 lines, 1624 loops, two
nested calls to `chain_ode_M5`, 24 seeds passed as a 576-entry table per stage of which exactly
24 entries are nonzero. Star coloring of a 24-variable stage block gives 24 colors, so every seed
is a unit vector and the compressed Hessian is the dense 24 × 45 block computed one column at a
time with full-length tangent vectors.

`scalarize_stage.py` on this function, 40 calls, bit-identical outputs:

| Form | µs | scalar ops per stage |
|---|---:|---:|
| generated | 1910 | |
| scalar SSA, hash-consed, seeds symbolic | 298 | 72,945 |
| scalar SSA, unit seeds folded | 247 | 27,421 |
| SX per stage, for reference | 2.4 per stage, 96 total | 10,500 arithmetic (12,437 assignments) |

The 6.4× between the first two rows is code shape alone: loops over materialized buffers,
zero-fill-then-scatter patterns, and 24 dense tangent vectors where the unrolled scalar graph
has one live value per nonzero. Folding the unit seeds is a further 2.7× in operation count but
only 1.2× in time. After both, Alloy still executes 2.6× SX's operations; 1738 divisions per stage
against SX's 674 is one visible piece (the DIV tangent rule divides once per seed where a shared
reciprocal would multiply), worth about 7% of the folded time. The rest is the forward-over-reverse
composition itself and is not resolved here.

The glue is 43 zero fills of 23,544 doubles followed by 41 scatters of 576 values into them, then
a sum: the per-piece contributions of the gradient's `jvp_many` each land in their own full
`(24, 981)` buffer. That is the 1,075,248-double workspace, and 8 MB of memset per call. It wants a
single accumulation buffer, or C-8's per-stage fusion. The constant-seed edit alone moves the cell
from 2600 to 2458 µs.

## What the evidence recommends, in order of measured payoff

1. **Matmul lowering by layout.** Column sweep when the reduction axis is strided, blocked dot with
   four unrolled accumulators when it is contiguous, and the `A.T @ v` fold so the adjoint products
   need no transpose. Twenty lines in `passes/lowering.py` and `passes/expr.py`, bit-identical
   results, 1.8× on npmpc and 4.2× on unbumpercars, and it turns the unbumpercars margin over MX
   from 2.9× to 12×. The tests to add: a matmul fixture per shape class pinned against NumPy, and
   the C snapshot updated.
2. **Scalarize small stage bodies.** For a callee whose tensors are all below some size, unroll to
   scalar SSA, hash-cons, fold `0`, `1` and constant arithmetic, and emit straight-line C. This is
   what SX is and what `scalarize_stage.py` does after the fact: 1.27× on the race-car stage and
   7.7× on the chain stage, bit-identical. tinygrad's expansion of small tensor ops into scalar
   UOps followed by its symbolic simplifier is the reference design. It subsumes most of C-10 and
   the "fold the seeds" half of C-9, and it is the only item that moves chain.
3. **Bake stage-invariant constant tangents into the VMAP callee.** The 23-line edit in
   `experiments.patch`; on its own 4% on race-car, but it removes the `nseed × length × slice`
   seed tables that C-9 targets and is what makes item 2 fold the seeds. Extend to a few distinct
   tiles when the coloring is not exactly periodic.
4. **Hoist loop-invariant callee work.** The npmpc weight transposes are recomputed per stage
   because the callee cannot know its argument is stage-invariant; 3.5 of 47 µs. A program-dialect
   pass, or emitting `v @ W` against the untransposed parameter, both work.
5. **One accumulation buffer for a sum of scatters.** Chain's 43 zero-filled buffers. An expression
   rewrite `scatter(a) + scatter(b) -> scatter_add`, or the fusion in C-8.
6. **Glue fusion (C-8) and affine index maps (C-9)** remain right for the workspace and metadata
   gates, and are worth 22% of race-car's runtime and 26% of chain's. They are not what closes the
   runtime gap with SX on race-car, and they do nothing for npmpc.

On the `-march=native` question: with the patch, npmpc is 45.7 against MX's 51.1 at the protocol
flags and 47.2 against 41.9 with native flags, because the row-major products are still in dot form.
Item 4 changes that. The comparison should be reported both ways until the protocol decides.

## Feedback on the text assembly

`al.render_expr_assembly(fn)` was readable and useful for one thing the C could not show: which
identities the simplifier leaves behind. In the race-car N=5 module, 2 of 63 divisions have a
constant-zero numerator and 49 of 873 multiplies have a mixed 0/1 vector constant as one operand.
For the rest of this investigation the generated C plus a driver was the better artifact, because
the question was runtime, not structure. What would have made the assembly the first stop:

- a per-function operation histogram at the end of each `expr.func`, and for the program dialect
  the same weighted by trip count, so "how much arithmetic is in this stage" is one line;
- marking constants that are neither all-zero nor all-one, and constants that a scalar view would
  fold, since those are exactly the ones the tensor-level simplifier cannot remove;
- a one-call way to get the assembly for a benchmark cell, next to the `.c` the harness already
  writes; and
- for `expr.vmap` and `expr.call`, printing the callee's per-iteration window of each argument
  (start, stride, size), which is the information needed to see that a mapped tangent is a
  stage-periodic constant.

## Follow-up, 2026-09-08

Answers to the review of the first version, each measured.

**`perf`.** Not usable from this session: `kernel.perf_event_paranoid` is 4 and even `cycles:u` and
`cpu-clock:u` fail to open, inside and outside the tool sandbox. It works in a shell with
`CAP_PERFMON` or after `sysctl kernel.perf_event_paranoid=1`, which this note did not change. It
would have shortened two steps, attributing time inside the chain stage body (memset versus
arithmetic) and confirming the libm share without the identity-stub trick, and `perf annotate` would
have shown the dependent add chains in the matvec directly. The driver-and-ablation route reached
the same numbers, so the conclusions do not depend on it.

**Why `-march=native` helps Alloy's race-car kernel and not SX's: FMA contraction, not vector
width.** Race-car N=50, µs:

| Flags | Alloy | SX |
|---|---:|---:|
| `-O3` | 32.9 | 21.3 |
| `-O3 -mfma` | 27.8 | 20.7 |
| `-O3 -mavx2 -mfma` | 27.9 | 20.6 |
| `-O3 -march=native` | 26.9 | 20.7 |
| `-O3 -march=native -ffp-contract=off` | 35.2 | 20.5 |
| `-O3 -march=native -ffp-contract=fast` | 27.9 | 18.8 |

`-mfma` alone gives Alloy almost the whole gain and vector width adds one microsecond; with
contraction disabled the native build is slower than the baseline. The native Alloy object has 439
fused multiply-adds and 758 AVX instructions, mostly in the glue loops (10 loops vectorized against
5 at baseline, and 14 gathers from the index tables). The native SX object has zero fused
multiply-adds and zero vector instructions: SX emits one operation per statement, `a=(a*b);
a=(a+c);`, and clang's default `-ffp-contract=on` contracts only within one expression, so the
straight-line code never forms an FMA. Alloy's renderer writes compound expressions and gets
contraction for free. This is a code-shape effect worth keeping when C-44 scalarizes: render
expression trees, not one op per line. There is no evidence here that the horizon loops
auto-vectorize; the stage body is called per iteration through a `noinline` function, so the
compiler never sees the horizon as a loop of independent bodies. Tiling the horizon so that four
stages run in lockstep with a stage-major layout is the only way to get SIMD across stages, and it
is a data-layout change, as suspected.

**The `lowering` hint is inert today.** `Expr.scalar()`, `.block()` and `.opaque()` set the
`lowering` attribute and every constructor propagates it, but no pass, lowering rule or renderer
reads its value; `grep` for a comparison against `"scalar"` or `"block"` finds nothing outside
`ir/expr.py`. Hooking C-44 to it is the right design: scalarize a callee when its body is marked
`scalar` or when every tensor in it is below a size threshold under `auto`, and never under
`block`. That keeps the SX failure mode, unbounded straight-line code, a local choice.

**Lean division rules, cheap and effective.** The forward rule was `(dx*y - x*dy) / y**2` and the
reverse rule `adj_y = -cot * x / y**2`; the tangent of the reverse rule differentiated a quotient
with a squared denominator. Replacing them with `(dx - f*dy) * (1/y)` and `q = cot/y; adj_x = q;
adj_y = -q*f`, where `f` is the primal quotient already in the graph, changes nothing else:

| Cell | Before | After | Note |
|---|---:|---:|---|
| race-car N=50 | 31.4 | 28.0 | 26.1 with `-march=native`; SX 21.4 |
| chain M=5 | 2458 | 2277 | 2012 with `-march=native`; SX 96, 83 native |

Chain divisions per stage in the scalarized stage body fell from 1738 to 335 while the total
operation count stayed at about 28,000, so the gain is the division latency, as estimated. Results
are not bit-identical to the old rules; the harness check passes on both cells. This is the
`jvp_many` and forward-over-reverse audit the review asked for, in its first instance.

**Chain: where the remaining 2.6× against SX sits.** Splitting the scalarized stage body by
dependence on the seeds, with the seeds kept symbolic:

| Part | Ops per stage |
|---|---:|
| primal plus adjoint, seed-independent | 993 |
| tangents of the adjoint, 24 seeds | 71,952, so 2,998 per seed |
| same with the unit seeds folded | about 26,400, so 1,100 per seed |
| SX, whole Hessian per stage | 10,500, so at most about 400 per seed |

So the adjoint is small and the forward-over-reverse tangents are what is large: each unit-seed
tangent of the adjoint costs more than the entire primal plus adjoint, even after the seeds fold,
where one would expect a fraction. Two candidate causes remain and neither is settled here: the
tangent rules themselves, of which DIV was one and `neg`/`sub` handling (2,749 negations per stage)
may be another; and the composition, since the tangent of a 4-stage RK4 adjoint through a chain
whose coupling is local should stay sparse for the first substeps. The direct test is to build the
same stage in CasADi SX as an explicit forward-over-reverse with the same 24 unit seeds and
compare the per-operation histogram; that is an afternoon and would say whether the gap is rule
quality or something structural.

**Race-car's own share of this.** With the seeds folded, its stage has 226 seed-independent and
1,392 seed-dependent operations for 4 seeds, 348 per seed, 1.5 times the primal plus adjoint. SX's
whole per-stage Hessian is about 970 operations, so the same question applies at a smaller scale.

**A principled loop compiler versus ad hoc matmul rules.** `tinygrad_rangeify.md` in this directory
is the study of tinygrad's current design. The parts relevant to the decision: tinygrad has no
loop-order cost model either, its C backend emits a plain i-j-k nest with the reduce innermost, and
it gets multiple accumulators from a generic mechanism (an UPCAST split of an output axis becomes
extra accumulator lanes when the reduce is turned into `acc init / acc op= x / END`) rather than a
matmul rule. Its fusion policy is "share the range variable", three cases and a 3-buffer cap, in
about 40 lines. The honest reading for Alloy: the two matmul rules in C-43 are the special case of
two generic transforms, a range split with an accumulator per lane and a choice of which range is
outermost, and both generalize to `W @ [v1 v2 v3]` and to the matrix-matrix products a vmapped
neural dynamics would produce. C-43 as written is the right first step because it is twenty lines
and measured; the generic form is what C-8 should become, and the study lists the six pieces to port.

## C-44 validation, 2026-09-08

`src/alloy/passes/program/scalarize.py` expands selected procedures before fusion and workspace packing.
It follows the expansion, hash-consing, and single-use expression emission described in
`tinygrad_rangeify.md` sections 4 and 6. The corresponding tinygrad `codegen/__init__.py`,
`uop/symbolic.py`, and `renderer/cstyle.py` were read before implementation. Reusing the existing
lowered indices avoids a second registry of expression lowering rules. Parsing generated C
remains an investigation tool, not part of the compiler.

The stage comparison reuses the saved original kernels, hand-scalarized kernels, and sample
inputs from this investigation. Compile flags are `clang++ -O3 -std=c++17`, with no native-target
flag or fast math. The reference machine uses the performance governor with boost disabled.
Each number is the best of five timing loops, with 20,000 repetitions for race-car and 300 for
chain. Compiler jobs and tests were stopped during the final timing run.

| Stage, calls | Seed binding | Hand scalarizer, µs | C-44, µs |
|---|---|---:|---:|
| Race-car, 50 | Runtime arguments | 20.24 | 20.82 |
| Race-car, 50 | Constants inside the body | 18.17 | 19.63 |
| Chain M=5, 40 | Runtime arguments | 307.93 | 265.06 |

The original tensor stages take 22.92 µs and 1908.23 µs respectively. C-44's chain stage also
falls within 10% of the folded-seed hand kernel, which takes 246.33 µs in this run. Both C-44
stage outputs match the original kernels bit for bit on these inputs.

The race-car constant-seed control substitutes the known tile into the expression body before
lowering. It checks C-44's folding at the same seed binding as the hand rewrite. The mapped
forward rule still passes runtime seeds. Exposing these constants automatically remains C-45.
Thus the 19.63 µs control is not the stage time of an ordinary mapped race-car build yet.

The full benchmark cells pass both the Python-call and compiled-kernel dense-reference checks.
With the current harness flags, `-O3 -march=native -fno-math-errno`, their full-kernel times are
26.24 µs for race-car N=50, 1068.01 µs for chain M=5, and 46.56 µs for NPMPC N=12.
These flags differ from the isolated stage comparison above. The chain entry-point workspace
remains 1,075,248 doubles; scalarization removes the stage's buffers, not the glue buffers.

Local artifacts are in `benchmarks/results/c44/`: the two `*_driver.cpp` files and `*_stages.log`
files hold the stage comparison, and `cells/results.json` and the cell directories hold the
full-kernel checks, generated C, compile logs, and timings. Core coverage lives in
`tests/passes/test_scalarize.py`, including an analytic spring Hessian and runtime versus constant
seeds. Disabling scalarization fails the loop-removal gates. Disabling floating-point folding
leaves five sine operations where the folding gate requires one.

### C-44 closeout

The final automatic policy caps each procedure at 4,096 unique arithmetic nodes after folding
and sharing, expansion work at 65,536 units, and program-wide growth at 16,384 arithmetic nodes,
assignments, and stores. Tests cover each limit separately, including the exact operation-count
boundary and a narrow tensor with too many nested sine operations. Explicit scalar selection
bypasses the limits. The fixed pass pipeline now lives in `src/alloy/passes/program/`.

This closeout regenerated only the race-car N=50 and chain M=5 Hessian kernels and their controls.
Each control disables the scalarization pass and retains every other pass. Timings are medians
of three fresh Google Benchmark binary processes with a 0.2-second minimum per process. Each
generated cell and compiled binary passed the independent dense Hessian check at absolute and
relative tolerances of `1e-9`. No compiler or test jobs ran alongside timed kernels.

The reference Ryzen 9 7940HS used the performance governor with boost disabled. Clang 20.1.8 used
the harness flags `-O3 -march=native -fno-math-errno -std=c++17`; GCC 13.3.0 used the default JIT
flags `-O2 -march=native -fno-math-errno`. Compile times below are single cold kernel compilations,
excluding the benchmark wrapper and link. This is a minimal diagnostic, not the full frozen
benchmark protocol. The full rerun is deferred until more Track C work lands.

| Kernel and selection | Runtime, µs | Source, bytes | Clang compile, s | GCC compile, s |
|---|---:|---:|---:|---:|
| Race-car, pass disabled | 26.019 | 136,087 | 0.359 | 0.32 |
| Race-car, auto | 25.909 | 112,637 | 0.286 | 0.20 |
| Chain, pass disabled | 2,216.842 | 1,274,426 | 4.917 | 4.98 |
| Chain, explicit derived procedure | 1,081.824 | 1,906,598 | 16.808 | 62.77 |

Race-car runtime is unchanged within noise. Its source is 17.2% smaller and its single Clang
compile is 20.3% faster. The automatic policy admits its equality stage at 1,618 arithmetic nodes
and 2,004 arithmetic nodes plus statements. Chain is 2.05 times faster with explicit expansion,
but its 72,913 arithmetic nodes and 86,416 nodes plus statements explain why it exceeds the
automatic budgets. Clang compilation costs 3.42 times as much and GCC about 12.6 times as much.
The entry-point workspace remains 1,075,248 doubles in both chain variants.

The chain experiment sets `lowering="scalar"` on the generated
`chain_eq_stage_M5_adj0_0_1_fwd24_adj:eq_z` procedure immediately before calling the shipped
scalarization pass. A `.scalar()` hint on the primal stage output did not reach this derived
procedure. That first diagnostic retained automatic selection and measured 1,862.941 µs.
Thus this experiment establishes the explicit pass's benefit; propagating the primal hint through
derivative construction remains separate work. The earlier stage gate above still records the
comparison with the hand scalarizer at matching seed binding.

Raw samples, generated C, the temporary runner, and compile logs are under
`benchmarks/results/c44/closeout/`. The rejected hint-placement attempt is retained separately as
`chain_5_auto_diagnostic`. The suite passed with 682 tests and one expected failure; Ruff formatting,
lint, type checking, and the documentation build also passed.

## Track C follow-up, 2026-09-08

After C-45, C-55, C-12, C-13, C-53 and C-10 landed together. Same method as the C-44 closeout:
`sweep.run_cell` with the harness flags, then two more benchmark processes, median of three; the
pre-change tree ran from a detached worktree of the same commit; no compiler, test or agent jobs
ran during the timed cells; each cell passed the harness dense-reference check.

| Cell | Before, µs | After, µs | Source, bytes | Static metadata, bytes |
|---|---:|---:|---:|---:|
| Race-car N=50 Hessian | 26.01 | 24.02 | 112,637 -> 90,070 | 107,137 -> 90,115 |
| Chain M=5 Hessian | 1,869.7 | 1,769.0 | 1,265,670 -> 1,174,015 | 1,331,095 -> 1,264,721 |
| Chain M=5 Hessian, `.scalar()` on the stage output | - | 835.2 | 1,273,458 | 1,205,031 |

The hinted chain row is a diagnostic of C-55: the hint reaches the adjoint-tangent procedure
automatically, where the C-44 closeout had to force it after the fact (1,081.8 µs then). The chain
stage now carries the hint in `benchmarks/problems/chain`. Race-car was tried with the same hint and
produced byte-identical source, since the automatic policy already selects its stage. The entry-point
workspace is unchanged at 1,075,248 doubles on chain: C-8 owns that.

## C-46 hoist, 2026-09-09

`hoist_invariant` alone, measured with `sweep.run_cell` and the harness flags on the same tree, the
before variant with the pass removed from `PASS_PIPELINE` in the probe process; median of three
alternating runs, each cell passing the harness dense-reference check.

| Cell | Before, µs | After, µs | Executable, bytes |
|---|---:|---:|---:|
| npmpc N=12 Hessian | 47.47 | 44.55 | 36,385 -> 38,108 |

The weight transposes this item named are gone since the C-43 layout fold, so nothing of the
3.5 µs remains to hoist. What the pass moves out of the `x` stage kernel is the tangent of the
sigmoid input under the constant unit seeds: two 32-element columns of `-(W0[:, k] * scale_k)`,
recomputed per stage before. The `u` kernel has no invariant buffer at all; its duplicate primal
and adjoint passes are the callee-merge half of C-46, still open.

