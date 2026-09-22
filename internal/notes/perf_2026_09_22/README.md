# From a hand-optimized race-car Hessian to compiler passes, 2026-09-22

Another agent rewrote the generated `race_car_closed_loop_N200_hess_lower` by hand on an x86
AVX-512 machine (95 to 11 µs) and listed what it changed. This note re-measures those changes on an
Apple M4 Max, separates the general from the problem-specific, and proposes the passes and the
shape of the loop compiler that would produce the hand-written code automatically. The hand-written
file and the original answer live outside the repository (`~/dev/scaly/answer.md`,
`~/dev/scaly/race_car_closed_loop_N200_hess_lower_fast.c`); the scripts here rebuild every variant
from them.

## 1. Measurements on the M4 Max

Apple clang 21, `-O3 -mcpu=native -fno-math-errno`, one performance core, best of 5 × 20000 calls.
Homebrew clang 23 agrees within 10% everywhere except where noted.

| Variant | µs/call | What it isolates |
| --- | --- | --- |
| Generated code as of today | 43.5 | baseline |
| Same, the `noinline` on `_fwd` callees removed | 43.0 | inlining alone buys nothing |
| The 200 stage kernels alone, out of line | 28.9 | kernel share of the baseline |
| Fused per-stage assembly, original out-of-line kernel | 28.4 (25.7 on clang 23) | the 15 assembly passes cost about 15 µs |
| Fused assembly, kernel inlined, scalar | 18.0 | consumers fused, so 6 of 16 block entries die |
| Same, 2 NEON lanes, libm called per lane | 17.9 | width alone buys nothing here |
| Same, 4 lanes | 15.6 | |
| Same, 8 lanes | 14.2 | |
| Fused scalar with `-ffast-math` | 15.5 | mostly reciprocal of invariant divisors |
| Fused, trig replaced by 3-term polynomials, any width 1 to 8 | 8.6 | arithmetic floor; flat across widths |
| 2000 `sin`/`cos` and 800 `tanh` scalar libm calls | 5.7 | 2.0 ns per call on Apple libm |

Every correct variant matches the baseline to 7e-14 absolute on entries up to 1400 (the reference
data are `bench_hess.c`'s pseudo-random inputs, not a trajectory).

What the table says, in order of size:

1. **Assembly is 35% of the baseline and all of it is avoidable.** The other agent's first finding
   holds on this machine: 43.5 to 28.4 µs from writing the 13 entries per stage straight into
   `res[0]`.
2. **Inlining is worthless without fusion.** Removing the `noinline` changes nothing because the
   kernel still writes all 48 outputs to a workspace buffer. Once the consumers are fused, the
   compiler sees that only 10 of the 16 block entries are read and deletes the rest: 28.4 to
   18.0 µs. The macOS `_fwd` noinline workaround in `codegen/c.py` is therefore not what is holding
   this kernel back; the call boundary is, and only because nothing is fused across it.
3. **On Apple silicon, SIMD width is not the lever.** With the transcendentals replaced by
   polynomials the fused loop runs at 8.6 µs whether it is scalar or 8 lanes wide. The out-of-order
   core already overlaps independent stages and saturates its four FP pipes; the RK4 kernel is
   latency bound within a stage and throughput bound across stages either way. The remaining 9 µs
   between 8.6 and 18.0 is 2800 scalar libm calls (5.7 µs measured in isolation) plus the lane
   extract and insert around them. The 2 to 8 lane gains (18.0 to 14.2) come from a longer
   independent-work window, not from vector arithmetic. "8 lanes" here is a
   `vector_size(64)` type, which clang legalizes to four NEON `.2d` registers per operation (the
   assembly has 3096 `.2d` ops, no SVE `z` registers; the M4 has SME but no SVE and the compiler
   uses neither). So W = 8 on this machine is W = 2 unrolled four times with the trig calls
   batched, which is why it helps a little and why it is not a width effect.
4. **On x86 the picture is different and the other agent's 3 to 4× from width is real.** glibc's
   scalar `sin` is 15 to 25 ns where Apple's is 2 ns, and AVX-512 has 8 lanes. Width and vector
   transcendentals must stay a per-target choice, and the scalar path stays first class.
5. **Homebrew clang 23 auto-vectorizes the fused loop when given a vector math library.**
   `-fveclib=SLEEF` on the fused scalar file produced 7 `_ZGVnN2v_sin`, 7 `_ZGVnN2v_cos` and 4
   `_ZGVnN2v_tanh` calls: the whole stage body, transcendentals included, went 2 wide with no
   vector types in the source. Apple clang has no arm64 vector-libm mapping (`-fveclib=Accelerate`
   yields scalar `__sincos_stret`), and the `sincos` merge the other agent hit on gcc did not block
   LLVM here. Whether it is a win on this machine is unmeasured: SLEEF is not installed, and the
   decision in section 3.6 is not to depend on any vector libm, so it stays unmeasured.

## 2. The hand optimizations, sorted by how general they are

| Hand change | General form | Level | Payoff here |
| --- | --- | --- | --- |
| One pass, write `res[0]` directly | Fusion of the sparse-derivative assembly (`gather(transpose(jvp_many(...)))` in `ad/sparse.py`) into the mapped producer loop | program dialect, the C-8 range propagation | 15 µs of 43.5 |
| Only the lower triangle | Dead-store elimination across the call boundary once the kernel body is a fused range body | falls out of fusion plus existing `fold_arith` | 10 µs |
| Closed-form cost block, masks and all-ones tables gone | Constant-tile folding: a `static const` table that is a tile of period P becomes `k[i % P]`, and a scalar when P = 1. `k0` (4800 ones), `k22` and `k26` (period 6 masks) are exactly what C-57 lists as unexplained | program dialect, generalizes the C-45 bake | small alone, frees `fold_arith` to kill the masked terms |
| Recovery gather `k44` | Already affine: values `0,4,5,8,12,13,16,17,18,20,21,22,23` then `+24` per stage, a residual of 13 under `AffineIndexMap`; the terminal stage's 7 entries break the period. Peel the last trip of a mapped axis when its slice differs, then C-9's map applies | lowering, `passes/affine.py` | metadata, not time |
| Replace 14 divisions by reciprocals | Both divisors (`params[2]`, `0.5 * wheelbase`) are loop invariant. `x / y` with `y` invariant becomes `x * inv_y` with `inv_y` hoisted; a policy flag, since it moves the last bit | program dialect, `hoist_invariant` plus an arith rule | about 2 µs (the `-ffast-math` gap) |
| Cache the cost block across SQP iterations | Split every oracle into a parameter-only prologue and a body; the solver plugin runs the prologue once per solve | Function or solver level, not a compiler pass | unmeasured, problem dependent |
| SoA staging arrays, explicit vector types, libmvec | Lane widening of the mapped range | program dialect plus renderer, see section 3 | 4 µs here, 30 µs on the x86 machine |
| Reference-heading trig computed once, not 402 times | CSE across the two mapped callees that both compute `cos(ref[2])`, `sin(ref[2])`; needs the callees fused into one range first | falls out of fusion | 1 µs |

Not general: the `2σ Rᵀ diag(w) R` algebra. It is trig-identity rewriting on one problem's
structure; do not build a pass for it.

## 3. The loop compiler and where vectorization sits in it

C-8 (`internal/todo.md`) already plans a rangeify-shaped loop compiler in three steps: ranges with
the three-case propagation rule, reductions to per-lane accumulators, the reduce-under-broadcast
rule. The measurements say what to put before and after those steps.

### 3.0 Make the mapped callee a range body

Today `_lower_vmap` emits `FOR(GLOBAL) { CALL callee(views) }` and every later pass stops at the
`CALL`. `fuse_elementwise` fuses only single-store loops, so nothing downstream of a `VMAP` fuses,
which is why the assembly exists at all. The missing step is small: after `scalarize`, a callee
whose body is scalar form becomes the loop body of every `FOR` that calls it, with its views turned
into index expressions over the loop variable. Then step (1) of C-8 propagates the consumers'
ranges into that body and the assembly collapses. Gate: `race_car_closed_loop_N200_hess_lower`
under 22 µs on this machine with no vectorization, and `workspace` fixed across N.

### 3.1 Widen the mapped range

The single tinygrad idea to port is `shift_to`: `r -> r_outer * W + r_lane` with `r_lane` tagged
`RangeKind.VECTOR`, which `ir/program.py` already declares and nothing yet produces. Under a
`VECTOR` range of static width W every scalar value in the body is W wide, a load whose index is
affine in `r_lane` with stride 1 is a vector load, any other stride is a gather or a staging
transpose, a store the same, and reductions over inner ranges keep vector accumulators. This is
outer-loop vectorization in the SPMD (single program, multiple data) sense: one lane per stage,
all inner control flow uniform because shapes are static, no masks except at the tail.

Choosing the axis: always the mapped axis first. Trips of a `VMAP` are independent by
construction, so there is no dependence analysis and no cross-lane reduction. Only when no mapped
axis exists (chain at M = 9 is one stage with big matmuls) fall back to the same substitution on a
contiguous output axis (stride-1 loads) and then on a reduction axis (unroll with one accumulator
per lane and a horizontal add, which is what C-8 step 2 does anyway). When both a long horizon and
an inner matmul exist (npmpc's MLP stage), the horizon takes the lanes and the matmul's inner
loops run per lane unchanged; that composes because the inner ranges are simply carried under the
outer `VECTOR` range.

Choosing W: from the target, not the problem. AVX-512 8, AVX2 4, NEON 2, scalar 1;
`codegen/jit.py` already detects the native flag. Then cap by register pressure: the race-car
kernel has about 260 live scalars, and W = 8 means 260 vector registers against 32, which the x86
run survived because spills hit L1, and which is one reason W = 2 buys nothing on NEON. A rule of
the form `live_values × W ≤ 4 × register_file` is enough to start; measure before refining.
Tail: N mod W ≠ 0 peels a scalar remainder (the cost `VMAP` has 201 trips). Expose the width as
`SCALY_LANES` and a compile option; default to the target's.

Layout: the ABI arrays are the solver's and stay stage major (`z[6k + j]`). Inside, a buffer
created under a `VECTOR` range is lane major, that is, one contiguous run per variable across
stages (`Z[j][k]`), and the layout change happens exactly at the ABI boundary: a transpose of the
inputs a widened body reads with non-unit stride, and a scalar epilogue that writes the per-stage
block back in `res` order. The other agent's `Z[4][N]`, `L[4][N]`, `H[10][N]` are these staging
buffers, and they cost under 1 µs here. This is tinygrad's "materialize on the axis where index
expressions disagree", applied at the boundary between the widened range and the ABI. NEON's
`ld2`/`ld3`/`ld4` cover strides up to 4 without staging; stride 6 does not, so staging is the
general answer and the structured loads an optimization for small strides.

### 3.2 Render: the modes considered, and the transcendental question

This section records the options as they were weighed; section 3.6 has the decision (two modes,
`gnu` and `c`). Assume gcc or clang everywhere, `zig cc` included, and keep one plain mode:

- `gnu` mode: `__attribute__((vector_size))` types for the widened values,
  `__builtin_shufflevector` for the staging transposes, `restrict` and `__builtin_assume_aligned`
  on the staging buffers, `always_inline` on fused callees. All three compilers accept these.
  What is common to gcc 16 and clang 21 to 23 (verified 2026-09-22 with `vb2.c`, kept beside this note):
  `vector_size` types with `+ - * /` and comparisons, `v[i]` element access,
  `__builtin_shufflevector` (gcc ≥ 12) and `__builtin_convertvector` (gcc ≥ 9). Not common: the
  ternary `a > b ? a : b` on vectors compiles on gcc but is rejected by clang in C, so `minimum`
  and `maximum` render per lane or through the integer mask of the comparison. Clang only: `__builtin_elementwise_sin/cos/exp/...` and `__builtin_elementwise_fma`,
  which lower to LLVM intrinsics and become scalar calls unless a vector libm is mapped, and
  `__builtin_reduce_*`, integer only. `ext_vector_type` is clang only; `vector_size` is the
  portable spelling and the one `codegen/c.py` already uses. A transcendental on a widened value
  therefore renders as a per-lane scalar call in the common subset, and `__builtin_elementwise_*`
  is a clang-mode nicety, not a requirement.
- `c` mode (called `ansi` while this was being weighed): the same program rendered as an inner `for (lane < W)` loop over the scalar body
  with the staging buffers in place. Correct everywhere, and on clang with a vector libm flag it
  auto-vectorizes anyway (measured in section 1, point 5).
- Scalar mode: W = 1, today's output.

The transcendentals decide whether widening pays, and there are three ways to get them wide:

1. `libm` per lane. What the 2-lane variant measured: no gain, and gcc's `sincos` merging can
   block auto-vectorization of the `c` form.
2. A vector libm. glibc libmvec (`_ZGVdN4v_sin`, `_ZGVeN8v_sin`) on x86 Linux only; SLEEF's
   gnuabi symbols (`_ZGVnN2v_sin`) on every target if we vendor SLEEF, which is a small CMake
   build and about 1 MB, and the only portable choice under `zig cc`. Declare the prototypes
   ourselves per target rather than relying on `-fveclib`, whose mapping differs per compiler.
   Accuracy is 1 ulp (`u10` variants); the other agent saw 1.7e-12 relative from that.
3. Our own polynomials, in the program dialect. tinygrad's `codegen/decomp/transcendental.py`
   (about 250 lines, SLEEF-derived `xsin`, `xexp2`, `xlog2`, Payne-Hanek and Cody-Waite reduction)
   expresses `sin`, `exp`, `log` as ordinary arithmetic on its own ops, so they vectorize with
   everything else and give bit-identical results on every platform and compiler. It needs
   int-float reinterpretation, shifts and bit masks as program ops, which do not exist yet. The
   risk is speed on Apple silicon, where the system `sin` is 2 ns; measure a C port of `xsin`
   against it before choosing.

Decision 2026-09-22: no vendored SLEEF and no own polynomials. Widen the arithmetic and leave the
transcendentals as per-lane scalar calls (mode 1). That is the 18.0 to 14.2 µs measured above, it
keeps the generated C free of any dependency beyond libm, and the effort budget is reserved for
the fusion work and for the GPU backend. Where the platform ships a vector libm for free, glibc's
libmvec on x86 Linux, mode 2 costs nothing more than declaring `_ZGVdN4v_sin` and friends in the
generated source and linking `-lmvec`; that is a rendering option, not a dependency, and the x86
re-run decides whether it is worth even that.

### 3.3 What else the extensions buy

With the gcc-or-clang assumption made explicit, a few more things become available and should be
listed in the design as opt-in, each behind the same mode switch:

- `__attribute__((simd))` on gcc generates `_ZGV` variants of a scalar function, so a scalar stage
  kernel called from a loop can be auto-widened by the compiler. Clang needs `-fopenmp-simd` and
  `#pragma omp declare simd`. Cheaper than our own widening for a first cut, but it removes the
  lane count from our control and does nothing for the staging layout.
- `#pragma clang loop vectorize(assume_safety)` and `#pragma GCC ivdep` on mapped loops state
  the independence we know by construction. Free and harmless in `c` mode.
- `-ffp-contract=fast` and `__builtin_fma` for explicit fused multiply-adds; expression-tree
  rendering (C-44) already lets the compiler contract, and this only matters for gcc, which defaults
  to `on` per statement only.
- Statement expressions and `__builtin_expect` add nothing measurable to this workload.

### 3.4 How tinygrad and XLA actually render vectors, and why Scaly stays with C

tinygrad's C renderer (`ClangRenderer` in `renderer/cstyle.py`) emits
`typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));`, literals as
`(float4){a,b,c,d}`, lanes as `.x/.y/.z/.w` up to 4 and `v[i]` above. The struct-of-array
spelling exists only in its CUDA renderer, for packed `half` and `bfloat16`. More important than
the spelling: on CPU `devectorizer2` unpacks every elementwise op, load and store into scalar
UOps, and `memory_coalescing` re-merges adjacent loads and stores late, so only memory accesses
reach the C as vectors and the arithmetic is scalar SSA that LLVM's SLP vectorizer repacks. That
is the shape C-51 already gave Scaly's stores. Transcendentals never reach libm there:
`decomp/transcendental.py` rewrites `SIN`, `EXP2`, `LOG2` into arithmetic and bit ops first.

XLA's CPU backend emits LLVM IR directly, native `<N x double>` types, scalar loops for the loop
vectorizer plus hand-emitted vector IR in the dot emitter and in its own polynomial math
(Cephes coefficients, Cody-Waite reduction); the newer emitters go through MLIR's `vector`
dialect. Halide's C++ backend has the struct you may remember, `CppVector<T, Lanes>`, as a
fallback only; its default is `vector_size`. ISPC, Triton and MLIR never see C.

The pattern is that compilers owning their toolchain emit LLVM IR, and nobody who emits C for
speed uses a struct, because an aggregate is passed through memory, has no operators and only
becomes a register if SROA and SLP rebuild it. Scaly stays with C: clang's translation of GNU
vector C to LLVM IR is one to one (`vector_size` to `<N x double>`, `__builtin_shufflevector` to
`shufflevector`, `__builtin_elementwise_sin` to `llvm.sin.v2f64`), a second backend would cost
a renderer, a verifier target and an LLVM dependency, and it would lose the two reasons for C:
compiling with any toolchain including embedded vendor compilers, and dropping into C and C++
applications. The one-sentence version is now in `docs/how_it_works/architecture.md`.

### 3.5 `zig cc` and the clang builtins, verified 2026-09-22

`zig cc` (zig 0.16, clang 21) compiled `ew.c`, a file of `__builtin_elementwise_sin/exp/fma/
max/sqrt` on 2- and 4-wide `vector_size` doubles plus vector arithmetic, and the whole fused
race-car kernel, for seven targets:

| Target | Vector arithmetic | `__builtin_elementwise_sin`, `exp` |
| --- | --- | --- |
| aarch64-macos, `-mcpu=apple_m4` | NEON `.2d` | scalar `_sin` per lane |
| aarch64-linux-gnu | NEON `.2d` | per lane; `-fveclib=SLEEF` gives `_ZGVnN2v_sin` |
| x86_64-linux-gnu, `x86_64_v3` | `vmulpd` and friends | scalar `sin@PLT` per lane |
| same, `-fveclib=libmvec` | same | `_ZGVdN4v_sin`, `_ZGVbN2v_exp`, one call per vector |
| x86_64-linux-gnu, `x86_64_v4` | same | as v3 for a 4-wide value |
| x86_64-windows-gnu, `-fveclib=libmvec` | same | `_ZGVdN4v_sin` emitted |
| thumb, `cortex_m7`, musl hard float | soft-float `__aeabi_dmul` calls | per lane |
| arm-linux-musleabihf, `cortex_a72` | scalar VFP (no f64 NEON) | per lane |

The fused kernel cross-compiled for AVX2 with `-fveclib=libmvec` has exactly three undefined
symbols: `_ZGVdN4v_cos`, `_ZGVdN4v_sin` and scalar `tanh` (LLVM's libmvec table has no `tanh`
although glibc 2.35+ ships one). So `__builtin_elementwise_*` is a one-line entry point to
vectorized transcendentals wherever a vector libm exists, and degrades to per-lane scalar calls,
never to an error, everywhere else. Linking is the catch: `zig cc` cannot find `-lmvec` when
cross-compiling because its glibc stubs omit libmvec; a native Linux `zig cc` with the system
library path is untested (C-81). No `-ffast-math` is needed for the mapping.

What is common to gcc 16 and clang 21 to 23 and what is not, from `vb2.c`:
common are `vector_size` types with `+ - * /` and comparisons, `v[i]`, `__builtin_shufflevector`
(gcc ≥ 12), `__builtin_convertvector` (gcc ≥ 9); clang only are `__builtin_elementwise_*`,
`__builtin_reduce_*` (integers only) and `ext_vector_type`; the vector ternary `a > b ? a : b`
compiles on gcc and is rejected by clang in C.

### 3.6 Decision: two render modes, no third

The transcendentals are a third of the fused race-car kernel here and less elsewhere: npmpc's
time is in matmuls, chain's in the colored sweeps, and both are already ahead of CasADi. So the
lane work stops at vector arithmetic, and the render has two modes:

- `gnu`, the default: `vector_size` types, `v[i]`, `__builtin_shufflevector`,
  `__builtin_convertvector`, `restrict`; gcc ≥ 12, clang, `zig cc`, armclang. Transcendentals,
  `minimum` and `maximum` render per lane.
- `c`, opt in: the same widened program as a scalar body inside an inner lane loop over the same
  staging buffers; any C99 compiler, MSVC and vendor compilers included. Correct rather than fast.

Both render from one program, so they must produce byte-identical outputs and the tests pin that.
The `clang`-only `__builtin_elementwise_*` plus `-fveclib` path is not a mode; if C-81 shows the
per-lane trig dominating on x86 after widening, it becomes a one-line option inside `gnu`,
guarded by `#ifdef __clang__`. `zig cc` ties in as the compiler the JIT can always ship, on all
three operating systems and cross-compiling for embedded targets, which is what makes `gnu` a
safe default there; AOT users with another compiler pass the `c` mode.

Draft for `docs/api/codegen.md`, to land with C-79 and not before (the published docs describe
what exists):

> **Render modes.** `render_c_module(..., dialect="gnu")` is the default and emits C99 plus the
> GNU vector extension: `__attribute__((vector_size))` types for values computed across several
> stages at once, `__builtin_shufflevector` and `__builtin_convertvector` for the layout changes
> at the interface, and `restrict` on scratch buffers. It compiles with gcc 12 or newer, any
> clang including Apple clang, `zig cc` and armclang, and it is what the JIT uses. `dialect="c"`
> emits the same computation as plain C99 with an inner loop over lanes; it compiles with any
> C99 compiler, including MSVC and embedded vendor compilers, at the cost of leaving
> vectorization to that compiler. The two modes produce identical results bit for bit. Both keep
> `libm` as the only dependency; transcendental functions are always scalar calls.

## 4. Items recorded in `internal/todo.md` as C-77 to C-81

All five sit in the Deferred list beside C-8 and are not required for 0.1.0; the todo entry says
how each relates to C-8's steps (C-77 is its steps 0 and 1, C-79 one more range rewrite after
them, C-78 independent). In the M4's payoff order: C-77 fusion of the derivative assembly, C-78 the three small passes,
C-79 explicit lanes with the two render modes, C-80 the parameter-only prologue, C-81 the x86
re-run that may swap C-77 and C-79. Expected differences on x86: a similar assembly share, a much
larger width gain, and per-lane scalar trig roughly ten times more expensive than Apple's libm,
which is what decides whether the `__builtin_elementwise_*` option earns its line.

## Reproduction

```sh
uv run internal/notes/perf_2026_09_22/gen_hess_kernel.py 200 /tmp/hess200
cd /tmp/hess200
cp ~/dev/scaly/race_car_closed_loop_N200_hess_lower_fast.c .
uv run --project <repo> <repo>/internal/notes/perf_2026_09_22/mkvariant.py 2 lane   # W in {1,2,4,8}, mode lane|orig
cc -O3 -mcpu=native -fno-math-errno -DFN=race_car_closed_loop_N200_hess_lower \
   -o bench variant_w2_lane.c <repo>/internal/notes/perf_2026_09_22/bench_hess.c -lm && ./bench 0 out.bin
```

`mode orig` needs the original stage kernel extracted from the generated source into
`orig_kernel.c` (lines of `..._hoisted_1_raw`, `static` dropped). `kern_only.c` times the 200
kernels alone; `trigcost.c` times the libm calls alone.
