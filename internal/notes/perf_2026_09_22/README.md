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
| One pass, write `res[0]` directly | Fusion of the sparse-derivative assembly (`gather(transpose(jvp_many(...)))` in `ad/sparse.py`) into the mapped producer loop | program dialect, the [#69] range propagation | 15 µs of 43.5 |
| Only the lower triangle | Dead-store elimination across the call boundary once the kernel body is a fused range body | falls out of fusion plus existing `fold_arith` | 10 µs |
| Closed-form cost block, masks and all-ones tables gone | Constant-tile folding: a `static const` table that is a tile of period P becomes `k[i % P]`, and a scalar when P = 1. `k0` (4800 ones), `k22` and `k26` (period 6 masks) are exactly what [#94] lists as unexplained | program dialect, generalizes the C-45 bake | small alone, frees `fold_arith` to kill the masked terms |
| Recovery gather `k44` | Already affine: values `0,4,5,8,12,13,16,17,18,20,21,22,23` then `+24` per stage, a residual of 13 under `AffineIndexMap`; the terminal stage's 7 entries break the period. Peel the last trip of a mapped axis when its slice differs, then C-9's map applies | lowering, `passes/affine.py` | metadata, not time |
| Replace 14 divisions by reciprocals | Both divisors (`params[2]`, `0.5 * wheelbase`) are loop invariant. `x / y` with `y` invariant becomes `x * inv_y` with `inv_y` hoisted; a policy flag, since it moves the last bit | program dialect, `hoist_invariant` plus an arith rule | about 2 µs (the `-ffast-math` gap) |
| Cache the cost block across SQP iterations | Split every oracle into a parameter-only prologue and a body; the solver plugin runs the prologue once per solve | Function or solver level, not a compiler pass | unmeasured, problem dependent |
| SoA staging arrays, explicit vector types, libmvec | Lane widening of the mapped range | program dialect plus renderer, see section 3 | 4 µs here, 30 µs on the x86 machine |
| Reference-heading trig computed once, not 402 times | CSE across the two mapped callees that both compute `cos(ref[2])`, `sin(ref[2])`; needs the callees fused into one range first | falls out of fusion | 1 µs |

Not general: the `2σ Rᵀ diag(w) R` algebra. It is trig-identity rewriting on one problem's
structure; do not build a pass for it.

## 3. The loop compiler and where vectorization sits in it

[#69] already plans a rangeify-shaped loop compiler in three steps: ranges with
the three-case propagation rule, reductions to per-lane accumulators, the reduce-under-broadcast
rule. The measurements say what to put before and after those steps.

### 3.0 Make the mapped callee a range body

Today `_lower_vmap` emits `FOR(GLOBAL) { CALL callee(views) }` and every later pass stops at the
`CALL`. `fuse_elementwise` fuses only single-store loops, so nothing downstream of a `VMAP` fuses,
which is why the assembly exists at all. The missing step is small: after `scalarize`, a callee
whose body is scalar form becomes the loop body of every `FOR` that calls it, with its views turned
into index expressions over the loop variable. Then step (1) of [#69] propagates the consumers'
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
per lane and a horizontal add, which is what [#69] step 2 does anyway). When both a long horizon and
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
generated source and linking `-lmvec`; that is a rendering option, not a dependency. The x86
re-run (section 5) measured it at 26.6 against 12.7 µs, and the decision after it was still no:
the generated C stays free of anything but libm on every operating system and compiler.

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
The `clang`-only `__builtin_elementwise_*` plus `-fveclib` path is not a mode. C-81 (section 5)
showed the per-lane trig dominating on x86 after widening, and that declaring glibc's `_ZGV*`
prototypes would close the gap on all three compilers; it stays out, with every other vector
libm, for portability of the generated C. `zig cc` ties in as the compiler the JIT can always ship, on all
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

### 3.7 Revision the same evening: target-aware opt-ins, portable by default

Ted's supervisor proposed the opposite of 3.6: a user-supplied *target* (OS × arch × compiler ×
libc, seven names plus `generic`) for which Scaly emits the fastest code, auto-detected in the
JIT. Three models were asked to weigh it against sections 1 and 5 (gpt-6-astra through `codex
exec`, Fable 5.1 and Opus 5.5 through `claude -p`). All three, and Ted, landed on the same
middle: target awareness earns its place, the seven-name taxonomy does not, and the libmvec ban
in 3.6 is reversed as an opt-in.

- Only three things change the emitted C: the dialect, the lane width and the vector libm.
  Compiler and OS change flags only; `macos_arm64_gcc` and `macos_arm64_appleclang` would render
  byte-identical files. So the design is three render options (`dialect`, `lanes`,
  `vector_libm`) plus a build recipe keyed by CPU level, not named targets.
- `lanes="auto"` keeps the preprocessor block of 3.2 for AOT files (one file, several CPU
  builds, embedded targets fall to W = 1 untold); the JIT passes a fixed integer. Once external
  vector calls exist W is no longer a pure performance knob: `_ZGVeN8v_*` needs a 512-bit
  register, so the libmvec guard must require both the width and the ISA macro.
- `vector_libm="glibc"` declares the prototypes in the source, which section 5 showed is
  compiler-neutral (12.7, 12.8, 11.6 µs on gcc, clang, `zig cc`). Off by default in AOT because
  it moves results (4 ulp); on in the JIT on glibc x86-64; guarded by `__GLIBC_PREREQ(2, 35)`
  for `tanh` with an `#error` naming the option and `-lmvec`. Accelerate stays out (vForce is
  array-based and would undo the fusion; Apple's scalar `sin` is 2 ns), as do SLEEF and own
  polynomials. LLVM documents AArch64 libmvec with glibc 2.40+, a later measurement.
- The byte-identical gate is narrowed to same compiler, same flags, `vector_libm="none"`;
  bitwise equality across targets never existed.
- Two side findings became items: the JIT cache key hashes `-march=native` as a string, not the
  CPU it resolves to ([#63]), and `zig cc` moves from last fallback to preferred JIT compiler
  ([#82]). The C-79 entry, moved to `c77_c79_implementation.md` when it closed, holds the full
  design and gates; the docs draft above gains the `lanes` and `vector_libm` paragraphs when it
  lands.

## 4. Historical tasks C-77 to C-81

All five sit in the Deferred list beside [#69] and are not required for 0.1.0; the todo entry says
how each relates to [#69]'s steps (C-77 is its steps 0 and 1, C-79 one more range rewrite after
them, C-78 independent). In the M4's payoff order:

Added after the x86 run: [#76], a frame budget in `pack_workspace`. The hand-written kernel's
"no workspace, about 40 kB of stack" is the memory side of fusion: the fifteen full-length
intermediates disappear and per-stage locals plus lane staging remain, which is fine on a host
and wrong on a solver thread with a small stack or a Cortex-M, where a caller-provided `w[]` in a
static section is what an integrator wants. The per-buffer 1024-double spill threshold bounds no
frame; a per-procedure estimate against a budget does, and `-Wframe-larger-than=` makes the
estimate testable. C-77 fusion of the derivative assembly, C-78 the three small passes,
C-79 explicit lanes with the two render modes, C-80 the parameter-only prologue, C-81 the x86
re-run that may swap C-77 and C-79. Expected differences on x86: a similar assembly share, a much
larger width gain, and per-lane scalar trig roughly ten times more expensive than Apple's libm,
which is what decides whether the `__builtin_elementwise_*` option earns its line.

## 5. The x86 reference machine, 2026-09-22 (C-81)

Same variants on the reference machine of `docs/benchmarks/fairness.md` (Ryzen 9 7940HS, Zen 4 with
AVX-512, glibc 2.39, `performance` governor, boost off, pinned to one core), protocol flags
`-O3 -march=native -fno-math-errno`, gcc 13.3 (the JIT's compiler), clang 20.1 and a native
`zig cc` (zig 0.16, clang 21). Best of 5 × 20000 calls; every number repeated within 3% except the `-ffast-math` rows, which vary by 10%. All
variants match the gcc baseline to 9.1e-13 absolute on entries up to 1400; the per-lane libm
variants to 5.3e-14. `x86_variants.sh` beside this note rebuilds and times everything.

| Variant | gcc 13 | clang 20 | `zig cc` | Vector libm calls |
| --- | ---: | ---: | ---: | --- |
| Generated code as of today | 108.5 | 77.3 | 84.1 | |
| Same, `noinline` removed | 110.1 | 76.2 | | |
| The 200 stage kernels alone, out of line | 71.7 | 60.3 | | |
| Fused per-stage assembly, original out-of-line kernel | 62.0 | 52.5 | | |
| Fused assembly, kernel inlined, scalar source | 59.4 | 24.6 | 24.4 | |
| Same, 2 lanes, libm per lane | 38.5 | 32.4 | | |
| Same, 4 lanes | 30.3 | 25.6 | | |
| Same, 8 lanes | 26.6 | 25.8 | 24.5 | |
| Fused scalar source, `-ffast-math` | 11.3 to 13.4 | 23.0 | 22.8 | gcc: `_ZGVeN8v_{sin,cos,tanh}` |
| 4 lanes, glibc `_ZGVdN4v_*` prototypes declared, `-lmvec` | 17.5 | 15.6 | | 4 wide |
| 8 lanes, glibc `_ZGVeN8v_*` prototypes declared, `-lmvec` | 12.7 | 12.8 | 11.6 | 8 wide, `tanh` included |
| Same with `-ffast-math` | 11.1 to 12.3 | 10.8 | 11.7 | isolates the reciprocal gain, 0.4 to 1.6 µs |
| The other agent's hand-written AVX-512 file | 12.65 | 12.6 | 13.4 | 8 wide |
| Fused scalar source, `__attribute__((simd("notinbranch")))` on `sin`, `cos`, `tanh`, no fast-math | 12.6 to 13.1 | 24.6 (ignored) | | gcc: 8 wide, `tanh` included |
| Fused scalar source, `-fveclib=libmvec` | | 21.9 | 44.8 | 4 wide, no `tanh`; the loop is 8 wide |
| Same, `-mprefer-vector-width=256` | | 14.6 | | 4 wide loop, 4 wide calls |
| 4 lanes, `__builtin_elementwise_*`, `-fveclib=libmvec` | rejected | 14.7 | 14.8 | 4 wide, `tanh` scalar |
| 8 lanes, `__builtin_elementwise_*`, `-fveclib=libmvec` | rejected | 24.5 | | none: LLVM's libmvec table stops at 4 lanes |
| 2000 `sin`/`cos` and 800 `tanh` scalar libm calls | 23.5 | 23.6 | | 8.4 ns per call, against 2.0 on Apple libm |

All times in µs per call. What it says, against the M4 table in section 1:

1. **Fusion is still the first step, and larger here.** 108.5 to 59.4 on gcc, 45% of the
   baseline against 35% on the M4. The kernel share is also larger: the 200 kernels alone are 66%
   of the gcc baseline. C-77 keeps its place before C-79.
2. **Inlining alone buys nothing on either compiler**, as on the M4 (108.5 to 110.1, 77.3 to
   76.2). But once fused, the two compilers part ways: clang vectorizes the fused scalar loop 8
   wide by itself, scalarizing the 14 `sin`/`cos`/`tanh` calls per stage into per-lane calls
   (24.6 µs, the same as our explicit 8-lane variant at 25.8), while gcc refuses to vectorize any
   loop containing a call without a simd clone and stays scalar at 59.4.
3. **Per-lane scalar trig dominates after widening**, which is the question section 3.6 left to
   this run. Explicit 8 lanes with libm per lane is 26.6 on gcc; the identical source with the
   three glibc `_ZGVeN8v_*` prototypes declared is 12.7. That 14 µs is the trig, and it is the
   whole gap to the hand-written kernel (12.65): the explicit-lane render with per-lane trig, as
   C-79 was decided, would sit at 2.1× the hand kernel and miss its own 1.5× gate on this machine.
   With the prototypes declared it is at 1.0×.
4. **The right one line is not `__builtin_elementwise_*`.** On clang 20 and zig's clang 21 the
   libmvec mapping stops at 4 lanes and has no `tanh`, so the 8-wide `__builtin_elementwise_*`
   variant degrades to per-lane scalar calls (24.5) and the 4-wide one reaches only 14.7. Declaring
   the glibc prototypes ourselves works on gcc, clang and `zig cc` alike, at 8 lanes, `tanh`
   included: 12.7, 12.8, 11.6. That replaces the `#ifdef __clang__` option of section 3.6.
5. **On gcc, the compiler does the widening itself given a declaration.** Three
   `__attribute__((__simd__("notinbranch")))` declarations on the fused *scalar* source, no
   vector types, no `-ffast-math`, give 12.6 to 13.1 µs with `_ZGVeN8v_*` calls, equal to the
   hand-written kernel. This is exactly what glibc's `bits/math-vector.h` does under
   `__FAST_MATH__`, which is why `-ffast-math` on the scalar source jumps from 59.4 to 11.3; the
   remaining 0.4 to 1.6 µs of fast-math is the reciprocal of invariant divisors, C-78. Clang ignores the
   attribute and `#pragma omp declare simd` on external functions, so on clang the explicit
   `_ZGV` prototypes remain the way. So the C-79 `c` mode (scalar body, inner lane loop) plus a
   per-target block of simd declarations reaches the hand-written kernel on gcc, and the `gnu`
   mode plus the same declarations reaches it on clang and `zig cc`.
6. **`zig cc` native links `-lmvec`.** Every `-lmvec` build linked and ran; `ldd` shows
   `libmvec.so.1`. The cross-compile limitation of section 3.5 does not apply to the JIT's native
   use. Its own `-fveclib=libmvec` behaves like clang's (4 wide, no `tanh`).
7. **Side finding: gcc 13 is 40% slower than clang 20 on today's generated code** (108.5 against
   77.3 at the same flags), and 8% slower than `zig cc`. The M4 numbers are all clang. The JIT
   picks `cc`, which is gcc on this machine, and the published Scaly timings come from it. Worth
   one measurement across the sweep before the release, not settled here.

What changes in the todo: C-81 is done and C-77 stays first. Points 3 to 5 show how to reach the
hand-written kernel, and the decision is still not to: libmvec is x86 glibc only, and the
portability of the generated C across operating systems and compilers outranks the 14 µs. So
transcendentals stay per-lane scalar libm calls everywhere, and C-79's gate is measured against
`variant_w8_lane.c` (26.6 µs on gcc 13 here) rather than the hand-written file. The same
discussion settled two more things for C-79: W is chosen by the preprocessor from the compiler's
target macros (`__AVX512F__` 8, `__AVX__` 4, `__SSE2__` or `__aarch64__` 2, else 1, override
`-DSCALY_LANES`), and the stage body is rendered once as an `always_inline` function called for
the `N / SCALY_LANES` full trips plus one remainder call under `#if (N % SCALY_LANES) != 0` with
clamped loads and guarded stores, so there is no scalar tail. Staging is sized for W = 8 to keep
the header's workspace size macro independent. Two alternatives were weighed and dropped: a
fixed W = 4 everywhere is correct (the compiler legalizes any width) but costs 10 to 12% against
the per-target width on both machines, and a peeled scalar tail costs N mod W scalar trips, 38% of
the stages at N = 13 and W = 8, plus a second copy of the body in the source. The guarded store at
the ABI boundary is a stride-13 scatter of extracted lanes, so no masked vector store is expected
or wanted from either compiler; with the lane count a constant at each call site the guards fold
away in the full trips and survive only in the remainder call.

## Reproduction

```sh
uv run internal/notes/perf_2026_09_22/gen_hess_kernel.py 200 /tmp/hess200
cd /tmp/hess200
cp ~/dev/scaly/race_car_closed_loop_N200_hess_lower_fast.c .
uv run --project <repo> <repo>/internal/notes/perf_2026_09_22/mkvariant.py 2 lane   # W in {1,2,4,8}, mode lane|orig
cc -O3 -mcpu=native -fno-math-errno -DFN=race_car_closed_loop_N200_hess_lower \
   -o bench variant_w2_lane.c <repo>/internal/notes/perf_2026_09_22/bench_hess.c -lm && ./bench 0 out.bin
```

`bench_hess.c` includes the generated header, so add `-I.` when it is not compiled from the kernel's
directory. `x86_variants.sh <repo>` derives the vector-libm variants and runs the whole section 5
table with gcc, clang and `zig cc`. `mode orig` needs the original stage kernel extracted from the generated source into
`orig_kernel.c` (lines of `..._hoisted_1_raw`, `static` dropped). `kern_only.c` times the 200
kernels alone; `trigcost.c` times the libm calls alone.

[#69]: https://github.com/PREDICT-EPFL/scaly/issues/69
[#94]: https://github.com/PREDICT-EPFL/scaly/issues/94
[#63]: https://github.com/PREDICT-EPFL/scaly/issues/63
[#82]: https://github.com/PREDICT-EPFL/scaly/issues/82
[#76]: https://github.com/PREDICT-EPFL/scaly/issues/76
