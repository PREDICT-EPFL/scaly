# Seed selector tables of overlapping stage windows, 2026-10-15

Measured on a2r-main (Intel Core Ultra 9 285K, AVX2, no AVX-512), not on the reference machine,
with Clang 18 and glibc vector math. The kernels are npmpc's Scaly Lagrangian Hessian, generated
from the `t3code/issue-19-one-forward-traversal` branch (#166); `main` emits the same selector.

## What grows with the horizon

npmpc's kernel grows linearly with N: `kernel.o` is 237 kB at N=50, 675 kB at N=200 and 6.3 MB at
N=2000, almost all of it `.rodata`. The growth is six `static const double` tables of
7 directions x N stages x 9 entries (3,150 entries each at N=50), all 0 or 1:

```c
for (long long i = 0; i < 3150; ++i) {                    // direction x stage x entry
  int64_t v = (((i / 9) % 50) * 9) + (i % 9);
  s4[i] = k0[i] * s1[v] + k6[i] * s2[v] + k10[i] * s3[v] + k14[i] * t16[v];
}
```

They come from the local-coloring path of the map JVP rule (`_vmap_jvp_many`): each stage is
differentiated in its own few local directions (`s1`, `s2`, `s3`, `t16`), and the result is mapped
into the global coloring's 7 directions by multiplying with the constant global seed gathered at
each local color's representative column. The stage windows `(x_i, x_{i+1}, u_i)` overlap, so the
global coloring is used, and which local result feeds which global direction changes from stage to
stage. The coloring repeats with period 4 along the stages, but its first few stages and the
horizon length break exact periodicity, so the existing periodic-tile folds do not apply.

The tables cost binary size, not much time: the selector loops are a small part of the kernel.

## Alternatives tried and set aside

| approach | binary at N=200 | eval at N=50 | why not |
|---|---|---|---|
| today | 675 kB | 44.8 µs | |
| bake peeled periodic seeds into the stage derivative | 564 kB | 125 µs | 7–8 specialized copies of a large stage body, fusion lost, compile 10 s |
| `fold_tiles` with a peeled head, period and tail along the stage axis | 88 kB | 62.9 µs | per-element index remap turns the contiguous mask loads into gathers |
| scatter each local result into its global color with an index table | 157 kB | 52.6 µs | scatter-adds do not vectorize; stops small masks folding away (`spjac` snapshot grows) |

Each measured the whole kernel, three runs, all cells correct.

## Mask storage, whole kernel

Hand-edited generated kernels (`mask_variants.py`): the format barely moves whole-kernel time
(within about ±5 %), because these loops are small and, as generated, scalar: Clang reports
"cannot identify array bounds", since all candidates and the output are workspace pointers read
through a `/` and `%` index. Forcing vectorization turns the candidate reads into `vgatherdpd`.

## Micro-benchmark of the selector loop

`bench.cpp` with `kernels.c` (and `kernels_avx2.c`) times the loop alone on the real masks
(`extract_masks.py`), pinned to one performance core, median of five, every variant equal to the
`double` reference as numbers (the select variants differ only in signed zeros).

| variant | mask data at N=50 | ns per element, N=50 | N=200 |
|---|---|---|---|
| as generated (flat index, scalar) | 100.8 kB | 1.34 | 1.34 |
| `double`, contiguous loop, auto-vectorized | 100.8 kB | 0.28–0.33 | 0.39–0.41 |
| `uint8` converted to `double` | 12.6 kB | 0.22–0.23 | 0.26–0.33 |
| bits, best auto-vectorized form | 1.6 kB | 0.61 | 0.62 |
| bits, hand-written AVX2 intrinsics | 1.6 kB | 0.25–0.26 | 0.26–0.27 |
| **bits, portable vector types (`sel_K_vector_bits`)** | **1.6 kB** | **0.27** | **0.28** |

Two findings. The loop shape matters most: a direction loop around a contiguous stage x entry loop
vectorizes and is 3–4x faster than today in any format. And bits are as fast as bytes when the
vector code is written explicitly; Clang's auto-vectorizer does not find the bit idiom.

`sel_K_vector_bits` uses the GCC/Clang vector types the generated lane code already uses. Each
direction's row of bits is padded to whole bytes, so 8 consecutive entries share one byte:

```c
hit = ((vi64)(byte + (vi64){}) & bits) == bits;    // broadcast, test each lane's bit: -1 or 0
sum += (vf64)((vi64)candidate & hit);               // keep or zero, no multiply, no branch
```

It compiles to the natural idiom on each target (compiled for AArch64 and Zen 4, not run):

- AVX2, 4 lanes: `vpbroadcastq`, `vpand` with {1,2,4,8}, `vpcmpeqq`, `vandpd`.
- AVX-512, 8 lanes: `vpbroadcastq`, `vptestmq` into `k1`, `vmovdqu64 ... {k1}{z}`, `vaddpd`.
- NEON, 2 lanes: `and`, `cmeq`, `bic`. The `uint8`-to-`double` cast compiles poorly on NEON
  (`umov` per byte, then `ucvtf`), another reason to prefer bits.

## Implementation

`passes/program/pack_masks.py` runs after `fold_arith`. It takes a loop
`out[i] = sum_l mask_l[i] * x_l[g(i)]` whose masks are constant 0/1 tables,
read nowhere else, shaped `(directions, ...)`, and whose other reads depend on the position
`i % positions` only, contiguously or with a constant stride (checked for every `i`). It stores each
mask as `uint8` bits, a direction's row padded to whole bytes, and splits the loop into directions,
blocks of eight positions and the eight lanes of a block, plus a tail. Each `mask * x` becomes
`BIT_SELECT(byte, lane, x)`, which renders as `((byte >> lane) & 1) ? x : 0.0` in scalar code and as
broadcast, shift, compare and AND in lane code. The rewritten reads go to the owner of an aliased
buffer, because `widen_ranges` does not widen a loop that reads through an alias; that is also why
the flat selector loop never vectorized. `uint8` was added to the public dtypes for the tables.

## Result

Scaly Hessian kernels, `main` (`edb3cbb`) against the branch. Size is `kernel.o` from Clang 18 at
`-O3 -march=native -fno-math-errno`; time is the median of three fresh processes (six, interleaved,
at N=200), every cell correct:

| workload | size | `kernel.o` (`.rodata`) kB | eval µs |
|---|---|---|---|
| npmpc | 6 | 120 (19) -> 104 (1) | 10.1 -> 10.6 |
| npmpc | 50 | 257 (157) -> 109 (9) | 49.7 -> 47.5 |
| npmpc | 200 | 686 (629) -> 91 (34) | 220.6 -> 200.5 |
| unbumpercars | 16 | 227 (117) -> 171 (61) | 999.9 -> 999.8 |

race_cars (N = 5, 50, 200) and chain (M = 3) have no such tables and are unchanged. unbumpercars
keeps some tables the pass does not match.

Packing has no size threshold. With `-fno-math-errno` and no fast-math, C compilers keep
`x * 0.0` (it is not `+0.0` for a NaN, infinite or negative `x`), while a known-zero bit selects
`0.0` and folds. On the test loop, packed was smaller at every size from 35 entries up (35 entries:
525 -> 176 bytes of text and data, the packed loop folded entirely into code).

Lane helpers keep an index that is the same in every lane as a lane array. Emitting it as one
scalar instead compiled the test loop to the same instructions, but made npmpc at N=6 take 16-17 µs
instead of 10-11 µs, through some change in Clang's inlining or unrolling of the other helpers
it touched, so it was dropped.
