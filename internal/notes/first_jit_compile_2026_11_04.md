# The first JIT compile on a fresh machine

Measured and decided on 2026-11-04 for [#252]. This settles the question left open in
[What the Windows machine should settle](windows_toolchain_options_2026_10_09.md#what-the-windows-machine-should-settle):
document the first-compilation delay rather than add a warm-up mechanism.

## What the first compile builds

The JIT compiles with `-march=native`. On first use, zig builds its C runtime for that exact
target: compiler_rt, and on Windows the MinGW-w64 C runtime (`libmingw32.lib` and its
neighbours). It keeps that build in its global cache, keyed on the zig version and the CPU model
and features. The path of the zig installation does not enter the key. On Linux, a second
virtual environment's zig reused the runtime built by the first. A C++ link builds libc++,
libc++abi and libunwind in addition. Only the benchmark driver links C++. No user path does.

## Measurements

zig 0.17.0 from `zig-bin`, with an empty `ZIG_GLOBAL_CACHE_DIR` and Scaly cache, and each row in a
new process. The Windows rows come from a temporary step in PR #256's CI, runs 41811995716 and
41823146142, and the Linux row from a laptop.

| Machine | Cores | First `sc.function` call, cold zig cache | Next new function | Cache hit, whole process | First C++ link, C runtime warm |
|---|---|---|---|---|---|
| Linux, AMD Ryzen 9 7940HS (zig model `znver4`) | 16 | 3.1 s | 0.17 s | 0.35 s | 16.7 s |
| GitHub Windows runner, AMD EPYC 7763 (`znver3`) | 4 | 88 s | 0.52 s | 0.71 s | 153 s |
| GitHub Windows runner, Intel Xeon Platinum 8370C (`icelake_server`) | 4 | 77 s | 0.54 s | 0.75 s | 139 s |

A bare `zig cc -march=native -shared` of a small file took 76 to 87 s cold and 0.15 s warm on the
runners, and 3.0 s and 0.03 s on Linux, so the runtime build is nearly all of the first call. The
installation guide already gave about 13 s on a recent Windows desktop and 2 to 3 s on Linux and
macOS, from #235. CI's "Warm zig's cache" step, which builds both runtimes, took 3.5 to 4.3 min
before PR #256 cached it. On a hit it takes about 10 s (see that pull request).

The CPU model is part of the key. In a cache already warm for the host, a one-step
`zig cc -march=skylake -shared` built the runtime again: 82 to 84 s on the runners and 2.6 s on
Linux. A cache copied between machines with different CPUs is therefore no help, which is why CI
keys its zig cache on the CPU model.

## A runtime built once for a generic CPU

zig can build its runtime for a generic CPU while the kernel keeps `-march=native`. The kernel is
compiled with `zig cc -c -march=native` and linked with `zig cc -mcpu=baseline -shared`. After one
baseline runtime build, a kernel compiled for another CPU model linked in 0.25 to 0.30 s instead
of rebuilding the runtime. It costs:

- A larger first build: 124 to 148 s instead of 76 to 87 s on the runners, and 11 s instead of
  3 s on Linux. With an explicit CPU, zig treats the target as a cross target and builds more of
  its C library.
- Slower math on Windows. zig links `sin`, `cos` and `exp` statically into the DLL from its
  runtime, so these lose the native CPU. Per element on the Xeon runner: `sin` +11%, `cos` +16%,
  `exp` +3%. `atan2`, `log1p`, `pow` and `tanh` are imported from the Universal C Runtime (UCRT)
  DLL and did not change. `fma` was +13% on a loop of 0.5 ns per element, which is too short to
  read much into. On Linux every math function comes from the system glibc, and all were within 3%.
- Two compiler processes for every JIT compile instead of one.

A fresh machine still pays one runtime build. The generic build helps only where a zig cache
travels between CPU models, as in CI, and CI already gets most of that from keying on the model.

## Decision

Document the delay. Each machine pays it once per CPU model and zig version. On a desktop it takes
seconds, and only small CI runners take minutes. A warm-up mechanism, such as a
`scaly_toolchain` option that builds the runtime, would not shorten the delay. It would only move
it, and compiling any function once already does that. The generic runtime is rejected for the
costs above. The installation guide's *Compiler selection* section states the delay, the
per-processor build and the CI runner time.

In CI, `test / windows-x86_64` restores zig's global cache keyed on the zig version, zig's CPU
model and a hash of its CPU features. It saves the cache on a miss outside the merge queue, whose
caches only its own short-lived branch can read. The nightly runs on main, so its saves seed the
cache for every pull request. `install / windows-x86_64` keeps a cold zig cache. It is the one
place where CI compiles on a fresh machine the way a user does, and it is not the slowest job.

[#252]: https://github.com/PREDICT-EPFL/scaly/issues/252
