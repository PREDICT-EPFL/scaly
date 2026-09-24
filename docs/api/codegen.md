# Code generation

The render functions produce a C kernel, a C or C++ header, and a build recipe. The recipe
records the CPU target, lane width, C dialect, and numerical options used by the module.

## Rendering

::: scaly.codegen.aot.CModule

::: scaly.codegen.aot.render_c_module

::: scaly.codegen.aot.render_c_source

::: scaly.codegen.aot.render_c_api_header

::: scaly.codegen.aot.write_module

::: scaly.codegen.aot.workspace_size

## Render options

These keyword options apply to `render_c_module`, `render_c_source`, `render_c_api_header`,
`write_module`, and `workspace_size`.

| Option | Default | Meaning |
| --- | --- | --- |
| `cpu` | `"generic"` | CPU baseline recorded in the recipe. Accepted values are `generic`, `native`, `x86-64-v3`, `x86-64-v4`, and `apple-m4`. |
| `lanes` | `"auto"` | Target-selected lane width. An integer `1`, `2`, `4`, or `8` requests a fixed width. `1` disables range widening. |
| `dialect` | `"gnu"` | GNU vector extensions. `"c"` emits scalar operations inside lane loops using C99 syntax. |
| `vector_libm` | `"none"` | Per-lane scalar math calls. `"glibc"` selects supported x86-64 libmvec functions and requires `dialect="gnu"`. |
| `reciprocal` | `False` | Permits invariant `x / y` to become `x * (1 / y)`. This can change rounding, overflow, and underflow. |

The `lang` option selects the header language. It does not select the kernel dialect.
`module.recipe.cpu_flags` contains the target compiler flags. `module.link_flags` includes
solver dependencies and the selected math library. Generated headers and sources carry a recipe
comment with GCC and Clang build commands. The recipe's target flags must match the object build.

### Lane selection and tails

With `lanes="auto"`, generated source selects `SCALY_LANES` from the compiler's target macros.
The widths are eight for AVX-512, four for AVX or fixed SVE widths of at least 256 bits,
two for SSE2 or AArch64, and one otherwise. `-DSCALY_LANES=1`, `2`, `4`, or `8` overrides
that selection. Fixed integer options emit a fixed width instead of target detection.

Each widened range can cap the requested width to limit register pressure. The cap uses the
peak number of live scalar values and an estimate of the target's register capacity. It also
caps the width at the next power of two of the static iteration count. Width is
therefore a request, not a promise that every loop uses that many lanes. Unsupported dependencies
retain their scalar loops.

A widened range has one helper body. Full chunks and the final partial chunk call the same helper
with different valid-lane counts. Tail loads use clamped indices, and only valid lanes store
results. Local lane storage reserves capacity for eight lanes, so changing `SCALY_LANES` does
not change the header's workspace requirement. Active elements use the helper's effective width
as their stride, including in the final partial chunk. Input and output arrays retain their
original application binary interface layout.

### C dialects and math libraries

`dialect="gnu"` uses GNU vector types and compiler builtins. Its compiler family is GCC 12 or
later, Clang, `zig cc`, and Armclang with the required GNU extensions. `dialect="c"` uses scalar
lane loops and is intended for C99 compilers without those extensions. Both modes use the same
Program IR transformations, buffer layouts, and tail rules. Typed buffers use 16-byte alignment
in C11 and C++, and natural alignment in C99. `minimum` and `maximum` remain
per-lane operations.

With `vector_libm="none"`, transcendentals call scalar libm once per lane. The `glibc` option
emits guarded declarations for the supported `sin`, `cos`, `tan`, `exp`, `log`, `pow`, and `tanh`
vector symbols. Guards check x86-64, glibc, and the target feature required by the vector width.
`tanh` also requires glibc 2.35 or later. Incompatible builds fail at compile time. Cross builds
whose C library lacks libmvec use `vector_libm="none"`.

Vector math is a separate numerical policy. The glibc 2.39 manual documents a maximum error of
four units in the last place for x86-64 libmvec functions. Results can differ from scalar libm.
See [glibc's math accuracy reference](https://sourceware.org/glibc/manual/2.39/html_node/Errors-in-Math-Functions.html).

The `glibc` recipe includes `-lmvec`. On the reference Ubuntu host, the `libm.so` linker script
already includes libmvec through `AS_NEEDED`, so `-lm` also resolves these symbols. The explicit
recipe flag records the dependency without relying on that linker-script arrangement.

Range widening preserves the original order of reduction additions. It does not replace a scalar
sum with a tree of partial sums. With the same compiler flags and scalar math policy, reduction
results retain the scalar evaluation's bit pattern. Reciprocal and vector-libm options have their
own rounding behavior. General algebraic simplification has the limits described under
[arithmetic semantics](../how_it_works/lowering.md#arithmetic-semantics).

### CPU recipes and the command line

`generic` adds no CPU-specific flag. `native` uses `-march=native` on x86 and `-mcpu=native` on
AArch64. The x86-64 levels use `-march=x86-64-v3` and `-march=x86-64-v4`; `apple-m4` uses
`-mcpu=apple-m4`. `native` describes a build for the current host. Distributable binaries need a
CPU baseline supported by every destination machine.

The command-line equivalents are `--cpu`, `--lanes`, `--dialect`, `--vector-libm`, and
`--reciprocal`:

```bash
uv run scaly_codegen my_model:kernel -o generated --cpu generic --lanes auto --vector-libm none
uv run scaly_codegen my_model:kernel -o generated --cpu x86-64-v3 --lanes 4 --dialect gnu
```

`scaly_toolchain` reports the compiler and its detected native recipe. The just-in-time compiler
uses that native recipe with a fixed lane width. It enables libmvec on supported glibc x86-64
hosts after checking the host version. `SCALY_VECTOR_LIBM=none` forces scalar libm;
`SCALY_VECTOR_LIBM=glibc` explicitly requests libmvec. The source recipe and compile and link flags
participate in the JIT cache key. See [environment variables](../guide/env_vars.md).

## The ABI

::: scaly.codegen.abi.c_api_signature

::: scaly.codegen.abi.BufferType

## The C++ header

::: scaly.codegen.cpp.render_cpp_header

## The CasADi layer

::: scaly.codegen.casadi.check_casadi_layout

::: scaly.codegen.casadi.casadi_sparsity

::: scaly.codegen.casadi.casadi_scratch

## Compiling and caching

::: scaly.codegen.jit.CompiledFunction

::: scaly.codegen.jit.invalidate_cache

::: scaly.codegen.jit.JitError

::: scaly.codegen.jit.JitUnavailable

## Toolchain

::: scaly.codegen.toolchain.BuildRecipe

::: scaly.codegen.toolchain.native_recipe

::: scaly.codegen.toolchain.find_c_compiler

::: scaly.codegen.toolchain.cache_root
