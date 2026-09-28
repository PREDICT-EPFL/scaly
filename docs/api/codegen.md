# Code generation

`write_module` writes a function's C source and header to disk. The rendering functions return
those artifacts in memory. See the [code generation guide](../guide/codegen.md) for an example
that compiles and calls the result.

Normal numerical calls to a `Function` compile and cache it automatically. Use
`Function.compile()` to prepare it before the first evaluation. The compilation and
toolchain interfaces below support direct control of that process.

`write_module` accepts a `Function` or a `Solver`. Other rendering functions and
`workspace_size` take a `Function`. Pass `solve.function` when exporting a solver.

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

### Lanes, dialects and math libraries

The [code generation guide](../guide/codegen.md#cpu-targets-vector-lanes-and-math-libraries)
explains these options, and [vector lanes](../how_it_works/lowering.md#vector-lanes) shows the
loops they produce. With `lanes="auto"`, generated source selects `SCALY_LANES` from the
compiler's target macros, and `-DSCALY_LANES=1`, `2`, `4`, or `8` overrides that selection. A
fixed integer option emits a fixed width instead. The width is a request. A loop can use fewer
lanes, and a loop that cannot be widened stays scalar. Input and output arrays keep the same
layout at every width, and the header's workspace size does not depend on it.

`dialect="gnu"` uses GNU vector types and compiler builtins, as provided by GCC and Clang.
`dialect="c"` uses scalar lane loops and is intended for C99 compilers without those extensions.
Both give the same buffer layouts. Typed buffers use 16-byte alignment in C11 and C++, and
natural alignment in C99.

With `vector_libm="none"`, transcendentals call scalar libm once per lane. The `glibc` option
emits guarded declarations for the supported `sin`, `cos`, `tan`, `exp`, `log`, `pow`, and `tanh`
vector symbols. Guards check x86-64, glibc, and the target feature required by the vector width.
`tanh` also requires glibc 2.35 or later. Incompatible builds fail at compile time. Cross builds
whose C library lacks libmvec use `vector_libm="none"`. The `glibc` recipe links `-lmvec`.

Vector math is a separate numerical policy. The glibc 2.39 manual documents a maximum error of
four units in the last place for x86-64 libmvec functions. Results can differ from scalar libm.
See [glibc's math accuracy reference](https://sourceware.org/glibc/manual/2.39/html_node/Errors-in-Math-Functions.html).

Widening keeps the order of the additions in a sum. With the same compiler flags and scalar math,
a reduction gives the same bits as the scalar code. Reciprocal and vector-libm options have their
own rounding behavior. General algebraic simplification has the limits described under
[arithmetic semantics](../how_it_works/lowering.md#arithmetic-semantics).

### CPU recipes and the command line

`generic` adds no CPU-specific flag. `native` uses `-march=native` on x86 and `-mcpu=native` on
AArch64. The x86-64 levels use `-march=x86-64-v3` and `-march=x86-64-v4`. `apple-m4` uses
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
hosts after checking the host version. `SCALY_VECTOR_LIBM=none` forces scalar libm.
`SCALY_VECTOR_LIBM=glibc` explicitly requests libmvec. The source recipe and compile and link flags
participate in the JIT cache key. See [environment variables](../guide/env_vars.md).

## Application binary interface

::: scaly.codegen.abi.c_api_signature

## The C++ header

::: scaly.codegen.cpp.render_cpp_header

## The CasADi layer

::: scaly.codegen.casadi.check_casadi_layout

::: scaly.codegen.casadi.casadi_sparsity

::: scaly.codegen.casadi.casadi_scratch

## Compiling and caching

::: scaly.codegen.jit.CompiledFunction

::: scaly.codegen.jit.load_library

::: scaly.codegen.jit.invalidate_cache

::: scaly.codegen.jit.JitError

::: scaly.codegen.jit.JitUnavailable

## Toolchain

::: scaly.codegen.toolchain.find_c_compiler

::: scaly.codegen.toolchain.cache_root
