# Code generation

Scaly compiles to C, and there is one path to it. Calling a function from Python and shipping one
in a C++ application run the same lowering, the same passes and the same renderer. The only
difference is whether the result is compiled in place or written to a file. There is no Python
evaluator, so what you test from Python is the artifact you deploy.

Compiling a function when it is first called from Python, then caching the result, is the
just-in-time (JIT) path. Rendering the C to files that you build and link yourself is the
ahead-of-time (AOT) path. Everything before the last step is shared between them.

## Calling from Python: the JIT path

```python
result = fn(x_value)
```

The first call lowers the graph, renders a translation unit, compiles it with `cc`, caches the
shared library, loads it, and dispatches through the C ABI (application binary interface). Every
later call in this process reuses the loaded handle. Every later call in any process reuses the
cached library, because the cache lives on disk.

The cache key is a SHA-256 over the cache format version, the ABI signature, the function name, the
generated source text and the compile flags. Two functions that render to the same source but have
different names or different flags get different entries; two that are identical share one.

```python
fn.recompile()    # drop the in-process handle and the on-disk entry for this function
```

Cached artifacts live under `$XDG_CACHE_HOME/scaly/jit`, or `~/.cache/scaly/jit`, or wherever
`SCALY_CACHE_DIR` points. Deleting that directory is always safe.

There is no fallback path. A missing compiler raises `JitUnavailable`; a compiler that returns an
error, or a function placed on a non-host device, raises `JitError`. An operation the lowerer does
not cover raises `LoweringError` when you render files, and reaches you as `JitUnavailable` with
that `LoweringError` as its cause when you call the function from Python.

## Writing files: the AOT path

```python
from scaly.codegen import render_c_module

module = render_c_module(fn)                 # f.h and f.c
module = render_c_module(fn, lang="cpp")     # f.hpp and the same f.c
module = render_c_module(fn, casadi=True)    # plus the CasADi 3.8 compatible symbols
module.header           # the header text
module.source           # the .c text
module.header_name      # the filename it expects
module.source_name
module.workspace_size   # the f_SZ_W the header declares
module.link_flags       # link flags for any solver plugins reached
```

Or write the pair directly:

```python
from scaly.codegen import write_module
write_module(fn, out_dir, lang="cpp")
```

From a shell:

```bash
uv run scaly_codegen mymodule:my_function -o generated/ --lang cpp --casadi
```

The argument is `<module>:<attribute>`, an importable module and the name of a `Function` in it.
`--lang c` (the default) writes a C header with a struct per buffer, `--lang cpp` a C++ header with
`Buffer` types in a namespace; `--casadi` adds the symbols acados and `casadi.external` look for;
`--no-typed-buffers` strips the C header down to the pointer signature and the sparsity tables.

The AOT output and the JIT read the same `CModule`, produced from a single lowering. The header's
`SZ_W` and the source's scratch use cannot drift apart, because there is only one number.

## Using the result

The generated pair depends on nothing but libm. Every function is reachable through one signature:

```c
int f(const double** arg, double** res, int* iw, double* w, int mem);
```

```c
#include "f.h"

double x[3] = {1.0, 2.0, 3.0};
double y[1];
const double* arg[] = {x};
double* res[] = {y};
double w[f_SZ_W > 0 ? f_SZ_W : 1];

int rc = f(arg, res, NULL, f_SZ_W ? w : NULL, 0);
```

The caller owns all the storage, including the `w` scratch array whose required size the header
gives. The C header also declares a struct per buffer and a workspace struct, so the same call is
`f_call(&x, &y, &workspace)` with `f_x_t x`, `f_y_t y` and a `f_workspace_t` you place wherever you
like. The C++ header spells it `f::call(x, y, workspace)` with `f::x_t` and friends, which keep the
`Expr` shape and index as `x(i, j)`. Both headers declare the same kernel; pick the one your caller
is written in.

Compile the generated C for the machine that will run it:

```bash
cc -O3 -march=native -fno-math-errno -c f.c
```

The generated code is compound expressions such as `a*b + c`, and the portable x86-64 baseline
withholds the fused multiply-add instructions they contract into, so the native target matters.
The flag ablation behind that rule is in
[fairness](../results/fairness.md#why-the-native-compiler-flags-are-fair). `-fno-math-errno` lets
`sqrt` and the other libm calls inline, since nothing in the generated code reads `errno`. The JIT
compiles on the machine that runs the result and passes the same flags (`-mcpu=native` on AArch64).
A binary distributed to other machines is the exception: build it at the portable baseline, as the
solver plugin wheels are.

The full contract, with status codes, sparse output tables, the C++ `Buffer`, the CasADi layer
and what a translation unit contains, is in
[the generated interface](../how_it_works/generated_interface.md).

## How much scratch space

```python
from scaly.codegen import workspace_size
workspace_size(fn)
```

Small temporaries stay as C locals inside the generated function and never show up in this number.
Large ones spill into the caller's `w[]`, and this is the total of those. Small functions usually
report `0`.

The threshold is a pass decision, not a rendering one; see
[Lowering and optimization](../how_it_works/lowering.md#the-optimization-pipeline).

## Functions that call functions

When a function has callees, scaly renders them as `static inline` bodies in the same translation
unit and calls them directly. Only the root is exported through the ABI. A stage function used a
hundred times is one C function called a hundred times.

A solver is the one exception. Its wrapper comes from the solver plugin, and it drives its oracles,
which are rendered normally. See [Solvers](solvers.md#shipping-one-in-c).

## Checking the toolchain

```bash
uv run scaly_toolchain
```

Prints the cache root, the compiler and where it was found, the solver discovery source with its
include and library directories, the resolved path of each installed solver plugin with whether it
is discoverable and loadable, and the flags the JIT would add for those solvers. Run it first when
a compile fails for reasons that are not about your graph.
