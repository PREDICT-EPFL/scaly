# Code generation

Alloy compiles to C, and there is only one path to it. Calling a function from Python and shipping
one in a C++ application run the same lowering, the same passes and the same renderer — the
difference is only whether the result is compiled in place or written to a file.

That is deliberate. There is no Python evaluator to disagree with the generated code, so anything
you test from Python is the artifact you deploy.

The two ends of that one path have the usual names. Compiling a function when it is first called
from Python, then caching the result, is the **just-in-time** path (JIT). Rendering the C to files
that you build and link yourself is the **ahead-of-time** path (AOT). Both are described below, in
that order, and everything before the last step is shared between them.

## Calling from Python: the JIT path

```python
result = fn(x_value)
```

The first call lowers the graph, renders a translation unit, compiles it with `cc`, caches the
shared library, loads it, and dispatches through the C ABI. Every later call in this process reuses
the loaded handle. Every later call in *any* process reuses the cached library, because the cache
lives on disk.

The cache key is a SHA-256 over the cache format version, the ABI signature, the function name, the
generated source text and the compile flags. Two functions that render to the same source but have
different names or different flags get different entries; two that are genuinely identical share
one.

```python
fn.recompile()    # drop the in-process handle and the on-disk entry for this function
```

Cached artifacts live under `$XDG_CACHE_HOME/alloy/jit` — or `~/.cache/alloy/jit`, or wherever
`ALLOY_CACHE_DIR` points. Deleting that directory is always safe.

There is no fallback path. A missing compiler raises `JitUnavailable`; a compiler that returns an
error, or a function placed on a non-host device, raises `JitError`. An operation the lowerer does
not cover raises `LoweringError` when you render files, and reaches you as `JitUnavailable` with that
`LoweringError` as its cause when you call the function from Python.

## Writing files: the AOT path

```python
from alloy.codegen import render_c_module

module = render_c_module(fn)
module.header           # the .h text
module.source           # the .c text
module.header_name      # the filename it expects
module.source_name
module.workspace_size   # the f_SZ_W the header declares
module.link_flags       # link flags for any solver plugins reached
```

Or write the pair directly:

```python
from alloy.codegen import write_module
write_module(fn, out_dir)
```

From a shell:

```bash
uv run python -m alloy.codegen mymodule:my_function -o generated/
```

The argument is `<module>:<attribute>` — an importable module and the name of a `Function` in it.

Both the ahead-of-time output and the JIT read the *same* `CModule`, produced from a single
lowering. The header's `SZ_W` and the source's actual scratch use cannot drift apart, because there
is only one number.

## Using the result

The generated pair has no dependency on alloy, on Python, or on anything but libm. Every function
is reachable through one signature:

```c
int f(const double** arg, double** res, int* iw, double* w, void* mem);
```

```c
#include "f.h"

double x[3] = {1.0, 2.0, 3.0};
double y[1];
const double* arg[] = {x};
double* res[] = {y};
double w[f_SZ_W > 0 ? f_SZ_W : 1];

int rc = f(arg, res, NULL, f_SZ_W ? w : NULL, NULL);
```

The caller owns all the storage, including the `w` scratch array whose required size the header
tells you. From C++, the header also emits typed structs and an inline `f_call` wrapper that builds
the pointer arrays for you; pass `typed_buffers=False` to leave them out.

Compile the generated C for the machine that will run it:

```bash
cc -O3 -march=native -fno-math-errno -c f.c
```

The generated code is compound expressions such as `a*b + c`, and the portable x86-64 baseline
withholds the fused multiply-add instructions they contract into; on a race-car Hessian kernel the
native flags are worth about a fifth of the runtime. `-fno-math-errno` lets `sqrt` and the other
libm calls inline, since nothing in the generated code reads `errno`. The JIT compiles on the machine
that runs the result and passes the same flags. A binary distributed to other machines is the
exception: build it at the portable baseline, as the solver plugin wheels are.

The full contract — status codes, sparse output tables, memory hooks, what a translation unit
contains — is in [the C ABI](../how_it_works/c_abi.md).

## How much scratch space

```python
from alloy.codegen import workspace_size
workspace_size(fn)
```

Small temporaries stay as C locals inside the generated function and never show up in this number.
Large ones are spilled into the caller's `w[]`, and this is the total of those. Small functions
usually report `0`.

The threshold is a pass decision, not a rendering one — see
[Lowering and optimization](../how_it_works/lowering.md#the-optimization-pipeline).

## Functions that call functions

When a function has callees, they are rendered as `static inline` bodies in the same translation
unit and invoked directly. Only the root is exported through the ABI. That is what keeps the output
small: a stage function used a hundred times is one C function called a hundred times, not a hundred
inlined copies.

A solver is the one exception — its wrapper comes from the solver plugin rather than from the
program dialect, and it drives its oracles, which are rendered normally. See
[Solvers](solvers.md#shipping-one-in-c).

## Checking the toolchain

```bash
uv run python -m alloy.codegen.toolchain
```

Prints the active compiler, the cache root, the shared-library extension and the state of vendored
solver discovery. Run it first when a compile fails for reasons that are not about your graph.
