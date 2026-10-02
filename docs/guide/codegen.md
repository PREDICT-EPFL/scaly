# Code generation

To evaluate a `Function` on concrete numerical values, scaly generates C code
from the symbolic computational graph. This happens both in Python and at the
user's request for deployment in a C or C++ application. This allows, for
example, a controller in a ROS node to reuse the optimal controller written
during prototyping without having to rewrite it. The model is written once,
in Python, and both uses compile from it.

This page assumes familiarity with scaly [functions](functions.md) and basic C
concepts (such as pointers and how to compile C code).

## Exported source and compiled functions

Consider a function representing \(E(x)=\sum_i x_i^2\):

```python
import scaly as sc

@sc.function(sc.arg("x", 3), outputs=sc.arg("energy"))
def energy(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x)
```

The decorated `energy` is an `sc.Function`. Constructing it records the
expression graph without generating or compiling C. There are two ways to
use that graph:

- A numerical Python call automatically generates C, compiles a shared library,
  and evaluates it. Scaly caches the library for later calls. This is the
  _just-in-time_, or JIT, compilation path.
- Export the `Function` to source and header files in a specific directory and
  integrate them with your application's build system. This is the
  _ahead-of-time_, or AOT, compilation path.

Both paths use the same code generator. Exporting source does not compile it,
and exporting alone does not directly require numerical inputs or a C compiler.

```python
from pathlib import Path
from scaly.codegen import write_module

module = write_module(energy, Path("generated"))
print(module.header_name, module.source_name)
# energy.h energy.c
```

`write_module` writes `generated/energy.h` and `generated/energy.c` and returns
a `CModule`. This object holds the generated text, filenames, workspace size,
the build recipe, and link flags. Existing files with those names are
overwritten.

The command-line equivalent refers to the Python object by module and attribute
name. If `energy` is defined in an importable module named `model`, the command is:

```bash
uv run scaly_codegen model:energy -o generated/
```

`model:energy` means import `model` and use its `energy` attribute. The module
must be importable in the command's Python environment. A `model.py` in the
current directory is one way to provide it.

## The C header

The generated header declares one struct per input and output, a workspace
struct, and an `energy_call` helper that evaluates the function. These are
excerpts from `energy.h`:

```c
typedef struct { SCALY_ALIGNAS(16) double data[3]; } energy_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } energy_energy__t;
typedef struct { SCALY_ALIGNAS(16) double data[energy_SZ_W > 0 ? energy_SZ_W : 1]; } energy_workspace_t;

static inline int energy_call(const energy_x_t* x, energy_energy__t* energy_, energy_workspace_t* workspace);
```

Each struct holds one contiguous buffer of `double` values. Arrays use
row-major order, as in a contiguous NumPy array, and a scalar occupies one
entry. Python tuple groups do not become extra buffers. The output name becomes
`energy_` inside the generated interface to avoid a collision with the function
name, which accounts for the double underscore in `energy_energy__t`.

The structs own their storage. You can allocate them on the stack, statically,
or on the heap. C11 and C++ buffers request 16-byte alignment, while C99 uses
natural alignment. The workspace struct reserves at least one entry because
standard C does not allow a zero-length array. The header reports the real
requirement in doubles as `energy_SZ_W`.

For the `energy` model above, the following complete C caller supplies one
three-element input and reads one scalar output. The compilation command below
uses `main.c` as its filename:

```c
#include <stdio.h>
#include "energy.h"

int main(void) {
    energy_x_t x = {{1.0, 2.0, 3.0}};
    energy_energy__t result;
    energy_workspace_t workspace;

    int status = energy_call(&x, &result, &workspace);
    if (status != 0) return status;
    printf("%g\n", result.data[0]);
    return 0;
}
```

```bash
cc -O2 -Igenerated main.c generated/energy.c -lm -o energy_demo
./energy_demo
```

The program prints `14`. It links the C math library with `-lm` and needs no
Python runtime. The [interface reference](../how_it_works/generated_interface.md)
lists the argument checks and return codes.

### The pointer entry

`energy_call` is a header-only helper. The compiled symbol underneath it is a
pointer-based entry point with the same shape for every generated function:

```c
int energy(const double** arg, double** res, int* iw, double* w, int mem);
```

`arg` contains one pointer per input and `res` one pointer per output, in
declaration order. `w` is the workspace of `energy_SZ_W` doubles, or `NULL`
when that size is zero. `iw` and `mem` are unused and take `NULL` and `0`.
The [CasADi section](#casadi-compatible-exports) explains why they exist.
The helper only fills the two pointer arrays and forwards to this entry.

Call the entry directly in two situations. The first is memory that another
library already owns, such as a message array in a ROS node or an Eigen
matrix, which the typed structs would require copying. The second is loading
the compiled library with `dlopen` and no header, where this fixed signature
is the whole contract. The C++ header and the CasADi layer are both built on
the same symbol, so every generated module can be called this way.

## Generated text and export options

`render_c_module` returns a `CModule` without writing files:

```python
from scaly.codegen import render_c_module

module = render_c_module(energy)
print(module.workspace_size)  # 0
print(module.link_flags)      # ()
header = module.header       # C declarations and buffer metadata
source = module.source       # C implementation
```

The zero workspace size means this function needs no caller-provided scratch
buffer. Its local C variables still occupy memory. Empty `link_flags` means
there are no additional solver or vector-math libraries to link beyond `-lm`.

The main export choices are:

| Python keyword        | Command-line option  | Result                                                      |
| --------------------- | -------------------- | ----------------------------------------------------------- |
| `lang="c"`            | `--lang c`           | C header with typed buffer structs, the default             |
| `lang="cpp"`          | `--lang cpp`         | C++ header with buffer types and a namespace                |
| `casadi=True`         | `--casadi`           | Metadata functions for compatible external-function loaders |

## C++ buffers

`lang="cpp"` generates a C++17 header. The implementation remains C, which
lets the C and C++ interfaces use the same compiled numerical function:

```python
cpp_module = write_module(energy, Path("generated_cpp"), lang="cpp")
print(cpp_module.header_name, cpp_module.source_name)
# energy.hpp energy.c
```

The header contains a `Buffer` template with inline storage and compile-time
shape information. Its declaration begins as follows, with indexing methods
omitted from this excerpt:

```cpp
template <typename T, std::size_t... Ns>
struct alignas(16) Buffer {
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * std::size_t{1});
  T data[size > 0 ? size : 1];
  T* ptr() { return data; }
  const T* ptr() const { return data; }
  // Indexing methods omitted.
};
```

`Buffer<double, 2, 3>` holds a two-by-three matrix in row-major order.
`buffer(i, j)` accesses an element, with bounds assertions in debug builds.
`Buffer<double>` is a scalar, accessed as `buffer()`. Empty buffers reserve
one physical element while reporting a logical `size` of zero.

Each function has its own namespace. For `energy`, the generated aliases are:

```cpp
using x_t = Buffer<double, 3>;
using energy__t = Buffer<double>;
using workspace_t = Buffer<double, energy_SZ_W>;
```

These aliases live inside `namespace energy`. The output alias contains two
underscores for the same name-collision reason as the typed C struct.
A complete `main.cpp` using this header is:

```cpp
#include <iostream>
#include "energy.hpp"

int main() {
    energy::x_t x = {{1.0, 2.0, 3.0}};
    energy::energy__t result{};
    energy::workspace_t workspace{};
    int status = energy::call(x, result, workspace);
    if (status != 0) return status;
    std::cout << result() << '\n';
}
```

```bash
cc -O2 -c generated_cpp/energy.c -o energy.o
c++ -std=c++17 -Igenerated_cpp main.cpp energy.o -lm -o energy_cpp_demo
./energy_cpp_demo
```

This prints `14`. The `energy::call` helper constructs pointer arrays and calls
the C entry point, declared with `extern "C"` linkage. `Buffer` owns its data.
To use arrays already owned by another library without copying them into a
`Buffer`, call the pointer interface directly.

## Sparse output patterns

Sparse derivative functions write compact value arrays. The generated header
also contains fixed coordinate tables so the caller can identify the matrix
entry corresponding to each value. For example:

```python
@sc.function(sc.arg("x", 3), outputs=sc.arg("y"))
def measurements(x: sc.Expr) -> sc.Expr:
    return sc.stack([x[0] * x[2], x[1], x[0] + x[2]])

measurements_jac = sc.sparse_jacobian(measurements, "y", "x", name="measurements_jac")
sparse_module = write_module(measurements_jac, Path("generated_sparse"))
```

The Jacobian is

\[
J(x)=\begin{bmatrix}x_2 & 0 & x_0 \\ 0 & 1 & 0 \\ 1 & 0 & 1\end{bmatrix}.
\]

Its output has five values instead of nine. The following declarations are
emitted in `measurements_jac.h`:

```c
#define measurements_jac_spjac_y_x_NNZ 5
#define measurements_jac_spjac_y_x_NROW 3
#define measurements_jac_spjac_y_x_NCOL 3
static const int measurements_jac_spjac_y_x_rows[5] = {0, 0, 1, 2, 2};
static const int measurements_jac_spjac_y_x_cols[5] = {0, 2, 1, 0, 2};
static const int measurements_jac_spjac_y_x_csr_row_ptr[4] = {0, 2, 3, 5};
static const int measurements_jac_spjac_y_x_csr_col_ind[5] = {0, 2, 1, 0, 2};
static const int measurements_jac_spjac_y_x_csr_val_perm[5] = {0, 1, 2, 3, 4};
static const int measurements_jac_spjac_y_x_csc_col_ptr[4] = {0, 2, 3, 5};
static const int measurements_jac_spjac_y_x_csc_row_ind[5] = {0, 2, 1, 0, 2};
static const int measurements_jac_spjac_y_x_csc_val_perm[5] = {0, 3, 2, 1, 4};
```

The prefix combines the function name and derivative output name. For input
`[2, 3, 4]`, the value array is `[4, 2, 1, 1, 1]`. Entry `k` belongs at
`(rows[k], cols[k])`. Coordinates are not guaranteed to be sorted, even though
the row order happens to be sorted in this example.

Compressed sparse row and compressed sparse column formats, or CSR and CSC,
need their own value orders. Their `_val_perm` arrays give the conversion:
`values_csc[k] = values[csc_val_perm[k]]`. Here the CSC values become
`[4, 1, 1, 2, 1]`. The CSR values already have the required order.

C++ headers expose the same metadata as `constexpr` arrays inside a namespace
named after the derivative output. The [sparsity guide](sparsity.md) explains
the corresponding Python patterns and triangle choices for Hessians.

## CasADi-compatible exports

The [pointer entry](#the-pointer-entry) takes the same five arguments, in the
same order, as a function generated by CasADi 3.8, including its
integer-workspace argument `iw` and its memory-handle argument `mem`. Scaly
functions are stateless, so both are ignored. The one difference in the
signature is `iw`, which Scaly declares as `int*` where CasADi uses
`casadi_int*`. Because the entry never reads it, CasADi and acados can pass
their own pointer unchanged. The signature alone does not tell a loader how many inputs a
function has or how large its buffers are. `casadi=True` adds the metadata
queries of CasADi's external-function interface:

```python
casadi_module = write_module(energy, Path("generated_casadi"), casadi=True)
```

For this model, the additional header declarations include:

```c
casadi_int energy_n_in(void);
casadi_int energy_n_out(void);
const char* energy_name_in(casadi_int i);
const char* energy_name_out(casadi_int i);
const casadi_int* energy_sparsity_in(casadi_int i);
const casadi_int* energy_sparsity_out(casadi_int i);
int energy_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
int energy_checkout(void);
void energy_release(int mem);
```

`energy_n_in()` and `energy_n_out()` both return one. The name queries return
`"x"` and `"energy"` at index zero. The sparsity queries describe a dense
three-by-one input and one-by-one output. `energy_work` returns the required
pointer-array and workspace sizes. `energy_checkout()` returns zero and
`energy_release` does nothing because scaly does not use memory handles.

For example, on Linux this command builds a shared library for CasADi:

```bash
cc -O2 -fPIC -shared generated_casadi/energy.c -lm -o generated_casadi/libenergy.so
```

With CasADi installed, it can then be loaded and called:

```python
import casadi as ca

external_energy = ca.external("energy", "generated_casadi/libenergy.so")
print(float(external_energy([1.0, 2.0, 3.0])))  # 14.0
```

The generated `casadi_int` macro defaults to `long long int`, matching CasADi's
loader. Consumers expecting another integer width need a matching build.
In particular, acados uses `-Dcasadi_int=int`. The option works with either
`lang="c"` or `lang="cpp"`.

For sparse outputs, this mode writes values in CSC order. If native scaly
ordering differs, the generated entry point permutes the values using extra
workspace. The module's workspace size includes that storage, and its header
tables describe the converted output order.

!!! warning "Buffer compatibility"
    Dense matrices with both dimensions greater than one, and dense arrays of
    rank greater than two, are rejected by `casadi=True`. Scaly uses row-major
    storage while CasADi uses column-major storage. Scalars, vectors, and compact
    sparse outputs are supported. Every input and output pointer is still
    required, unlike CasADi's convention of treating null inputs as zero and
    skipping null outputs.

## Solver export and deployment dependencies

`write_module` and `scaly_codegen` accept an `sc.Solver` as well as a
`Function`. They export the solver's underlying `solve.function`, including
its objective, constraints, derivatives, and native solver wrapper.
`render_c_module` and `workspace_size` take a `Function`, so those calls use
`solve.function` explicitly.

The C interface always requires the initial variables, initial multipliers,
and parameters. Defaults supplied by the Python `Solver` wrapper do not
become defaults in C. Generated solver wrappers also keep static state, so
the same wrapper cannot be called concurrently.

Ordinary generated model code has no dependency on Python, NumPy, or SciPy.
Solver code additionally needs the corresponding native solver headers and
libraries. The `CModule.link_flags` property reports the flags discovered for
those dependencies and the selected math library in the current environment. Reading it requires the
libraries to be available, even though rendering their wrapper source does not.
For deployment on another machine, provide libraries and paths appropriate to
that target. See [Solver backends](solver_backends.md#native-libraries-and-deployment).

## CPU targets, vector lanes, and math libraries

A generated module carries a `BuildRecipe` in `module.recipe`. It records the
CPU target, lane width, C dialect, and numerical options used to generate the
source. The header and source begin with a comment showing matching GCC and
Clang compilation commands and required math libraries.

A lane is one element processed alongside other elements in a vector
operation. Scaly can group independent loop iterations into lanes without
changing the function's input and output layout. For example:

```python
vector_module = render_c_module(
    energy, cpu="x86-64-v3", lanes=4, dialect="gnu", vector_libm="glibc",
)
print(vector_module.recipe.cpu_flags)  # ('-march=x86-64-v3',)
print(vector_module.link_flags)        # ('-lmvec',)
```

These options request a target configuration. They do not guarantee that a
particular small model uses vector instructions. The compiler widens eligible
loops and can reduce their width to limit register pressure.

### CPU baseline and lane width

Export defaults to `cpu="generic"`, which adds no target-specific compiler
flags, and `lanes="auto"`. Automatic widths follow the target macros set by
the C compiler:

| Target features | Requested lanes |
| --- | --- |
| AVX-512 | 8 |
| AVX or fixed Arm SVE width of at least 256 bits | 4 |
| SSE2 or AArch64 | 2 |
| Other targets | 1 |

A fixed `lanes=1`, `2`, `4`, or `8` bypasses this detection. One lane disables
widening. With automatic selection, `-DSCALY_LANES=4` can select the width at
C compilation time. Partial groups use guarded stores, and input and output
arrays need no extra padding. Workspace capacity supports up to eight lanes,
so changing the lane selection does not change the header's workspace size.

`cpu="native"` selects `-march=native` on x86 or `-mcpu=native` on AArch64.
The other named targets are `x86-64-v3`, `x86-64-v4`, and `apple-m4`.
`module.recipe.cpu_flags` gives their matching flags.

!!! note "The target belongs to the application build"
    Export records the requested target without checking the deployment machine.
    Compile with the recipe's flags and choose a baseline supported by every
    destination. A `native` binary is intended for the build machine, rather
    than an arbitrary computer with the same operating system.

### C dialect and vector math

`dialect="gnu"` is the default and uses GNU vector extensions supported by
GCC and Clang. `dialect="c"` emits scalar lane loops in C99 syntax for
compilers without those extensions. This option is separate from `lang`,
which chooses the language of the caller's header.

Export defaults to `vector_libm="none"`, so each lane uses scalar math calls.
`vector_libm="glibc"` selects supported libmvec implementations of `sin`,
`cos`, `tan`, `exp`, `log`, `pow`, and `tanh`. It requires the GNU dialect,
x86-64 with compatible target features, and glibc. `tanh` additionally
requires glibc 2.35 or newer. Generated guards reject incompatible builds.
The recipe adds `-lmvec`, in addition to the usual `-lm` math-library link.

Vector math can produce different rounded results from scalar math. Likewise,
`reciprocal=True` allows invariant division `x / y` to become multiplication
by a precomputed `1 / y`, which changes rounding and can change overflow.
Both are separate numerical choices. Neither enables general fast-math flags.
See [arithmetic semantics](../how_it_works/lowering.md#arithmetic-semantics)
for simplifications that apply independently of these options.

The same choices are available to `scaly_codegen` as `--cpu`, `--lanes`,
`--dialect`, `--vector-libm`, and `--reciprocal`. The
[render-options reference](../api/codegen.md#render-options) lists their values.

## Compilation and caching in Python

A numerical call compiles on first use unless a matching artifact already
exists in the disk cache. Scaly then keeps the loaded library in memory.
The Python compiler detects a native CPU recipe with a fixed lane width.
On supported glibc x86-64 hosts it also selects vector math automatically.
`SCALY_VECTOR_LIBM=none` forces scalar math calls, as described in
[environment settings](env_vars.md#compiler-selection-and-optimization).
The cache key includes the generated source and its recipe, function name,
interface version, and compiler and link flags. New numerical inputs do not
cause recompilation. The key does not identify the compiler executable or the
exact CPU, so [environment settings](env_vars.md#compiler-selection-and-optimization)
describes when to clear the cache.

`compile()` prepares the library without evaluating the function:

```python
energy.compile()
```

This is useful before a time-sensitive calculation. It also reuses a cached
library when available. For a solver, `solve.function.compile()` prepares the
compiled wrapper without running an optimization.

The default cache directory is `$XDG_CACHE_HOME/scaly/jit`, or
`~/.cache/scaly/jit` when `XDG_CACHE_HOME` is unset. `SCALY_CACHE_DIR` overrides
it. Deleting cached files does not invalidate an already loaded function, but
future compilations rebuild missing artifacts.

```python
energy.recompile()
```

Despite its name, `recompile()` clears the function's compiled handle and disk
cache entry without immediately compiling. The next numerical call or explicit
`compile()` prepares the library again.

Scaly does not fall back to Python evaluation when compilation fails.
`uv run scaly_toolchain` reports the compiler, cache directory, and resolved
solver libraries when diagnosing such a failure.

## Working memory

`workspace_size` obtains the caller-provided workspace requirement without
rendering all the generated text:

```python
from scaly.codegen import workspace_size

print(workspace_size(energy))  # 0
```

The number counts `double` entries in `w`, not bytes or total temporary memory.
For an export with non-default options, use that module's `workspace_size` or
its header's `<name>_SZ_W` constant. In particular, CasADi sparse-output
conversion can require additional workspace.
