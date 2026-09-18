# The generated interface

Every function scaly generates is one C kernel behind one pointer signature:

```c
int f(const double** arg, double** res, int* iw, double* w, int mem);
```

That signature, the status codes and the symbol names are the application binary interface (ABI):
what lets separately compiled artifacts interoperate through `dlopen` without sharing a header. It
is how a generated solver drives generated oracles, how acados and CasADi's `external` load a
library, and it is the part that stays stable.

On top of it sit three source-level layers, each an application programming interface (API) a
caller includes rather than a binary contract: a C header with a struct per buffer, a C++ header
with a `Buffer` type per buffer in a namespace per function, and an optional set of CasADi 3.8
compatible symbols. Both header languages compile the same kernel. The header language changes what
a caller writes, not what runs.

```python
from scaly.codegen import render_c_module

render_c_module(fn)                          # f.h + f.c
render_c_module(fn, lang="cpp")              # f.hpp + f.c, the same f.c
render_c_module(fn, casadi=True)             # either header, plus the CasADi symbols
render_c_module(fn, typed_buffers=False)     # C header with the pointer ABI and tables only
```

## The pointer ABI

The caller owns all storage. Nothing is allocated inside.

| Argument | Contract |
| --- | --- |
| `arg` | array of `f_SZ_ARG` input pointers. Each `arg[i]` points to a contiguous row-major `double` buffer of the statically known flattened input size. Every input is required. There are no defaults, so no `arg[i]` may be null. |
| `res` | array of `f_SZ_RES` output pointers, each to caller-owned contiguous row-major storage of the statically known flattened output size. Every output is required. |
| `iw` | integer workspace. Currently unused (`f_SZ_IW` is always 0); pass null. |
| `w` | floating workspace of at least `f_SZ_W` doubles. May be null only when `f_SZ_W == 0`. |
| `mem` | a memory handle, following CasADi 3.8. Every scaly function is stateless and ignores it; pass `0`. |

Buffers are assumed to have ordinary C `double` and `int` alignment. The typed layers below align
their buffers to 16 bytes, which the kernel does not yet rely on.

The header carries the sizes as compile-time macros, and nothing else:

```c
#define f_SZ_ARG 1
#define f_SZ_RES 1
#define f_SZ_IW  0
#define f_SZ_W   0
```

`f_SZ_W` is the packed spill size decided by the workspace packer, not the total temporary
footprint. Small temporaries stay as C locals inside the function and never appear here, so small
functions routinely report `0`. There is no run-time size query in the native interface; a caller
that reaches the module through `dlopen` and needs one turns on the [CasADi layer](#the-casadi-layer),
whose `f_work` is that query.

### Status codes

| Code | Value | Meaning |
| --- | --- | --- |
| `SCALY_SUCCESS` | 0 | Evaluation succeeded. |
| `SCALY_ERR_NULL_ABI` | 1 | The `arg` or `res` array itself is null. |
| `SCALY_ERR_NULL_WORK` | 2 | The function needs floating workspace and `w` is null. |
| `SCALY_ERR_NULL_RESULT` | 3 | A required `res[i]` is null. |
| `SCALY_ERR_NULL_INPUT` | 4 | A required `arg[i]` is null. |

Headers guard these defines with `#ifndef`, so several generated modules can be included into one
translation unit without colliding.

### What a translation unit contains

One rendered module is one `.c` file and one header. Inside the `.c`:

1. `static` bodies for every callee reached from the root, named `<callee>_raw`. They are `static
   inline`, or `static __attribute__((noinline))` for the forward-AD helpers covered by the clang
   workaround in `codegen/c.py`;
2. any solver wrappers, in dependency order after the oracle bodies they drive;
3. the exported ABI entry for the root function;
4. with `casadi=True`, the query functions and their sparsity tables.

Only the root is exported. Nested calls become direct C calls to the `_raw` bodies, which is why
generated code stays small when the same block appears many times. The block is one C function.

The `.c` compiles on its own. With a C header it includes that header, so the pair builds as an
ordinary translation unit; with a C++ header it includes nothing, because C cannot include C++, and
the header is compiled only by the callers that include it.

## The C header (`lang="c"`)

The pointer ABI, the macros, and for each input and output a fixed-size struct so a C caller gets
a type per buffer:

```c
typedef struct { SCALY_ALIGNAS(16) double data[3]; } f_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } f_y_t;
typedef struct { SCALY_ALIGNAS(16) double data[f_SZ_W > 0 ? f_SZ_W : 1]; } f_workspace_t;

static inline int f_call(const f_x_t* x, f_y_t* y, f_workspace_t* workspace);
```

`SCALY_ALIGNAS` is `_Alignas` in C and `alignas` in C++, so the same header compiles under both.
The struct is `f_<name>_t`. Only when the same name is both an input and an output, as a solver
Function's warm start and solution are, does it split into `f_<name>_in_t` and `f_<name>_out_t`,
and the C++ aliases and `call` parameters follow (`w_in`, `w_out`). An empty buffer gets a
one-element array so the struct stays valid C.
The workspace is a struct the caller places where it likes: on the stack, in a `static`, on the
heap. `f_call` builds the pointer arrays and passes `workspace->data`; for a function with
`f_SZ_W == 0` it accepts `NULL`.

```c
#include "f.h"

static f_workspace_t workspace;
f_x_t x = {{1.0, 2.0, 3.0}};
f_y_t y;
int rc = f_call(&x, &y, &workspace);
```

`typed_buffers=False` leaves the structs and `f_call` out; the benchmark kernels use it because
their driver goes through the pointer ABI.

### Sparse outputs

A compact derivative output carries its pattern into the header as static tables:

```c
#define f_spjac_y_x_NNZ  4
#define f_spjac_y_x_NROW 3
#define f_spjac_y_x_NCOL 4
static const int f_spjac_y_x_rows[4] = {0, 1, 1, 2};
static const int f_spjac_y_x_cols[4] = {0, 2, 3, 1};
static const int f_spjac_y_x_csr_row_ptr[4]  = {0, 1, 3, 4};
static const int f_spjac_y_x_csr_col_ind[4]  = {0, 2, 3, 1};
static const int f_spjac_y_x_csr_val_perm[4] = {0, 1, 2, 3};
static const int f_spjac_y_x_csc_col_ptr[5]  = {0, 1, 2, 3, 4};
static const int f_spjac_y_x_csc_row_ind[4]  = {0, 2, 1, 1};
static const int f_spjac_y_x_csc_val_perm[4] = {0, 3, 1, 2};
```

The value buffer the function writes is in `(rows, cols)` order, and that order is not necessarily
sorted. The structured path through `VMAP` emits nonzeros piece by piece, so the coordinate list is
the authority on which value belongs where.

The `_val_perm` tables pair the values with a sorted compressed sparse row (CSR) or compressed
sparse column (CSC) structure: `values_csr[k] = values[csr_val_perm[k]]`, and likewise for CSC.
`SparsityType.to_csr()` and `to_csc()` return the same permutation as their third element, so
Python and C agree by construction.

## The C++ header (`lang="cpp"`)

A namespace per function and nothing above it. Every header carries the same guarded template, so
several generated headers share one translation unit:

```cpp
template <typename T, std::size_t... Ns>
struct alignas(16) Buffer {
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * std::size_t{1});
  T data[size > 0 ? size : 1];
  T* ptr();
  const T* ptr() const;
  T& operator()(I... idx);   // row-major, one index per dimension, bounds-asserted in debug builds
};
```

Storage is inline, so a `Buffer` is an aggregate: `f::x_t x = {{0.25, -0.75}};` on the stack, or
`static`, or `std::make_unique<f::x_t>()`. `shape` is the `Expr` shape, so a `(N, nx)` stage
trajectory keeps its two dimensions and is indexed `traj(k, i)` instead of `data[k * nx + i]`.
Wrapping memory someone else already owns is what the pointer entry is for.

```cpp
namespace f {
extern "C" int f(const double** arg, double** res, int* iw, double* w, int mem);

using x_t = Buffer<double, 3>;
using y_t = Buffer<double>;            // a scalar: ndim 0, size 1
using workspace_t = Buffer<double, f_SZ_W>;
constexpr int sz_arg = 1, sz_res = 1, sz_iw = 0, sz_w = f_SZ_W;

inline int call(const x_t& x, y_t& y, workspace_t& workspace);

namespace spjac_y_x {
constexpr int nrow = 3, ncol = 4, nnz = 4;
constexpr std::array<int, nnz> rows = {...}, cols = {...}, csr_col_ind = {...}, csr_val_perm = {...};
constexpr std::array<int, nrow + 1> csr_row_ptr = {...};
constexpr std::array<int, ncol + 1> csc_col_ptr = {...};
constexpr std::array<int, nnz> csc_row_ind = {...}, csc_val_perm = {...};
}
}
```

The kernel symbols are declared `extern "C"` inside the namespace: a namespace and a function
cannot share the global name `f`, and C linkage ignores the namespace, so `f::f` is the same symbol
a C caller reaches as `f`. A C++ caller that wants the pointer ABI calls it directly. Sparse
metadata is `constexpr`, so a consumer can size its own arrays from `nnz` at compile time. The
header needs C++17.

## The CasADi layer

`casadi=True` adds the symbols CasADi 3.8 exports from its own code generator, with either header
language:

```c
casadi_int f_n_in(void);
casadi_int f_n_out(void);
const char* f_name_in(casadi_int i);
const char* f_name_out(casadi_int i);
casadi_real f_default_in(casadi_int i);            /* always 0 */
const casadi_int* f_sparsity_in(casadi_int i);
const casadi_int* f_sparsity_out(casadi_int i);
int f_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
int f_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
int f_checkout(void);                              /* returns 0 */
void f_release(int mem);
void f_incref(void);
void f_decref(void);
```

acados resolves six of these: the entry, `f_work`, `f_sparsity_in`, `f_sparsity_out`, `f_n_in` and
`f_n_out`. The rest cost a line each and make `casadi.external("f", "libf.so")` load the library.
Sparsity uses CasADi's compressed encoding: `{nrow, ncol, 1}` for a dense buffer, and
`{nrow, ncol, colind[0..ncol], row[0..nnz)}` for a compact sparse output.

Three facts make this more than aliases:

- **`casadi_int` and `casadi_real` are guarded macros**, as in CasADi's own output, defaulting to
  `long long int` and `double`. CasADi's `external` reads the queries as `long long`; acados reads
  them as `int`. The two disagree, so one binary cannot serve both. The default serves CasADi; an
  acados build compiles the same `.c` with `-Dcasadi_int=int`, which is what acados-generated
  CasADi code hard-codes.
- **Compact sparse outputs are handed over in compressed-column order.** A CasADi consumer reads a
  sparse value buffer in CSC order, while scaly's native order is the coordinate list above. Under
  `casadi=True` the entry evaluates such an output into `w` past the packed workspace and gathers it
  into `res[i]` through `csc_val_perm`, adding `nnz` per gathered output to `f_SZ_W`. An output
  whose native order is already CSC, as the QP path's patterns are, is written straight to
  `res[i]` and adds nothing. The header's tables describe the buffer as written, so their
  `csc_val_perm` is the identity. Native CSC emission, which would remove the copy, is a separate
  later optimisation.
- **Dense matrices are rejected.** Scaly buffers are row-major and CasADi's are column-major.
  Scalars, vectors and compact sparse outputs agree; a dense input or output with both dimensions
  above one would need a transpose, and `render_c_module` refuses it with a clear error instead.
  acados only ever passes vectors and reads Jacobians through their sparsity, so this rarely bites.

Null handling stays scaly's: every `arg[i]` and `res[i]` is required. CasADi treats a null input
as zeros and skips a null output; acados does neither.

## Solver-bearing modules

A module containing a solver additionally defines the versioned, fixed-width `scaly_solver_stats`
struct and exports `int <solver_symbol>_stats(scaly_solver_stats* out)` for each wrapper in the
translation unit. The query copies the wrapper's latest process-local statistics. It adds no
symbolic output and does not change the entry signature.

Version 3 is a 136-byte layout: version, scaly status, native status and iteration count; then the
objective and the total, function-evaluation, solver, QP, globalization and glue times in seconds;
five evaluation counters with explicit padding; then primal violation, last step norm, accepted step
length, merit penalty, backtrack count and accumulated QP iterations.

Scaly status codes are backend-neutral: `OK=0`, `ACCEPTABLE=1`, `MAX_ITER=2`, `PRIMAL_INFEASIBLE=3`,
`DUAL_INFEASIBLE=4`, `NUMERICS=5`, `USER_STOP=6`, `ERROR=7`.

Timing is instrumented unconditionally with a monotonic clock, and the split adds up. `t_fe` covers
generated oracle work, `t_glue` is the remainder, and
`t_total = t_fe + t_solver + t_qp + t_globalization + t_glue` to within floating-point rounding.
Each backend fills the middle terms differently. PIQP reports setup, update and solve in `t_qp`;
IPOPT reports solve time outside callbacks in `t_solver`; the SQP plugin separates its PIQP
subproblems in `t_qp` from its line-search work in `t_globalization`.

Wrapper state, including the latest statistics and the solver workspace, lives in translation-unit
statics. Generated solver wrappers are not reentrant.

## Producing a module

```python
from scaly.codegen import render_c_module, workspace_size, write_module

module = render_c_module(fn, lang="c", casadi=False)
module.header          # the header text, f.h or f.hpp
module.source          # the .c text
module.workspace_size  # the f_SZ_W the header declares, gather scratch included
module.link_flags      # flags for any solver plugins reached (empty without a solver)

workspace_size(fn)     # the same number, without rendering the rest
write_module(fn, out_dir, lang="cpp")
```

From the command line:

```bash
uv run scaly_codegen mymodule:my_function -o generated/ --lang cpp --casadi
```

The just-in-time (JIT) path consumes exactly this object with the defaults. It compiles
`module.body`, keys its cache on that text, and takes the workspace size from the module rather
than from the library, so what you ship ahead of time and what runs when you call the function from
Python are the same translation unit. See [Code generation](../guide/codegen.md) for the usage side.
