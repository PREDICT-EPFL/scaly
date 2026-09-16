# The C ABI

Every function scaly generates is reachable through one C signature:

```c
int f(const double** arg, double** res, int* iw, double* w, void* mem);
```

This is CasADi's universal ABI in spirit, and the choice is deliberate: it is the lowest common
denominator that lets generated functions call each other, lets a generated solver drive generated
oracles, and lets an existing C or C++ codebase consume scaly output without knowing anything about
scaly. Typed wrappers exist on top of it, but this pointer signature is the stable interface.

## Calling convention

The caller owns all storage. Nothing is allocated inside.

| Argument | Contract |
| --- | --- |
| `arg` | array of `f_SZ_ARG` input pointers. Each `arg[i]` points to a contiguous row-major `double` buffer of the statically known flattened input size. Every input is required — there are no defaults, so no `arg[i]` may be null. |
| `res` | array of `f_SZ_RES` output pointers, each to caller-owned contiguous row-major storage of the statically known flattened output size. |
| `iw` | integer workspace. Currently unused (`f_SZ_IW` is always 0); pass null. |
| `w` | floating workspace: at least `f_SZ_W` doubles. May be null only when `f_SZ_W == 0`. |
| `mem` | reserved for stateful functions. Pass null after using the memory hooks below. |

Buffers are assumed to have ordinary C `double` and `int` alignment.

## What a header declares

```c
#define f_SZ_ARG 1
#define f_SZ_RES 1
#define f_SZ_IW  0
#define f_SZ_W   0

int f(const double** arg, double** res, int* iw, double* w, void* mem);
int f_sz_arg(void);
int f_sz_res(void);
int f_sz_iw(void);
int f_sz_w(void);
void* f_alloc_mem(void);
int   f_init_mem(void* mem);
void  f_free_mem(void* mem);
```

The `f_SZ_*` macros are for callers that need the sizes at compile time — to stack-allocate a
workspace, for instance. The `f_sz_*` functions are the same numbers at run time, for callers
reaching the module through `dlopen`.

`f_SZ_W` is the packed spill size decided by the workspace packer, not the total temporary
footprint: small temporaries stay as C locals inside the function and never appear here. Small
functions routinely report `0`.

The memory hooks are shaped for a future in which a generated function holds state. Today
`f_alloc_mem()` returns `NULL`, `f_init_mem(mem)` ignores its argument and returns `SCALY_SUCCESS`,
and `f_free_mem(mem)` does nothing. Having the hooks now means adding solver or integrator memory
later will not change the exported signature.

## Status codes

| Code | Value | Meaning |
| --- | --- | --- |
| `SCALY_SUCCESS` | 0 | Evaluation succeeded. |
| `SCALY_ERR_NULL_ABI` | 1 | The `arg` or `res` array itself is null. |
| `SCALY_ERR_NULL_WORK` | 2 | The function needs floating workspace and `w` is null. |
| `SCALY_ERR_NULL_RESULT` | 3 | A required `res[i]` is null. |
| `SCALY_ERR_NULL_INPUT` | 4 | A required `arg[i]` is null. |

Headers guard these defines with `#ifndef`, so several generated modules can be included into one
translation unit without colliding.

## Typed buffer wrappers

For statically known shapes the header also emits fixed-size structs and, under C++, an inline
wrapper that builds the pointer arrays for you:

```c
typedef struct { double data[3]; } f_x_in;
typedef struct { double data[1]; } f_y_out;

static inline int f_call(const f_x_in& in_x, f_y_out& out_y);
```

Generated C++ headers add `static_assert`s that each struct flattens to the expected number of
doubles. This is sugar — convenient and type-checked at the call site, but the pointer ABI
underneath is what stays stable. Pass `typed_buffers=False` to omit it.

## What a translation unit contains

One rendered module is one `.c` file and one `.h` file. Inside the `.c`:

1. `static inline` bodies for every callee reached from the root, named `<callee>_raw`;
2. any solver wrappers, in dependency order after the oracle bodies they drive;
3. the exported ABI entry for the root function.

Only the root is exported. Nested calls become direct C calls to the `_raw` bodies, which is why
generated code stays small when the same block appears many times — the block is one C function,
not a hundred inlined copies.

The `.c` file includes the generated header, so the pair compiles as an ordinary translation unit
and links into a C++ caller that includes the same header.

## Sparse outputs

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

**The value buffer the function writes is in `(rows, cols)` order, and that order is not
necessarily sorted.** The structured path through `VMAP` emits nonzeros piece by piece, so the
coordinate list — not row-major order — is the authority on what value belongs where.

The `_val_perm` tables are how you pair the values with a sorted index structure:
`values_csr[k] = values[csr_val_perm[k]]`, and likewise for CSC. `SparsityType.to_csr()` and
`to_csc()` return the same permutation as their third element, so Python and C agree by
construction.

## Solver-bearing modules

A module containing a solver additionally defines the versioned, fixed-width `scaly_solver_stats`
struct and exports `int <solver_symbol>_stats(scaly_solver_stats* out)` for each wrapper in the
translation unit. The query copies the wrapper's latest process-local statistics; it adds no
symbolic output and does not change the entry signature.

Version 3 is a 136-byte layout: version, scaly status, native status and iteration count; then the
objective and the total, function-evaluation, solver, QP, globalization and glue times in seconds;
five evaluation counters with explicit padding; then primal violation, last step norm, accepted step
length, merit penalty, backtrack count and accumulated QP iterations.

Scaly status codes are backend-neutral: `OK=0`, `ACCEPTABLE=1`, `MAX_ITER=2`, `PRIMAL_INFEASIBLE=3`,
`DUAL_INFEASIBLE=4`, `NUMERICS=5`, `USER_STOP=6`, `ERROR=7`.

Timing is instrumented unconditionally with a monotonic clock, and the split is designed to add up:
`t_fe` covers generated oracle work, `t_glue` is the remainder, and
`t_total = t_fe + t_solver + t_qp + t_globalization + t_glue` to within floating-point rounding.
Each backend fills the middle terms differently — PIQP reports setup, update and solve in `t_qp`;
IPOPT reports solve time outside callbacks in `t_solver`; the SQP plugin separates its PIQP
subproblems into `t_qp` from its line-search work in `t_globalization`.

One caveat worth stating plainly: wrapper state, including the latest statistics and the solver
workspace, lives in translation-unit statics. **Generated solver wrappers are not reentrant.**

## Producing a module

```python
from scaly.codegen import render_c_module, workspace_size, write_module

module = render_c_module(fn)
module.header          # the .h text
module.source          # the .c text
module.workspace_size  # the f_SZ_W the header declares
module.link_flags      # flags for any solver plugins reached (empty without a solver)

workspace_size(fn)     # the same number, without rendering the rest
write_module(fn, out_dir)
```

From the command line:

```bash
uv run python -m scaly.codegen mymodule:my_function -o generated/
```

The JIT consumes exactly this object — it compiles `module.body` and keys its cache on that text —
so what you ship ahead of time and what runs when you call the function from Python are the same
translation unit. See [Code generation](../guide/codegen.md) for the usage side.
