# Generated code and compilation

This page opens up the C file that Scaly writes and follows it through the
compiler and the Python cache. The [code generation
guide](../guide/codegen.md) covers how to export, build and call a module,
including the typed headers, the sparse coordinate tables and the CasADi
queries. This page explains what is inside the files and why they are built
the way they are.

## An annotated module

Here is a stage function mapped over a ten-step horizon. The `.block()` hint
keeps `euler` in loop form, so it survives lowering as its own procedure
instead of being expanded into its caller (see
[lowering](lowering.md) for the hints):

```python
import scaly as sc
from scaly.codegen import render_c_module

@sc.function(sc.arg("z", 2), sc.arg("u", ()), outputs=sc.arg("znext"))
def euler(z: sc.Expr, u: sc.Expr) -> sc.Expr:
    return sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * u.sin()]).block()

@sc.function(sc.arg("zs", 20), sc.arg("us", 10), outputs=sc.arg("zn"))
def rollout(zs: sc.Expr, us: sc.Expr) -> sc.Expr:
    return sc.vmap(euler, 10)(zs.reshape((10, 2)), us).vec()

print(render_c_module(rollout, lanes=1).source)
```

`lanes=1` turns off vector widening so the listing stays short. This is the
printed `rollout.c`:

```c
/* Scaly build recipe
 * CPU baseline: generic
 * lanes=1, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c rollout.c
 * clang -O3 -fno-math-errno -c rollout.c
 * Link with: -lm
 */
#include "rollout.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline void euler_raw(const double* z, const double* u, double* znext, double* w) {
  (void)w;
  const double* t0 = z;
  const double* t2 = z + 1;
  double v0 = t2[0];
  *(double2*)(znext) = (double2){(t0[0] + (0.10000000000000001 * v0)), (v0 + (0.10000000000000001 * sin(u[0])))};
}

int rollout(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  for (long long it_zn = 0; it_zn < 10; ++it_zn) {
    euler_raw((arg[0] + (2 * it_zn)), (arg[1] + it_zn), (res[0] + (it_zn * 2)), NULL);
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
```

The sections below take this file apart from top to bottom.

## Build recipe

The comment at the top is the `BuildRecipe` the module was rendered for. It
names the CPU baseline, the lane width, the C dialect, the math library and the
reciprocal policy, and it gives a GCC and a Clang command that build the file
the way Scaly expects. The same comment opens the header.

The recipe is not advice. Rendering made decisions that only hold under its
flags. With `lanes="auto"`, the lane width is chosen by the preprocessor from
the target macros the compiler defines, so building without the recipe's
`-march` flag silently gives narrower vectors. With glibc vector math the
generated guards refuse to build at all. For a 64-element `sin` rendered with
`cpu="x86-64-v4", lanes=8, vector_libm="glibc"`:

```console
$ cc -O3 -fno-math-errno -c waves.c
waves.c:45:2: error: #error "vector_libm=glibc width 8 requires __AVX512F__ and -lmvec"
$ cc -O3 -march=x86-64-v4 -fno-math-errno -c waves.c
```

The second command, with the recipe's flags, compiles. `-fno-math-errno` lets
the compiler inline and vectorize calls such as `sqrt`, because they no longer
have to set `errno`. Scaly never enables general fast-math flags. The
[guide](../guide/codegen.md#cpu-targets-vector-lanes-and-math-libraries)
lists the recipe options and their values.

## Module layout

Every module is one C source and one header. The source contains, in order:

1. The recipe comment and, for a C header, `#include` of that header.
2. System includes, vector typedefs and lane macros, and the status codes.
3. One `static inline void <callee>_raw(...)` per retained callee.
4. For a module that reaches a solver, the solver wrappers.
5. The exported entry, named after the root function.

A `_raw` procedure takes one pointer per input, `const`-qualified, one pointer
per output and a trailing workspace pointer. It has no null checks and no
status code, so the entry pays for those once rather than at every stage. A
callee that lowering expanded into its caller has no `_raw` at all. Without
`.block()`, `euler` is expanded into the loop in `rollout` and `euler_raw`
disappears, so function boundaries in Python do not promise C procedures.

`euler_raw` is called with `w = NULL` because it spills nothing to the
workspace. When a callee does spill, the caller passes its own `w` advanced
past its own spill window.

A solver-bearing module adds the solver wrapper and a statistics accessor. The
top-level definitions of the SQP solver for a two-variable problem, printed
from its source with long signatures cut:

```c
#include "circle_sqp.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include "piqp/piqp.h"
static double scaly_clock_s(void) {
static inline void circle_base_raw(const double* x, const double* p, double* f, double* g, doubl ...
static inline void circle_grad_raw(const double* x, const double* p, double* grad_f_x, double* w) {
static inline void circle_jac_raw(const double* x, const double* p, double* spjac_g_x, double* w) {
static inline void circle_hess_upper_raw(const double* x, const double* p, const double* lam_f,  ...
static inline void circle_bounds_raw(const double* p, double* x_lb, double* x_ub, double* w) {
static scaly_solver_stats circle_sqp_stats_data;
static void circle_sqp_raw(const double* in0, const double* in1, const double* in2, const double ...
int circle_sqp_stats(scaly_solver_stats* out) {
int circle_sqp_with_options(const double** arg, double** res, int* iw, double* w, int mem, ...
const scaly_solver_option* circle_sqp_default_options(void) {
int circle_sqp(const double** arg, double** res, int* iw, double* w, int mem) {
```

The oracles are ordinary `_raw` procedures lowered like any other function.
Only `circle_sqp_raw` comes from a template in the solver plugin, and it is
placed after the oracles it calls. [Solvers](solvers.md) describes what that
wrapper does.

## The pointer ABI

The standard entry has the same five-argument signature for every function, and
the [guide](../guide/codegen.md#the-pointer-entry) explains each argument. A module
containing a solver also exports a `_with_options` entry with a sixth argument
for call-time option arrays. Python uses this entry. The standard entry calls it
with backend defaults. The [option interface](../guide/codegen.md#solver-options-in-c)
describes the arrays and their lifetime. Every
input and output buffer is an array of `double`, whatever the data type of the
matching value in the graph. An `int64` or `bool` input is read from `double`
values, and an integer or Boolean output is written as `double` values, so a
numerical call from Python returns `float64` arrays for them too. The body
starts with null checks that return one of five codes:

| Code                    | Value | Returned when                                         |
| ----------------------- | ----- | ----------------------------------------------------- |
| `SCALY_SUCCESS`         | 0     | evaluation finished                                   |
| `SCALY_ERR_NULL_ABI`    | 1     | `arg` or `res` is null                                |
| `SCALY_ERR_NULL_WORK`   | 2     | `w` is null and the header's `SZ_W` is not zero       |
| `SCALY_ERR_NULL_RESULT` | 3     | an output pointer is null                             |
| `SCALY_ERR_NULL_INPUT`  | 4     | an input pointer is null                              |

`rollout` has no workspace, so its entry casts `w` away and the
`SCALY_ERR_NULL_WORK` check is absent. The checks cannot tell whether a
non-null pointer refers to a buffer that is large enough. A solver that fails
to converge still returns `SCALY_SUCCESS`. Convergence is reported in the
statistics described [below](#solver-statistics).

### Alignment

The pointer interface needs only the natural alignment of `double`. Stores of
several values at once go through vector types declared `aligned(8)` and
`may_alias`, like the `double2` store in `euler_raw`, so the compiler never
assumes 16-byte alignment of a caller's buffer. The typed structs in the header
request 16 bytes regardless.

### C++ linkage

The source wraps everything in `extern "C"` when compiled as C++, so the file
builds with either compiler and exports the same symbol. The C++ header
declares the entry `extern "C"` inside a namespace of the same name, so
`rollout::rollout` is the symbol a C caller reaches as `rollout`. With
`lang="cpp"` the source does not include the header, because a C++ header
cannot be included from C.

## Sparse outputs

A sparse derivative output is written as compact values in the coordinate
order that lowering produced. That order is not sorted and can change when
coloring or `vmap` lowering changes, which is why the header carries
`_csr_val_perm` and `_csc_val_perm` tables rather than promising an order. The
[guide](../guide/codegen.md#sparse-output-patterns) shows the tables for a
concrete Jacobian, and [versioning](../dev/versioning.md) says what a release
promises about them.

## The CasADi layer

CasADi expects sparse values in compressed-column order. With `casadi=True`,
an output whose native order differs is written to extra workspace and then
permuted into `res`. For the three-by-three Jacobian from the guide, the plain
module needs no workspace and the CasADi module needs five doubles, one per
nonzero. Its entry, with the body of the computation trimmed:

```c
int measurements_jac(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  if (!w) return SCALY_ERR_NULL_WORK;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  double* spjac_y_x_native = w + 0;
  /* ... computation trimmed, writing into spjac_y_x_native ... */
  static const int spjac_y_x_csc_val_perm[5] = {0, 3, 2, 1, 4};
  for (int k = 0; k < 5; ++k) res[0][k] = spjac_y_x_native[spjac_y_x_csc_val_perm[k]];
  return SCALY_SUCCESS;
}
```

The scratch starts after the function's own packed workspace, and the header's
`SZ_W` and the `_work` query both include it. Outputs already in column order,
such as the matrices extracted for a quadratic program, need no copy. The
header's sparsity tables describe the order actually written.

## Solver statistics

A module that reaches a solver defines a fixed-width statistics struct in its
header and exports one `<solver>_stats` accessor per wrapper, which copies the
latest statistics of that wrapper into the caller's struct. The struct as it
appears in the header:

```c
#define SCALY_SOLVER_STATS_VERSION 3
#define SCALY_SOLVE_OK 0
#define SCALY_SOLVE_ACCEPTABLE 1
#define SCALY_SOLVE_MAX_ITER 2
#define SCALY_SOLVE_PRIMAL_INFEASIBLE 3
#define SCALY_SOLVE_DUAL_INFEASIBLE 4
#define SCALY_SOLVE_NUMERICS 5
#define SCALY_SOLVE_USER_STOP 6
#define SCALY_SOLVE_ERROR 7
typedef struct {
  int32_t version;
  int32_t status;
  int32_t native_status;
  int32_t iter;
  double obj;
  double t_total;
  double t_fe;
  double t_solver;
  double t_qp;
  double t_globalization;
  double t_glue;
  int32_t n_eval_f;
  int32_t n_eval_grad_f;
  int32_t n_eval_g;
  int32_t n_eval_jac_g;
  int32_t n_eval_h;
  int32_t _pad0;
  double primal_viol;
  double step_inf;
  double alpha;
  double merit_penalty;
  int32_t backtracks;
  int32_t qp_iter;
} scaly_solver_stats;
```

The layout is 136 bytes, and new fields are only ever appended with a version
bump. `version` is zero until the wrapper has run once. Fields a backend has no
use for stay zero.

Times are in seconds from a monotonic clock. `t_fe` is function evaluation,
and the wrapper sets `t_glue` so that
`t_total = t_fe + t_solver + t_qp + t_globalization + t_glue`. PIQP puts its
time in `t_qp`, IPOPT puts the time spent outside oracle calls in `t_solver`,
and Scaly SQP splits `t_qp` from the line search in `t_globalization`.

The statistics live in a `static` variable next to the wrapper's native
workspace, which is why a wrapper
[must not be called concurrently](../guide/solvers.md#solvers-in-generated-c).

## Compilation and caching

A numerical call such as `energy(x)` reaches the same `render_c_module` as an
export, with a recipe for the host machine. The first call on a fresh cache
compiles a shared library, and later processes reuse it. Timing
`energy.compile()` for the `energy` function from the guide, with
`SCALY_CACHE_DIR=/tmp/scaly-cache`:

```python
import time

start = time.perf_counter()
energy.compile()
print(f"compile() took {1e3 * (time.perf_counter() - start):.0f} ms")
```

On a fresh cache the first process prints 42 ms and a second process 9 ms.
Running it again with `SCALY_CC_OPT=-O3` prints 41 ms, because the new flag
gives a new cache key and a new compilation. The cache root then holds one
directory per key, each with `energy.c` and `libenergy.so`.

### The host recipe

The JIT probes the compiler once per process with `cc -march=native -dM -E`
and reads the target macros. The widest supported vector width becomes a
fixed lane count, 8 with AVX-512, and glibc vector math is selected on x86-64
with glibc 2.35 or newer. On such a machine the rendered recipe reads:

```c
/* Scaly build recipe
 * CPU baseline: host-local native CPU
 * lanes=8, dialect=gnu, vector_libm=glibc, reciprocal=False
 * Math library: glibc x86-64; tanh requires glibc >= 2.35
 * gcc -O3 -march=native -fno-math-errno -c energy.c
 * clang -O3 -march=native -fno-math-errno -c energy.c
 * Link with: -lmvec -lm
 */
```

The compiler commands in the recipe are for building an exported module
ahead of time. The JIT itself compiles with `-O2 -march=native -fno-math-errno`, or with
`SCALY_CC_OPT` in place of `-O2`, followed by `-fPIC -shared`, the solver
include, library and `rpath` flags, `-lmvec` when selected, and `-lm`.

### The cache key

The key is a SHA-256 hash over, in order:

- a cache version string in `src/scaly/codegen/jit.py`, raised whenever
  generated code or the cache layout changes
- the pointer ABI signature
- the function's name
- the rendered translation unit, which includes the recipe comment
- the optimization, CPU and `-fno-math-errno` flags, and the link flags
  (solver paths, `-lmvec`). The fixed `-fPIC -shared` and `-lm` are left out.

Changing the optimization level therefore gives a new key, as the `-O3` run
shows. The compiler binary is not part of the key, so pointing `SCALY_CC` at a
different compiler with the same flags reuses artifacts built by the old one.
Numerical inputs and runtime solver tuning never enter the key. Structural
choices such as PIQP's `sparse` and SQP's `qp` change the generated interface and
therefore the key.

### Where artifacts live

Each key gets a directory under the cache root holding the exact source that
was compiled, as `<name>.c`, and `lib<name>.so` (`.dylib` on macOS). Both are
written to a file named after the process ID and then renamed into place, so
parallel test workers building the same function on a cold cache never see a
half-written library.

### What a new process reuses

Nothing about tracing or rendering is cached on disk. A new process runs the
Python body again when the decorator executes, lowers and renders the C again,
and hashes it. Only the C compiler call is skipped when the library exists.
That is the difference between the 42 ms and 9 ms runs above. Within a process,
a second `Function` with the same generated source finds the artifact in an
in-memory table and does not touch the compiler either.

`compile()` does all of this and loads the library without evaluating. It
opens the library with `ctypes`, resolves the entry and any `_stats`
accessors, and keeps the handle on the `Function`. On Linux a library that
reaches a solver is opened with `dlmopen` into a separate linker namespace,
shared by all solver libraries, so the solvers' own dependencies cannot
collide with libraries already loaded by Python packages.

`recompile()` renders the source once more to recompute the key, drops the
handle and the in-memory entry, and deletes that key's directory. It does not
compile. The next call or `compile()` does. A library that is already loaded,
by this or any other `Function`, stays usable after its files are deleted.

The module map in [the codebase](../dev/codebase.md) points to the files
behind each stage.
