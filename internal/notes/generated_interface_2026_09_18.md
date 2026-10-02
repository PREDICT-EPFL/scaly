# The generated C and C++ interface for the first release

Design note, 2026-09-18, frozen. Decisions settled the same day; the historical tasks CAPI-72 to CAPI-76 are complete. The current interface is documented in [the C ABI](../../docs/how_it_works/generated_interface.md)
and rendered by `codegen/aot.py` (`_render_header`, `_typed_cpp_wrapper`) and `codegen/c.py`
(`_render_entry`).

## Where we are

The extraction from the previous framework replaced a Jinja-rendered C++ module (a namespace per
function, a generic `Buffer<T, Ns...>` template carrying the shape, `constexpr` sparsity metadata,
a caller-owned workspace type, user-declared `constexpr` constants) with a C-first design that has
not changed shape since the first commit:

- one exported pointer entry `int f(const double** arg, double** res, int* iw, double* w, void* mem)`;
- `f_SZ_ARG/RES/IW/W` macros and `f_sz_*()` functions returning the same numbers;
- `f_alloc_mem/init_mem/free_mem` stubs "for a future stateful function";
- under `#ifdef __cplusplus`, flat structs `f_x_in { double data[n]; }` and an inline `f_call`
  that builds the pointer arrays and stack-allocates the workspace on every call;
- sparse output metadata as macros and `static const int` tables in the header;
- `typed_buffers=False` as the only knob.

Consumers today: the JIT (`codegen/jit.py`, reads `f_sz_w()` through ctypes), the benchmark
kernels (`typed_buffers=False`, call the pointer entry from a C++ Google Benchmark driver), two
C++ smoke tests in `tests/codegen/test_c.py`, and the ABI doc.

What is wrong with it, in order of pain:

1. `f_call` puts `f_SZ_W` doubles on the stack. Fine for a stage oracle, wrong for a horizon-length
   Hessian (19,670 doubles in the N=100 benchmark header) and hostile to embedded targets.
2. It is neither a clean C interface nor a clean C++ one. C callers get macros and `void*` hooks
   that do nothing; C++ callers get prefix-mangled structs with no shape, no `constexpr`, no
   namespace, and a wrapper they cannot give memory to.
3. It is "CasADi-style" without being CasADi-compatible. Nothing that loads CasADi-generated code
   (acados, CasADi's own `external()`) can load it: the query functions have different names and
   signatures, sparsity is exposed as COO tables instead of CasADi's compressed column encoding,
   the `mem` hooks follow the shape CasADi retired, and dense matrix outputs are row-major where
   CasADi's are column-major.

## Target

One module is one kernel source plus one header, and the header comes in two languages. Three
independent choices at render time:

| Choice | Values | Default |
| --- | --- | --- |
| header language | `c`, `cpp` | `c` |
| CasADi-compatible symbols | off, on | off |
| typed wrappers | on, off (C only, kept for the benchmark kernels) | on |

Proposed spelling: `render_module(fn, *, lang="c" | "cpp", casadi=False)`, file names
`f.h`/`f.c` or `f.hpp`/`f.c`. The kernel source is always C and must compile without the header,
so the C++ header can be the only header shipped. The CLI grows `--lang` and `--casadi`.
`render_c_module` stays as the name until callers move; the CasADi flag is the only new public
behaviour, the rest is a reshaping of what a header contains.

### The pointer entry (both languages, always)

```c
int f(const double** arg, double** res, int* iw, double* w, int mem);
#define f_SZ_ARG 1
#define f_SZ_RES 1
#define f_SZ_IW  0
#define f_SZ_W   0
```

Changes against today:

- `mem` becomes `int`, following CasADi 3.8's signature `(..., casadi_real* w, int mem)`. In
  CasADi it is a memory handle: `f_checkout()` hands out an index into a static pool of per-instance
  memory objects (one per thread, typically), the entry uses that index to find its state, and
  `f_release(mem)` returns the slot. It is unrelated to `w`, which is scratch the caller owns. A
  stateless function ignores it, and every scaly function is stateless today, so the argument is
  still ignored. acados' `external_function_generic.h` declares the pointer as
  `void*`, but the CasADi code acados itself loads is generated with `int mem`, so acados already
  lives with that mismatch and it is harmless for an ignored trailing argument.
- The `f_sz_*()` functions and the `alloc_mem/init_mem/free_mem` stubs go. The macros carry the
  sizes at compile time; the JIT reads `module.workspace_size` from the Python object it already
  holds instead of calling into the library. Nothing else calls them. Run-time introspection then
  exists only with `casadi=True`, which is the full CasADi query set (`_work`, `_n_in`,
  `_sparsity_out`, ...), not scaly's own half-way version of it. The native interface is the
  pointer ABI plus compile-time macros, nothing more.
- Status codes stay as they are.

### The C header (`lang="c"`)

The pointer entry, the macros, and for each input and output a fixed-size struct so a C caller
gets a type per buffer:

```c
typedef struct { _Alignas(16) double data[3]; } f_x_t;
typedef struct { _Alignas(16) double data[1]; } f_y_t;
typedef struct { _Alignas(16) double data[f_SZ_W > 0 ? f_SZ_W : 1]; } f_workspace_t;
static inline int f_call(const f_x_t* x, f_y_t* y, f_workspace_t* w);
```

The `_in`/`_out` suffixes go where a name is unambiguous. Correction at implementation: a solver
Function's warm start and solution share names (`w` in, `w` out), so a name that is both an input
and an output keeps the suffix; the rest drop it. The workspace is a struct the caller places where it likes (stack, static,
heap). Sparse outputs keep the macros and static tables exactly as today; they are the right C
spelling.

### The C++ header (`lang="cpp"`)

A namespace per function, nothing above it. Every header carries the same guarded template:

```cpp
#ifndef SCALY_BUFFER_HPP
#define SCALY_BUFFER_HPP
template <typename T, std::size_t... Ns> struct alignas(16) Buffer {
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * 1);
  T data[size > 0 ? size : 1];
  T* ptr() { return data; }
  const T* ptr() const { return data; }
  T& operator()(auto... idx);   // row-major, ndim indices, bounds-asserted in debug
};
#endif
```

Storage is inline, so a `Buffer` is an aggregate: `f::x_t x = {{0.25, -0.75}};` on the stack, or
`static`, or `std::make_unique<f::x_t>()`, or `malloc(sizeof(f::x_t))` cast to the type; all of
them honour the `alignas`, since C++17 aligned `new` and every 64-bit `malloc` give 16 bytes.
What is lost against the previous framework's pointer wrapper is wrapping memory someone else
already owns; the pointer entry remains for that case. The 16-byte alignment costs nothing and
keeps the option of `__builtin_assume_aligned` in the kernel open. The kernel does not assume
alignment today, so none of the benchmarks moves either way until it does. The guard lets several
generated headers share one translation unit, the same trick the status defines already use.

```cpp
namespace f {
using x_t = Buffer<double, 2>;
using y_t = Buffer<double, 2>;
using workspace_t = Buffer<double, f_SZ_W>;
constexpr int sz_arg = 1, sz_res = 1, sz_w = f_SZ_W;
inline int call(const x_t& x, y_t& y, workspace_t& w);
namespace spjac_y_x {
  constexpr int nrow = 3, ncol = 4, nnz = 4;
  constexpr std::array<int, nnz> rows = {...}, cols = {...};
  constexpr std::array<int, nrow + 1> csr_row_ptr = {...};
  ... csr_col_ind, csr_val_perm, csc_col_ptr, csc_row_ind, csc_val_perm
}
}
```

`Buffer::shape` is the `Expr` shape, so a `(N, nx)` stage trajectory keeps its two dimensions
instead of flattening to `data[N*nx]`. The `static_assert`s on struct size stay. The pointer entry
and macros are declared `extern "C"` above the namespace as today, so a C++ caller can still go
through the pointer ABI.

### The CasADi layer (`casadi=True`, either language)

The symbols CasADi 3.8 exports from `CodeGenerator::add` and `FunctionInternal::codegen_meta`,
with acados' loader as the acceptance test:

```c
typedef int casadi_int;      /* guarded #ifndef, as CasADi does; int because acados requires it */
typedef double casadi_real;
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

acados needs exactly six of these: `f`, `f_work`, `f_sparsity_in`, `f_sparsity_out`, `f_n_in`,
`f_n_out`. The rest cost a line each and make `casadi.external("f", "libf.so")` work. The
`alloc_mem/init_mem/free_mem` trio is what CasADi itself only emits for functions that own memory
and is the part the user remembered as deprecated; `checkout/release` is the current pair.

Sparsity is CasADi's compressed column encoding: dense is `{nrow, ncol, 1}`, sparse is
`{nrow, ncol, colind[0..ncol], row[0..nnz)}`. `SparsityType.to_csc()` already produces the
two arrays.

Two layout facts that make this more than symbol aliases:

- **Sparse outputs.** Scaly writes the compact value buffer in its own COO order, which is why
  the header carries `csc_val_perm`. A CasADi consumer reads the values in compressed-column
  order. Why nobody noticed: IPOPT takes triplets in whatever order `iRow`/`jCol` were declared,
  so the IPOPT wrapper passes the buffer through untouched; the sparse PIQP wrapper needs CSC and
  gets it because the QP path builds its patterns sorted by column (`qp._qp_matrix_sparsity`) and
  gathers the symbolic matrix expression into that order (`qp._gathered`, a `gather` on the
  expression graph, so entries outside the pattern are never rendered), then asserts the
  permutation is the identity; and the
  benchmark drivers scatter both sides into a dense array before comparing. Nothing in the tree
  hands a scaly compact buffer to a CasADi-order consumer. The structured `VMAP` path
  (`ad/sparse.py`, the piece-wise assembly around `SparseJacobian`) emits its nonzeros tile by
  tile and permutes the *pattern* to match, precisely to avoid a full-nnz gather; forcing CSC
  there means reintroducing that gather, which is the mess remembered from the previous framework.
  Proposal: keep the native order as is, and in CasADi mode have the entry evaluate into `w` and
  gather through `csc_val_perm` into `res[i]`, adding `nnz` to `f_SZ_W`. It is one `O(nnz)` copy
  next to an oracle that is already `O(nnz)` work, and it keeps the fast path untouched. Native
  CSC emission stays a separate, later optimisation with its own measurement.
- **Dense matrices.** Scaly buffers are row-major, CasADi's column-major. Vectors and scalars
  agree. A 2-D input or output with both dimensions above one needs a transpose in CasADi mode.
  acados only ever passes vectors (`x`, `u`, `z`, `p`) and reads Jacobians through the sparsity
  description, so the first version rejects such a function with a clear error at render time
  and the transpose can come when someone needs it.

Null handling stays scaly's: every `arg[i]` and `res[i]` is required. CasADi treats a null input
as zeros and skips a null output; acados does neither, so the difference is documented, not
implemented.

## What this is called

"ABI" is the right word for the pointer entry: a fixed signature, symbol naming and calling
convention that let separately compiled artifacts interoperate through `dlopen` without sharing a
header. That is exactly how acados and `casadi.external` consume it. The typed C structs, the C++
namespaces and the `Buffer` template are an API: source-level, language-specific, free to change
without breaking a binary. The doc page should become "The generated interface", with one section
on the ABI and one per API layer, and the code should stop calling the typed structs part of the
ABI (`abi.py`'s docstring does today).

## Decisions (settled 2026-09-18, all as recommended)

1. `int mem` following CasADi 3.8, or keep `void*`. Recommendation: `int`.
2. Drop `f_sz_*()` and the memory stubs from the native interface. Recommendation: yes; the JIT
   moves to `module.workspace_size`.
3. `Buffer` with inline storage (aggregate, no allocator) rather than a pointer wrapper.
   Recommendation: inline.
4. CasADi mode on a function with a dense matrix input or output: reject, or transpose.
   Recommendation: reject in the first version.
5. Header file for C++: `.hpp` alongside a `.c` kernel, self-contained. Recommendation: yes.
   The doc must say plainly that both languages compile the same kernel; the header language
   changes what a caller writes, not what runs.
6. Type names carry a `_t` suffix (`f::x_t`, `f_x_t`) so a buffer type never shadows the argument
   named after it. Settled 2026-09-18.

## Phases

Each phase leaves the suite green and the docs matching the code.

1. **Native entry cleanup.** `int mem`; remove `_sz_*` and memory stubs from `c.py`, `aot.py`,
   `jit.py` and the ABI doc; JIT reads `workspace_size` from the module. Touches every rendered
   header, so run the whole suite and regenerate `benchmarks/results/smoke/**` fixtures if they
   are compared textually.
2. **C header.** Drop `_in/_out`, add `f_workspace`, make `f_call` take it. Update the two smoke
   tests and the doc.
3. **C++ header.** `lang="cpp"`, guarded `Buffer`, namespace per function, `constexpr` sparsity,
   `call(..., workspace&)`. New smoke test compiles a C++ caller against a `(N, nx)`-shaped
   function and a sparse Jacobian, checks `shape`, and reads a value through `csc_val_perm`.
4. **CasADi layer.** `casadi=True`: typedefs, twelve query functions, CSC sparsity tables, the
   CSC gather for compact outputs, the matrix-layout render-time check. Test: load the shared
   library with `casadi.external` (CasADi is already a benchmark dependency) and compare against
   `numerical_call`; a second test hands it to acados' loader when an acados checkout is present,
   otherwise asserts the six symbols resolve with `ctypes` and `f_sparsity_out` decodes to the
   expected pattern.
5. **Docs.** Rename and restructure the ABI page as above; update `guide/codegen.md` and the CLI
   help.

Phases 1 and 2 are sequential and small. Phase 3 and phase 4 are independent of each other once
phase 2 has landed. Semantic constants (user-named `constexpr`/`#define` integers) are deferred to
after the release; when they come they slot into the namespace of phase 3 and cost about ten
lines.
