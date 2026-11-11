# External C functions

A user can describe a C library, declare its concrete functions and call them inside a Scaly graph.
The just-in-time compiler compiles the library sources separately from the generated kernel and
links their objects into the same shared library. Ahead-of-time export writes the owned sources,
headers and a `<function>.build.json` manifest beside the generated C. The manifest keeps library
names rather than host-specific linker flags, so the consumer resolves them for its toolchain.

## Needs the maintainer's decision

Recommend the exported `sc.CLibrary` and `sc.extern` interface below. The names have docstrings but
are deliberately omitted from the published API reference, pending the section review. No guide
page ships with this change.

`CLibrary.sources` and `headers` map relative file names to source text. Construction copies them,
`defines` and `versions` into immutable tuples. `include_dirs` contains additional header paths,
`compile_flags` uses clang's dialect, and `link` contains names such as `"m"`. Flags apply only to
the library's translation units. A content-addressed object cache reuses them across symbols and
host Functions. Source, headers, expanded included headers, flags, the compiler and resolved linked
library content contribute to the cache key. `versions` also contributes to that key.

`extern(library, symbol, inputs=(arg(...), ...), outputs=arg(...) | group(...), workspace=0)` returns
a concrete callable. Each input tuple entry describes one positional argument. A group remains a
tuple-valued argument. The C function returns `void`, takes one `const double*` per input leaf, one
`double*` per output leaf, and then `double* w`. `workspace` counts doubles. Sparse buffers follow
the declared leaf layout. Calls must be deterministic and reentrant, read inputs only, fully write
outputs and retain no pointers. A failure status is an ordinary output.

## Example and observed result

The end-to-end example is `tests/function/test_extern.py`, especially `PAIR_SOURCE`, `pair_library`,
`pair`, `host` and `test_external_source_two_results_share_one_invocation_and_match_numpy`.
It declares one three-element input and two outputs, a three-element square and a scalar sum.
The source uses a supplied header, a library-local scaling macro and three workspace doubles.
With input `[0.2, -0.7, 1.1]`, the compiled result is:

```text
(array([0.04, 0.49, 1.21]), array(1.35))
```

The generated kernel has one external invocation for both outputs. Calling the returned extern
with numerical arrays compiles a wrapper. Symbolic calls build EXTERN nodes directly. The tests
also exercise nesting, mapping, an unused output, sparse storage conversion, a named `m` dependency
on Linux/macOS, and a consumer that builds and runs the exported manifest.

An active derivative request without a rule raises an error naming the C symbol and
`sc.custom_derivative`. Wrap the call in a Function and attach the needed forward or reverse rule.
The custom rule's derivative sparsity is its declaration, or dense when omitted. The example test
checks both forms against finite differences.

## Expected gaps

- #342 moves solver plugins onto externs and removes SOLVER_CALL. This change leaves plugins alone.
- #56 owns typed pointers and mixed-dtype workspaces. Extern leaves follow the dtypes the current
  generated calling convention supports, currently float64 at this task's base.
- #84 owns Windows acceptance beyond the existing CI jobs.

No roadmap departure was needed. Runtime timings are not evidence during the shared-machine run.

## Build manifest and review notes

Paths in `source`, each library's `sources` and `headers`, and the first `include_dirs` entry are
relative to the manifest directory. Additional include directories are absolute, resolved when
constructing `CLibrary`. `defines` is a JSON object, where `null` means `-DNAME` without `=value`.
`compile_flags` applies to that record's source files. Compile them as separate C translation
units, then link their objects with the generated source. `link` contains toolchain-resolved
library names, and `versions` is dependency-version metadata. The generated source's root flags
record the CPU recipe, but the consumer chooses compiler-specific vectorization tuning. The AOT
command reports the manifest and existing solver/math link flags. It does not require a compiler
or prepare a JIT cache entry.

Before #342, the interface had no library-directory field. Prebuilt libraries used the compiler's search path,
including `SCALY_CC="cc -L..."` or `LIBRARY_PATH`. Explicit compiler directories take precedence
over a system copy. A custom derivative attaches to a wrapping Function, and passing the extern
itself reports that requirement. Flatten order, NULL workspace and the current float64 boundary
are explicit in the extern docstring.

#357 makes compatible external kernels share Linux linker namespaces. Before loading, the JIT
asks the running process's ELF loader to list transitive dependencies without running
initializers, and reads shared-object names and content identities. It locates the loader and libc
through glibc's `getauxval` and `dladdr`, so loading needs neither `ldd` nor a shell. Metadata and
content hashes are cached by canonical path, file size and modification time. A namespace is
compatible when every common requested name, path and shared-object name identifies the same
canonical path and binary content. Identical copies at different paths stay separate because their
`$ORIGIN`-relative plugins can differ. Independent dependencies can join
it. Conflicting versions use separate namespaces, even when their files live in different
directories.

Each external namespace starts with the process's libc as its anchor. Kernels then load with
`RTLD_NOW | RTLD_LOCAL`. Only libc and the dynamic loader occupy the namespace's shared lookup
scope, so other symbols resolve through each kernel's own dependencies. Before #342 the solver
namespace remained separate, with its existing first-kernel anchor and load flags.

The correctness review rejected `RTLD_DEEPBIND`. It split `strdup`'s allocation in the shared libc
from a later kernel's `free`, which bound to a dependency's replacement allocator, and crashed in
both load orders. The extern buffer contract does not constrain allocation inside dependencies,
so it cannot justify that flag. A libc anchor keeps these allocation and release calls together.
Replacement allocator symbols in dependencies do not override libc through ordinary lookup.
#342 should retain this binding rule when moving solver libraries onto externs and must not rely
on an allocator interposer overriding libc.

Dependency inspection covers linked dependencies, not libraries opened later with `dlopen`.
Do not change `LD_LIBRARY_PATH` during the process: glibc retains its startup search path, while a
new inspection process sees the current environment. Use a fresh process after changing it.
C++ allocation and release pairs must follow the dependencies' own ABI, which the extern
buffer contract does not validate. C++ inline-function and template statics can have different copies across separately built
libraries unless their compiler gives them unique binding within the namespace. Namespace reuse
does not promise process-wide C++ singleton identity. AddressSanitizer-instrumented dependencies
and tools that require process-wide symbol interposition remain unsupported in isolated
namespaces, as before this change. Removing deep binding does not provide that support.

Loaded handles, namespace records and their accumulated dependency identities stay alive until
process exit. Dropping a Python compiled handle does not unload its code or free a namespace.
This preserves old callables after rebuilds and avoids reusing a namespace that still contains an
incompatible binary. Reclamation is unnecessary for compatible-library reuse and remains outside
this change. When incompatible live versions exhaust glibc's namespaces or static thread-local
storage, the error includes the loader's reason and advises using a fresh process with fewer
versions. #159 still owns environment changes after the last isolated load.

The namespace choice does not enter either cache key or change the artifact/index schema.
Both the graph index and the rendered key reuse the library build fingerprint. It hashes every
readable file below declared include directories, preprocessed sources and resolved linked-library
contents/versions, so changes between processes miss the index. An unchanged index hit does not
lower the graph. The key still preprocesses each source when a graph is looked up. `_Extern` supplies the
callee metadata the compiler walks, rather than being a concrete expression graph. These last two
choices remain implementation points for the section's later simplification pass.


## Solver bridges on externs (#342)

Every solver Function contains ordinary EXTERN nodes selecting the outputs of
one concrete bridge. Construction records the installed library and oracle graphs without
lowering. At module rendering, core lowers the oracles together with the selected build
recipe and renders fixed-signature oracle entries and the plugin wrapper into CLibrary
sources. Warm JIT index hits skip both host and oracle lowering. The library describes the
installed headers and named native dependency. SQP and Fatrop use the same
bridge contract so the retired operation and all special lowering can disappear.
The plugin protocol is version 9. No native build hooks change.

A host Function lowers normally, including when it mixes solver calls and
user externs. Each bridge declares its oracle workspace. Its source compiles
once through the external object cache and links into each host shared object.
The host's public option entry sets each bridge's options before evaluation.
The bridge keeps its native workspace and statistics, with the same exported
statistics accessor and default-option API. All internal oracle signatures
end in `w`; they no longer carry an option context. Fixed-signature entries
write only the oracle's declared outputs, even when its internal calls also
write custom-rule residuals. The lowerer accepts ordinary C name reservations
for the surrounding bridge and supplied external oracles.

Two qualifications are recorded in the roadmap. `CLibrary.library_dirs` gives
the installed library search directories, replacing the previous requirement
to change the compiler environment. It appears in the exported manifest.
Solver bridges keep their non-reentrant native workspace and statistics and add a
per-bridge option pointer. The general extern contract's ban on state cannot apply to them. This is
an explicit exception for solver bridges, not a broader extern promise.

The JIT has one isolated external-library path. Solver libraries use #357's
checked namespace pool, libc anchor and ordinary local lookup. The separate
solver pool, backend build flags and native-directory fingerprint in the JIT
are removed. Dependency lookup and extern fingerprints own these inputs, including
contents of declared library directories so Windows DLL changes invalidate an unchanged
import archive. Object builds compile from their fingerprinted source directory, so
a compiler cache cannot retain a header whose contents changed at the same timestamp.
The allocation, environment, interposition and lifetime limits described
above still apply.

Ahead-of-time consumers compile the main source and every library source
listed in the build manifest. Each library record supplies its own include
directories, defines, compilation flags, library directories and dependency
names. Native paths describe the exporting installation and may need changing
for a deployment target. The main source alone no longer contains the solver
wrapper and oracles.

The solver snapshot is split between `solver.c` and `solver.extern.c`, so the
bridge remains covered. All other existing snapshots, including `solver.h`,
stay byte-identical. `solver_externs/measurements.json` records generated source
size and C compile observations on shared la015. The combined source grows
from 10,932 to 13,099 bytes because separate translation units repeat their
C declarations. These are compile observations, not a runtime performance claim.

The later conflict rebase preserves typed entry/workspace handling from #56 and
ordinary support-specialized procedures from #225. Solver leaf validation runs
when preparing the bridge, before generic lowering. The solver host snapshot
adds only the `_support` procedure suffix after that rebase; its external bridge
source and all other snapshots stay byte-identical.
