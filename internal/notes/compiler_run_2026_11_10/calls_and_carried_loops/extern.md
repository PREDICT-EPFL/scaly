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

The interface has no library-directory field. Prebuilt libraries use the compiler's search path,
including `SCALY_CC="cc -L..."` or `LIBRARY_PATH`. Explicit compiler directories take precedence
over a system copy. A custom derivative attaches to a wrapping Function, and passing the extern
itself reports that requirement. Flatten order, NULL workspace and the current float64 boundary
are explicit in the extern docstring.

#357 owns reuse of compatible Linux linker namespaces. The operating system limits the number
of live namespaces, so many linked-binary rebuilds in one process can exhaust them. The loader now
reports its actual error. Source-only builds share one namespace. The key still preprocesses each
source when a graph is looked up, to detect changed included headers. `_Extern` supplies the
callee metadata the compiler walks, rather than being a concrete expression graph. These last two
choices remain implementation points for the section's later simplification pass.
