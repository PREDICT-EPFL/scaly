# scaly-fatrop

The `"fatrop"` Scaly backend solves nonlinear programs with stage structure.
Install `scaly[fatrop]`, declare a stage-ordered `sc.problem`, and select it with
`sc.solver(problem, "fatrop")`. See the [solver guide](https://github.com/PREDICT-EPFL/scaly/blob/main/docs/guide/solvers.md#stage-structured-problems-with-fatrop)
for the variable ordering and an example.

This package bundles the [Fatrop](https://github.com/meco-group/fatrop) shared library and C
headers for Linux, macOS and Windows. `scaly_fatrop.lib_dir()` and `scaly_fatrop.include_dir()` return
the directories needed to compile and link a C consumer of `fatrop/ocp/OCPCInterface.h`.

Link with `-lfatrop` and add `lib_dir()` to the runtime library search path. On Windows, compile
consumers with `FATROP_SYMBOL_EXPORT` defined as `__declspec(dllimport)`. Put `lib_dir()` on
`PATH` for executables. Python consumers can use `os.add_dll_directory` in a `with` block.

The included BLASFEO headers provide the matrix and vector types used by the C interface.
BLASFEO is linked statically with hidden symbols, so this package can share a process with
`scaly-piqp`.

Fatrop uses BSD-2-Clause and EPL-2.0 licenses. BLASFEO uses BSD-2-Clause. The wheel includes their
license texts and pinned source links under `scaly_fatrop/licenses/`.
