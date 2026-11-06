# scaly-fatrop

This package bundles the [Fatrop](https://github.com/meco-group/fatrop) shared library and C
headers for Linux, macOS and Windows. `scaly_fatrop.lib_dir()` and `scaly_fatrop.include_dir()` return
the directories needed to compile and link a C consumer of `fatrop/ocp/OCPCInterface.h`.

Link with `-lfatrop` and add `lib_dir()` to the runtime library search path. The included BLASFEO
headers provide the matrix and vector types used by the C interface. BLASFEO is linked statically
with hidden symbols, so this package can share a process with `scaly-piqp`.

Fatrop uses BSD-2-Clause and EPL-2.0 licenses. BLASFEO uses BSD-2-Clause. The wheel includes their
license texts and pinned source links under `scaly_fatrop/licenses/`.
