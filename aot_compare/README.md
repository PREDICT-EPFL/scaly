# The npmpc Hessian ahead of time: Scaly, XLA and IREE

One derivative, compiled three ways into a C/C++ program and timed in one harness: the lower
triangle of the Lagrangian Hessian of the neural-process MPC problem (`benchmarks/problems/npmpc`)
at horizon N, with respect to the decision vector.

- **Scaly**: `sc.sparse_hessian` of the Lagrangian, rendered to one C source.
- **XLA**: the same Lagrangian in JAX, differentiated by [asdex](https://adrianhill.de/asdex/),
  wrapped by `jax2tf` and compiled by TensorFlow's XLA ahead-of-time compiler
  (`saved_model_cli aot_compile_cpu`) into a C++ class.
- **IREE**: the same asdex Hessian, exported as StableHLO and compiled by `iree-compile` into a
  static library of dispatches plus a VM module emitted as C, which IREE's runtime runs.

Nothing is tuned by hand: each tool gets the flags it needs to build correct f64 code for this CPU
and nothing else. The harness follows Scaly's benchmark cells: the same inputs for every backend,
a check of every lower-triangle entry against a dense `jax.hessian` (1e-9 absolute plus 1e-9
relative), then Google Benchmark timing the call. The directory uses Scaly but not `benchmarks/`;
the problem is copied into `problem.py` and its trained checkpoint into `data/`.

## Running

```bash
uv run bench.py                        # N = 12, 50, 200, three fresh processes per binary
uv run bench.py --sizes 12 --reps 1
```

It needs clang, cmake and git. The first run fetches and builds Google Benchmark and Abseil,
downloads IREE's runtime (an 80 MB tarball, of which three libraries and the headers are kept) and
builds the runtime TensorFlow ships for its AOT classes, all into `build/third_party/`.

## The code

| file | |
|---|---|
| `problem.py` | the layout, decoder weights, sample inputs and dense reference; the Lagrangian in JAX and in Scaly |
| `hessian.py` | the lower-triangle Hessians: asdex on the JAX Lagrangian, `sc.sparse_hessian` on the Scaly one |
| `lower_scaly.py`, `lower_xla.py`, `lower_iree.py` | one lowering path each, returning what the harness links |
| `toolchain.py` | the third-party pieces and the flags everything is compiled with |
| `harness.py`, `cpp/` | `main.cpp` and one adapter per backend behind `kernel.h`; builds and runs the binaries |
| `bench.py` | runs it all and prints the table |

## The artifacts

`build/N{N}/<backend>/` holds exactly the generated files that backend's binary is built from;
`bin/` beneath it holds the harness build. Read an object with `objdump -d --no-show-raw-insn`.

| backend | file | |
|---|---|---|
| scaly | `npmpc_hess.c`, `npmpc_hess.h` | the kernel, and its pointer ABI with the workspace size |
| xla | `saved_model/` | the compiler's input: the asdex Hessian's StableHLO inside a TensorFlow graph |
| xla | `npmpc_hess.h` | the `NpmpcHess` class; `Run()` is the schedule of kernel calls, written out as C++ |
| xla | `npmpc_hess.o`, `npmpc_hess_constants.o` | the kernels' machine code, which imports only libm and libc, and the constants they read |
| iree | `stablehlo.mlir` | the compiler's input, from `jax.export` |
| iree | `module.h` | the VM program as C: it allocates the buffers, records 82 dispatches and 63 barriers at every horizon, and holds the constants |
| iree | `library.o`, `library.h` | the dispatches' machine code, and the function the runtime asks for the list of them |

## Results

Ryzen 9 7940HS laptop, performance governor, boost not controlled. CPU time per evaluation in µs,
best of three processes, with the ratio to Scaly:

| N | Scaly | XLA AOT | IREE |
|---:|---:|---:|---:|
| 12 | 22.4 | 37.9 (1.69×) | 45.4 (2.03×) |
| 50 | 84.8 | 146.3 (1.73×) | 137.1 (1.62×) |
| 200 | 242.2 | 562.5 (2.32×) | 462.4 (1.91×) |

Scaly's N = 50 cell varies most between processes, from 84.8 to 92.3 µs; every other cell stays
within 3%.

## What each needs at run time

Code and constants the linker keeps for a minimal program at N = 12 (setup and one evaluation,
`-O3`, `--gc-sections`):

| backend | generated | runtime | total | shared libraries |
|---|---:|---:|---:|---|
| Scaly | 145 KiB | none | 145 KiB | libc, libm |
| XLA AOT | 66 KiB of kernels and constants, 1.9 MiB of schedule and its inlined C++ | 2.9 MiB of Eigen matrix product, 0.3 MiB of Abseil | 5.1 MiB | libc, libm, libstdc++ |
| IREE | 191 KiB of VM module, 57 KiB of dispatches | 1.07 MiB | 1.3 MiB | libc, libm |

Most of IREE's runtime is the local device and its executable loaders (414 KiB), the VM (218 KiB),
HAL buffers and allocators (104 KiB), asynchronous I/O (100 KiB), base utilities (84 KiB) and the
HAL module the program calls (30 KiB). This is the prebuilt x86-64 Linux build, which links in
support for every driver and loader it was built with. An embedded target would use a runtime built from
IREE's runtime sources for that target, with one driver and one loader (IREE's bare-metal guide);
the prebuilt one cannot be used there.

IREE's `--iree-execution-model=inline-dynamic` would remove the device, its command buffers and
the asynchronous machinery: the module then calls the dispatches directly and imports only buffer
allocation and dispatch from the runtime. In 3.12 it runs a small function correctly but fails on
this Hessian with a null reference inside `hal_loader.executable.dispatch`, in IREE's own
`iree-run-module` as well as in this harness, so the IREE column uses the default model on the
single-threaded `local-sync` device.
