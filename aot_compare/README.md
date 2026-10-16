# The npmpc Hessian ahead of time: Scaly, XLA and IREE

One derivative, compiled three ways into a C/C++ program and timed in one harness: the lower
triangle of the Lagrangian Hessian of the neural-process MPC problem (`benchmarks/problems/npmpc`)
at horizon N, with respect to the decision vector.

- **Scaly**: `sc.sparse_hessian` of the Lagrangian, rendered to one C source.
- **XLA**: the same Lagrangian in JAX, differentiated by [asdex](https://adrianhill.de/asdex/),
  wrapped by `jax2tf` and compiled by TensorFlow's XLA ahead-of-time compiler
  (`saved_model_cli aot_compile_cpu`) into a C++ class.
- **IREE**: the same asdex Hessian, exported as StableHLO and compiled by `iree-compile` into a
  static library of dispatches plus a virtual machine (VM) module emitted as C, which IREE's
  runtime runs.

Nothing is tuned by hand: each tool gets the flags it needs to build correct f64 code for this CPU
and nothing else. The harness follows Scaly's benchmark cells: the same inputs for every backend,
a check of every lower-triangle entry against a dense `jax.hessian` (1e-9 absolute plus 1e-9
relative), then Google Benchmark timing the call. The directory uses Scaly but not `benchmarks/`;
the problem is copied into `problem.py` and its trained checkpoint into `data/`.

## Running

```bash
uv run bench.py                        # N = 12, 50, 200, three fresh processes per binary
uv run bench.py --sizes 12 --reps 1
uv run bench.py --backends scaly,iree  # without TensorFlow, which has Linux wheels only
```

The xla backend needs TensorFlow, in the `xla` dependency group: `uv run --group xla bench.py`.

It needs clang and git; CMake and Ninja come from PyPI. The first run fetches and builds Google
Benchmark, IREE's runtime and, for xla, Abseil and the runtime TensorFlow ships for its AOT classes,
all into `build/third_party/`. IREE's runtime takes about a minute, since the compiler, and with it
LLVM, is left out.

## The code

| file | |
|---|---|
| `problem.py` | the layout, decoder weights, sample inputs and dense reference; the Lagrangian in JAX and in Scaly |
| `hessian.py` | the lower-triangle Hessians: asdex on the JAX Lagrangian, `sc.sparse_hessian` on the Scaly one |
| `lower_scaly.py`, `lower_xla.py`, `lower_iree.py` | one lowering path each, returning what the harness links |
| `toolchain.py` | the third-party pieces and the flags everything is compiled with |
| `harness.py`, `cpp/` | `main.cpp` and one adapter per backend behind `kernel.h`; builds and runs the binaries |
| `bench.py` | runs it all and prints the table |

## How the two ahead-of-time paths work

Both start from the same JAX function, `hessian.asdex_lower(N)`, and end with machine code plus C or
C++ that a program links directly, with no Python or just-in-time compiler at run time.

### XLA, through TensorFlow

`jax.jit` compiles the same function with XLA, but its executable is not usable ahead of time.
Serialized, it does contain the kernels' machine code (35 objects at N = 12, 19 KiB in total,
needing only libm and libc), but not a program around them:

- the order in which the kernels run and where their buffers sit are data for XLA's runtime, and
  that runtime lives inside jaxlib (`libjax_common.so`, 353 MB);
- every matrix product is a YNNPACK fusion that jaxlib runs itself, so the executable has no
  machine code for them;
- jaxlib exports nothing a C or C++ program can call, only its Python modules, and no published
  library of XLA's CPU runtime (a PJRT plugin) was found to load the executable with.

What remains is building that runtime from XLA's sources or writing a driver for the schedule, both
of which mean building infrastructure rather than using a tool. TensorFlow packages one:
`saved_model_cli aot_compile_cpu` runs XLA:CPU on a SavedModel and writes a standalone C++ class.
Getting the JAX function there, and the class into a program, took:

- `jax2tf.convert`, which wraps the function's StableHLO in an `XlaCallModule` operation of a
  TensorFlow graph, saved as a SavedModel;
- the target CPU spelled out with `--target_cpu`, taken from what clang resolves `-march=native` to;
- the runtime TensorFlow ships as source for these classes (`xla_aot_runtime_src`), built with its
  own CMake file at `-O3` because some of its Eigen convolution sources do not compile with AVX-512;
  the generated schedule calls only its f64 matrix product, which is rebuilt with the harness flags;
- Abseil, whose mutexes, counters and logging the generated header uses: TensorFlow ships its
  headers but not its libraries, so the matching release, 20250814.1, is built from source;
- inputs aligned to 64 bytes and set by name, since the class requires the alignment and orders its
  arguments its own way.

TensorFlow, 1.3 GiB installed, is needed only to compile; the program needs libc, libm and
libstdc++. The XLA it uses is TensorFlow's snapshot rather than jaxlib's, and it computes matrix
products with Eigen rather than YNNPACK.

### IREE

`jax.export` writes the function's StableHLO, and the PyPI compiler, `iree-base-compiler`, takes it
unchanged. Besides the flags that select the CPU target, three choices matter, none of them tuning:

- `--iree-input-demote-f64-to-f32=false`, or IREE silently computes in f32;
- `--iree-llvmcpu-link-embedded=false` with `--iree-llvmcpu-link-static`: IREE's default
  self-contained ELF has no f64 `sin`, `cos` or `exp`, so the dispatches become a plain object
  whose calls resolve against libm;
- `--output-format=vm-c`, so the program that drives the dispatches comes out as C rather than as
  bytecode (a `.vmfb` file) for the runtime to load and interpret.

The runtime is the harder part. The PyPI runtime wheel has it compiled into its Python extension,
with no headers or libraries a C program can use, so the C runtime is built from source at the
compiler wheel's release, with only the `local-sync` driver. The host, `cpp/iree.c`, defines the system allocator IREE's CMake package
would otherwise set, and creates the single-threaded `local-sync` device with the loader for
dispatches linked into the binary, plus the asynchronous I/O pool that 3.12 requires every device
to have. Each call then allocates buffers, records the dispatches and barriers into a command
buffer, submits it and waits; `local-sync` runs the dispatches on the calling thread.

IREE's `--iree-execution-model=inline-dynamic` would remove the device, its command buffers and
the asynchronous machinery: the module then calls the dispatches directly and imports only buffer
allocation and dispatch from the runtime. In 3.12 it runs a small function correctly but fails on
this Hessian with a null reference inside `hal_loader.executable.dispatch`, in IREE's own
`iree-run-module` as well as in a host written for it, so the IREE column uses the default model.

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

No generated object contains a runtime: each only references one, and the final link pulls it in
from the archives in `build/third_party/`. The benchmark binary adds gbench and the harness on top,
so sizes come from a minimal program instead, one that sets up and makes one call, at N = 12 and
linked like the benchmarks with `--gc-sections`. The table counts the code, constants and data the
binary keeps, attributed to each input with the link map:

| backend | generated | runtime | total | stripped file | shared libraries |
|---|---:|---:|---:|---:|---|
| Scaly | 144 KiB: `npmpc_hess.o` | none | 145 KiB | 154 KiB | libc, libm |
| IREE | 133 KiB: `iree.o` 80 KiB (the VM module, compiled in with the adapter), `library.o` 52 KiB | about 680 KiB | 815 KiB | 986 KiB | libc, libm |
| XLA AOT | 83 KiB: kernels 47 KiB, constants 17 KiB, `xla.o` 20 KiB (the schedule, compiled in with the adapter) | Eigen matrix product 1,054 KiB, Abseil and the AOT runtime about 280 KiB | 1,447 KiB | 1,655 KiB | libc, libm, libstdc++ |

The XLA row has assertions compiled out with `-DNDEBUG`, as TensorFlow's own Release build of its
runtime does. `toolchain.py` currently rebuilds the matrix product without that define, which keeps
Eigen's assertions and 1.8 MiB of their messages: 3,316 KiB in total, 3,531 KiB stripped. Timing is
the same either way, within 0.2%.

IREE's runtime is 618 KiB of code from its archives (the VM, the module of the hardware
abstraction layer (HAL) that the program calls, HAL buffers and allocators, the `local-sync` device
and its loaders, asynchronous I/O, base utilities and flatcc), plus most of the 64 KiB of strings,
small constants and padding the linker merges across inputs. Creating the runtime instance registers
every driver the runtime was built with, so only `local-sync` is built. IREE's prebuilt x86-64 Linux
runtime also has the multithreaded `local-task` driver and the Vulkan driver, and with it the same
program is 1,199 KiB, 1,418 KiB stripped.
