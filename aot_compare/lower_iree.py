"""The IREE path: export the asdex Hessian's StableHLO with `jax.export`, then compile it with `iree-compile` to a static dispatch library and a VM module emitted as C."""

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

import hessian
import problem
import toolchain
from harness import Kernel

# Only what the build needs; none of these tune the generated code.
IREE_FLAGS = (
  "--iree-input-type=stablehlo",
  # IREE otherwise rewrites every f64 to f32, silently.
  "--iree-input-demote-f64-to-f32=false",
  "--iree-hal-target-device=local",
  "--iree-hal-local-target-device-backends=llvm-cpu",
  "--iree-llvmcpu-target-cpu=host",
  # The dispatches as a plain object to link, whose sin, cos and exp resolve against libm, instead of
  # IREE's self-contained ELF, which has no f64 versions of them.
  "--iree-llvmcpu-link-embedded=false",
  "--iree-llvmcpu-link-static",
  # The VM program as C rather than as bytecode for the runtime to interpret.
  "--output-format=vm-c",
)


def lower(N: int, out: Path) -> Kernel:
  """Writes `stablehlo.mlir`, the compiler's input, and its output: `module.h`, the VM program as C,
  and `library.o` with `library.h`, the dispatches' machine code and the function that lists them."""
  function, rows, cols = hessian.asdex_lower(N)
  specs = [jax.ShapeDtypeStruct((size,), jnp.float64) for size in (problem.n_z(N), problem.N_P, problem.n_g(N))]
  (out / "stablehlo.mlir").write_text(jax.export.export(jax.jit(function))(*specs).mlir_module())
  iree_compile = Path(sys.executable).parent / "iree-compile"
  subprocess.run(
    [iree_compile, out / "stablehlo.mlir", *IREE_FLAGS, f"--iree-llvmcpu-static-library-output-path={out / 'library.o'}", "-o", out / "module.h"],
    check=True,
  )
  runtime = toolchain.iree_runtime()
  return Kernel(
    out,
    rows,
    cols,
    adapter="iree.c",
    objects=(out / "library.o",),
    # The allocator IREE's CMake package would define for its users.
    flags=("-isystem", runtime / "include", "-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl"),
    libs=tuple(runtime / "lib" / f"lib{name}.a" for name in toolchain.IREE_LIBRARIES),
  )
