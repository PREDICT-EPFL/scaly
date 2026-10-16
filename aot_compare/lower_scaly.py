"""Scaly's path: render the lower-triangle Hessian Function to one C source and its header."""

from pathlib import Path

from scaly.codegen.aot import render_c_module
from scaly.codegen.toolchain import find_c_compiler, native_recipe

import hessian
from harness import Kernel


def lower(N: int, out: Path) -> Kernel:
  """Writes `npmpc_hess.c` and `npmpc_hess.h`; the harness compiles the source with its own flags."""
  function, rows, cols = hessian.scaly_lower(N)
  # The math library Scaly's benchmarks select on this host: libmvec when glibc is recent enough,
  # so that sin, cos and exp vectorize.
  compiler = find_c_compiler()
  vector_libm = native_recipe(compiler.command).vector_libm if compiler is not None else "none"
  module = render_c_module(function, header_name="npmpc_hess.h", source_name="npmpc_hess.c", vector_libm=vector_libm)
  (out / module.header_name).write_text(module.header)
  (out / module.source_name).write_text(module.source)
  return Kernel(out, rows, cols, adapter="scaly.c", sources=(out / module.source_name,))
