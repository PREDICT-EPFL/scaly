"""The XLA path: wrap the asdex Hessian in a SavedModel with jax2tf, then compile it ahead of time with TensorFlow's `saved_model_cli aot_compile_cpu`.

jaxlib cannot do this itself: the executable `jax.jit` builds runs only inside jaxlib's own runtime,
which exports no C interface. TensorFlow's AOT compiler is the same XLA:CPU emitting a standalone C++
class instead.
"""

import subprocess
import sys
from pathlib import Path

import tensorflow as tf
from jax.experimental import jax2tf

import hessian
import problem
import toolchain
from harness import Kernel


def lower(N: int, out: Path) -> Kernel:
  """Writes `saved_model/`, the compiler's input, and its output: `npmpc_hess.h`, which declares the
  `NpmpcHess` class and holds the schedule of kernel calls as C++, `npmpc_hess.o` with the kernels'
  machine code, and `npmpc_hess_constants.o` with the constants they read."""
  function, rows, cols = hessian.asdex_lower(N)
  npmpc_hess = tf.function(
    jax2tf.convert(function, with_gradient=False),
    autograph=False,
    input_signature=[
      tf.TensorSpec((size,), tf.float64, name=name) for name, size in (("z", problem.n_z(N)), ("p", problem.N_P), ("lam_g", problem.n_g(N)))
    ],
  )
  tf.saved_model.save(tf.Module(), str(out / "saved_model"), signatures={"serving_default": npmpc_hess})
  subprocess.run(
    [
      Path(sys.executable).parent / "saved_model_cli",
      "aot_compile_cpu",
      "--dir",
      out / "saved_model",
      "--tag_set",
      "serve",
      "--signature_def_key",
      "serving_default",
      "--output_prefix",
      out / "npmpc_hess",
      "--cpp_class",
      "NpmpcHess",
      "--target_triple",
      "x86_64-pc-linux",
      "--target_cpu",
      toolchain.host_cpu(),
    ],
    check=True,
    capture_output=True,
  )
  # Its other two outputs are not needed: a makefile fragment repeating the include path and ABI flag
  # passed below, and the program-shape metadata, which nothing in the harness asks for.
  (out / "npmpc_hess_makefile.inc").unlink()
  (out / "npmpc_hess_metadata.o").unlink()
  return Kernel(
    out,
    rows,
    cols,
    adapter="xla.cpp",
    objects=(out / "npmpc_hess.o", out / "npmpc_hess_constants.o", *toolchain.xla_aot_runtime()),
    # The generated header uses TensorFlow types that TensorFlow itself has since deprecated.
    flags=("-isystem", toolchain.tensorflow_dir() / "include", "-D_GLIBCXX_USE_CXX11_ABI=1", "-Wno-deprecated-declarations"),
    libs=("-Wl,--start-group", *toolchain.abseil(), "-Wl,--end-group"),
  )
