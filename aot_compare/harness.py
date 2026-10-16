"""The gbench harness: link one backend's generated Hessian behind cpp/main.cpp, then time the binaries as fresh, interleaved processes."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import problem
import toolchain

CPP = Path(__file__).parent / "cpp"


@dataclass(frozen=True)
class Kernel:
  """One backend's generated artifacts in `dir`, and what it takes to link them behind the harness.

  `rows` and `cols` place each output value in the Hessian. `adapter` is the file in cpp/ that puts
  the backend's calling convention behind cpp/kernel.h; `flags` are what compiling it needs beyond
  the harness flags.
  """

  dir: Path
  rows: np.ndarray
  cols: np.ndarray
  adapter: str
  sources: tuple[Path, ...] = ()
  objects: tuple[Path, ...] = ()
  flags: tuple[str | Path, ...] = ()
  libs: tuple[str | Path, ...] = ()


def write_inputs(directory: Path, N: int) -> Path:
  """The sample inputs and the dense reference Hessian, as raw doubles the harness reads at startup."""
  directory.mkdir(parents=True, exist_ok=True)
  z, p, lam_g = problem.sample_inputs(N)
  for name, values in (("z", z), ("p", p), ("lam_g", lam_g), ("hessian", problem.dense_hessian(N, z, p, lam_g))):
    np.asarray(values, dtype=np.float64).tofile(directory / f"{name}.bin")
  return directory


def build(kernel: Kernel, N: int, inputs: Path) -> Path:
  """Compile the adapter, cpp/main.cpp and any generated C, and link them with the backend's objects into `dir/bin/benchmark`."""
  out = kernel.dir / "bin"
  shutil.rmtree(out, ignore_errors=True)
  out.mkdir()
  (out / "layout.h").write_text(
    "\n".join(
      [
        "// Input sizes and where each output value sits in the Hessian, for this backend at this horizon.",
        f"#define NZ {problem.n_z(N)}",
        f"#define NP {problem.N_P}",
        f"#define NLAM {problem.n_g(N)}",
        f"#define NNZ {kernel.rows.size}",
        f"static const int ROWS[NNZ] = {{{', '.join(map(str, kernel.rows))}}};",
        f"static const int COLS[NNZ] = {{{', '.join(map(str, kernel.cols))}}};",
        "",
      ]
    )
  )
  gbench = toolchain.gbench()
  flags = [*toolchain.FLAGS, "-I", CPP, "-I", out, "-I", kernel.dir, *kernel.flags]
  objects = []
  for source in (*kernel.sources, CPP / kernel.adapter, CPP / "main.cpp"):
    obj = out / f"{source.stem}.o"
    if source.suffix == ".c":
      toolchain.run(["clang", *flags, "-c", source, "-o", obj])
    else:
      toolchain.run(["clang++", *flags, "-std=c++17", "-I", gbench / "include", f'-DINPUTS_DIR="{inputs}"', "-c", source, "-o", obj])
    objects.append(obj)
  binary = out / "benchmark"
  toolchain.run(
    ["clang++", "-pthread", *objects, *kernel.objects, "-o", binary, *kernel.libs, gbench / "lib" / "libbenchmark.a", *toolchain.VECTOR_LIBM, "-lm", "-ldl"]
  )
  return binary


def run(binary: Path, min_time: str) -> float:
  """CPU time of one evaluation in microseconds, as Google Benchmark reports it; the binary checks its output first."""
  out = subprocess.run([binary, f"--benchmark_min_time={min_time}", "--benchmark_format=json"], capture_output=True, text=True)
  if out.returncode:
    raise RuntimeError(f"{binary} failed ({out.returncode}): {out.stderr.strip()}")
  data = json.loads(out.stdout[out.stdout.index("{") :])
  return next(b["cpu_time"] for b in data["benchmarks"] if b["run_type"] == "iteration") / 1e3
