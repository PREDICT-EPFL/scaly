"""Compare C generation, object compilation and declared storage without running kernels."""

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path

from benchmarks.harness import sweep
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import main_proc
from tests.passes.lowering.test_kkt_assembly import kkt_assembly

CELLS = (("chain", 5), ("race_cars", 5), ("npmpc_jac", 100), ("npmpc", 100), ("npmpc_jac", 6), ("kkt", 1))


def storage(module):
  root = main_proc(module.program)
  arrays = [
    (stmt.attrs["name"], stmt.dtype.c_type, max(1, math.prod(stmt.attrs["shape"])))
    for stmt in root.args[root.attrs["param_count"] :]
    if stmt.op == ProgramOp.BUFFER
    and stmt.attrs["address_space"] == "private"
    and "alias_of" not in stmt.attrs
    and "workspace_offset" not in stmt.attrs
  ]
  return {
    "workspace_bytes": module.workspace_size,
    "workspace_elements": {dtype.c_type: count for dtype, count in root.attrs["workspace_layout"]},
    "declared_local_double_slots": sum(count for _, dtype, count in arrays if dtype == "double"),
    "root_declarations": arrays,
    "mapped_lane_caps": [
      {"trips": stmt.attrs["vector_count"], "caps": stmt.args[1].args[0].attrs["lane_caps"]}
      for stmt in walk_program(root)
      if stmt.op == ProgramOp.FOR and stmt.attrs.get("vector_mapped")
    ],
  }


def measure(out, clang):
  original_render = sweep.render_c_module
  captured = []

  def render(*args, **kwargs):
    module = original_render(*args, **kwargs)
    captured.append(module)
    return module

  sweep.render_c_module = render
  cells = []
  neural = []
  for workload, size in CELLS:
    folder = out / f"{workload}_{size}"
    folder.mkdir(parents=True, exist_ok=True)
    if workload == "kkt":
      start = time.perf_counter()
      fun, _, _ = kkt_assembly()
      build_ms = (time.perf_counter() - start) * 1000
      _, render_ms = sweep._render_scaly(fun, fun.name, folder)
      build = {"source": Path(f"{fun.name}.c"), "build_ms": build_ms, "render_ms": render_ms}
    else:
      build = sweep.build_kernel(workload, size, "scaly", folder)
    module = captured.pop()
    path = folder / build["source"]
    command = [clang, "-O3", "-march=native", "-fno-math-errno", "-fveclib=libmvec", "-c", path.name, "-o", "kernel.o"]
    start = time.perf_counter()
    subprocess.run(command, cwd=folder, check=True)
    compile_ms = (time.perf_counter() - start) * 1000
    cells.append({
      "workload": workload,
      "size": size,
      "source_lines": module.source.count("\n") + 1,
      "source_bytes": path.stat().st_size,
      "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "header_sha256": hashlib.sha256((folder / module.header_name).read_bytes()).hexdigest(),
      "build_ms": build["build_ms"],
      "render_ms": build["render_ms"],
      "generation_ms": build["build_ms"] + build["render_ms"],
      "compile_ms": compile_ms,
      "compile_command": command,
      **storage(module),
    })
    if workload == "npmpc_jac":
      for lanes in ("auto", 2, 4):
        variant = module if lanes == "auto" else original_render(build["callable"], lanes=lanes, vector_libm=sweep.vector_libm())
        (folder / f"lanes_{lanes}.c").write_text(variant.source)
        neural.append({"horizon": size, "lanes": lanes, **storage(variant)})
  return cells, neural


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  clang = shutil.which("clang")
  assert clang is not None
  cells, neural = measure(args.output, clang)
  result = {
    "machine": platform.node(),
    "python": platform.python_version(),
    "compiler": subprocess.check_output([clang, "--version"], text=True).splitlines()[0],
    "cells": cells,
    "neural_storage": neural,
  }
  (args.output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
  main()
