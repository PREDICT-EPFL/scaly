"""Large tables: what they cost to generate and compile, against CasADi's code generator.

    uv run internal/notes/perf_2026_09_28_interp/bench_codegen.py [--only NAME]

For each table, one Function of a point giving the value: its lowering and rendering time, the C
source size, and the compile time at the JIT's flags; CasADi 3.8's `interpolant` on the same data
through `CodeGenerator`. Tables: a 1-D linear table of 1e6 points; a 1-D cubic of 1e4; a 2-D bicubic
of 256 x 256 (per-cell polynomials and local bases both); a 3-D tricubic of 64 x 64 x 64 (local
bases) and a 3-D trilinear of 64 x 64 x 64.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from _harness import BUILD, ROOT

OUT = BUILD / "codegen"


def tables() -> dict[str, dict]:
  rng = np.random.default_rng(0)
  g1 = np.linspace(0.0, 1.0, 1_000_000)
  g2 = np.linspace(0.0, 1.0, 256)
  g3 = np.linspace(0.0, 1.0, 64)
  return {
    "1d_linear_1e6": {"grid": (g1,), "values": np.sin(7 * g1), "kind": "linear"},
    "1d_cubic_1e4": {"grid": (g1[::100],), "values": np.sin(7 * g1[::100]), "kind": "cubic"},
    "2d_bicubic_256_pp": {"grid": (g2, g2), "values": rng.normal(size=(256, 256)), "kind": "cubic", "strategy": "pp"},
    "2d_bicubic_256_basis": {"grid": (g2, g2), "values": rng.normal(size=(256, 256)), "kind": "cubic", "strategy": "basis"},
    "3d_tricubic_64_basis": {"grid": (g3, g3, g3), "values": rng.normal(size=(64, 64, 64)), "kind": "cubic", "strategy": "basis"},
    "3d_trilinear_64": {"grid": (g3, g3, g3), "values": rng.normal(size=(64, 64, 64)), "kind": "linear"},
  }


def compile_seconds(source, out) -> float:
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler

  t0 = time.perf_counter()
  subprocess.run([find_c_compiler().cc, *compile_flags(), "-fPIC", "-shared", str(source), "-lm", "-o", str(out)], check=True)
  return time.perf_counter() - t0


def scaly_side(name: str, spec: dict) -> dict:
  import scaly as sc
  from scaly import interp
  from scaly.codegen import render_c_module

  folder = OUT / name / "scaly"
  folder.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  f = interp.interpolant(spec["grid"] if len(spec["grid"]) > 1 else spec["grid"][0], spec["values"], kind=spec["kind"], strategy=spec.get("strategy", "auto"))
  x = sc.sym("x", () if len(spec["grid"]) == 1 else (len(spec["grid"]),))
  fn = sc.Function._from_exprs(f"cg_{name}", [x], [f(x)], ["x"], ["y"])
  fit = time.perf_counter() - t0
  t0 = time.perf_counter()
  module = render_c_module(fn)
  render = time.perf_counter() - t0
  (folder / "f.c").write_text(module.body)
  return {"fit_s": fit, "render_s": render, "c_bytes": len(module.body), "compile_s": compile_seconds(folder / "f.c", folder / "lib.so"), "strategy": f.strategy}


def casadi_side(name: str, spec: dict) -> dict:
  import casadi as ca

  folder = OUT / name / "casadi"
  folder.mkdir(parents=True, exist_ok=True)
  method = "linear" if spec["kind"] == "linear" else "bspline"
  t0 = time.perf_counter()
  itp = ca.interpolant(f"ca_{name}", method, [list(g) for g in spec["grid"]], np.ravel(spec["values"], order="F"))
  fit = time.perf_counter() - t0
  x = ca.MX.sym("x", len(spec["grid"]))
  fn = ca.Function(f"cg_{name}", [x], [itp(x)])
  t0 = time.perf_counter()
  gen = ca.CodeGenerator("f.c", {"casadi_int": "long long", "casadi_real": "double"})
  gen.add(fn)
  gen.generate(str(folder) + "/")
  render = time.perf_counter() - t0
  body = (folder / "f.c").read_text()
  return {"fit_s": fit, "render_s": render, "c_bytes": len(body), "compile_s": compile_seconds(folder / "f.c", folder / "lib.so")}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--only")
  args = parser.parse_args()
  rows = []
  print("| table | side | fit s | render s | C MB | compile s |")
  print("| --- | --- | --- | --- | --- | --- |")
  for name, spec in tables().items():
    if args.only and args.only != name:
      continue
    for side, run in (("scaly", scaly_side), ("casadi", casadi_side)):
      if side == "casadi" and name.endswith("_basis") and "2d" in name:
        continue  # the same CasADi table as the pp row
      r = run(name, spec)
      label = f"{side} ({r['strategy']})" if "strategy" in r else side
      print(f"| {name} | {label} | {r['fit_s']:.2f} | {r['render_s']:.2f} | {r['c_bytes'] / 1e6:.1f} | {r['compile_s']:.1f} |", flush=True)
      rows.append({"table": name, "side": side, **r})
  (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
