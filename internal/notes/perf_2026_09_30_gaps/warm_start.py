"""C-225 (Tier 7): what a second process pays before its first result, with the JIT's library found
from the rendered C (``SCALY_JIT_KEY=source``, the only way before) and from the graph (``structure``).

Each corpus kernel is built and called once in a fresh process per mode, after a first process has
built its library: the time to build the graph, the time of the first call (the key or the render,
then loading the library, then the call), and the structural key's own time. The benchmark problems'
builders compile their oracle while they build it, so their render is in the build's time, and the
table compares build and first call together.

  uv run internal/notes/perf_2026_09_30_gaps/warm_start.py [--kernels a,b] [--repeats 3] [--out results/warm_start.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT = "rosen_hess,chol_solve_200,sparse_ldl_mpc50,riccati_50,race_cars_40,race_cars_200,chain_5,chain_9,npmpc_12,ipm_hs118_dense,ipm_qafiro_sparse,ipm_cvxqp1_dense,ipm_mpc_12_4_20_sparse"


def child(name: str) -> None:
  start = time.perf_counter()
  sys.path.insert(0, str(ROOT))
  sys.path.insert(0, str(HERE.parent / "perf_2026_09_30_codegen"))
  import corpus  # the codegen corpus's builders

  imported = time.perf_counter()
  fn, args = corpus.KERNELS[name]()
  fn = fn.concrete
  args = [value.reshape(shape) for value, shape in zip(args, fn.input_shapes, strict=True)]
  built = time.perf_counter()
  fn._flat_numerical_call(*args)
  called = time.perf_counter()
  from scaly.codegen.structure import code_digest, graph_digest

  found = graph_digest(fn)
  walked = time.perf_counter()
  code = code_digest(found[1]) if found is not None else None
  scanned = time.perf_counter()
  print(
    json.dumps(
      dict(
        imports=imported - start,
        build=built - imported,
        first_call=called - built,
        graph_digest=walked - called,
        code_digest=scanned - walked,
        keyed=found is not None and code is not None,
      )
    )
  )


def run(name: str, mode: str) -> dict:
  env = {**os.environ, "SCALY_JIT_KEY": mode}
  out = subprocess.run([sys.executable, __file__, "--child", name], env=env, capture_output=True, text=True, check=True)
  return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--child")
  parser.add_argument("--kernels", default=DEFAULT)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  if args.child:
    child(args.child)
    return
  rows = {}
  heads = f"{'kernel':26s} {'source':>9s} {'structure':>9s} {'ratio':>6s} {'build':>8s} {'call':>7s} {'graph key':>9s} {'code key':>9s}"
  print(f"{heads}   seconds to the first result (build and first call), best of {args.repeats}; build, call and keys are the structure run's")
  for name in args.kernels.split(","):
    run(name, "structure")  # the library and its index entry exist from here on
    best: dict[str, dict] = {}
    for _ in range(args.repeats):
      for mode in ("source", "structure"):
        row = run(name, mode)
        row["total"] = row["build"] + row["first_call"]
        if mode not in best or row["total"] < best[mode]["total"]:
          best[mode] = row
    source, structure = best["source"], best["structure"]
    rows[name] = best
    print(
      f"{name:26s} {source['total']:9.3f} {structure['total']:9.3f} {structure['total'] / source['total']:6.3f} {structure['build']:8.3f} "
      f"{structure['first_call']:7.3f} {structure['graph_digest']:9.4f} {structure['code_digest']:9.4f}{'' if structure['keyed'] else '   no key'}",
      flush=True,
    )
  if args.out:
    args.out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
  main()
