"""Count materialized dense support after scalar expansion and inspect solver oracle paths."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scaly as sc
from scaly.codegen import aot
from scaly.ir.program import ProgramOp, walk_program
from study import Observer, benchmark_functions, program_counts, ROOT


class Census(Observer):
  def __init__(self):
    super().__init__()
    self.allocations = {}

  def allocate(self, ctx, expr):
    buffer = super().allocate(ctx, expr)
    if expr.type.pattern is None and expr.support is not None and len(expr.support) < expr.size:
      key = (ctx.fun.name, buffer.attrs["name"])
      self.allocations[key] = {
        "function": ctx.fun.name,
        "buffer": buffer.attrs["name"],
        "op": expr.op.value,
        "shape": expr.shape,
        "entries": expr.size,
        "support": len(expr.support),
        "scalarize": ctx.scalarize,
      }
    return buffer


def inspect(label, fun):
  stages = {}
  constants = []
  original = aot.lower_function

  def normalized(stage, function):
    for node in function.nodes:
      if node.op.value == "const" and node.type.pattern is None and node.support is not None and len(node.support) < node.size:
        constants.append({"function": function.name, "shape": node.shape, "entries": node.size, "support": len(node.support)})

  def lower(*args, **kwargs):
    kwargs["observe"] = lambda stage, program: stages.__setitem__(stage, program) if stage == "pass:fold_arith_after_unroll" else None
    kwargs["observe_expr"] = normalized
    return original(*args, **kwargs)

  with Census() as census, patch.object(aot, "lower_function", lower):
    module = aot.render_c_module(fun, lanes=1, dialect="c", cpu="generic")
  prepack = stages["pass:fold_arith_after_unroll"]
  counts = program_counts(prepack, stores=True)
  if "count_error" in counts:
    raise RuntimeError(counts["count_error"])
  buffers = {
    f"{proc.attrs['name']}:{node.attrs['name']}": node
    for proc in prepack.args[: prepack.attrs["proc_count"]]
    for node in walk_program(proc)
    if node.op == ProgramOp.BUFFER
  }
  for row in census.allocations.values():
    key = f"{row['function']}:{row['buffer']}"
    row["writes"] = counts["executed"].get("writes:" + key, 0)
    row["computed_writes"] = counts["executed"].get("computed:" + key, 0)
    row["remaining_buffer"] = key in buffers
  return {
    "name": label,
    "allocations": list(census.allocations.values()),
    "batches": census.batches,
    "constants": constants,
    "program": program_counts(module.program),
    "source_bytes": len(module.body.encode()),
  }


def special_functions():
  from benchmarks.problems.chain import chain_ocp

  solver = sc.solver(chain_ocp(3, 4).stage_problem(), "fatrop")
  descriptor = solver.function.instantiate().descriptor
  stage = descriptor.stage_oracles
  for name, fn in (("base", descriptor.base), ("jac", stage.jac), ("hess", stage.hess)):
    yield "chain_ocp_M3_N4_fatrop/" + name, fn

  path = ROOT / "examples/sparse_constant_mpc.py"
  tree = ast.parse(path.read_text())
  tree.body = tree.body[: next(i for i, node in enumerate(tree.body) if isinstance(node, ast.For))]
  namespace = {}
  exec(compile(tree, str(path), "exec"), namespace)
  yield "example_piqp_sparse/oracle", namespace["solve"].function.instantiate().descriptor.oracle
  descriptor = sc.solver(namespace["tracking_mpc"], "piqp", name="tracking_mpc_dense", options={"sparse": False}).function.instantiate().descriptor
  yield "example_piqp_dense/oracle", descriptor.oracle

  p = sc.SparsityPattern((2, 3), (0, 1), (0, 2))
  q = sc.SparsityPattern((2, 3), (0, 1), (1, 0))
  yield (
    "test_sparse_select/stack",
    sc.function(sc.arg("v", 2), name="heterogeneous_stack")(lambda v: sc.stack((sc.pack(v, p).sqrt(), sc.pack(sc.const(np.ones(2)), q))).block()),
  )
  dense = np.zeros((2, 3, 4))
  dense[0, 0, 3], dense[1, 2, 2] = 2.0, 4.0
  yield "test_sparse_boundary/sparsify", sc.function(name="heterogeneous_sparsify")(lambda: sc.sparsify(sc.const(dense)).block())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--extras-only", action="store_true")
  args = parser.parse_args()
  results = json.loads(Path(__file__).with_name("census.json").read_text()) if args.extras_only else []
  if not args.extras_only:
    for family in ("chain", "race_cars", "npmpc", "unbumpercars"):
      for label, fun, _ in benchmark_functions(family):
        print(label, flush=True)
        results.append(inspect(label, fun))
        Path(__file__).with_name("census.json").write_text(json.dumps(results, indent=2) + "\n")
  for label, fun in special_functions():
    print(label, flush=True)
    results.append(inspect(label, fun))
  Path(__file__).with_name("census.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
  main()
