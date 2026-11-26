"""Count, per kernel, the stored nodes that lowering computes at fewer than all of their entries.

Usage: uv run narrowed.py OUTPUT_JSON, from the repository root with ``PYTHONPATH=src:.``."""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.harness import sweep
from scaly.ir.expr import ExprOp, topo
from scaly.passes.lowering import ctx

CELLS = (("chain", 5), ("race_cars", 5), ("npmpc_jac", 100), ("npmpc", 100), ("chain_jac", 5))
found: Counter = Counter()
emit_body = ctx.LowerCtx.emit_body


def record(self):
  emit_body(self)
  leaves = {leaf.id for leaves in self.sums.values() for leaf in leaves}
  for node in topo(self._computed):
    if node.id in self.domains and node.id not in self.deferred and node.id not in leaves:
      key = ExprOp(node.op).name
      found[key, "nodes"] += 1
      found[key, "entries"] += node.size
      found[key, "computed"] += len(self.domains[node.id])


ctx.LowerCtx.emit_body = record
result = {}
for workload, size in CELLS:
  found.clear()
  with tempfile.TemporaryDirectory() as folder:
    sweep.build_kernel(workload, size, "scaly", Path(folder))
  ops = sorted({op for op, _ in found})
  result[f"{workload}_{size}"] = {op: {field: found[op, field] for field in ("nodes", "entries", "computed")} for op in ops}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
