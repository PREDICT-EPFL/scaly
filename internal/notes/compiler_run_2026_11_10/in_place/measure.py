"""Count and compile a four-entry update of a 100k loop carry on la015."""

from collections import Counter
from pathlib import Path
import json
import subprocess
import time
from unittest.mock import patch

import scaly as sc
from scaly.codegen import render_c_module
from scaly.ir.program import ProgramOp
from scaly.passes.lowering import lower_function


@sc.function(sc.arg("carry", 100_000), name="update_four")
def update(carry):
  return sc.scatter(carry, [11, 29, 47, 83], sc.gather(carry, [11, 29, 47, 83]) + 1, combine="set")


@sc.function(sc.arg("initial", 100_000), name="update_loop")
def loop(initial):
  result, _ = sc.scan(update, initial, length=100)
  return result


def counts(node, factor=1):
  result = Counter()
  if node.op == ProgramOp.FOR:
    start, stop, step = (bound.attrs["value"] for bound in node.args[0].args)
    for child in node.args[1:]:
      result.update(counts(child, factor * len(range(start, stop, step))))
    return result
  if node.op in (ProgramOp.LOAD, ProgramOp.STORE, ProgramOp.ADD):
    result[node.op.value] += factor
  elif node.op == ProgramOp.STORE_PAIR:
    result["store"] += 2 * factor
  for child in node.args:
    result.update(counts(child, factor))
  return result


def measure(label):
  program = lower_function(loop)
  module = render_c_module(loop)
  source = module.source
  path = Path(__file__).parent / f"{label}.c"
  path.write_text(source)
  (path.parent / module.header_name).write_text(module.header)
  t0 = time.perf_counter()
  subprocess.run(["cc", "-O3", "-c", str(path), "-o", str(path.with_suffix(".o"))], check=True)
  elapsed = time.perf_counter() - t0
  path.with_suffix(".o").unlink()
  body = next(proc for proc in program.args if proc.attrs["name"].startswith("update_four"))
  body_counts = Counter()
  for stmt in body.args[body.attrs["param_count"]:]:
    body_counts.update(counts(stmt))
  return {"source_bytes": len(source.encode()), "compile_seconds": elapsed, "step_operations": dict(body_counts),
          "workspace_elements": program.args[-1].attrs["sz_w"],
          "parameters": [{"name": param.attrs["name"], "shape": param.attrs["shape"], "intent": param.attrs.get("intent", "input" if i < body.attrs["input_count"] else "output")}
                         for i, param in enumerate(body.args[:body.attrs["param_count"]])],
          "carry_slots": 1 if body.attrs.get("inout_carries") else 2,
          "carry_buffer_bytes": 100_000 * 8 * (1 if body.attrs.get("inout_carries") else 2)}


if __name__ == "__main__":
  with patch("scaly.passes.lowering.calls.carry_reuse", return_value={}):
    baseline = measure("alternating")
  reused = measure("reused")
  result = {"carry_entries": 100_000, "updated_entries_per_step": 4, "steps": 100,
            "baseline": baseline, "reused": reused}
  report = json.dumps(result, indent=2) + "\n"
  (Path(__file__).parent / "measurement.json").write_text(report)
  print(report)
