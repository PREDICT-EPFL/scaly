"""Section 5 demo: one loop per mapped invocation, and a carried range that survives the optimizer.

Run from a checkout with ``uv run internal/notes/compiler_run_2026_11_10/calls_and_carried_loops/demo.py``.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import render_c_api_header, render_c_source
from scaly.codegen.c import _render_raw_callee
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.program_spec import verify_program
from scaly.ir.text import format_program
from scaly.ir.types import dtypes
from scaly.passes.lowering import lower_function
from scaly.passes.program import optimize_program

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRIPS = 5


def mapped(printing: bool) -> sc.Function:
  """The ``vmap_multi`` snapshot's function, with an optional print in the mapped body."""

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("vector"), sc.arg("total")))
  def pair(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    if printing:
      x = sc.print("trip x={}", x)
    return x.sin().block(), x.sum().block()

  @sc.function(sc.arg("xs", (TRIPS, 2)), outputs=sc.group(sc.arg("vectors"), sc.arg("totals")))
  def multi_output(xs: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return sc.vmap(pair, TRIPS)(xs)

  return multi_output


def c_stdout(run) -> str:
  """What ``run`` writes to the process's standard output, generated C included."""
  sys.stdout.flush()
  saved = os.dup(1)
  with tempfile.TemporaryFile() as capture:
    os.dup2(capture.fileno(), 1)
    try:
      run()
      sys.stdout.flush()
    finally:
      os.dup2(saved, 1)
      os.close(saved)
    capture.seek(0)
    return capture.read().decode()


def compile_seconds(fun: sc.Function, compiler: str) -> list[float]:
  samples = []
  with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, "multi_output.h").write_text(render_c_api_header(fun))
    Path(tmp, "multi_output.c").write_text(render_c_source(fun))
    for _ in range(3):
      start = time.perf_counter()
      subprocess.run([compiler, "-O3", "-fno-math-errno", "-c", "multi_output.c", "-o", "multi_output.o"], cwd=tmp, check=True)
      samples.append(time.perf_counter() - start)
  return samples


print("== #28: a mapped Function with two used outputs and a print ==")
fun = mapped(printing=True)
source = render_c_source(fun)
entry = source[source.index("int multi_output(") :]
lines = entry.splitlines()
calls = [i for i, line in enumerate(lines) if "pair_raw(" in line]
print("\n".join("\n".join(lines[i - 1 : i + 2]) for i in calls))
print(f"loops in the entry that call the mapped body: {len(calls)}")
xs = np.arange(2.0 * TRIPS).reshape(TRIPS, 2)
printed = c_stdout(lambda: fun(xs))
print(printed, end="")
print(f"prints: {len(printed.splitlines())} for {TRIPS} trips and two used outputs (before #28: {2 * TRIPS})")

print("\n== #28: the snapshot that changed, tests/baseline/c/vmap_multi.c ==")
print((HERE / "invocations" / "snapshots.diff").read_text(), end="")
snapshot = mapped(printing=False)
assert render_c_source(snapshot) == (ROOT / "tests" / "baseline" / "c" / "vmap_multi.c").read_text()
recorded = json.loads((HERE / "invocations" / "measurements.json").read_text())
before, after = (recorded[when]["rows"]["vmap_multi"] for when in ("before", "after"))
operations = Counter(node.op.value for node in walk_program(lower_function(snapshot)))
compiler = shutil.which("cc")
rows = [
  ("program nodes", *(sum(row["operations"].values()) for row in (before, after)), sum(operations.values())),
  *(
    (label, before["operations"][op], after["operations"][op], operations[op])
    for label, op in (("calls", "call"), ("loops", "for"), ("buffers", "buffer"))
  ),
  ("generated C, bytes", before["source_bytes"], after["source_bytes"], len(render_c_source(snapshot).encode())),
]
if compiler is not None:
  ms = lambda samples: f"{1e3 * statistics.median(samples):.1f}"
  rows.append(("cc -O3 -c, median of 3, ms", ms(before["compile_seconds"]), ms(after["compile_seconds"]), ms(compile_seconds(snapshot, compiler))))
print(f"{'':28}{'before #28':>12}{'after #28':>12}{'this checkout':>15}")
for label, *values in rows:
  print(f"{label:28}{values[0]!s:>12}{values[1]!s:>12}{values[2]!s:>15}")
print(f"The first two columns are recorded in invocations/measurements.json (baseline {recorded['baseline_commit'][:8]}). Counts include the")
print("mapped body's own procedure. The recorded renderings are 27 bytes longer than the snapshot file, for a reason the")
print("record does not give. The diff above is the 142 bytes between them. The compile times come from a shared machine")
print("and show no difference.")

print("\n== #29: a hand-built carried range with BREAK_IF ==")
flags = p.buffer("flags", dtypes.bool_, (6,))
out = p.buffer("out", dtypes.float64, (1,))
count = p.buffer("count", dtypes.int64, (1,))
first = lambda buf: p.view(buf, [p.const_int(0)])
loop = p.for_(
  p.range_("i", 0, 6, carried=True),
  [p.break_if(p.load(p.view(flags, [p.var("i")]))), p.store(first(out), p.add(p.div(p.load(first(out)), p.const_float(2)), p.const_float(1)))],
  exit_var=True,
)
proc = p.proc("recurrence", [flags, out, count], [p.store(first(out), p.const_float(0)), loop, p.store(first(count), p.var("i"))])
# The scalar hint asks scalarize to unroll every constant-bound loop of the procedure.
proc = ProgramNode(proc.op, proc.args, {**proc.attrs, "input_count": 1, "lowering": "scalar", "scalarize_mode": "procedure"}, proc.dtype)
print(format_program(p.program([proc])))
passes = []


def intact(name: str, stage: ProgramNode) -> None:
  verify_program(stage)
  kept = [n for n in walk_program(stage) if n.op == ProgramOp.FOR and n.args[0].attrs.get("carried")]
  assert len(kept) == 1 and kept[0].attrs["exit_var"] is True, name
  assert kept[0].args[0].args[1] == p.const_int(6) and any(n.op == ProgramOp.BREAK_IF for n in walk_program(kept[0])), name
  passes.append(name.removeprefix("pass:"))


optimized = optimize_program(p.program([proc]), observe=intact, lanes=2, reciprocal=True)
print(f"\nthe loop keeps its six trips, exit variable and break after each of {len(passes)} passes (lanes=2, reciprocal=True):")
print(", ".join(passes))
print("\n" + "\n".join(_render_raw_callee(optimized.args[0])))
