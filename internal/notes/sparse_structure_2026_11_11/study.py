"""Inspect main's sparse workloads without executing or timing generated functions."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.codegen.aot import render_c_module
from scaly.function.model import as_concrete
from scaly.ir.program import ProgramOp as O, walk_program
from scaly.ir.support import support_indices
from scaly.passes.affine import affine_index_map
from scaly.passes.lowering.ctx import LowerCtx
from scaly.passes.lowering import plans
from structure import inspect_pattern, self_check

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
ARITH = {
  O.ADD,
  O.SUB,
  O.MUL,
  O.DIV,
  O.NEG,
  O.SIN,
  O.COS,
  O.TAN,
  O.EXP,
  O.LOG,
  O.SQRT,
  O.POW,
  O.ABS,
  O.MINIMUM,
  O.MAXIMUM,
  O.TANH,
  O.SINH,
  O.COSH,
  O.ASIN,
  O.ACOS,
  O.ATAN,
  O.ATAN2,
  O.ERF,
  O.FLOOR,
  O.CEIL,
}


class Observer:
  def __init__(self):
    self.maps = Counter()
    self.indices = Counter()
    self.dense = []
    self.batches = []
    self.seen = set()

  def regions(self, writes, reads):
    result = self.original_regions(writes, reads)
    for region in result:
      arrays = [region.writes, *(r for r in region.reads if len(r) and r[0] >= 0)]
      if len(region.writes) <= 4:
        kind = "straight"
      elif any(len(affine_index_map(a).residual) > 1 for a in arrays):
        kind = "table"
      elif all(len(a) < 2 or np.all(np.diff(a) == 1) for a in arrays):
        kind = "dense"
      else:
        kind = "affine"
      self.maps[kind] += len(region.writes)
      self.maps[kind + "_regions"] += 1
    return result

  def index(self, ctx, indices, k, **kwargs):
    if k.op != O.CONST_INT and len(indices):
      amap = affine_index_map(indices)
      kind = "affine" if len(amap.residual) == 1 else "table"
      self.indices[kind + "_positions"] += len(indices)
      self.indices[kind + "_requests"] += 1
      self.indices["residual_entries"] += len(amap.residual) if kind == "table" else 0
    return self.original_index(ctx, indices, k, **kwargs)

  def allocate(self, ctx, expr):
    key = (ctx.fun.name, expr.id)
    if key not in self.seen:
      self.seen.add(key)
      support = expr.support
      if expr.type.pattern is None and support is not None and len(support) < expr.size:
        self.dense.append(
          {
            "function": ctx.fun.name,
            "op": expr.op.value,
            "shape": expr.shape,
            "entries": expr.size,
            "support": len(support),
            "scalarize": ctx.scalarize,
          }
        )
      if expr.type.pattern is not None and len(expr.shape) > 2:
        width = int(np.prod(expr.shape[-2:]))
        batch = int(np.prod(expr.shape[:-2]))
        different = False
        if support is not None and batch > 1:
          entries = support_indices(support)
          first = entries[entries < width]
          different = any(not np.array_equal(first, entries[(entries // width) == b] % width) for b in range(1, batch))
        self.batches.append(
          {
            "function": ctx.fun.name,
            "op": expr.op.value,
            "shape": expr.shape,
            "stored": expr.type.storage_size,
            "support": None if support is None else len(support),
            "different_supports": different,
          }
        )
    return self.original_allocate(ctx, expr)

  def __enter__(self):
    self.original_regions, self.original_index, self.original_allocate = plans.map_regions, LowerCtx.index_at, LowerCtx.alloc_tmp
    self.patches = [
      patch.object(plans, "map_regions", self.regions),
      patch.object(LowerCtx, "index_at", lambda ctx, a, k, **kw: self.index(ctx, a, k, **kw)),
      patch.object(LowerCtx, "alloc_tmp", lambda ctx, e: self.allocate(ctx, e)),
    ]
    for p in self.patches:
      p.start()
    return self

  def __exit__(self, *args):
    for p in reversed(self.patches):
      p.stop()


def program_counts(program, *, stores=False, coverage=False):
  procs = {p.attrs["name"]: p for p in program.args[: program.attrs["proc_count"]]}
  loops = [n for n in walk_program(program) if n.op == O.FOR]
  loaded = lambda r: any(n.op == O.LOAD for arg in r.args for n in walk_program(arg))
  root = next(reversed(procs))
  root_proc = procs[root]
  output_masks = {
    buf.attrs["name"]: np.zeros(int(np.prod(buf.attrs["shape"])), dtype=np.int8)
    for buf in root_proc.args[root_proc.attrs["input_count"] : root_proc.attrs["param_count"]]
  }

  @lru_cache(None)
  def expression(node):
    if node.op == O.BUFFER:
      return Counter()
    counts = Counter()
    if node.dtype.is_floating and node.op in ARITH:
      counts[node.op.value] += 1
    if node.op == O.LOAD and node.args[0].attrs.get("index_table"):
      counts["index_loads"] += 1
    for arg in node.args:
      counts.update(expression(arg))
    return counts

  def evaluate(node, env, buffers):
    if node.op == O.CONST_INT:
      return node.attrs["value"]
    if node.op == O.VAR:
      return env[node.attrs["name"]]
    if node.op == O.LOAD:
      view = node.args[0]
      return np.asarray(buffers[view.attrs["buffer"]].attrs["values"], dtype=np.int64)[evaluate(view.args[0], env, buffers)]
    args = [evaluate(a, env, buffers) for a in node.args]
    rules = {
      O.ADD: np.add,
      O.SUB: np.subtract,
      O.MUL: np.multiply,
      O.DIV: np.floor_divide,
      O.MOD: np.remainder,
      O.FLOORDIV: np.floor_divide,
      O.FLOORMOD: np.remainder,
      O.NEG: np.negative,
      O.MINIMUM: np.minimum,
      O.MAXIMUM: np.maximum,
    }
    return rules[node.op](*args)

  def statements(body, env, size, buffers, proc_name, depth=0, table_bound=False):
    total = Counter()
    for node in body:
      if node.op == O.BUFFER:
        continue
      if node.op == O.FOR:
        rng = node.args[0]
        lo, hi, step = [np.broadcast_to(evaluate(x, env, buffers), (size,)) for x in rng.args]
        lengths = np.maximum(0, (hi - lo + step - 1) // step).astype(np.int64)
        count = int(lengths.sum())
        if count > 10_000_000:
          raise ValueError("static iteration expansion exceeds ten million coordinates")
        take = np.repeat(np.arange(size), lengths)
        expanded = {k: np.broadcast_to(v, (size,))[take] for k, v in env.items()}
        offsets = np.arange(count) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        expanded[rng.attrs["name"]] = lo[take] + step[take] * offsets
        if loaded(rng):
          total["loaded_loop_invocations"] += size
          total["loaded_loop_iterations"] += count
        total.update(statements(node.args[1:], expanded, count, buffers, proc_name, depth + 1, table_bound or loaded(rng)))
      elif node.op == O.CALL:
        callee = node.attrs["callee"]
        child = procedure(callee)
        total.update({k: v * size for k, v in child.items()})
        if stores:
          params = procs[callee].args[: procs[callee].attrs["param_count"]]
          for formal, actual in zip(params, node.args, strict=True):
            name = actual.attrs.get("buffer", actual.attrs.get("name"))
            for prefix in ("writes", "computed"):
              total[f"{prefix}:{proc_name}:{name}"] += size * child[f"{prefix}:{callee}:{formal.attrs['name']}"]
      elif node.op == O.BLOCK:
        total.update(statements(node.args, env, size, buffers, proc_name, depth, table_bound))
      elif node.op in (O.IF, O.BREAK_IF):
        raise ValueError("data-dependent control flow has no exact static operation count")
      else:
        total.update({k: v * size for k, v in expression(node).items()})
        if coverage and proc_name == root and node.op in (O.STORE, O.STORE_PAIR) and node.args[0].attrs["buffer"] in output_masks:
          mask = output_masks[node.args[0].attrs["buffer"]]
          slots = np.broadcast_to(evaluate(node.args[0].args[0], env, buffers), (size,))
          views = [n for n in walk_program(node) if n.op == O.VIEW]
          if depth == 0:
            form = 1
          elif table_bound or any(v.attrs.get("index_table") for v in views):
            form = 4
          else:
            indices = [np.broadcast_to(evaluate(v.args[0], env, buffers), (size,)) for v in views]
            form = 2 if all(np.all(np.isin(np.diff(i), (0, 1))) for i in indices) else 3
          for offset in range(len(node.args) - 1):
            mask[slots + offset] = np.maximum(mask[slots + offset], form)
        if stores and node.op in (O.STORE, O.STORE_PAIR) and node.dtype.is_floating:
          target = f"{proc_name}:{node.args[0].attrs['buffer']}"
          for value in node.args[1:]:
            total[f"writes:{target}"] += size
            if value.op != O.CONST_FLOAT or value.attrs["value"] != 0:
              total[f"computed:{target}"] += size
        if node.op == O.ASSIGN and node.args[0].dtype.is_integer:
          env[node.attrs["target"]] = evaluate(node.args[0], env, buffers)
    return total

  @lru_cache(None)
  def procedure(name):
    proc = procs[name]
    buffers = {n.attrs["name"]: n for n in walk_program(proc) if n.op == O.BUFFER}
    return statements(proc.args[proc.attrs["param_count"] :], {}, 1, buffers, name)

  result = {"loop_sites": len(loops), "loaded_loop_sites": sum(loaded(n.args[0]) for n in loops)}
  try:
    result["executed"] = dict(procedure(root))
  except (KeyError, ValueError, TypeError) as error:
    result["count_error"] = str(error)
  tables = [n for n in walk_program(program) if n.op == O.BUFFER and n.attrs.get("index_table")]
  result["table_entries"] = sum(int(np.prod(n.attrs["shape"])) for n in tables)
  result["table_arrays"] = len(tables)
  if coverage:
    result["output_coverage"] = {
      name: dict(zip(("unknown", "straight", "dense", "affine", "table"), map(int, np.bincount(mask, minlength=5)), strict=True))
      for name, mask in output_masks.items()
    }
  return result


def measure(label, fun, pattern, compile_c):
  with Observer() as obs:
    module = render_c_module(fun, lanes=1, dialect="c", cpu="generic")
  result = {
    "name": label,
    "function": as_concrete(fun).name,
    "pattern": inspect_pattern(pattern.shape, pattern.rows, pattern.cols),
    "maps": dict(obs.maps),
    "indices": dict(obs.indices),
    "dense": obs.dense,
    "batches": obs.batches,
    "program": program_counts(module.program),
    "source_bytes": len(module.body.encode()),
    "workspace_entries": module.workspace_size,
  }
  if compile_c:
    with tempfile.TemporaryDirectory(prefix="scaly-91-") as tmp:
      path = Path(tmp) / "kernel.c"
      path.write_text(module.body)
      command = ["cc", "-O3", "-fno-math-errno", "-std=c99", "-c", str(path), "-o", str(path.with_suffix(".o"))]
      start = time.perf_counter()
      completed = subprocess.run(command, capture_output=True, text=True)
      result["compile_seconds"] = round(time.perf_counter() - start, 6)
      result["compile_returncode"] = completed.returncode
      if completed.returncode:
        result["compile_error"] = completed.stderr[-2000:]
  return result


def kkt_functions(name):
  spec = importlib.util.spec_from_file_location("kkt_workload", ROOT / "tests/integration/test_sparse_kkt.py")
  test = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(test)
  P, A, G = test._problem(name)
  n, p, m = P.shape[0], A.shape[0], G.shape[0]
  upper = sparse.triu(P, format="csc")
  sizes = [upper.nnz, A.nnz, G.nnz, n, m, 1]
  offsets = np.cumsum([0, *sizes])
  rows, cols = map(np.asarray, test._coordinates(upper))
  off = np.flatnonzero(rows != cols)
  mirrored = sc.SparsityPattern(P.shape, tuple(map(int, np.r_[rows, cols[off]])), tuple(map(int, np.r_[cols, rows[off]])))
  sources = np.r_[np.arange(upper.nnz), off]

  def assemble(v, triangular):
    parts = [v[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    pv = sc.pack(sc.gather(parts[0], sources), mirrored)
    av, gv = test._leaf(A, parts[1]), test._leaf(G, parts[2])
    top = (test._leaf(upper, parts[0]) if triangular else pv) + sc.pack(parts[3], sc.SparsityPattern((n, n), tuple(range(n)), tuple(range(n))))
    bottom = sc.pack(-parts[4], sc.SparsityPattern((m, m), tuple(range(m)), tuple(range(m))))
    zero = sc.const(sparse.csc_array((p, m)))
    return sc.concat(
      (
        sc.concat((top, av.T, gv.T), axis=1),
        sc.concat((sc.const(sparse.csc_array((p, n))) if triangular else av, -parts[5][0] * sc.const(sparse.eye_array(p)), zero), axis=1),
        sc.concat((sc.const(sparse.csc_array((m, n))) if triangular else gv, zero.T, bottom), axis=1),
      ),
      axis=0,
    )

  for triangular in (False, True):
    label = "upper" if triangular else "kkt"
    fun = sc.function(sc.arg("v", int(offsets[-1])), name=f"{label}_{name}")(lambda v: assemble(v, triangular).block())
    yield f"{name}/{label}", fun, as_concrete(fun).outputs[0].type.pattern
  for label, matrix in (("P", P), ("A", A), ("G", G)):
    pattern = sc.SparsityPattern(matrix.shape, *test._coordinates(matrix)).canonicalize()[0]
    decl = sc.arg("a", sc.TensorType(matrix.shape, pattern=pattern), layout="csc")
    fun = sc.function(decl, sc.arg("x", matrix.shape[1]), name=f"mv_{label}_{name}")(lambda a, x: (a @ x).block())
    yield f"{name}/{label}x", fun, pattern
    if label != "P":
      fun = sc.function(decl, sc.arg("y", matrix.shape[0]), name=f"mtv_{label}_{name}")(lambda a, y: (a.T @ y).block())
      yield f"{name}/{label}Ty", fun, pattern
      diagonal = sc.SparsityPattern((matrix.shape[0], matrix.shape[0]), tuple(range(matrix.shape[0])), tuple(range(matrix.shape[0])))
      fun = sc.function(decl, sc.arg("w", matrix.shape[0]), name=f"gram_{label}_{name}")(lambda a, w: (a.T @ sc.pack(w, diagonal) @ a).block())
      yield f"{name}/{label}gram", fun, as_concrete(fun).outputs[0].type.pattern


def benchmark_functions(family):
  if family == "chain":
    from benchmarks.problems.chain import chain_nlp, HORIZON

    solver = chain_nlp(3, HORIZON)
    label = f"chain_M3_N{HORIZON}"
  elif family == "race_cars":
    from benchmarks.problems.race_cars.closed_loop import EpisodeConfig, _race_car_nlp

    solver = _race_car_nlp(EpisodeConfig(horizon=25))
    label = "race_cars_N25"
  elif family == "npmpc":
    from benchmarks.problems.npmpc import npmpc_nlp

    solver = npmpc_nlp(12)
    label = "npmpc_N12"
  elif family == "unbumpercars":
    from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig
    from benchmarks.problems.unbumpercars.filters import build_scaly_nlp

    solver = build_scaly_nlp(ClosedLoopConfig(ncars=8), FilterConfig(model="dt"))
    label = "unbumpercars_C8"
  else:
    raise ValueError(family)
  descriptor = solver.function.instantiate().descriptor
  for kind in ("jac", "hess"):
    yield f"{label}/{kind}", getattr(descriptor, kind), getattr(descriptor, kind + "_sparsity")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("group", choices=("kkt", "bench", "self-check"))
  parser.add_argument("--select", default="")
  parser.add_argument("--compile", action="store_true")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  self_check()
  if args.group == "self-check":
    print("NumPy structure checks passed")
    return
  output = args.output or HERE / (args.group + ".jsonl")
  if args.group == "kkt":
    names = sorted(p.stem for p in (ROOT / "tests/integration/data/maros_meszaros").glob("*.npz") if not p.stem.endswith("_patterns"))
    source = ((label, fn, pattern) for name in names if not args.select or name == args.select for label, fn, pattern in kkt_functions(name))
  else:
    source = (
      (label, fn, pattern)
      for family in ("chain", "race_cars", "npmpc", "unbumpercars")
      if not args.select or family == args.select
      for label, fn, pattern in benchmark_functions(family)
    )
  with output.open("w") as stream:
    for label, fun, pattern in source:
      print(label, flush=True)
      try:
        result = measure(label, fun, pattern, args.compile)
      except Exception as error:
        result = {"name": label, "error": f"{type(error).__name__}: {error}"}
        print(result, flush=True)
      stream.write(json.dumps(result) + "\n")
      stream.flush()


if __name__ == "__main__":
  main()
