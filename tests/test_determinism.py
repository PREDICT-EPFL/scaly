"""Generated C stays independent of hash seeds and process-global counters.

Subprocesses compare generated source and headers across different ``PYTHONHASHSEED`` values.

A module-level counter makes a name depend on what the process built before, so the same
Function would generate different C, and a different cache key, depending on what ran first.
The check reads the source with ``ast`` and refuses an ``itertools.count`` object and an integer
global that a function rebinds.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scaly as sc
from scaly.codegen import render_c_api_header, render_c_source
from scaly.function.concrete import ConcreteFunction
from scaly.function.sugar import _mapped_call
from tests.test_c_snapshot import CORPUS

SRC = Path(__file__).resolve().parents[1] / "src" / "scaly"


def _mapped_stage() -> ConcreteFunction:
  @sc.function(sc.group(sc.arg("x"), sc.arg("p", ())), outputs=sc.group(sc.arg("y"), sc.group(sc.arg("cost", ()))))
  def stage(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, tuple[sc.Expr]]:
    x, p = inputs
    return x * p, ((x * x).sum(),)

  return sc.vmap(stage, 3).instantiate(((3, 2), (3,)))


def _nested_call_map(layout: str, nseed: int) -> sc.Function:
  @sc.function(sc.arg("nested_map_x", 2), outputs=sc.arg("y"), name="nested_map_leaf")
  def leaf(x):
    return sc.stack([x[0] * x[1], x[1].sin()])

  @sc.function(sc.arg("nested_map_z", 6), outputs=sc.arg("y"), name="nested_call_of_map")
  def call_of_map(z):
    return _mapped_call(leaf, 3, [(z, 0, 2)]) * z

  @sc.function(sc.arg("nested_map_w", 2), outputs=sc.arg("y"), name="nested_stage_with_call")
  def stage_with_call(w):
    return leaf(w) + w.exp()

  @sc.function(
    sc.group(sc.arg("outer_map_z", 6), sc.arg("outer_map_seeds", (nseed, 6)), sc.arg("outer_map_cot", 6)),
    outputs=sc.group(sc.arg("value"), sc.arg("single"), sc.arg("many"), sc.arg("adj")),
  )
  def products(inputs):
    z, seeds, cot = inputs
    y = call_of_map(z) if layout == "call_of_map" else _mapped_call(stage_with_call, 3, [(z, 0, 2)])
    return y, sc.stack([sc.jvp(y, z, seeds[i]) for i in range(nseed)]), sc.jvp_many(y, z, seeds), sc.vjp((y,), (z,), (cot,))[0]

  return products


def _render_corpus() -> dict[str, tuple[str, str]]:
  functions: dict[str, sc.Function | ConcreteFunction] = {name: build() for name, build in CORPUS.items()}
  functions["mapped_stage"] = _mapped_stage()
  for layout in ("call_of_map", "map_of_call"):
    for nseed in (1, 3):
      functions[f"{layout}_{nseed}"] = _nested_call_map(layout, nseed)
  return {name: (render_c_source(fun), render_c_api_header(fun)) for name, fun in functions.items()}


def test_generated_c_is_independent_of_hash_seed() -> None:
  expected = None
  for seed in (0, 1, 4, 5, 7, 42):
    result = subprocess.run(
      [sys.executable, "-c", "import json; from tests.test_determinism import _render_corpus; print(json.dumps(_render_corpus()))"],
      env={**os.environ, "PYTHONHASHSEED": str(seed)},
      cwd=SRC.parents[1],
      capture_output=True,
      check=True,
      timeout=120,
    )
    rendered = json.loads(result.stdout)
    if expected is None:
      expected = rendered
    for name, artifacts in rendered.items():
      assert artifacts == expected[name], f"{name} changed under PYTHONHASHSEED={seed}"


def module_counters(source: str) -> list[str]:
  tree = ast.parse(source)
  found = []
  int_globals = {
    t.id
    for stmt in tree.body
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Constant) and type(stmt.value.value) is int
    for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
    if isinstance(t, ast.Name)
  }
  itertools_names = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if a.name == "itertools"}
  count_names = {
    a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "itertools" for a in n.names if a.name == "count"
  }
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      fn = node.func
      if (isinstance(fn, ast.Attribute) and fn.attr == "count" and isinstance(fn.value, ast.Name) and fn.value.id in itertools_names) or (
        isinstance(fn, ast.Name) and fn.id in count_names
      ):
        found.append(f"line {node.lineno}: itertools.count")
    if isinstance(node, ast.Global):
      found += [f"line {node.lineno}: global {name}" for name in node.names if name in int_globals]
  return found


def test_no_module_level_counters() -> None:
  found = [f"{path.relative_to(SRC)} {hit}" for path in sorted(SRC.rglob("*.py")) for hit in module_counters(path.read_text())]
  assert found == []


@pytest.mark.parametrize(
  "source",
  [
    "import itertools\n_ids = itertools.count()\n",
    "from itertools import count\n_ids = count()\n",
    "import itertools as it\n_ids = it.count()\n",
    "from itertools import count as fresh\n_ids = fresh()\n",
    "_n = 0\ndef fresh():\n  global _n\n  _n += 1\n  return f't{_n}'\n",
  ],
)
def test_the_check_finds_a_counter(source: str) -> None:
  assert module_counters(source)
