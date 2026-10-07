"""Generated names stay content-derived: no module in ``src/scaly`` keeps a counter across calls.

A module-level counter makes a name depend on what the process built before, so the same
Function would generate different C, and a different cache key, depending on what ran first.
The check reads the source with ``ast`` and refuses an ``itertools.count`` object and an integer
global that a function rebinds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "scaly"


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
