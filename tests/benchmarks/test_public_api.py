"""Keep benchmark formulations on the exported modelling APIs."""

import ast
import importlib
from pathlib import Path

import pytest

import scaly as sc

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = {
  "scaly": set(sc.__all__),
  **{f"scaly.{name}": set(importlib.import_module(f"scaly.{name}").__all__) for name in ("codegen", "solvers", "utils")},
  "scaly.factory": set(sc.factory.__all__),
}
# These run experiments or episodes rather than defining their mathematical models.
DRIVERS = {"checks.py", "run_closed_loop.py", "width_study.py"}
FORMULATIONS = [
  path
  for path in sorted((ROOT / "benchmarks/problems").glob("*/*.py"))
  if path.name not in DRIVERS
  and path.relative_to(ROOT).as_posix()
  not in {
    "benchmarks/problems/chain/closed_loop.py",
    "benchmarks/problems/npmpc/closed_loop.py",
  }
]
# PR #111 uses the internal conversion to inspect concrete derivative functions.
KNOWN_EXCEPTIONS = {
  "benchmarks/problems/npmpc/__init__.py": {"scaly.function.model.as_concrete"},
  "benchmarks/problems/unbumpercars/filters.py": {"scaly.function.model.as_concrete"},
}


def private_uses(source: str) -> list[tuple[int, str]]:
  tree = ast.parse(source)
  bindings = {}
  violations = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        if alias.name == "scaly" or alias.name.startswith("scaly."):
          if alias.name not in PUBLIC:
            violations.append((node.lineno, alias.name))
          bindings[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else "scaly"
    elif isinstance(node, ast.ImportFrom) and node.module and (node.module == "scaly" or node.module.startswith("scaly.")):
      for alias in node.names:
        name = f"{node.module}.{alias.name}"
        if alias.name not in PUBLIC.get(node.module, set()) and name not in PUBLIC:
          violations.append((node.lineno, name))
        bindings[alias.asname or alias.name] = name
  for node in ast.walk(tree):
    if not isinstance(node, ast.Attribute):
      continue
    parts = []
    base = node
    while isinstance(base, ast.Attribute):
      parts.insert(0, base.attr)
      base = base.value
    if not isinstance(base, ast.Name) or base.id not in bindings:
      continue
    name = bindings[base.id]
    for part in parts:
      child = f"{name}.{part}"
      if (name in PUBLIC and part not in PUBLIC[name] and child not in PUBLIC) or part.startswith("_"):
        violations.append((node.lineno, child))
        break
      name = child
  return sorted(set(violations))


@pytest.mark.parametrize("path", FORMULATIONS, ids=lambda path: str(path.relative_to(ROOT / "benchmarks/problems")))
def test_formulations_use_public_api(path: Path) -> None:
  relative = path.relative_to(ROOT).as_posix()
  violations = [
    f"{relative}:{line}: {name} is not public" for line, name in private_uses(path.read_text()) if name not in KNOWN_EXCEPTIONS.get(relative, set())
  ]
  assert not violations, "\n".join(violations)
