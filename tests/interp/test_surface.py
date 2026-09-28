"""The public surface of ``scaly.interp``: its names, and that ``sc.interp`` is the package, loaded on first use."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import scaly as sc


def test_the_interp_surface() -> None:
  interp = importlib.import_module("scaly.interp")
  assert sc.interp is interp
  assert interp.__all__ == ["BOUNDARIES", "KINDS", "PP_BUDGET", "Axis", "BSpline", "Index", "Inverse", "constrained", "interpolant", "smoothing"]
  assert "interp" not in sc.__all__ and not hasattr(sc, "interpolant")


def _attribute_docstring(module_name: str, name: str) -> str | None:
  """The string literal after ``name = ...`` at a module's top level, which the API pages render."""
  body = ast.parse(Path(importlib.import_module(module_name).__file__ or "").read_text()).body
  for node, after in zip(body, body[1:], strict=False):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    if any(isinstance(t, ast.Name) and t.id == name for t in targets):
      return (
        after.value.value if isinstance(after, ast.Expr) and isinstance(after.value, ast.Constant) and isinstance(after.value.value, str) else None
      )
  return None


def test_every_interp_name_is_documented() -> None:
  """The API page is generated from these docstrings: each public name, each public method and
  property of its classes, and each constant (an attribute docstring) carries its own."""
  interp = importlib.import_module("scaly.interp")
  missing = []
  for name in interp.__all__:
    obj = getattr(interp, name)
    if isinstance(obj, type):
      missing += [name] if not obj.__dict__.get("__doc__") else []
      for member, value in vars(obj).items():
        target = value.fget if isinstance(value, property) else value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
        if not member.startswith("_") and callable(target) and not getattr(target, "__doc__", None):
          missing.append(f"{name}.{member}")
    elif callable(obj):
      missing += [name] if not obj.__doc__ else []
    elif not any(_attribute_docstring(f"scaly.interp.{module}", name) for module in ("fit", "spline", "grid", "constrained")):
      missing.append(name)
  assert not missing, f"undocumented: {missing}"
