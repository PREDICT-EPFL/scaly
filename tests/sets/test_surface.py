"""The public surface of ``scaly.sets``: its names, and that ``sc.sets`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_sets_surface() -> None:
  sets = importlib.import_module("scaly.sets")
  assert sc.sets is sets
  assert sets.__all__ == ["Constraint", "Ellipsoid", "Polytope"]
  assert "sets" not in sc.__all__ and not hasattr(sc, "Polytope")
  assert not hasattr(importlib.import_module("scaly.ocp"), "Polytope")
