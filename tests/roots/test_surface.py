"""The public surface of ``scaly.roots``: its names, its methods in the registry, and that ``sc.roots`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_roots_surface() -> None:
  roots = importlib.import_module("scaly.roots")
  assert sc.roots is roots
  assert roots.__all__ == [
    "LINEAR",
    "METHOD_API",
    "REGISTRY",
    "GaussNewton",
    "Info",
    "LeastSquares",
    "LevenbergMarquardt",
    "Linear",
    "Newton",
    "NewtonBisection",
    "Residual",
    "Root",
    "RootSpec",
    "custom_root",
    "least_squares",
    "root",
    "solver",
  ]
  assert "roots" not in sc.__all__ and not hasattr(sc, "Newton")
  assert sorted(roots.REGISTRY.installed()) == ["gauss_newton", "levenberg_marquardt", "newton", "newton_bisection"]
  for name, cls in (("newton", roots.Newton), ("newton_bisection", roots.NewtonBisection), ("gauss_newton", roots.GaussNewton)):
    assert roots.REGISTRY.get(name) is cls and cls.name == f"roots.{name}" and cls.api == roots.METHOD_API
