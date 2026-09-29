"""The public surface of ``scaly.geometry``: its names, and that ``sc.geometry`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_geometry_surface() -> None:
  geometry = importlib.import_module("scaly.geometry")
  assert sc.geometry is geometry
  assert geometry.__all__ == ["SO3", "Euclidean", "Pose3", "cross", "quaternion", "skew"]
  assert geometry.quaternion.__all__ == ["EPS", "conj", "exp", "log", "matrix", "mul", "rotate"]
  assert "geometry" not in sc.__all__
  for name in geometry.__all__:
    assert getattr(geometry, name).__doc__, name
