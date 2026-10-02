"""Compose symbolic functions, evaluate the result, and export its C source."""

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module


@sc.function(sc.arg("x", 3), outputs=sc.arg("square", ...))
def square(x: sc.Expr) -> sc.Expr:
  return x * x


@sc.function(sc.arg("x", 3), outputs=sc.arg("energy", ...))
def energy(x: sc.Expr) -> sc.Expr:
  return square(x).sum()


print(energy(np.array([1.0, 2.0, 3.0])))  # 14.0
print(sc.gradient(energy, "energy", "x")(np.ones(3)))  # [2. 2. 2.]
write_module(energy, Path(__file__).parent / "generated")
