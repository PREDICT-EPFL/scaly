"""Mapped stage calculations and their block-diagonal sparse Jacobian."""

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

N = 4


@sc.function(sc.arg("z", 2), outputs=sc.arg("residual"))
def stage(z: sc.Expr) -> sc.Expr:
  return sc.stack([z[0].sin() * z[1], z[0] + z[1] ** 2])


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("residuals"))
def stages(zs: sc.Expr) -> sc.Expr:
  return sc.vmap(stage, N)(zs).vec()


jac = sc.sparse_jacobian(stages, "residuals", "zs")
zs = np.arange(1.0, 2 * N + 1) / 10.0
pattern = jac.sparsity()
assert pattern is not None
values = jac(zs)
print(stages(zs))
print("Jacobian shape:", pattern.shape)
print("Stored entries:", pattern.nnz)
print("Row coordinates:", pattern.rows)
print("Column coordinates:", pattern.cols)
print("Values:", values)

generated = Path(__file__).parent / "generated"
write_module(stages, generated)
write_module(jac, generated)
