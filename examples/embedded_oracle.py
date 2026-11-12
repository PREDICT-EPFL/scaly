"""The mapped-derivatives example in float32, exporting values and a sparse Jacobian together."""

from pathlib import Path

import scaly as sc
from scaly.codegen import write_module


@sc.function(sc.arg("z", sc.TensorType((2,), dtype=sc.dtypes.float32)), outputs=sc.arg("residual"))
def stage(z: sc.Expr) -> sc.Expr:
  """The ``.block()`` hint forces a workspace for this check, and omitting it gives an object of 206 bytes with no workspace."""
  return sc.stack([z[0].sin() * z[1], z[0] + z[1] ** 2]).block()


@sc.function(sc.arg("zs", sc.TensorType((8,), dtype=sc.dtypes.float32)), outputs=sc.arg("residuals"))
def stages(zs: sc.Expr) -> sc.Expr:
  return sc.vmap(stage, 4)(zs.reshape((4, 2))).vec()


oracle = stages.factory("embedded_oracle", ["zs"], ["residuals", sc.factory.SpJac("residuals", "zs")])


if __name__ == "__main__":
  module = write_module(oracle, Path(__file__).parent / "generated" / "embedded", dialect="c", lanes=1, frame_budget=0)
  print(f"Caller workspace: {4 * module.workspace_size} bytes")
