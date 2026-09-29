"""The graph view colours a registered op by its traits: one elementwise and expensive like ``SIN`` takes
``SIN``'s colour."""

from __future__ import annotations

import numpy as np

from scaly.ir.expr import ExprOp, register_op
from scaly.ir.program import ProgramOp
from scaly.viz.graph import _expr_color

HALF_SINE = register_op(
  "test_viz_half_sine",
  arity=1,
  numpy=lambda x: np.sin(x),
  jvp=lambda e, d: e.args[0].cos() * d[0],
  traits={"elementwise": ProgramOp.SIN, "expensive": True},
)


def test_an_elementwise_op_takes_the_colour_of_its_program_op() -> None:
  assert _expr_color(HALF_SINE.name) == _expr_color(ExprOp.SIN) != _expr_color(ExprOp.ADD)
