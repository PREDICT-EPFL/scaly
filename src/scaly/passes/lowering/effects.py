"""Lower prints into statements that read their values in place."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp, print_pieces
from .ctx import LowerCtx, lowers


@lowers(ExprOp.PRINT)
def _lower_print(ctx: LowerCtx, node: Expr) -> None:
  """Print every value, a tensor flat as ``[a, b, ...]``, then let the node share its first value's buffer."""
  pieces = print_pieces(node.attrs["format"])
  text, values = [pieces[0]], []
  for value, after in zip(node.args, pieces[1:], strict=True):
    buf = ctx.buf_of(value)
    elements = [p.load(p.view(buf, [p.const_int(i)])) for i in range(value.size)]
    values += elements
    if not value.shape:
      text.append(after)
    elif not elements:
      text[-1] += "[]" + after
    else:
      text[-1] += "["
      text += [", "] * (len(elements) - 1) + ["]" + after]
  text[-1] += "\n"
  ctx.statements.append(p.print_(text, values))
  ctx.value_buffers[node.id] = ctx.value_buffers[node.args[0].id]
