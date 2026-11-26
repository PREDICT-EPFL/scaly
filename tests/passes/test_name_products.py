"""A product of two locals that several statements repeat is written once, outside widened loops."""

import scaly as sc
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.types import dtypes
from scaly.passes.lowering import lower_function
from scaly.passes.program.name_products import name_products

F64 = dtypes.float64


def _block(*statements: ProgramNode) -> ProgramNode:
  out = p.buffer("out", F64, (2,), intent="out")
  x = p.buffer("x", F64, (2,))
  a = p.assign("a", p.load(p.view(x, [p.const_int(0)])), declare=True)
  b = p.assign("b", p.load(p.view(x, [p.const_int(1)])), declare=True)
  return p.program([p.proc("block", [x, out], [a, b, *statements])])


def _store(at: int, value: ProgramNode) -> ProgramNode:
  return p.store(p.view(p.buffer("out", F64, (2,), intent="out"), [p.const_int(at)]), value)


def _products(node: ProgramNode) -> int:
  """Products the C writes, each occurrence counted: interned nodes repeat in the tree."""
  count, stack = 0, [node]
  while stack:
    node = stack.pop()
    count += node.op == ProgramOp.MUL and node.dtype.is_floating
    stack.extend(node.args)
  return count


A, B, C = p.var("a", F64), p.var("b", F64), p.const_float(3.0)


def test_a_product_two_statements_repeat_is_named_before_the_first():
  prog = _block(_store(0, p.sub(C, p.mul(A, B))), _store(1, p.add(p.mul(A, B), A)))
  body = name_products(prog).args[0].args[4:]
  assert [stmt.op for stmt in body] == [ProgramOp.ASSIGN, ProgramOp.STORE, ProgramOp.STORE]
  assert body[0].args[0] == p.mul(A, B)
  assert _products(p.block(*body)) == 1


def test_a_product_a_statement_repeats_is_named():
  prog = _block(_store(0, p.add(p.mul(A, B), p.div(C, p.mul(A, B)))))
  assert _products(p.block(*name_products(prog).args[0].args[4:])) == 1


def test_a_product_used_once_is_left_alone():
  prog = _block(_store(0, p.mul(A, B)), _store(1, p.mul(A, C)))
  assert name_products(prog) == prog


def test_an_operand_assigned_between_two_uses_ends_the_group():
  prog = _block(_store(0, p.mul(A, B)), p.assign("a", C), _store(1, p.mul(A, B)))
  assert name_products(prog) == prog


def test_a_serial_loop_names_its_products_inside_the_loop():
  serial = p.for_(p.range_("i", 0, 2), [_store(0, p.mul(A, B)), _store(1, p.sub(C, p.mul(A, B)))])
  (loop,) = name_products(_block(serial)).args[0].args[4:]
  assert loop.args[1].op == ProgramOp.ASSIGN and _products(loop) == 1


def test_a_name_never_shadows_a_procedure_or_external_function():
  out, x = p.buffer("out", F64, (2,), intent="out"), p.buffer("x", F64, (2,))
  prog = _block(_store(0, p.mul(A, B)), p.call("v0", [x, out]), _store(1, p.sub(C, p.mul(A, B))))
  (named,) = [stmt for stmt in name_products(prog).args[0].args[4:] if stmt.op == ProgramOp.ASSIGN]
  assert named.attrs["target"] != "v0"


def test_the_code_widening_produces_keeps_its_products():
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def stage(x):
    product = x[0] * x[1]
    return sc.stack([product - x[0], product + x[1]])

  @sc.function(sc.arg("xs", 32), outputs=sc.arg("ys"))
  def horizon(xs):
    return sc.vmap(stage, 16)(xs.reshape((16, 2))).ravel()

  # lower_function verifies the program the pass returns.
  prog = lower_function(horizon, lanes=4)
  widened = [node for node in walk_program(prog) if node.op == ProgramOp.FOR and node.args[0].attrs.get("lanes")]
  assert widened and all(_products(loop) == 2 for loop in widened)
