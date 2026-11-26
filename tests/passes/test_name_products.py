"""A product of two locals that several statements repeat is written once, outside widened loops."""

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.types import dtypes
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
    count += node.op == ProgramOp.MUL
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


def test_a_widened_range_keeps_its_products_at_their_uses():
  lane = p.range_("lane", 0, 2, kind=p.RangeKind.VECTOR)
  lane = ProgramNode(lane.op, lane.args, {**lane.attrs, "lanes": 2}, dtype=lane.dtype)
  widened = p.for_(lane, [_store(0, p.mul(A, B)), _store(1, p.sub(C, p.mul(A, B)))])
  serial = p.for_(p.range_("i", 0, 2), [_store(0, p.mul(A, B)), _store(1, p.sub(C, p.mul(A, B)))])
  widened_body = name_products(_block(widened)).args[0].args[4:]
  serial_body = name_products(_block(serial)).args[0].args[4:]
  assert _products(p.block(*widened_body)) == 2
  assert _products(p.block(*serial_body)) == 1
