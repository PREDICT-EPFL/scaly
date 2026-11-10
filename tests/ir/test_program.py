"""Phase 4: hand-built Program IR examples — verify and pretty-print.

These tests pin the Program IR contract before lowering (Phase 5) starts to
build real Program IR from expression IR. Each example walks one piece of the
vocabulary:

- elementwise loop with LOAD/STORE through VIEWs;
- negative cases: malformed nodes are rejected with a clear diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, RangeKind
from scaly.ir.text import format_program
from scaly.ir.program_spec import verify_program
from scaly.ir.spec import VerifyError
from scaly.ir.types import dtypes, frozen


def _elementwise_neg_proc() -> ProgramNode:
  """A tiny elementwise proc: ``out[i] = -in[i]`` for ``i in [0, N)``."""
  in_buf = p.buffer("in_", dtypes.float64, (16,), address_space="global")
  out_buf = p.buffer("out_", dtypes.float64, (16,), address_space="global")
  rng = p.range_("i", 0, 16, kind=RangeKind.GLOBAL)
  i = p.var("i", dtype=dtypes.int64)
  body = (
    p.for_(
      rng,
      [
        p.store(
          p.view(out_buf, [i]),
          p.neg(p.load(p.view(in_buf, [i]))),
        )
      ],
    ),
  )
  return p.proc("k_neg", [in_buf, out_buf], body)


def test_elementwise_proc_verifies_and_prints() -> None:
  k = _elementwise_neg_proc()
  verify_program(k)
  dump = format_program(k)
  assert "proc k_neg" in dump
  assert "for i in [0, 16) step 1 kind=global" in dump
  assert "out_[i] <- (-in_[i])" in dump


def test_for_first_arg_must_be_range() -> None:
  # forge a malformed FOR whose first arg is a CONST_INT instead of a RANGE
  bad = ProgramNode(ProgramOp.FOR, (p.const_int(0),), attrs={"body_len": 0}, dtype=dtypes.float64)
  with pytest.raises(VerifyError, match="for-body"):
    verify_program(bad)


def test_view_requires_buffer() -> None:
  with pytest.raises(TypeError, match="view requires a BUFFER"):
    p.view(p.const_int(0), [p.const_int(0)])


def test_buffer_address_space_validated() -> None:
  with pytest.raises(ValueError, match="unknown address space"):
    p.buffer("b", dtypes.float64, (4,), address_space="weird")


def test_scalar_declaration_verifies_and_prints() -> None:
  stmt = p.assign("shared", p.const_float(2), declare=True)
  proc = p.proc("scalar", [], [stmt])
  verify_program(proc)
  assert "float64 shared = 2" in format_program(proc)


def test_paired_store_verifies_and_prints() -> None:
  out = p.buffer("out", dtypes.float64, (2,))
  stmt = p.store_pair(p.view(out, [p.const_int(0)]), p.const_float(1), p.const_float(2))
  proc = p.proc("paired", [out], [stmt])
  verify_program(proc)
  assert "out[0] <- pair(1, 2)" in format_program(proc)


def test_paired_store_rejects_non_double_values() -> None:
  out = p.buffer("out", dtypes.float64, (2,))
  with pytest.raises(TypeError, match="requires float64"):
    p.store_pair(p.view(out, [p.const_int(0)]), p.const_float(1), p.const_float(2, dtypes.float32))


def test_program_rejects_duplicate_procedure_names() -> None:
  with pytest.raises(VerifyError, match="duplicate procedure name 'same'"):
    verify_program(p.program([p.proc("same", [], []), p.proc("same", [], [])]))


def test_every_observed_program_stage_verifies() -> None:
  import scaly as sc
  from scaly.passes.lowering import lower_function

  @sc.function(sc.arg("x", 5), outputs=sc.arg("y"), name="observed")
  def fun(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x * x).scalar()

  stages: list[ProgramNode] = []
  lower_function(fun, observe=lambda _name, prog: stages.append(prog))
  assert stages
  for stage in stages:
    verify_program(stage)


@pytest.mark.parametrize("attrs,args", [({}, (p.const_float(1),)), ({"target": "v"}, ()), ({"target": "v", "declare": "yes"}, (p.const_float(1),))])
def test_invalid_scalar_assignment_rejected(attrs, args) -> None:
  with pytest.raises(VerifyError, match="assign-attrs"):
    verify_program(ProgramNode(ProgramOp.ASSIGN, args, attrs, dtype=dtypes.float64))


def test_range_kind_value_must_be_enum_member() -> None:
  # forge a RANGE with a non-enum 'kind' attr
  bad = ProgramNode(ProgramOp.RANGE, (p.const_int(0), p.const_int(4), p.const_int(1)), attrs={"name": "i", "kind": "bogus"}, dtype=dtypes.float64)
  with pytest.raises(VerifyError, match="range-attrs"):
    verify_program(bad)


def test_scalar_dtype_must_match_operands() -> None:
  a = p.const_float(1.0, dtype=dtypes.float32)
  b = p.const_float(2.0, dtype=dtypes.float64)
  with pytest.raises(TypeError, match="operand dtype mismatch"):
    p.add(a, b)


@pytest.mark.parametrize("count", [2])
def test_view_requires_one_flat_index(count: int) -> None:
  buf = p.buffer("matrix", dtypes.float64, (3, 4))
  indices = [p.const_int(0)] * count
  with pytest.raises(ValueError, match="exactly one flat index"):
    p.view(buf, indices)
  with pytest.raises(VerifyError, match="exactly one flat index"):
    verify_program(ProgramNode(ProgramOp.VIEW, tuple(indices), {"buffer": "matrix"}, dtype=dtypes.float64))


def test_view_rejects_non_scalar_index() -> None:
  buf = p.buffer("matrix", dtypes.float64, (3, 4))
  with pytest.raises(TypeError, match="index must be a scalar"):
    p.view(buf, [buf])
  with pytest.raises(VerifyError, match="index op=.*not scalar"):
    verify_program(ProgramNode(ProgramOp.VIEW, (buf,), {"buffer": "matrix"}, dtype=dtypes.float64))


def test_signed_zeros_are_two_constants() -> None:
  plus, minus = p.const_float(0.0), p.const_float(-0.0)
  assert plus is not minus
  assert math.copysign(1.0, plus.attrs["value"]) == 1.0 and math.copysign(1.0, minus.attrs["value"]) == -1.0
  assert p.const_float(float("nan")) is p.const_float(float("nan"))


def test_interning_hit_returns_the_node_as_it_was_built() -> None:
  """The key matches values that are equal and not identical, so a hit must not assign them:
  programs already hold the node."""
  x = p.var("intern_x", dtypes.float64)
  node = ProgramNode(ProgramOp.NEG, (x,), {"width": 3}, dtype=dtypes.float64)
  attrs = node.attrs

  again = ProgramNode(ProgramOp.NEG, (x,), {"width": np.int64(3)}, dtype=dtypes.float64)

  assert again is node and node.attrs is attrs and type(attrs["width"]) is int


@pytest.mark.parametrize(
  "field,value",
  [
    ("name", "float_alias"),
    ("bits", 32),
    ("c_type", "float"),
    ("is_floating", False),
    ("is_integer", True),
    ("is_bool", True),
    ("is_unsigned", True),
  ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_interning_distinguishes_every_dtype_field(field: str, value: object, reverse: bool) -> None:
  original = dtypes.float64
  distinct = replace(original, **{field: value})
  first_dtype, second_dtype = (distinct, original) if reverse else (original, distinct)
  name = f"dtype_intern_{field}_{reverse}"
  first = p.buffer(name, first_dtype, (3,))
  second = p.buffer(name, second_dtype, (3,))
  assert first is not second
  assert first.dtype is first_dtype and second.dtype is second_dtype
  assert (first.dtype.c_type, second.dtype.c_type) == (first_dtype.c_type, second_dtype.c_type)
  assert p.buffer(name, replace(first_dtype), (3,)) is first
  assert p.buffer(name, replace(second_dtype), (3,)) is second


def test_node_holds_a_frozen_copy_of_its_attributes() -> None:
  given = {"name": "intern_v"}
  node = ProgramNode(ProgramOp.VAR, (), given, dtype=dtypes.int64)
  given["name"] = "intern_w"

  assert node.attrs == {"name": "intern_v"}
  assert ProgramNode(ProgramOp.VAR, (), {"name": "intern_v"}, dtype=dtypes.int64) is node
  with pytest.raises(TypeError):
    node.attrs["name"] = "intern_w"  # ty: ignore[invalid-assignment]


def test_a_constant_table_is_copied_all_the_way_down() -> None:
  values = [1.0, 2.0]
  node = ProgramNode(ProgramOp.BUFFER, (), {"name": "intern_k", "values": values, "nested": {"rows": [0, 1]}}, dtype=dtypes.float64)
  values[0] = 9.0

  assert node.attrs["values"] == (1.0, 2.0) and node.attrs["nested"]["rows"] == (0, 1)
  assert ProgramNode(ProgramOp.BUFFER, (), {"name": "intern_k", "values": [1.0, 2.0], "nested": {"rows": [0, 1]}}, dtype=dtypes.float64) is node
  assert p.const_buffer("intern_t", dtypes.float64, (2,), [1.0, 2.0]) is p.const_buffer(
    "intern_t", dtypes.float64, (2,), [np.float64(1), np.float64(2)]
  )
  with pytest.raises(TypeError):
    node.attrs["nested"]["rows"] = ()


@pytest.mark.parametrize(("first", "second"), [(1, True), (1, 1.0), ((1, 0), (True, False)), ((0.0, 1.0), (-0.0, 1.0))])
def test_interning_tells_attributes_apart_by_kind_and_bits(first, second) -> None:
  assert first == second
  a = ProgramNode(ProgramOp.CONST_INT, (), {"value": first}, dtype=dtypes.int64)
  b = ProgramNode(ProgramOp.CONST_INT, (), {"value": second}, dtype=dtypes.int64)

  assert a is not b
  assert repr(a.attrs["value"]) == repr(frozen(first)) and repr(b.attrs["value"]) == repr(frozen(second))
  assert ProgramNode(ProgramOp.CONST_INT, (), {"value": first}, dtype=dtypes.int64) is a


def test_interning_shares_numpy_scalars_with_their_python_kind() -> None:
  flag = ProgramNode(ProgramOp.CONST_INT, (), {"value": True}, dtype=dtypes.bool_)
  assert ProgramNode(ProgramOp.CONST_INT, (), {"value": np.True_}, dtype=dtypes.bool_) is flag
  assert p.const_float(np.float64(1.5)) is p.const_float(1.5)


def test_bit_select_builds_verifies_and_prints():
  from scaly.ir.text import render_program_assembly

  mask = p.const_buffer("mask", dtypes.uint8, (2,), [5, 0])
  x = p.buffer("x", dtypes.float64, (16,))
  y = p.buffer("y", dtypes.float64, (16,))
  i = p.var("i")
  word = p.load(p.view(mask, [p.div(i, p.const_int(8))]))
  value = p.bit_select(word, p.mod(i, p.const_int(8)), p.load(p.view(x, [i])))
  prog = p.program([p.proc("kernel", [x, y], [mask, p.for_(p.range_("i", 0, 16), [p.store(p.view(y, [i]), value)])])])
  verify_program(prog)
  assert "y[i] <- bit_select(mask[(i / 8)], (i % 8), x[i])" in format_program(prog)
  assert "prog.bit_select(prog.load %mask[prog.div(%i, 8)], prog.mod(%i, 8), prog.load %x[%i])" in render_program_assembly(prog)
  with pytest.raises(TypeError, match="integer word and bit"):
    p.bit_select(p.load(p.view(x, [i])), i, p.load(p.view(x, [i])))


def test_carried_loop_builds_verifies_and_prints():
  from scaly.ir.text import render_program_assembly

  flag = p.buffer("flag", dtypes.bool_, (1,))
  loop = p.for_(p.range_("i", 0, 3, carried=True), [p.break_if(p.load(p.view(flag, [p.const_int(0)])))], exit_var=True)
  prog = p.program([p.proc("carried", [flag], [loop])])
  verify_program(prog)
  assert loop.args[0].attrs["kind"] == RangeKind.SERIAL
  assert loop.args[0].attrs["carried"] is True
  assert "carried, exit_var" in format_program(prog)
  assert "break_if flag[0]" in format_program(prog)
  assert "carried, exit_var" in render_program_assembly(prog)
  assert "prog.break_if prog.load %flag[0]" in render_program_assembly(prog)
  assert "carried" not in p.range_("chunk", 0, 3).attrs


@pytest.mark.parametrize("where", ["bare", "block", "serial", "no_exit", "nested", "other_proc", "shared"])
def test_break_requires_the_nearest_loop_to_be_carried_with_an_exit_variable(where):
  stop = p.break_if(p.var("flag", dtypes.bool_))
  carried = lambda body: p.for_(p.range_("i", 0, 3, carried=True), body, exit_var=True)
  if where == "bare":
    root = stop
  elif where == "block":
    root = p.block(stop)
  elif where == "serial":
    root = p.for_(p.range_("i", 0, 3), [stop], exit_var=True)
  elif where == "no_exit":
    root = p.for_(p.range_("i", 0, 3, carried=True), [stop])
  elif where == "nested":
    root = carried([p.for_(p.range_("j", 0, 3), [p.block(stop)])])
  elif where == "other_proc":
    root = p.program([p.proc("valid", [], [carried([stop])]), p.proc("invalid", [], [stop])])
  else:
    root = p.proc("shared", [], [carried([stop]), p.for_(p.range_("j", 0, 3), [stop])])
  with pytest.raises(VerifyError, match="BREAK_IF.*carried.*exit_var|exit_var.*carried"):
    verify_program(root)


def test_nested_carried_breaks_and_blocks_verify():
  stop = p.break_if(p.var("flag", dtypes.bool_))
  inner = p.for_(p.range_("j", 0, 2, carried=True), [p.block(stop)], exit_var=True)
  verify_program(p.for_(p.range_("i", 0, 3, carried=True), [inner, stop], exit_var=True))


@pytest.mark.parametrize("where", ["procedure", "blocks", "loop_body"])
def test_exit_loops_cannot_redeclare_a_variable_in_one_scope(where):
  flags = p.buffer("flags", dtypes.bool_, (4,))
  count = p.buffer("count", dtypes.int64, (2,))
  loop = p.for_(p.range_("i", 0, 4, carried=True), [p.break_if(p.load(p.view(flags, [p.var("i")])))], exit_var=True)
  first = [loop, p.store(p.view(count, [p.const_int(0)]), p.var("i"))]
  second = [loop, p.store(p.view(count, [p.const_int(1)]), p.var("i"))]
  body = [*first, *second]
  if where == "blocks":
    body = [p.block(*first), p.block(p.block(*second))]
  elif where == "loop_body":
    body = [p.for_(p.range_("j", 0, 2), body)]
  with pytest.raises(VerifyError, match="duplicate C identifier 'i'"):
    verify_program(p.program([p.proc("twice", [flags, count], body)]))


@pytest.mark.parametrize("where", ["different_names", "procedures", "sibling_loops"])
def test_exit_loops_can_declare_distinct_or_separately_scoped_variables(where):
  loop = p.for_(p.range_("i", 0, 4, carried=True), [], exit_var=True)
  if where == "different_names":
    root = p.proc("distinct", [], [loop, p.for_(p.range_("j", 0, 4, carried=True), [], exit_var=True)])
  elif where == "procedures":
    root = p.program([p.proc("first", [], [loop]), p.proc("second", [], [loop])])
  else:
    root = p.proc("siblings", [], [p.for_(p.range_("j", 0, 2), [loop]), p.for_(p.range_("j", 0, 3), [loop])])
  verify_program(root)


@pytest.mark.parametrize("value", [p.const_int(1), p.const_float(1), p.buffer("not_scalar", dtypes.bool_, (1,))])
def test_break_if_requires_a_boolean_scalar(value):
  with pytest.raises(TypeError, match="bool scalar"):
    p.break_if(value)
  with pytest.raises(VerifyError, match="bool scalar"):
    verify_program(ProgramNode(ProgramOp.BREAK_IF, (value,)))


@pytest.mark.parametrize("args", [(), (p.var("flag", dtypes.bool_),) * 2])
def test_break_if_requires_exactly_one_condition(args):
  with pytest.raises(VerifyError, match="one bool scalar"):
    verify_program(ProgramNode(ProgramOp.BREAK_IF, args))


@pytest.mark.parametrize("attr,value", [("carried", 1), ("carried", "yes"), ("exit_var", 1), ("exit_var", "yes")])
def test_carried_and_exit_markers_are_boolean(attr, value):
  rng = p.range_("i", 0, 3, carried=True)
  node = rng if attr == "carried" else p.for_(rng, [])
  malformed = ProgramNode(node.op, node.args, {**node.attrs, attr: value}, node.dtype)
  with pytest.raises(VerifyError, match=f"{attr}.*bool"):
    verify_program(malformed)


@pytest.mark.parametrize("start,step", [(1, 1), (0, 2)])
def test_exit_variable_is_a_trip_count(start, step):
  loop = p.for_(p.range_("i", start, 5, step=step, carried=True), [], exit_var=True)
  with pytest.raises(VerifyError, match="exit_var.*start.*0.*step.*1"):
    verify_program(loop)


@pytest.mark.parametrize("lanes", [False, True])
def test_carried_ranges_cannot_be_vectorized(lanes):
  rng = p.range_("i", 0, 3, kind=RangeKind.VECTOR, carried=True)
  if lanes:
    rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "lanes": 2, "lane_cap": 2}, rng.dtype)
  with pytest.raises(VerifyError, match="carried.*VECTOR"):
    verify_program(rng)
