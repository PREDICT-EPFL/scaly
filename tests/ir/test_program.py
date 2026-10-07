"""Phase 4: hand-built Program IR examples — verify and pretty-print.

These tests pin the Program IR contract before lowering (Phase 5) starts to
build real Program IR from expression IR. Each example walks one piece of the
vocabulary:

- elementwise loop with LOAD/STORE through VIEWs;
- nested loop with reduction-kind range;
- negative cases: malformed nodes are rejected with a clear diagnostic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, RangeKind
from scaly.ir.text import format_program
from scaly.ir.program_spec import verify_program
from scaly.ir.spec import VerifyError
from scaly.ir.types import dtypes


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
  bad = ProgramNode(ProgramOp.FOR, (p.const_int(0),), attrs={"body_len": 0})
  with pytest.raises(VerifyError, match="for-body"):
    verify_program(bad)


def test_view_requires_buffer() -> None:
  with pytest.raises(TypeError, match="view requires a BUFFER"):
    p.view(p.const_int(0), [p.const_int(0)])


def test_buffer_address_space_validated() -> None:
  with pytest.raises(ValueError, match="unknown address space"):
    p.buffer("b", dtypes.float64, (4,), address_space="weird")


def test_buffer_dtype_and_device_recorded() -> None:
  b = p.buffer("x", dtypes.float32, (3, 4), address_space="local")
  assert b.dtype == dtypes.float32
  assert b.attrs["address_space"] == "local"
  assert str(b.attrs["device"]) == "host"


def test_no_gpu_vocabulary() -> None:
  assert not {"KERNEL", "LAUNCH", "BARRIER"} & set(ProgramOp.__members__)
  assert set(RangeKind.__members__) == {"SERIAL", "VECTOR", "GLOBAL", "REDUCE", "UNROLL"}
  for name in ("kernel", "launch", "barrier"):
    assert not hasattr(p, name)


def test_full_program_verifies() -> None:
  prog = p.program([_elementwise_neg_proc()])
  verify_program(prog)
  text = format_program(prog)
  assert text.startswith("program")
  assert "proc k_neg" in text


def test_reduction_kind_range_verifies() -> None:
  """A REDUCE-kind range is allowed today (lowering pass will decide how to emit it)."""
  buf = p.buffer("buf", dtypes.float64, (8,))
  rng = p.range_("i", 0, 8, kind=RangeKind.REDUCE)
  i = p.var("i", dtype=dtypes.int64)
  acc = p.assign("acc", p.add(p.load(p.view(buf, [i])), p.var("acc", dtype=dtypes.float64)))
  proc = p.proc("reduce_proc", [buf], [p.for_(rng, [acc])])
  verify_program(proc)


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
    verify_program(ProgramNode(ProgramOp.ASSIGN, args, attrs))


def test_range_kind_value_must_be_enum_member() -> None:
  # forge a RANGE with a non-enum 'kind' attr
  bad = ProgramNode(ProgramOp.RANGE, (p.const_int(0), p.const_int(4), p.const_int(1)), attrs={"name": "i", "kind": "bogus"})
  with pytest.raises(VerifyError, match="range-attrs"):
    verify_program(bad)


def test_scalar_dtype_must_match_operands() -> None:
  a = p.const_float(1.0, dtype=dtypes.float32)
  b = p.const_float(2.0, dtype=dtypes.float64)
  with pytest.raises(TypeError, match="operand dtype mismatch"):
    p.add(a, b)


def test_hash_consing_makes_structurally_equal_nodes_identical() -> None:
  a = p.const_int(7)
  b = p.const_int(7)
  assert a is b


@pytest.mark.parametrize("count", [0, 2])
def test_view_requires_one_flat_index(count: int) -> None:
  buf = p.buffer("matrix", dtypes.float64, (3, 4))
  indices = [p.const_int(0)] * count
  with pytest.raises(ValueError, match="exactly one flat index"):
    p.view(buf, indices)
  with pytest.raises(VerifyError, match="exactly one flat index"):
    verify_program(ProgramNode(ProgramOp.VIEW, tuple(indices), {"buffer": "matrix"}, dtypes.float64))


def test_view_uses_flat_index_without_rank() -> None:
  buf = p.buffer("matrix", dtypes.float64, (3, 4))
  index = p.add(p.mul(p.var("row"), p.const_int(4)), p.var("column"))
  view = p.view(buf, [index])
  verify_program(view)
  assert view.args == (index,)
  assert view.attrs == {"buffer": "matrix"}


def test_view_rejects_non_scalar_index() -> None:
  buf = p.buffer("matrix", dtypes.float64, (3, 4))
  with pytest.raises(TypeError, match="index must be a scalar"):
    p.view(buf, [buf])
  with pytest.raises(VerifyError, match="index op=.*not scalar"):
    verify_program(ProgramNode(ProgramOp.VIEW, (buf,), {"buffer": "matrix"}, dtypes.float64))


def test_signed_zeros_are_two_constants() -> None:
  plus, minus = p.const_float(0.0), p.const_float(-0.0)
  assert plus is not minus
  assert math.copysign(1.0, plus.attrs["value"]) == 1.0 and math.copysign(1.0, minus.attrs["value"]) == -1.0
  assert p.const_float(float("nan")) is p.const_float(float("nan"))


def test_interning_hit_returns_the_node_as_it_was_built() -> None:
  """The key matches values that are equal and not identical, so a hit must not assign them:
  programs already hold the node."""
  x = p.var("intern_x", dtypes.float64)
  node = ProgramNode(ProgramOp.NEG, (x,), {"width": 3}, dtypes.float64)
  attrs = node.attrs

  again = ProgramNode(ProgramOp.NEG, (x,), {"width": np.int64(3)}, dtypes.float64)

  assert again is node and node.attrs is attrs and type(attrs["width"]) is int


def test_node_holds_a_frozen_copy_of_its_attributes() -> None:
  given = {"name": "intern_v"}
  node = ProgramNode(ProgramOp.VAR, (), given, dtypes.int64)
  given["name"] = "intern_w"

  assert node.attrs == {"name": "intern_v"}
  assert ProgramNode(ProgramOp.VAR, (), {"name": "intern_v"}, dtypes.int64) is node
  with pytest.raises(TypeError):
    node.attrs["name"] = "intern_w"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(("first", "second"), [(1, True), (True, 1), (0, False), (1, 1.0), ((1, 0), (True, False)), ((0.0, 1.0), (-0.0, 1.0))])
def test_interning_tells_attributes_apart_by_kind_and_bits(first, second) -> None:
  assert first == second
  a = ProgramNode(ProgramOp.CONST_INT, (), {"value": first}, dtypes.int64)
  b = ProgramNode(ProgramOp.CONST_INT, (), {"value": second}, dtypes.int64)

  assert a is not b
  assert a.attrs["value"] is first and b.attrs["value"] is second
  assert ProgramNode(ProgramOp.CONST_INT, (), {"value": first}, dtypes.int64) is a


def test_interning_shares_numpy_scalars_with_their_python_kind() -> None:
  flag = ProgramNode(ProgramOp.CONST_INT, (), {"value": True}, dtypes.bool_)
  assert ProgramNode(ProgramOp.CONST_INT, (), {"value": np.True_}, dtypes.bool_) is flag
  assert p.const_float(np.float64(1.5)) is p.const_float(1.5)
