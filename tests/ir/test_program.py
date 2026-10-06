"""Phase 4: hand-built Program IR examples — verify and pretty-print.

These tests pin the Program IR contract before lowering (Phase 5) starts to
build real Program IR from expression IR. Each example walks one piece of the
vocabulary:

- elementwise kernel-like loop with LOAD/STORE through VIEWs;
- nested loop with reduction-kind range;
- host PROC that LAUNCHes a KERNEL and synchronizes;
- negative cases: malformed nodes are rejected with a clear diagnostic.
"""

from __future__ import annotations

import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, RangeKind
from scaly.ir.text import format_program
from scaly.ir.program_spec import spec_host_program, spec_kernel_program, spec_program_full, verify_program
from scaly.ir.spec import VerifyError
from scaly.ir.types import dtypes


def _elementwise_neg_kernel() -> ProgramNode:
  """A tiny elementwise kernel: ``out[i] = -in[i]`` for ``i in [0, N)``."""
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
  return p.kernel("k_neg", [in_buf, out_buf], body, grid_dims=1, device="cuda:0")


def test_elementwise_kernel_verifies_and_prints() -> None:
  k = _elementwise_neg_kernel()
  verify_program(k, spec=spec_kernel_program)
  dump = format_program(k)
  assert "kernel k_neg" in dump
  assert "device=cuda:0" in dump
  assert "for i in [0, 16) step 1 kind=global" in dump
  assert "out_[i] <- (-in_[i])" in dump


def test_host_proc_with_launch_and_barrier_verifies() -> None:
  in_buf = p.buffer("in_", dtypes.float64, (16,))
  out_buf = p.buffer("out_", dtypes.float64, (16,))
  host = p.proc(
    "host_drive",
    [in_buf, out_buf],
    [
      p.launch("k_neg", grid=[16], block_dims=[1], args=[in_buf, out_buf]),
    ],
  )
  verify_program(host, spec=spec_host_program)
  text = format_program(host)
  assert "proc host_drive" in text
  assert "launch k_neg<<<(16), (1)>>>" in text


def test_host_proc_rejects_device_only_op() -> None:
  in_buf = p.buffer("in_", dtypes.float64, (16,))
  bad = p.proc(
    "host_bad",
    [in_buf],
    [
      p.barrier("device"),  # BARRIER must live inside a KERNEL, not a host PROC
    ],
  )
  with pytest.raises(VerifyError, match="host-proc-no-device-only"):
    verify_program(bad, spec=spec_host_program)


def test_kernel_rejects_host_only_op() -> None:
  in_buf = p.buffer("in_", dtypes.float64, (16,))
  bad = p.kernel(
    "k_bad",
    [in_buf],
    [
      p.launch("k_neg", grid=[16], block_dims=[1], args=[in_buf]),  # LAUNCH only lives in host code
    ],
  )
  with pytest.raises(VerifyError, match="kernel-no-host-only"):
    verify_program(bad, spec=spec_kernel_program)


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
  b = p.buffer("x", dtypes.float32, (3, 4), address_space="local", device="metal:0")
  assert b.dtype == dtypes.float32
  assert b.attrs["address_space"] == "local"
  assert str(b.attrs["device"]) == "metal:0"


def test_full_program_verifies() -> None:
  k = _elementwise_neg_kernel()
  in_buf = p.buffer("in_", dtypes.float64, (16,))
  out_buf = p.buffer("out_", dtypes.float64, (16,))
  host = p.proc(
    "host_drive",
    [in_buf, out_buf],
    [p.launch("k_neg", grid=[16], block_dims=[1], args=[in_buf, out_buf])],
  )
  prog = p.program([host], [k])
  verify_program(prog, spec=spec_program_full)
  text = format_program(prog)
  assert text.startswith("program")
  assert "kernel k_neg" in text
  assert "proc host_drive" in text


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
