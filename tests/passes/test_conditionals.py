"""Program passes preserve conditional execution and prepare each branch in its own scope."""

import ctypes
import subprocess

import numpy as np
import pytest

from scaly.codegen.c import _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.program_spec import verify_program
from scaly.ir.spec import VerifyError
from scaly.ir.types import dtypes
from scaly.passes.program import PASS_PIPELINE
from scaly.passes.program.hoist_reciprocals import hoist_reciprocals
from scaly.passes.program.prepare_scalar import prepare_scalar_expressions
from scaly.passes.program.scheduling import MAX_SCALAR_DEPTH
from scaly.passes.program.widen_ranges import widen_ranges
from tests.compilers import c_compiler


def conditional_program():
  x, y = (p.buffer(name, dtypes.float64, (8,)) for name in ("x", "y"))
  tmp = p.buffer("tmp", dtypes.float64, (8,), address_space="private")
  i = p.var("i")
  value = p.load(p.view(x, [i]))
  flag = ProgramNode(ProgramOp.LT, (value, p.const_float(0)), dtype=dtypes.bool_)
  then = [p.store(p.view(tmp, [i]), p.mul(value, value))]
  otherwise = [p.store(p.view(tmp, [i]), p.div(p.const_float(1), p.add(value, p.const_float(2))))]
  loop = p.for_(p.range_("i", 0, 8), [p.if_(flag, then, otherwise), p.store(p.view(y, [i]), p.load(p.view(tmp, [i])))])
  raw = p.proc("branch", [x, y], [tmp, loop])
  proc = ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 1, "scalarize_mode": "procedure", "lowering": "scalar"}, dtype=raw.dtype)
  return p.program([proc])


PASSES = (
  *(entry for entry in PASS_PIPELINE if entry[0] not in ("hoist_scalar_invariant", "fold_arith_after_unroll")),
  ("hoist_reciprocals", hoist_reciprocals),
)


@pytest.mark.parametrize("name,transform", PASSES, ids=[name for name, _ in PASSES])
def test_each_pass_preserves_both_branches(tmp_path, name, transform):
  prog = transform(conditional_program())
  verify_program(prog)
  assert any(node.op == ProgramOp.IF for node in walk_program(prog))
  source = "\n".join([*_includes(dialect="c", prog=prog), *_render_raw_callee(prog.args[-1], dialect="c")])
  path, library = tmp_path / "branch.c", tmp_path / "branch.so"
  path.write_text(source + "\nvoid run(double *x, double *y) { branch_raw(x, y, NULL); }\n")
  subprocess.run([*c_compiler(), "-shared", "-fPIC", "-std=c99", str(path), "-lm", "-o", str(library)], check=True, capture_output=True)
  run = ctypes.CDLL(str(library)).run
  pointer = ctypes.POINTER(ctypes.c_double)
  run.argtypes = [pointer, pointer]
  x = np.array([-2.0, 1.0, -3.0, 0.0, -4.0, 2.0, -5.0, 4.0])
  y = np.full(8, np.nan)
  run(x.ctypes.data_as(pointer), y.ctypes.data_as(pointer))
  expected = [v * v if v < 0 else 1 / (v + 2) for v in x]
  np.testing.assert_allclose(y, expected)


def test_widening_keeps_a_conditional_body_scalar():
  prog = conditional_program()
  assert widen_ranges(prog, lanes=2) is prog


def test_preparation_bounds_branch_values_and_the_condition():
  x, y = (p.buffer(name, dtypes.float64, (1,)) for name in ("x", "y"))
  value = p.load(p.view(x, [p.const_int(0)]))
  for _ in range(MAX_SCALAR_DEPTH + 4):
    value = ProgramNode(ProgramOp.SIN, (value,), dtype=dtypes.float64)
  flag = ProgramNode(ProgramOp.LT, (value, p.const_float(0)), dtype=dtypes.bool_)
  store = p.store(p.view(y, [p.const_int(0)]), value)
  prog = p.program([p.proc("prepared", [x, y], [p.if_(flag, [store], [store])])])
  prepared = prepare_scalar_expressions(prog).args[0]
  pc = prepared.attrs["param_count"]
  assert any(stmt.op == ProgramOp.ASSIGN for stmt in prepared.args[pc:])
  branch = next(stmt for stmt in prepared.args[pc:] if stmt.op == ProgramOp.IF)
  assert all(any(stmt.op == ProgramOp.ASSIGN for stmt in block.args) for block in branch.args[1:])


@pytest.mark.parametrize("args", [(p.const_int(0), p.block(), p.block()), (p.const_int(1), p.block())])
def test_if_verifier_rejects_bad_condition_or_blocks(args):
  with pytest.raises(VerifyError, match="IF takes"):
    verify_program(ProgramNode(ProgramOp.IF, args, dtype=dtypes.float64))


def test_if_does_not_hide_an_invalid_break():
  flag = ProgramNode(ProgramOp.CONST_INT, attrs={"value": 1}, dtype=dtypes.bool_)
  with pytest.raises(VerifyError, match="BREAK_IF requires"):
    verify_program(p.if_(flag, [p.break_if(flag)]))
