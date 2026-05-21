"""Phase 5 (first slice): lower semantic IR to Program IR for elementwise + reshape.

Lowering for ``SUM`` / ``MATMUL`` / ``GATHER`` / ``CALL`` / ``MAP`` /
``SOLVER_CALL`` lands in subsequent commits.
"""

from __future__ import annotations

import pytest

import alloy as al
from alloy.lowering import LoweringError, lower_function, main_proc
from alloy.program import POps, format_program, verify_program


def test_lower_unary_elementwise() -> None:
  x = al.sym("x", 4)
  y = x.sin()
  fn = al.Function("f", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  assert "proc f(x:float64(4,), y:float64(4,))" in text
  assert " for i_t0 in [0, 4) step 1 kind=global" in text
  # output should be copied from t0 into y
  assert "y[i_y] <- t0[i_y]" in text


def test_lower_binary_elementwise() -> None:
  x = al.sym("x", 3)
  y = al.sym("y", 3)
  z = x * y
  fn = al.Function("f", [x, y], [z], ["x", "y"], ["z"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  assert "(x[i_t0] * y[i_t0])" in text


def test_lower_chained_unary_binary() -> None:
  x = al.sym("x", 5)
  y = x.sin() + x
  fn = al.Function("f", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  # two for-loops: one for sin into t0, one for add into t1
  assert text.count("for i_t") == 2
  assert "sin(x[i_t0])" in text
  assert "(t0[i_t1] + x[i_t1])" in text


def test_reshape_is_alias() -> None:
  x = al.sym("x", 6)
  y = x.reshape((2, 3))
  fn = al.Function("f", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  # only the output copy loop emits a FOR — the reshape itself does not
  assert text.count("for i_") == 1


def test_unsupported_op_raises() -> None:
  # SLICE with step != 1 is rejected by the Phase 5 lowerer.
  x = al.sym("x", 8)
  fn = al.Function("f_stride", [x], [x[::2]], ["x"], ["y"])
  with pytest.raises(LoweringError, match="step=1"):
    lower_function(fn)


def test_lower_transpose_emits_nested_loops() -> None:
  a = al.sym("a", (2, 3))
  fn = al.Function("f_T", [a], [a.T], ["a"], ["y"])
  prog = lower_function(fn)
  text = format_program(prog)
  # two GLOBAL loops, no REDUCE
  assert text.count("kind=global") >= 3  # 2 transpose + 1 output copy
  assert "kind=reduce" not in text


def test_lower_map_emits_for_loop_with_call() -> None:
  x = al.sym("x", 3)
  stage = al.Function("stage_lower", [x], [x.sin()], ["x"], ["y"])
  batch = al.sym("batch", 9)
  mapped = al.map_(stage, length=3, inputs=[(batch, 0, 3)])
  fn = al.Function("f_map_lower", [batch], [mapped], ["batch"], ["m"])
  prog = lower_function(fn)
  assert prog.op == POps.PROGRAM
  assert int(prog.attrs["proc_count"]) == 2
  text = format_program(prog)
  assert "for it_t0 in [0, 3)" in text
  assert "call stage_lower(" in text


def test_lower_call_emits_callee_proc_and_call_statement() -> None:
  x = al.sym("x", 3)
  inner = al.Function("inner_call_lower", [x], [(x * x).sum()], ["x"], ["s"])
  z = al.sym("z", 3)
  (out,) = inner.call([z])
  fn = al.Function("f_call_lower", [z], [out], ["z"], ["y"])
  prog = lower_function(fn)
  assert prog.op == POps.PROGRAM
  assert int(prog.attrs["proc_count"]) == 2  # one callee + one main
  callee_proc = prog.args[0]
  assert callee_proc.attrs["name"] == "inner_call_lower"
  text = format_program(prog)
  assert "call inner_call_lower(" in text


def test_lower_stack_emits_per_input_global_loop() -> None:
  x = al.sym("x", 3)
  y = al.sym("y", 3)
  fn = al.Function("f_stack", [x, y], [al.stack([x, y])], ["x", "y"], ["z"])
  proc = lower_function(fn)
  text = format_program(proc)
  # one per-input loop, all GLOBAL
  assert text.count("for j_t0_0") == 1
  assert text.count("for j_t0_1") == 1


def test_lower_concat_emits_per_input_global_loop_with_offset() -> None:
  x = al.sym("x", 2)
  y = al.sym("y", 3)
  fn = al.Function("f_concat", [x, y], [al.concat([x, y])], ["x", "y"], ["z"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "for j_t0_0 in [0, 2)" in text
  assert "for j_t0_1 in [0, 3)" in text


def test_lower_gather_unrolls() -> None:
  x = al.sym("x", 6)
  y = x.gather([0, 2, 4])
  fn = al.Function("f_gather", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "t0[0] <- x[0]" in text
  assert "t0[1] <- x[2]" in text
  assert "t0[2] <- x[4]" in text


def test_lower_scatter_initializes_then_writes() -> None:
  v = al.sym("v", 2)
  y = al.scatter(v, [1, 3], 6)
  fn = al.Function("f_scatter", [v], [y], ["v"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  # zero-init all six output cells then write two
  assert text.count("<- 0") == 6
  assert "t0[1] <- v[0]" in text
  assert "t0[3] <- v[1]" in text


def test_lower_matmul_matvec_emits_outer_global_inner_reduce() -> None:
  a = al.sym("a", (3, 4))
  v = al.sym("v", 4)
  fn = al.Function("f_matvec", [a, v], [a @ v], ["a", "v"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "kind=global" in text
  assert "kind=reduce" in text
  assert "(a[((i_t0 * 4) + k_t0)]" in text


def test_lower_matmul_matmat_emits_three_loops() -> None:
  a = al.sym("a", (2, 3))
  b = al.sym("b", (3, 4))
  fn = al.Function("f_matmat", [a, b], [a @ b], ["a", "b"], ["c"])
  proc = lower_function(fn)
  text = format_program(proc)
  # outer global, inner global, innermost reduce, plus one output copy loop
  assert text.count("kind=global") == 3
  assert text.count("kind=reduce") == 1


def test_lower_sum_emits_reduce_loop() -> None:
  x = al.sym("x", 4)
  fn = al.Function("f_sum", [x], [x.sum()], ["x"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "kind=reduce" in text
  assert "t0[0] <- (t0[0] + x[i_t0])" in text


def test_unsupported_device_raises() -> None:
  x = al.sym("x", 3)
  fn = al.Function("f", [x], [x.sin()], ["x"], ["y"], device="cuda:0")
  with pytest.raises(LoweringError, match="only emits host PROC"):
    lower_function(fn)


def test_lowered_proc_uses_input_buffer_directly() -> None:
  x = al.sym("x", 3)
  fn = al.Function("f", [x], [x.sin()], ["x"], ["y"])
  proc = main_proc(lower_function(fn))
  buffer_nodes = [a for a in proc.args if a.op == POps.BUFFER]
  names = [b.attrs["name"] for b in buffer_nodes]
  assert "x" in names and "y" in names


def test_const_inlined_into_lowered_proc() -> None:
  x = al.sym("x", 3)
  c = al.const([1.0, 2.0, 3.0])
  y = x + c
  fn = al.Function("f", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  # CONST stores three scalar values into t0
  assert "t0[0] <- 1" in text
  assert "t0[1] <- 2" in text
  assert "t0[2] <- 3" in text
