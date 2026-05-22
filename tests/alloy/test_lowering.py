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
  # The output expression is aliased directly to the ``y`` output buffer —
  # no workspace + copy loop.
  assert " for i_y in [0, 4) step 1 kind=global" in text
  assert "y[i_y] <- sin(x[i_y])" in text


def test_lower_binary_elementwise() -> None:
  x = al.sym("x", 3)
  y = al.sym("y", 3)
  z = x * y
  fn = al.Function("f", [x, y], [z], ["x", "y"], ["z"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  assert "(x[i_z] * y[i_z])" in text


def test_lower_chained_unary_binary() -> None:
  x = al.sym("x", 5)
  y = x.sin() + x
  fn = al.Function("f", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  # intermediate sin lands in a private t0; final ADD lands directly in y.
  assert "sin(x[i_t0])" in text
  assert "(t0[i_y] + x[i_y])" in text


def test_reshape_is_alias() -> None:
  x = al.sym("x", 6)
  y = x.reshape((2, 3))
  fn = al.Function("f", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  verify_program(proc)
  text = format_program(proc)
  # RESHAPE aliases the source buffer; output is a copy loop from x into y.
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
  # two GLOBAL loops (one per output axis), no REDUCE
  assert text.count("kind=global") == 2
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
  # MAP's output now aliases directly to `m`, so the loop var is named ``it_m``.
  assert "for it_m in [0, 3)" in text
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
  # per-input loops now write directly into the ``z`` output buffer.
  assert text.count("for j_z_0") == 1
  assert text.count("for j_z_1") == 1


def test_lower_concat_emits_per_input_global_loop_with_offset() -> None:
  x = al.sym("x", 2)
  y = al.sym("y", 3)
  fn = al.Function("f_concat", [x, y], [al.concat([x, y])], ["x", "y"], ["z"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "for j_z_0 in [0, 2)" in text
  assert "for j_z_1 in [0, 3)" in text


def test_lower_gather_unrolls() -> None:
  x = al.sym("x", 6)
  y = x.gather([0, 2, 4])
  fn = al.Function("f_gather", [x], [y], ["x"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "y[0] <- x[0]" in text
  assert "y[1] <- x[2]" in text
  assert "y[2] <- x[4]" in text


def test_lower_scatter_initializes_then_writes() -> None:
  v = al.sym("v", 2)
  y = al.scatter(v, [1, 3], 6)
  fn = al.Function("f_scatter", [v], [y], ["v"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  # zero-init all six output cells then write two — directly into ``y``.
  assert text.count("<- 0") == 6
  assert "y[1] <- v[0]" in text
  assert "y[3] <- v[1]" in text


def test_lower_matmul_matvec_emits_outer_global_inner_reduce() -> None:
  a = al.sym("a", (3, 4))
  v = al.sym("v", 4)
  fn = al.Function("f_matvec", [a, v], [a @ v], ["a", "v"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "kind=global" in text
  assert "kind=reduce" in text
  # matvec output aliases to ``y``; the inner index is ``k_y``.
  assert "(a[((i_y * 4) + k_y)]" in text


def test_lower_matmul_matmat_emits_three_loops() -> None:
  a = al.sym("a", (2, 3))
  b = al.sym("b", (3, 4))
  fn = al.Function("f_matmat", [a, b], [a @ b], ["a", "b"], ["c"])
  proc = lower_function(fn)
  text = format_program(proc)
  # outer global, inner global, innermost reduce. No extra output-copy loop
  # — the matmat result writes directly into the ``c`` output buffer.
  assert text.count("kind=global") == 2
  assert text.count("kind=reduce") == 1


def test_lower_sum_emits_reduce_loop() -> None:
  x = al.sym("x", 4)
  fn = al.Function("f_sum", [x], [x.sum()], ["x"], ["y"])
  proc = lower_function(fn)
  text = format_program(proc)
  assert "kind=reduce" in text
  assert "y[0] <- (y[0] + x[i_y])" in text


def test_non_host_device_emits_kernel() -> None:
  # Phase 7: device-placed Functions now lower to a host driver + KERNEL.
  # Backend codegen is Phase 8; for now the JIT path raises ``only host lowering``.
  x = al.sym("x", 3)
  fn = al.Function("f_dev", [x], [x.sin()], ["x"], ["y"], device="cuda:0")
  prog = lower_function(fn)
  assert int(prog.attrs["kernel_count"]) == 1


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
  # CONST stores three scalar values into a private workspace buffer.
  assert "t0[0] <- 1" in text
  assert "t0[1] <- 2" in text
  assert "t0[2] <- 3" in text
