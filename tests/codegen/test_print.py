"""``sc.print`` end to end: what the JIT prints, where a print runs, and compiling prints out."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_module, render_c_source

pytestmark = pytest.mark.skipif(shutil.which(os.environ.get("SCALY_CC", "cc")) is None, reason="cc is required to run generated C")


@sc.function(sc.group(sc.arg("x", 2), sc.arg("k", ())), outputs=sc.arg("y"))
def scaled(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, k = inputs
  return (sc.print("x={} k={}", x, k) * k).sum()


def test_print_returns_its_value_and_prints_at_each_call(capfd) -> None:
  x = np.array([1.5, -2.0])
  np.testing.assert_allclose(scaled((x, 3.0)), (x * 3.0).sum())
  # Read straight after the call: C stdout is buffered when it is not a terminal, so the line is
  # only there because the JIT flushed it.
  assert capfd.readouterr().out == "x=[1.5, -2] k=3\n"
  scaled((np.array([0.1, 0.0]), -1.0))
  assert capfd.readouterr().out == "x=[0.10000000000000001, 0] k=-1\n"


def test_format_text_reaches_stdout_unchanged(capfd) -> None:
  @sc.function(sc.arg("x", ()), outputs=sc.arg("y"))
  def f(x: sc.Expr) -> sc.Expr:
    return sc.print('100% "{}" \\ {{ok}}\t%s', x) + 1.0

  assert f(2.5) == 3.5
  assert capfd.readouterr().out == '100% "2.5" \\ {ok}\t%s\n'


def test_identical_prints_intern_to_one(capfd) -> None:
  @sc.function(sc.arg("x", ()), outputs=sc.arg("y"))
  def f(x: sc.Expr) -> sc.Expr:
    return sc.print("x={}", x) * sc.print("x={}", x)

  assert f(3.0) == 9.0
  assert capfd.readouterr().out == "x=3\n"


@sc.function(sc.group(sc.arg("z", 2), sc.arg("p", ())), outputs=sc.arg("y", 2))
def stage(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  z, p = inputs
  return sc.print("z={}", z) * sc.print("p={}", p).sin()


def test_mapped_print_runs_once_per_trip(capfd) -> None:
  """``p`` is the same at every trip, so the ``sin`` of it is loop-invariant work that would leave
  the loop; its print must still run at every trip, in trip order."""

  @sc.function(sc.group(sc.arg("z", (8, 2)), sc.arg("p", ())), outputs=sc.arg("y", (8, 2)))
  def rollout(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    return sc.vmap(stage, 8)((z, sc.broadcast(p)))

  z = np.arange(16.0).reshape(8, 2)
  np.testing.assert_allclose(rollout((z, 0.5)), z * np.sin(0.5))
  lines = capfd.readouterr().out.splitlines()
  assert sorted(lines) == sorted([*(f"z=[{2 * i:g}, {2 * i + 1:g}]" for i in range(8)), *["p=0.5"] * 8])
  assert [line for line in lines if line.startswith("z")] == [f"z=[{2 * i:g}, {2 * i + 1:g}]" for i in range(8)]


def test_derivative_prints_when_it_computes_the_primal(capfd) -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("f"))
  def f(x: sc.Expr) -> sc.Expr:
    return sc.print("x={}", x).sin().sum()

  x = np.array([0.25, -1.0])
  np.testing.assert_allclose(sc.gradient(f, "f", "x")(x), np.cos(x))
  assert capfd.readouterr().out == "x=[0.25, -1]\n"


def test_source_defines_scaly_printf_only_when_it_prints() -> None:
  @sc.function(sc.arg("x", ()), outputs=sc.arg("y"))
  def quiet(x: sc.Expr) -> sc.Expr:
    return x * 2.0

  assert "SCALY_PRINTF" not in render_c_source(quiet)
  assert "#ifndef SCALY_PRINTF" in render_c_source(scaled)


@pytest.mark.parametrize(
  ("define", "expected"),
  [((), "x=[0.25, -0.75] k=2\n"), (("-DSCALY_PRINTF(...)=",), ""), (("-DSCALY_PRINTF=uart_printf", "-include", "uart.h"), "uart: x=[0.25, -0.75] k=2\n")],
)
def test_scaly_printf_compiles_prints_out_or_reroutes_them(tmp_path, define, expected) -> None:
  module = render_c_module(scaled)
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  (tmp_path / "uart.h").write_text("int uart_printf(const char* format, ...);\n")
  main = tmp_path / "main.c"
  main.write_text(
    """
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include "scaled.h"
#include "uart.h"

int uart_printf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  fputs("uart: ", stdout);
  int n = vprintf(format, args);
  va_end(args);
  return n;
}

int main(void) {
  scaled_x_t x = {{0.25, -0.75}};
  scaled_k_t k = {{2.0}};
  scaled_y_t y = {{0}};
  int err = scaled_call(&x, &k, &y, NULL);
  if (err) return err;
  return fabs(y.data[0] + 1.0) > 1e-12;
}
"""
  )
  exe = tmp_path / "main"
  cc = os.environ.get("SCALY_CC", "cc")
  subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", *define, str(main), str(source), "-lm", "-o", str(exe)], check=True, cwd=tmp_path)
  assert subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout == expected
