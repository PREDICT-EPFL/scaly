"""Generated-C snapshots: the gate that says a refactor did not change what scaly emits.

A refactor of the IR, the passes or the renderer must leave the generated C identical, so this
pins it byte for byte. Each corpus entry covers one lowering path — a forward function, a dense
Jacobian, a compact sparse Jacobian, a VMAP workload, a wide function using most of the math surface, a workspace spill, and a solver-bearing graph — and every run re-renders and diffs
against the recorded source and header. A diff means something semantic moved with the code. Byte
equality also keeps the JIT cache key stable: it hashes the source, and the header pins the ABI
signature that goes in with it.

Take a fresh snapshot before starting a refactor and leave it alone until the work lands:
regenerating a baseline mid-flight is the defect, not the fix, because it is the one thing that can
turn a real behavior change green. After a *deliberate* codegen change, regenerate in the same diff
and review what moved: ``uv run python -m tests.test_c_snapshot``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
from tests.solvers.problem_helpers import build_qp
from scaly.codegen import render_c_api_header, render_c_source
from scaly.solvers.registry import available_backends

BASELINE = Path(__file__).resolve().parent / "baseline" / "c"
N_STAGES = 3
WEIGHTS = (np.arange(40 * 40, dtype=np.float64).reshape(40, 40) % 7 - 3.0) / 11.0


def _dynamics() -> sc.Function:
  """Elementwise math, slicing, a reduction and a concat."""

  @sc.function(sc.G(sc.L("z", 4), sc.L("u", 2)), output=sc.L("znext", ...), name="dynamics")
  def dynamics(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, u = inputs
    pos, vel = z[:2], z[2:]
    drag = 0.1 * sc.sumsqr(vel)
    return sc.concat([pos + 0.05 * vel, vel + 0.05 * (u - drag * vel)])

  return dynamics


def _shooting() -> sc.Function:
  """Multiple shooting defect over ``N_STAGES`` VMAP iterations."""
  z = sc.sym("z", 4 * (N_STAGES + 1))
  u = sc.sym("u", 2 * N_STAGES)
  defect = sc.vmap(_dynamics(), N_STAGES, [(z, 0, 4), (u, 0, 2)]) - z[4:]
  return sc.Function._from_exprs("shooting", [z, u], [defect], ["z", "u"], ["eq"])


def _wide() -> sc.Function:
  """Two outputs, a 40x40 matmul either way round, the transcendental surface, and a scatter."""

  @sc.function(sc.G(sc.L("x", 40), sc.L("y", 40)), output=sc.G(sc.L("z", ...), sc.L("tail", ...)), name="wide")
  def wide(inputs):
    x, y = inputs
    h = sc.const(WEIGHTS) @ x
    a = sc.maximum(h, 0.0) - sc.minimum(h, 0.0) * 0.5
    b = sc.atan2(a, y) + (a**3.0) / (1.0 + y.abs())
    trig = a.sin() * b.cos() + a.tan() + (a * 0.1).asin() + (b * 0.1).acos() + b.atan()
    hyp = a.sinh() + b.cosh() + b.tanh() + a.erf()
    c = (a.exp() + b.sqrt().log()) * (trig + hyp) + (a.floor() + b.ceil())
    z = sc.const(WEIGHTS).T @ (c / (1.0 + y * y))
    return (z, sc.scatter(z[:4], np.array([3, 1, 2, 0]), (4,)))

  return wide


def _workspace() -> sc.Function:
  """A shared intermediate large enough to spill after expression normalization."""
  x = sc.sym("x", 2048)
  value = x.sin()
  return sc.Function._from_exprs("workspace", [x], [value.sum(), (value * value).sum()], ["x"], ["sum", "sumsqr"])


def _control() -> sc.Function:
  """A scan whose step makes a data-dependent choice, takes a max and accumulates a repeated scatter."""
  c, u = sc.sym("c", 3), sc.sym("u", 2)
  clipped = sc.where(c > 1.0, 1.0, c) + sc.scatter(u, np.array([0, 2]), (3,))
  step = sc.Function._from_exprs(
    "control_step", [c, u], [clipped + sc.scatter(u * u, np.array([1, 1]), (3,)), sc.stack([clipped.max()])], ["c", "u"], ["n", "m"]
  )
  c0, us = sc.sym("c0", 3), sc.sym("us", 2 * N_STAGES)
  final, peaks = sc.scan(step, c0, [(us, 0, 2)], length=N_STAGES)
  return sc.Function._from_exprs("control", [c0, us], [final, peaks], ["c0", "us"], ["final", "peaks"])


def _qp_host() -> sc.Function:
  """A host function whose graph reaches a solver through a nested call."""
  mu = sc.sym("mu", 2)
  qp = build_qp(P=sc.const(np.eye(2)), c=-mu, name="corpus_qp")
  x = qp.symbolic_call(sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), mu)[0]
  return sc.Function._from_exprs("qp_host", [mu], [sc.sumsqr(x)], ["mu"], ["cost"])


def _table() -> sc.Function:
  """A 1-D linear lookup table of five points: the binary search, one take, the end segments continued."""
  return sc.interp.interpolant(np.arange(5.0), np.array([0.0, 1.0, 0.5, -0.25, 2.0]), kind="linear").function("table")


CORPUS = {
  "forward": _dynamics,
  "jac": lambda: sc.jacobian(_dynamics(), "znext", "z"),
  "vmap": _shooting,
  "spjac": lambda: sc.sparse_jacobian(_shooting(), "eq", "z"),
  "wide": _wide,
  "workspace": _workspace,
  "control": _control,
  "table": _table,
}
SOLVER_CORPUS = {"solver": _qp_host}


def _rendered(fun: sc.Function) -> dict[str, str]:
  return {".c": render_c_source(fun), ".h": render_c_api_header(fun)}


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_generated_c_matches_snapshot(name: str) -> None:
  for suffix, text in _rendered(CORPUS[name]()).items():
    assert text == (BASELINE / f"{name}{suffix}").read_text(), f"{name}{suffix} moved"


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name", sorted(SOLVER_CORPUS))
def test_generated_solver_c_matches_snapshot(name: str) -> None:
  for suffix, text in _rendered(SOLVER_CORPUS[name]()).items():
    assert text == (BASELINE / f"{name}{suffix}").read_text(), f"{name}{suffix} moved"


def test_baseline_holds_exactly_the_corpus() -> None:
  """A dropped corpus entry is a silently narrower gate, and an orphan baseline is never read."""
  expected = {f"{name}{suffix}" for name in (*CORPUS, *SOLVER_CORPUS) for suffix in (".c", ".h")}
  assert {p.name for p in BASELINE.iterdir()} == expected


def main() -> int:
  BASELINE.mkdir(parents=True, exist_ok=True)
  builders = dict(CORPUS)
  if "piqp" in available_backends():
    builders |= SOLVER_CORPUS
  else:
    print("skipping the solver entry: the scaly-piqp plugin is not installed")
  for name, build in sorted(builders.items()):
    fun = build()
    for suffix, text in _rendered(fun).items():
      path = BASELINE / f"{name}{suffix}"
      changed = not path.exists() or path.read_text() != text
      path.write_text(text)
      print(f"{'CHANGED' if changed else 'unchanged'}: {path.name} ({len(text.splitlines())} lines)")
  for path in sorted(BASELINE.iterdir()):
    if path.stem not in (*CORPUS, *SOLVER_CORPUS):
      path.unlink()
      print(f"removed orphan: {path.name}")
  print("\nA CHANGED line during phases 1-6 of the restructure is a bug in the refactor, not an update.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
