"""Generated-C snapshots: the gate that says a refactor did not change what alloy emits.

A refactor of the IR, the passes or the renderer must leave the generated C identical, so this
pins it byte for byte. Each corpus entry covers one lowering path — a forward function, a dense
Jacobian, a compact sparse Jacobian, a VMAP workload, a wide function that spills to the workspace
and uses most of the math surface, and a solver-bearing graph — and every run re-renders and diffs
against the recorded source and header. A diff means something semantic moved with the code. Byte
equality also keeps the JIT cache key stable: it hashes the source, and the header pins the ABI
signature that goes in with it.

Take a fresh snapshot before starting a refactor and leave it alone until the work lands:
regenerating a baseline mid-flight is the defect, not the fix, because it is the one thing that can
turn a real behavior change green. After a *deliberate* codegen change, regenerate in the same diff
and review what moved: ``uv run python tests/test_c_snapshot.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import alloy as al
from alloy.solvers.qp import _legacy_qp
from alloy.codegen import render_c_api_header, render_c_source
from alloy.solvers.registry import available_backends

BASELINE = Path(__file__).resolve().parent / "baseline" / "c"
N_STAGES = 3
WEIGHTS = (np.arange(40 * 40, dtype=np.float64).reshape(40, 40) % 7 - 3.0) / 11.0


def _dynamics() -> al.Function:
  """Elementwise math, slicing, a reduction and a concat."""

  @al.function(al.G(al.L("z", 4), al.L("u", 2)), al.L("znext", ...), name="dynamics")
  def dynamics(inputs):
    z, u = inputs
    pos, vel = z[:2], z[2:]
    drag = 0.1 * al.sumsqr(vel)
    return al.concat([pos + 0.05 * vel, vel + 0.05 * (u - drag * vel)])

  return dynamics


def _shooting() -> al.Function:
  """Multiple shooting defect over ``N_STAGES`` VMAP iterations."""
  z = al.sym("z", 4 * (N_STAGES + 1))
  u = al.sym("u", 2 * N_STAGES)
  defect = al.vmap(_dynamics(), N_STAGES, [(z, 0, 4), (u, 0, 2)]) - z[4:]
  return al.Function._from_exprs("shooting", [z, u], [defect], ["z", "u"], ["eq"])


def _wide() -> al.Function:
  """Two outputs, a 40x40 matmul either way round, the transcendental surface, and a scatter.

  Big enough that ``passes.pack_workspace`` spills to ``w[]`` (``sz_w`` is 1600 doubles), so the
  workspace packing and spill rendering are inside the gate too.
  """

  @al.function(al.G(al.L("x", 40), al.L("y", 40)), al.G(al.L("z", ...), al.L("tail", ...)), name="wide")
  def wide(inputs):
    x, y = inputs
    h = al.const(WEIGHTS) @ x
    a = al.maximum(h, 0.0) - al.minimum(h, 0.0) * 0.5
    b = al.atan2(a, y) + (a**3.0) / (1.0 + y.abs())
    trig = a.sin() * b.cos() + a.tan() + (a * 0.1).asin() + (b * 0.1).acos() + b.atan()
    hyp = a.sinh() + b.cosh() + b.tanh() + a.erf()
    c = (a.exp() + b.sqrt().log()) * (trig + hyp) + (a.floor() + b.ceil())
    z = al.const(WEIGHTS).T @ (c / (1.0 + y * y))
    return (z, al.scatter(z[:4], np.array([3, 1, 2, 0]), (4,)))

  return wide


def _qp_host() -> al.Function:
  """A host function whose graph reaches a solver through a nested call."""
  mu = al.sym("mu", 2)
  qp = _legacy_qp(P=al.const(np.eye(2)), c=-mu, name="corpus_qp")
  x = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])[0]
  return al.Function._from_exprs("qp_host", [mu], [al.sumsqr(x)], ["mu"], ["cost"])


CORPUS = {
  "forward": _dynamics,
  "jac": lambda: al.jacobian(_dynamics(), "znext", "z"),
  "vmap": _shooting,
  "spjac": lambda: al.sparse_jacobian(_shooting(), "eq", "z"),
  "wide": _wide,
}
SOLVER_CORPUS = {"solver": _qp_host}


def _rendered(fun: al.Function) -> dict[str, str]:
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
    print("skipping the solver entry: the alloy-piqp plugin is not installed")
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
