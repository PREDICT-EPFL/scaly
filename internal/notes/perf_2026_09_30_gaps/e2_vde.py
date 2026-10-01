"""A8 (Tier 8): the E2 case study's sensitivity kernel with the network inline and called.

acados' explicit integrator calls ``expl_vde_forw``: the model's value and its Jacobians applied
to the sensitivities. With the network written into the body the forward pass is computed once;
with the network called as a ``Function``, the call's own value and the derivative of the call
each compute it (CS-10). This times both, per call, on random weights.

  uv run python internal/notes/perf_2026_09_30_gaps/e2_vde.py [--sizes 2x16,5x128]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
NX, NU = 2, 1


def _models(sc, nn, net, layers: int, width: int):
  """The model's right-hand side with the network written inline, and with it called."""

  def network(x):
    return nn.mlp(x, net)

  model = sc.function(sc.L("x", NX), name=f"net_{layers}_{width}")(network)

  def inline(x, u, p):
    return sc.concat([x[1:2], u]) + nn.mlp(x, net)

  def called(x, u, p):
    return sc.concat([x[1:2], u]) + model(x)

  return (("inline", inline), ("called", called))


def main() -> None:
  import scaly as sc
  from e2_surrogate import best, params
  from scaly import export, nn

  parser = argparse.ArgumentParser()
  parser.add_argument("--sizes", default="2x16,2x128,5x16,5x128,12x32")
  args = parser.parse_args()
  rng = np.random.default_rng(0)
  print(f"{'layers x width':>14s} {'inline':>10s} {'called':>10s} {'called / inline':>15s}")
  for size in args.sizes.split(","):
    layers, width = (int(v) for v in size.split("x"))
    net = params(layers, width, rng)
    xdots = _models(sc, nn, net, layers, width)

    point = (rng.standard_normal(NX), rng.standard_normal(NX * NX), rng.standard_normal(NX * NU), rng.standard_normal(NU), np.zeros(0))
    times, outs = [], []
    for label, xdot in xdots:
      fn = export.acados_functions(xdot, NX, NU, name=f"vde_{label}_{layers}_{width}")[f"vde_{label}_{layers}_{width}_expl_vde_forw"].concrete
      outs.append(np.concatenate([np.ravel(o) for o in fn._flat_numerical_call(*point)]))
      times.append(best(lambda fn=fn: fn._flat_numerical_call(*point), floor=0.1))
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-10, atol=1e-12)
    print(f"{size:>14s} {times[0] * 1e6:8.1f} us {times[1] * 1e6:8.1f} us {times[1] / times[0]:15.2f}", flush=True)


if __name__ == "__main__":
  main()
