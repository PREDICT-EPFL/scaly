"""A9 (Tier 8): the E2 case study's surrogate kernel without its PyTorch side.

RTN-MPC's surrogate evaluates a network and its Jacobian at the ten shooting nodes. Scaly maps the
node over the ten (``examples/case_studies/neural_mpc/scaly_impl.py``: ``taylor_function``), so each
layer's weights are read once per node and per tangent column; a batched library multiplies each
layer by all the nodes' columns at once. This times Scaly's kernel per call at the study's sizes,
with random weights, beside NumPy's batched float64 form of the same computation (one matrix
product per layer over the 30 columns, on the machine's BLAS), and checks the two agree.

  uv run python internal/notes/perf_2026_09_30_gaps/e2_surrogate.py [--sizes 2x16,5x128,12x512]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "case_studies" / "neural_mpc"))
NX, NODES = 2, 10


def params(layers: int, width: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
  widths = [NX, *([width] * layers), NX]
  return [(rng.standard_normal((o, i)) / np.sqrt(i), rng.standard_normal(o) * 0.1) for i, o in zip(widths[:-1], widths[1:], strict=True)]


def batched(net, xs: np.ndarray) -> np.ndarray:
  """The same rows by one product per layer: the value and the two tangents of every node together."""
  h = xs.reshape(NODES, NX).T  # (nx, nodes)
  d = np.broadcast_to(np.eye(NX)[:, :, None], (NX, NX, NODES)).copy()  # (nx, seed, nodes)
  for k, (w, b) in enumerate(net):
    z = w @ h + b[:, None]
    dz = (w @ d.reshape(d.shape[0], NX * NODES)).reshape(w.shape[0], NX, NODES)
    if k < len(net) - 1:
      h = np.tanh(z)
      d = dz * (1.0 - h * h)[:, None, :]
    else:
      h, d = z, dz
  rows = [np.concatenate([xs.reshape(NODES, NX)[n], h[:, n], d[:, :, n].T.reshape(-1)]) for n in range(NODES)]
  return np.concatenate(rows)


def best(call, floor: float = 0.2) -> float:
  call()
  times = []
  for _ in range(5):
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < floor:
      call()
      n += 1
    times.append((time.perf_counter() - t0) / n)
  return min(times)


def main() -> None:
  import scaly_impl

  parser = argparse.ArgumentParser()
  parser.add_argument("--sizes", default="2x16,2x128,5x16,5x128,12x32,12x512")
  args = parser.parse_args()
  rng = np.random.default_rng(0)
  print(
    f"{'layers x width':>14s} {'scaly':>11s} {'numpy batched':>14s} {'ratio':>7s} {'build s':>8s} {'first call s':>12s}   per call; ratio is scaly / numpy"
  )
  for size in args.sizes.split(","):
    layers, width = (int(v) for v in size.split("x"))
    net = params(layers, width, rng)
    xs = rng.standard_normal(NODES * NX)
    t0 = time.perf_counter()
    fn = scaly_impl.taylor_function(net).concrete
    t1 = time.perf_counter()
    got = fn._flat_numerical_call(xs)[0]
    t2 = time.perf_counter()
    want = batched(net, xs)
    np.testing.assert_allclose(np.asarray(got).reshape(-1), want, rtol=1e-9, atol=1e-9)
    scaly, numpy = best(lambda fn=fn, xs=xs: fn._flat_numerical_call(xs)), best(lambda net=net, xs=xs: batched(net, xs))
    print(f"{size:>14s} {scaly * 1e6:9.1f} us {numpy * 1e6:11.1f} us {scaly / numpy:7.2f} {t1 - t0:8.2f} {t2 - t1:12.2f}", flush=True)


if __name__ == "__main__":
  main()
