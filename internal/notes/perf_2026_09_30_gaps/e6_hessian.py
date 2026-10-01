"""A10 (Tier 8): the AC-OPF case study's Lagrangian Hessian on synthetic networks, Scaly alone.

The case study (``examples/case_studies/acopf``) evaluates the Hessian by colouring: one
forward-over-reverse sweep of the whole model for each colour, and the colours follow the largest
bus degree. Its large PGLib cases need the Julia baseline to export them, so this builds networks
of the same model with the two things that matter set by hand: the number of buses, and the
degree of a few hub buses (the rest are a random tree with half as many chords again, 1.5 branches
a bus, as ``case118_ieee`` has). For each it reports the variables, the Hessian's nonzeros and
colours, the size of the generated C, the seconds to render and to compile it, and the time of
one evaluation, also per nonzero and per variable and colour: the first is what a local-derivative
scheme scales with, the second what colouring does. Then the constraint Jacobian's nonzeros,
colours and time.

  uv run python internal/notes/perf_2026_09_30_gaps/e6_hessian.py [--sizes 300x12,300x40,1200x12,1200x40]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "case_studies" / "acopf"))


def network(n_bus: int, hub_degree: int, seed: int = 0):
  """A connected network of ``n_bus`` buses: a random tree, half as many chords again, and four
  hubs joined to ``hub_degree`` buses each; a generator at every fifth bus."""
  import scaly_impl as si

  rng = np.random.default_rng(seed)
  edges = {(int(rng.integers(0, k)), k) for k in range(1, n_bus)}
  while len(edges) < int(1.5 * n_bus) - 4 * hub_degree:
    a, b = (int(v) for v in rng.integers(0, n_bus, 2))
    if a != b:
      edges.add((min(a, b), max(a, b)))
  for hub in rng.choice(n_bus, 4, replace=False):
    for other in rng.choice(n_bus, hub_degree, replace=False):
      if int(other) != int(hub):
        edges.add((min(int(hub), int(other)), max(int(hub), int(other))))
  f, t = (np.array(v) for v in zip(*sorted(edges), strict=True))
  n_branch = len(f)
  gen_bus = np.arange(0, n_bus, 5)
  n_gen = len(gen_bus)
  rate = rng.uniform(1.0, 5.0, n_branch)
  u = lambda lo, hi, n: rng.uniform(lo, hi, n)  # noqa: E731
  return si.Case(
    name=f"synthetic_{n_bus}_{hub_degree}",
    n_bus=n_bus,
    n_gen=n_gen,
    n_branch=n_branch,
    vmin=np.full(n_bus, 0.94),
    vmax=np.full(n_bus, 1.06),
    pd=u(0.0, 1.0, n_bus),
    qd=u(0.0, 0.3, n_bus),
    gs=np.zeros(n_bus),
    bs=u(0.0, 0.1, n_bus),
    gen_bus=gen_bus,
    pmin=np.zeros(n_gen),
    pmax=np.full(n_gen, 8.0),
    qmin=np.full(n_gen, -3.0),
    qmax=np.full(n_gen, 3.0),
    cost=np.stack([u(0.0, 0.1, n_gen), u(10.0, 40.0, n_gen), np.zeros(n_gen)], axis=1),
    f=f,
    t=t,
    f_arc=np.arange(n_branch),
    t_arc=n_branch + np.arange(n_branch),
    arc_bus=np.concatenate([f, t]),
    arc_rate=np.concatenate([rate, rate]),
    g=u(1.0, 10.0, n_branch),
    b=-u(5.0, 30.0, n_branch),
    tr=u(0.95, 1.05, n_branch),
    ti=u(-0.02, 0.02, n_branch),
    g_fr=np.zeros(n_branch),
    b_fr=u(0.0, 0.05, n_branch),
    g_to=np.zeros(n_branch),
    b_to=u(0.0, 0.05, n_branch),
    angmin=np.full(n_branch, -np.pi / 6),
    angmax=np.full(n_branch, np.pi / 6),
    rate=rate,
    ref=np.array([0]),
  )


def main() -> None:
  import scaly as sc
  import scaly_impl as si
  from scaly.codegen import render_c_source

  parser = argparse.ArgumentParser()
  parser.add_argument("--sizes", default="300x12,300x40,1200x12,1200x40")
  parser.add_argument("--repeats", type=int, default=20)
  args = parser.parse_args()
  print(
    f"{'buses x hub':>12s} {'variables':>9s} {'nonzeros':>9s} {'colours':>7s} {'build s':>8s} {'C MB':>7s} {'render s':>8s} {'compile s':>9s}"
    f" {'call ms':>8s} {'ns / nonzero':>12s} {'ns / (variable x colour)':>24s} {'jac nonzeros':>12s} {'colours':>7s} {'call ms':>8s}"
  )
  for size in args.sizes.split(","):
    n_bus, hub = (int(v) for v in size.split("x"))
    case = network(n_bus, hub)
    t0 = time.perf_counter()
    oracles = sc.opt.nlp_oracles(si.problem(case))
    hess = oracles.hess.triangle("lower")
    inputs = list(oracles.hess_inputs)
    fn = sc.Function.from_exprs(f"e6_hess_{n_bus}_{hub}", inputs, [hess.values], [str(e.name) for e in inputs], ["h"]).concrete
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    megabytes = len(render_c_source(fn)) / 1e6
    render = time.perf_counter() - t0
    rng = np.random.default_rng(1)
    values = [rng.uniform(0.9, 1.1, e.shape) for e in inputs]
    t0 = time.perf_counter()
    fn._flat_numerical_call(*values)
    compile_s = time.perf_counter() - t0 - render
    calls = []
    for _ in range(args.repeats):
      t0 = time.perf_counter()
      fn._flat_numerical_call(*values)
      calls.append(time.perf_counter() - t0)
    call = min(calls)
    n_var, nnz, colours = int(oracles.x.size), len(hess.sparsity.rows), int(hess.coloring_width or 0)
    jac = oracles.jac
    assert jac is not None
    jac_values = values[: len(jac.inputs)]
    jac._flat_numerical_call(*jac_values)
    jac_calls = []
    for _ in range(args.repeats):
      t0 = time.perf_counter()
      jac._flat_numerical_call(*jac_values)
      jac_calls.append(time.perf_counter() - t0)
    widths = jac.output_coloring_widths
    print(
      f"{size:>12s} {n_var:9d} {nnz:9d} {colours:7d} {build:8.1f} {megabytes:7.1f} {render:8.1f} {max(compile_s, 0.0):9.1f}"
      f" {call * 1e3:8.3f} {call * 1e9 / nnz:12.1f} {call * 1e9 / (n_var * max(colours, 1)):24.2f}"
      f" {oracles.jac_sparsity.nnz:12d} {int(widths[0] or 0) if widths else 0:7d} {min(jac_calls) * 1e3:8.3f}",
      flush=True,
    )


if __name__ == "__main__":
  main()
