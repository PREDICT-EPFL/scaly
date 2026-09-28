"""The AC-OPF study's model in Scaly: rosetta-opf's polar formulation, term for term, over PowerModels' data.

rosetta-opf (lanl-ansi/rosetta-opf, `jump.jl`) states the AC optimal power flow the way PowerModels'
reference model does, and the ExaModels paper benchmarks that model:

- variables: voltage angle `va` and magnitude `vm` per bus, generation `pg`, `qg` per generator, and the
  flows `p`, `q` at both ends of every branch (the "arcs");
- objective: the generators' quadratic costs;
- equalities: the reference bus's angle, active and reactive balance per bus (flows out equal
  generation less load less shunt), and the four flow equations of each branch (pi model with tap and
  phase shift);
- inequalities: each branch's angle-difference window and its thermal limit `p^2 + q^2 <= rate^2` at
  both ends; bounds on `vm`, `pg`, `qg` and on every flow (`|p|, |q| <= rate`).

`load_case` reads the case as `baseline/export_case.jl` writes it: PowerModels' processing (per-unit
values, corrected angle limits, thermal limits where the file has none) with every collection in the
order JuMP's model iterates it, so the two models have the same variables in the same order. Every
term is written over all buses, generators or branches at once with `sc.gather` and `sc.segment_sum`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import scaly as sc


@dataclass
class Case:
  name: str
  n_bus: int
  n_gen: int
  n_branch: int
  vmin: np.ndarray
  vmax: np.ndarray
  pd: np.ndarray
  qd: np.ndarray
  gs: np.ndarray
  bs: np.ndarray
  gen_bus: np.ndarray  # bus position of each generator
  pmin: np.ndarray
  pmax: np.ndarray
  qmin: np.ndarray
  qmax: np.ndarray
  cost: np.ndarray  # (n_gen, 3): c2, c1, c0 for pg in per unit
  f: np.ndarray  # bus positions of each branch's ends
  t: np.ndarray
  f_arc: np.ndarray  # arc positions of each branch's two ends
  t_arc: np.ndarray
  arc_bus: np.ndarray  # bus position of each arc
  arc_rate: np.ndarray
  g: np.ndarray
  b: np.ndarray
  tr: np.ndarray
  ti: np.ndarray
  g_fr: np.ndarray
  b_fr: np.ndarray
  g_to: np.ndarray
  b_to: np.ndarray
  angmin: np.ndarray
  angmax: np.ndarray
  rate: np.ndarray
  ref: np.ndarray  # bus positions of the reference buses

  @property
  def n_arc(self) -> int:
    return 2 * self.n_branch


def load_case(path: Path) -> Case:
  d = json.loads(Path(path).read_text())
  bus_pos = {b["id"]: k for k, b in enumerate(d["bus"])}
  arc_pos = {tuple(a): k for k, a in enumerate(d["arcs"])}
  br = d["branch"]
  f = np.array([bus_pos[x["f_bus"]] for x in br])
  t = np.array([bus_pos[x["t_bus"]] for x in br])
  f_arc = np.array([arc_pos[(x["id"], x["f_bus"], x["t_bus"])] for x in br])
  t_arc = np.array([arc_pos[(x["id"], x["t_bus"], x["f_bus"])] for x in br])
  branch_pos = {x["id"]: k for k, x in enumerate(br)}
  arr = lambda rows, key: np.array([r[key] for r in rows], dtype=np.float64)  # noqa: E731
  rate = arr(br, "rate_a")
  return Case(
    name=d["case"].removesuffix(".m"),
    n_bus=len(d["bus"]),
    n_gen=len(d["gen"]),
    n_branch=len(br),
    vmin=arr(d["bus"], "vmin"),
    vmax=arr(d["bus"], "vmax"),
    pd=arr(d["bus"], "pd"),
    qd=arr(d["bus"], "qd"),
    gs=arr(d["bus"], "gs"),
    bs=arr(d["bus"], "bs"),
    gen_bus=np.array([bus_pos[g["bus"]] for g in d["gen"]]),
    pmin=arr(d["gen"], "pmin"),
    pmax=arr(d["gen"], "pmax"),
    qmin=arr(d["gen"], "qmin"),
    qmax=arr(d["gen"], "qmax"),
    cost=np.array([g["cost"] for g in d["gen"]], dtype=np.float64),
    f=f,
    t=t,
    f_arc=f_arc,
    t_arc=t_arc,
    arc_bus=np.array([bus_pos[a[1]] for a in d["arcs"]]),
    arc_rate=np.array([rate[branch_pos[a[0]]] for a in d["arcs"]]),
    g=arr(br, "g"),
    b=arr(br, "b"),
    tr=arr(br, "tr"),
    ti=arr(br, "ti"),
    g_fr=arr(br, "g_fr"),
    b_fr=arr(br, "b_fr"),
    g_to=arr(br, "g_to"),
    b_to=arr(br, "b_to"),
    angmin=arr(br, "angmin"),
    angmax=arr(br, "angmax"),
    rate=rate,
    ref=np.array([bus_pos[i] for i in d["ref_buses"]]),
  )


def problem(c: Case) -> sc.opt.NLP:
  """rosetta-opf's model of case `c` as a Scaly problem, variables `(va, vm, pg, qg, p, q)`."""
  C = sc.const
  ttm = c.tr**2 + c.ti**2
  # The flow equations' coefficients, as jump.jl writes them.
  fr_vv, fr_c, fr_s = (c.g + c.g_fr) / ttm, (-c.g * c.tr + c.b * c.ti) / ttm, (-c.b * c.tr - c.g * c.ti) / ttm
  frq_vv = -(c.b + c.b_fr) / ttm
  to_vv, to_c, to_s = c.g + c.g_to, (-c.g * c.tr - c.b * c.ti) / ttm, (-c.b * c.tr + c.g * c.ti) / ttm
  toq_vv = -(c.b + c.b_to)

  @sc.opt.problem(vars=sc.G(sc.L("va", c.n_bus), sc.L("vm", c.n_bus), sc.L("pg", c.n_gen), sc.L("qg", c.n_gen), sc.L("p", c.n_arc), sc.L("q", c.n_arc)), name=f"acopf_{c.name}")
  def opf(variables):
    va, vm, pg, qg, p, q = variables
    vm_fr, vm_to = sc.gather(vm, c.f), sc.gather(vm, c.t)
    dva = sc.gather(va, c.f) - sc.gather(va, c.t)
    vv = vm_fr * vm_to
    cs, sn = vv * dva.cos(), vv * dva.sin()
    p_fr, q_fr, p_to, q_to = sc.gather(p, c.f_arc), sc.gather(q, c.f_arc), sc.gather(p, c.t_arc), sc.gather(q, c.t_arc)
    # cos(va_to - va_fr) = cos(dva), sin(va_to - va_fr) = -sin(dva)
    eq_p_fr = p_fr - (C(fr_vv) * vm_fr * vm_fr + C(fr_c) * cs + C(fr_s) * sn)
    eq_q_fr = q_fr - (C(frq_vv) * vm_fr * vm_fr - C(fr_s) * cs + C(fr_c) * sn)
    eq_p_to = p_to - (C(to_vv) * vm_to * vm_to + C(to_c) * cs - C(to_s) * sn)
    eq_q_to = q_to - (C(toq_vv) * vm_to * vm_to - C(to_s) * cs - C(to_c) * sn)
    gen_p, gen_q = sc.segment_sum(pg, c.gen_bus, c.n_bus), sc.segment_sum(qg, c.gen_bus, c.n_bus)
    bal_p = sc.segment_sum(p, c.arc_bus, c.n_bus) - gen_p + C(c.pd) + C(c.gs) * vm * vm
    bal_q = sc.segment_sum(q, c.arc_bus, c.n_bus) - gen_q + C(c.qd) - C(c.bs) * vm * vm
    cost = (C(c.cost[:, 0]) * pg * pg + C(c.cost[:, 1]) * pg + C(c.cost[:, 2])).sum()
    return sc.opt.ProblemSpec(
      minimize=cost,
      eq=(sc.gather(va, c.ref), bal_p, bal_q, eq_p_fr, eq_q_fr, eq_p_to, eq_q_to),
      ineq=(
        sc.opt.bounded(dva, lo=C(c.angmin), hi=C(c.angmax), name="angle_difference"),
        sc.opt.bounded(p * p + q * q, hi=C(c.arc_rate**2), name="thermal"),
      ),
      lb=(sc.opt.NO_LB, C(c.vmin), C(c.pmin), C(c.qmin), C(-c.arc_rate), C(-c.arc_rate)),
      ub=(sc.opt.NO_UB, C(c.vmax), C(c.pmax), C(c.qmax), C(c.arc_rate), C(c.arc_rate)),
    )

  return opf


def start(c: Case) -> tuple[np.ndarray, ...]:
  """jump.jl's starting point: `vm = 1`, every other variable at zero moved into its bounds (what
  Ipopt.jl does for a variable without a start value)."""
  clip = lambda lo, hi: np.clip(np.zeros_like(lo), lo, hi)  # noqa: E731
  return (np.zeros(c.n_bus), np.ones(c.n_bus), clip(c.pmin, c.pmax), clip(c.qmin, c.qmax), clip(-c.arc_rate, c.arc_rate), clip(-c.arc_rate, c.arc_rate))
