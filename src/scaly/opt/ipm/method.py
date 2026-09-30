"""``IPM``: the opt method that generates PIQP's interior-point algorithm for a quadratic problem, with the signature every opt solver has."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import numpy as np

from ...function.method import Status, Support
from ...function.model import ConcreteFunction
from ...function.tree import G, L, param_list
from ...ir.expr import Expr, equal, substitute, where
from ...ir.types import TensorType
from ..method import METHOD_API, Info
from ..nlp import nlp_oracles
from ..problem import NLP
from ..qp import NotQuadratic, extract_qp, prove_qp
from .algorithm import DUAL_INFEASIBLE, INFO_FIELDS, MAX_ITER_REACHED, NUMERICS, PRIMAL_INFEASIBLE, SOLVED, Settings, Solver
from .cost import choose_backend
from .structure import QPStructure, QPValues

_STATUS = {
  SOLVED: Status.OK,
  MAX_ITER_REACHED: Status.MAX_ITER,
  PRIMAL_INFEASIBLE: Status.PRIMAL_INFEASIBLE,
  DUAL_INFEASIBLE: Status.DUAL_INFEASIBLE,
  NUMERICS: Status.NUMERICS,
}
_SETTINGS = {f.name for f in fields(Settings)}


def _status(code: Expr) -> Expr:
  """PIQP's status code as a ``Status``; anything else (``INVALID_BOUNDS``) is ``ERROR``."""
  out: Expr | float = float(Status.ERROR)
  for piqp, status in _STATUS.items():
    out = where(equal(code, float(piqp)), float(status), out)
  return out if isinstance(out, Expr) else Expr.const(out)


@dataclass(frozen=True)
class IPM:
  """PIQP 0.6.2's proximal interior-point algorithm (Ruiz equilibration, the initial point, the
  Mehrotra loop with proximal updates, unscaling), generated as C specialised to the problem's
  sparsity and to which of its bounds are finite: no solver library behind it.

  A QP method, as ``PIQP`` is: it solves a problem Scaly proves quadratic. ``sparse=True`` factors
  the KKT system whole with ``linalg.SparseLDL``; ``sparse=False`` condenses it and factors it by a
  dense Cholesky; None, the default, chooses the one whose iteration costs less, from the problem's
  structure alone, for the target in force when the solver is built (``opt.ipm.cost``): both
  backends follow PIQP's path, so only their speed differs. ``options`` are PIQP's settings by
  name (``Settings``: ``eps_abs``, ``max_iter``, ...), so a PIQP method's options carry over;
  PIQP's ``verbose`` is accepted and has no effect. Like PIQP it takes no warm start."""

  name: ClassVar[str] = "opt.ipm"
  problem: ClassVar[type] = NLP
  api: ClassVar[int] = METHOD_API

  sparse: bool | None = None
  options: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    object.__setattr__(self, "options", dict(self.options))
    unknown = sorted(set(self.options) - _SETTINGS - {"verbose"})
    if unknown:
      raise TypeError(f"IPM takes PIQP's settings, not {unknown}; the settings are {sorted(_SETTINGS)}")
    Settings(**{k: v for k, v in self.options.items() if k != "verbose"})  # checks the values

  @property
  def settings(self) -> Settings:
    """The options as the generated solver's ``Settings``."""
    return Settings(**{k: v for k, v in self.options.items() if k != "verbose"})

  def supports(self, problem: Any) -> Support:
    if not isinstance(problem, NLP):
      return Support((f"{type(problem).__name__} is not an opt problem",))
    try:
      prove_qp(problem)
    except NotQuadratic as exc:
      return Support((str(exc),))
    return Support()

  def build(self, problem: NLP[Any, Any, Any, Any], *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    form = extract_qp(problem)
    n, p, m = form.P.shape[0], form.A.shape[0], form.G.shape[0]
    params = list(form.params)
    # The patterns ``PIQP(sparse=True)`` finds, and which bounds are finite at one random probe: an
    # entry or bound that depends on a parameter counts as present whatever value it takes.
    P_sp, A_sp, G_sp = form.patterns(name)
    probe = ConcreteFunction.from_exprs(
      f"{name}_probe", params, [form.g_lb, form.g_ub, form.x_lb, form.x_ub], form.param_names, ("g_lb", "g_ub", "x_lb", "x_ub")
    )
    rng = np.random.default_rng(0)
    bounds = probe.numerical_call(*probe.input_tree.unflatten(tuple(rng.standard_normal(e.shape) for e in params)))
    h_l, h_u, x_l, x_u = (np.asarray(v, dtype=float) for v in bounds)
    A_pat, G_pat = (np.zeros(shape, dtype=bool) if sp is None else sp for sp, shape in ((A_sp, (p, n)), (G_sp, (m, n))))
    s = QPStructure.from_patterns(P_sp, A_pat, G_pat, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)
    data = {
      "P": form.in_pattern(form.P, P_sp),
      "c": form.c,
      "A": form.in_pattern(form.A, A_sp) if A_sp is not None else Expr.const(np.zeros(0)),
      "b": form.b,
      "G": form.in_pattern(form.G, G_sp) if G_sp is not None else Expr.const(np.zeros(0)),
      "h_l": form.g_lb,
      "h_u": form.g_ub,
      "x_l": form.x_lb,
      "x_u": form.x_ub,
    }
    oracles = nlp_oracles(problem)
    objective = oracles.base.outputs[0]  # the problem's own objective, constants included
    backend = choose_backend(s) if self.sparse is None else "sparse" if self.sparse else "dense"
    settings = self.settings
    variables = problem.vars.with_types(tuple(TensorType(e.shape, e.type.dtype, e.type.sparsity, diff=False) for e in problem._var_symbols))
    sizes = [e.size for e in problem._var_symbols]
    shapes = [e.shape for e in problem._var_symbols]

    def blocks(flat: Expr) -> Any:
      offsets = np.cumsum([0, *sizes])
      return variables.unflatten(tuple(flat[a:b].reshape(shape) for a, b, shape in zip(offsets[:-1], offsets[1:], shapes, strict=True)))

    def body(_x0: Any, _lam_box0: Any, _lam_eq0: Expr, _lam_ineq0: Expr, values: Any) -> Any:
      # The form's data are expressions of the problem's parameter symbols; the Function has its own.
      swap = dict(zip(params, problem.params.flatten_symbolic(values, name), strict=True))
      out = Solver(s, backend, settings, name=name).solve(QPValues.preprocess(s, **{k: substitute(v, swap) for k, v in data.items()}))
      x = out["x"]
      info = Info(
        status=_status(out["status"]),
        iter=out["iter"],
        objective=substitute(objective, {**swap, oracles.x: x}),
        primal_residual=out["info"][INFO_FIELDS.index("primal_res")],
      )
      return blocks(x), blocks(out["z_bu"] - out["z_bl"]), out["y"], out["z_u"] - out["z_l"], info

    inputs = param_list(
      variables,
      variables.relabel("lam:"),
      L("lam_eq", TensorType((p,), diff=False)),
      L("lam_ineq", TensorType((m,), diff=False)),
      problem.params,
    )
    outputs = G(
      variables, variables.relabel("lam:"), L("lam_eq", TensorType((p,), diff=False)), L("lam_ineq", TensorType((m,), diff=False)), Info.tree()
    )
    return ConcreteFunction(name, body, inputs, outputs)


__all__ = ["IPM"]
