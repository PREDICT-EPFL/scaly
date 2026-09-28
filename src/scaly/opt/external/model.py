"""Solver descriptors: the extern callee of a solver Function, and the Function they build."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...function import ConcreteFunction
from ...function.extern import BuildRequirements, ExternRenderCtx, ExternSource, ExternState, extern_function
from ...function.tree import Tree, _G, flat_tree
from ...ir.types import SparsityType, TensorType
from ...utils.names import c_ident
from ..method import Info
from . import wrapper


@dataclass(frozen=True, slots=True)
class ExternalOracle:
  """A C-ABI oracle supplied by a plugin consumer instead of an Scaly graph.

  ``source`` must define ``raw_symbol`` with the same flat-buffer convention
  as generated Scaly kernels: one ``const double*`` per input, one ``double*``
  per output, and a trailing ``double*`` workspace argument. ``workspace_size``
  declares the number of doubles available through that final argument.
  """

  name: str
  raw_symbol: str
  source: str
  input_signature: tuple[tuple[str, tuple[int, ...]], ...]
  output_signature: tuple[tuple[str, tuple[int, ...]], ...]
  workspace_size: int = 0

  def __post_init__(self) -> None:
    if self.workspace_size < 0:
      raise ValueError(f"external oracle workspace_size must be nonnegative, got {self.workspace_size}")


@dataclass(frozen=True)
class SolverDescriptor:
  """Everything a solver plugin's generated C wrapper needs to drive a solve, and the extern callee
  (``scaly.function.extern``) of the solver Function it builds.

  Stored as the ``extern`` attr on every ``ExprOp.EXTERN_CALL`` node, and as ``Function.extern``,
  so that nodes for different outputs of the same solve share one identity. Frozen + identity
  hash (via ``id``) so it can live inside ``Expr.attrs`` without surprising
  structural equality.
  """

  name: str
  backend: str  # the method's short name ("piqp" for the ``scaly.methods`` entry ``opt.piqp``)
  n: int
  n_eq: int
  n_ineq: int
  # call-time input signature, in call order
  input_signature: tuple[tuple[str, tuple[int, ...]], ...]
  # output names + shapes, in solve-result order
  output_signature: tuple[tuple[str, tuple[int, ...]], ...]
  # param names (ordered parameter leaves after the fixed warm-start groups)
  param_names: tuple[str, ...]
  # Number of variable leaves at each end of the typed solver signature.
  n_var_blocks: int
  # Functions
  oracle: ConcreteFunction | None = None  # QP only
  # QP only: the objective's constant as a Function of the parameters, when it is not zero. The
  # solver sees 1/2 x'Px + c'x; core's frame adds this to the objective it reports.
  objective_constant: ConcreteFunction | None = None
  base: ConcreteFunction | ExternalOracle | None = None  # NLP only
  grad: ConcreteFunction | ExternalOracle | None = None
  jac: ConcreteFunction | ExternalOracle | None = None
  hess: ConcreteFunction | ExternalOracle | None = None
  bounds: ConcreteFunction | ExternalOracle | None = None
  # Sparsity (NLP)
  jac_sparsity: SparsityType | None = None
  hess_sparsity: SparsityType | None = None
  # Sparse QP (PIQP sparse interface): structural CSC patterns of P (upper
  # triangle), A_eq, G_ineq, baked into the generated wrapper as static
  # tables; the oracle emits compact CSC-ordered value buffers. None => dense.
  sparse: bool = False
  P_sparsity: SparsityType | None = None
  A_sparsity: SparsityType | None = None
  G_sparsity: SparsityType | None = None
  # Solver-specific options
  options: tuple[tuple[str, Any], ...] = ()
  # The external method (``opt.external.External``) whose wrapper, header and library this uses;
  # the installed one with default options when left out.
  method: Any = field(default=None, hash=False, compare=False, repr=False)
  # Oracle output naming (QP); the order in which the oracle's outputs encode
  # the QP data buffers.
  oracle_output_names: tuple[str, ...] = ()
  # Hash key used as a stable identifier (set in __post_init__)
  _key: int = field(default=0, hash=False, compare=False, repr=False)
  # Mutable per-descriptor runtime state (e.g. PIQP workspace handle). Frozen
  # is fine — we mutate the dict's contents, not the binding itself.
  runtime: dict[str, Any] = field(default_factory=dict, hash=False, compare=False, repr=False)

  def __post_init__(self) -> None:
    object.__setattr__(self, "_key", id(self))

  def __hash__(self) -> int:
    return self._key

  def __eq__(self, other: object) -> bool:
    return self is other

  def structural_key(self) -> tuple[Any, ...]:
    """Stable per-instance key used by ``Expr.structural_key`` for EXTERN_CALL nodes."""
    return ("SolverDescriptor", self._key, self.name, self.backend, self.n, self.n_eq, self.n_ineq)

  def dependencies(self) -> tuple[ConcreteFunction, ...]:
    """The oracle and derivative Functions the wrapper calls, deduplicated, in descriptor order."""
    out: list[ConcreteFunction] = []
    for candidate in (self.oracle, self.base, self.grad, self.jac, self.hess, self.bounds, self.objective_constant):
      if isinstance(candidate, ConcreteFunction) and candidate not in out:
        out.append(candidate)
    return tuple(out)

  def external_oracles(self) -> tuple[ExternalOracle, ...]:
    """The oracles supplied as C instead of as Functions, deduplicated by identity."""
    out: list[ExternalOracle] = []
    for oracle in (self.base, self.grad, self.jac, self.hess, self.bounds):
      if isinstance(oracle, ExternalOracle) and oracle not in out:
        out.append(oracle)
    return tuple(out)

  def extern_sources(self) -> tuple[ExternSource, ...]:
    return tuple(ExternSource(o.raw_symbol, o.source, o.workspace_size) for o in self.external_oracles())

  def render(self, fun: ConcreteFunction, ctx: ExternRenderCtx) -> list[str]:
    return wrapper.render_solver(fun, self, ctx)

  def build_requirements(self, fun: ConcreteFunction) -> BuildRequirements:
    return wrapper.solver_requirements(c_ident(fun.name), self)

  def state(self, fun: ConcreteFunction) -> ExternState:
    return wrapper.solver_state(c_ident(fun.name), fun.name)


def descriptor_function(
  descriptor: SolverDescriptor,
  input_tree: _G | None = None,
  output_tree: Tree[Any, Any] | None = None,
) -> ConcreteFunction[Any, Any, Any, Any]:
  """Build the plain Function whose opaque outputs share ``descriptor``.

  ``input_tree`` is its parameter list: for a solver, the five slots of the warm start, the
  multipliers and the parameters. Without one the inputs are a single group, as for ``from_exprs``.
  The outputs are the descriptor's (``output_tree``, or flat) and then an ``opt.Info``, which the
  wrapper frame fills from the solver's statistics (``wrapper.render_solver``).
  """
  info = Info.tree()
  signature = descriptor.output_signature
  solution = output_tree or flat_tree(tuple(n for n, _ in signature), tuple(TensorType(s, diff=False) for _, s in signature))
  parts = solution.parts if isinstance(solution, _G) else (solution,)
  outputs = (*signature, *zip(info.names, info.shapes, strict=True))
  return extern_function(
    descriptor.name, descriptor, descriptor.input_signature, outputs, input_tree=input_tree, output_tree=_G((*parts, info), public=False)
  )
