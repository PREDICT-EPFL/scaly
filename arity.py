"""Prototype of the typed pytree API for D3 (internal/notes/refactorings.md, "Solver problem construction").

Sketch only: ``Expr`` and ``Buffer`` are stand-ins and nothing lowers. It exists to prove, before the
refactoring starts, that the design behaves exactly as expected, both at type-check time
(``uv run ty check --error-on-warning arity.py``: every ``ty: ignore`` marks an expected error, and an
unused one fails the check) and at run time (``uv run arity.py`` runs every ``test_*``).

What the design guarantees, and where each guarantee is tested:

- The declared structure is the source of truth; nothing is inferred from the body's signature. Ty
  checks that decorator and body agree, and rejects calls with the wrong structure or count
  (typing tests, "functions").
- Symbolic and numeric calls share one structure with different leaf types, ``Expr`` and
  ``Buffer``; mixing them is a type error ("functions"). Typed numeric calls are the connection to
  the rest of a user's code and are not negotiable.
- Structures are built from two names: ``L(name, shape)`` and ``G(*trees)`` at any width from 2 to 8
  and any depth. Grouping is for readability; grouped and flat declarations of the same leaves are
  the same C signature (``test_grouping_is_not_a_signature``). ``G``'s width overloads are the one
  ladder in the design: deriving the ``Buffer`` structure from the ``Expr`` structure needs a
  type-level map and Python has none, which is also why a bare ``("u", 2)`` literal cannot be the
  declaration (its type is ``tuple[str, int]`` whatever classes exist).
- Names are metadata, one per leaf, unique within a tree, never how a value is addressed. They
  spell the generated header's typed buffers and sparse tables and the text asm, and they are the
  ``of``/``wrt`` strings of the derivative wrappers. The name a body binds is independent of the
  declared one (``blocked_quadratic`` binds ``vs``/``sc``). Unknown ``of``/``wrt`` fails when the
  derivative is built, with the declared choices in the message, never at evaluation time.
- Output shapes may be declared ``...`` and are then traced; a written shape is checked against the
  trace. Wrong output count or shape fails at the decorator.
- Derivatives keep the source's inputs and are therefore typed: ``gradient`` is
  ``Function[SI, NI, Expr, Buffer]``. Seeded modes pair the inputs with the new group,
  ``forward``/``adjoint`` with a seed, ``lagrangian_hessian`` with multipliers typed as the source's
  output tree, one multiplier per output.
- A problem is a ``ProblemSpec`` of expressions (cost, equality groups, bounded inequality groups,
  box bounds shaped like the variables) returned from a body traced over declared ``vars`` and
  ``params``; ``Problem`` carries the spec and both trees. Multi-block variables are just a grouped
  tree, and the solution and warm start have that structure.
- A solver is a plain ``Function`` with inputs ``(vars_init, lam_box0, lam_eq0, lam_ineq0, params)``
  and outputs ``(vars, lam_box, lam_eq, lam_ineq)``. Multipliers are always present, size 0 when the
  category is absent, so the signature never depends on the problem. Box multipliers have the
  variables' structure. The backend is positional; ``name`` belongs to the solver call.
- A QP backend is gated by a structural proof that the cost is quadratic and the constraints affine
  in the variables; the sketch tracks polynomial degree on ``Expr`` for it. An NLP backend takes any
  spec. ``qp_problem(n, n_eq, n_ineq)`` is the typed data form and goes through the same proof.
- Shape mismatches at call time are runtime ``ValueError``s; the type system checks structure, not
  shape, and that was accepted from the start.

How to read this file when implementing it. It is an interface sketch, not a reference implementation:
every body below is a stand-in, and the real ``src/alloy`` machinery behind each name must be kept and
have its interface swapped, not be rewritten from this file. Specifically:

- ``Buffer`` is ``np.ndarray``. It exists here only so ty can tell the numeric side from the symbolic
  one; there is no new runtime type to add.
- ``Expr`` is ``alloy.Expr``; the stub's ``degree`` field stands in for the structural dependency
  analysis that ``ad/sparsity.py``'s ``_jac_mask`` already does. The QP proof in ``solver`` must be
  written on ``_jac_mask`` over the real gradient/Hessian expressions, not on a degree counter.
- ``Function.__init__`` traces the body exactly as ``@al.function`` does today; ``symbolic_call`` is
  today's ``Function.call`` (a CALL node) and ``numerical_call`` is today's ``__call__`` through the
  JIT. Here they only check structure and fabricate buffers of the right shape.
- ``gradient``, ``jacobian``, ``hessian``, ``forward``, ``adjoint``, ``lagrangian_hessian`` are
  shape bookkeeping here. The real ones are the existing wrappers in ``function/api.py`` and the
  specs in ``function/factory.py`` (``Grad``, ``Jac``, ``Hess``, ``Fwd``, ``Adj``, ``SpHess`` with its
  ``triangle``), with two interface changes: the result is typed as this file says, and
  ``extra_inputs`` is gone because the source's whole input tree is kept.
- ``solver`` builds the descriptor and ``SOLVER_CALL`` nodes exactly as ``solvers/nlp.py`` and
  ``solvers/qp.py`` do today (base function, gradient, sparse Jacobian, sparse Lagrangian Hessian
  with ``backend.hess_triangle``, bounds function, the plugin registry). Here its body returns its
  inputs. The multipliers, the size-0 convention and the input order are the real ones.
- ``BACKENDS`` is the entry-point registry in ``solvers/registry.py``; ``NotQuadratic`` is the one
  new error.
- ``ProblemSpec.eq``/``ineq`` groups are concatenated into today's ``h_eq``/``g_ineq`` with
  ``l_ineq``/``u_ineq``; ``lb``/``ub`` become today's ``x_lb``/``x_ub`` over the concatenated
  variable. Multi-block ``vars`` need one new IR utility: substituting each block symbol with a
  slice of one internal decision symbol.
- ``c_signature`` is illustrative; the real header comes from ``codegen/aot.py`` unchanged, with the
  leaf names in tree order.

Left for the implementation, not the sketch: ``L`` taking a ``TensorType`` for dtype and ``diff``;
``params=None`` inference for a problem built inside an ``@al.function`` body; derivative caching on
``Problem`` and the Hessian triangle per backend; ``vmap`` typed by its callee's trees; the real
solver outputs (stats, status).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast, overload

import numpy as np

########################################################################################################
# Expr and Buffer stand-ins
########################################################################################################

type Shape = tuple[int, ...]
type ShapeDecl = Shape | EllipsisType


def _as_shape(shape: int | Shape) -> Shape:
  return (shape,) if isinstance(shape, int) else shape


def _size(shape: Shape) -> int:
  return int(np.prod(shape)) if shape else 1


def _degree_add(a: int | None, b: int | None) -> int | None:
  return None if a is None or b is None else max(a, b)


def _degree_mul(a: int | None, b: int | None) -> int | None:
  return None if a is None or b is None else a + b


@dataclass(frozen=True)
class Expr:
  """A symbolic tensor. ``degree`` is its polynomial degree in the problem variables (``None`` when
  not polynomial); it stands in for the structural dependency analysis the real proof uses."""

  shape: Shape
  name: str | None = None
  degree: int | None = 0

  @property
  def size(self) -> int:
    return _size(self.shape)

  def _binary(self, other: Expr | float, degree: int | None) -> Expr:
    other_shape = other.shape if isinstance(other, Expr) else ()
    if self.shape and other_shape and self.shape != other_shape:
      raise ValueError(f"shape mismatch {self.shape} vs {other_shape}")
    return Expr(self.shape or other_shape, degree=degree)

  def _degree_of(self, other: Expr | float) -> int | None:
    return other.degree if isinstance(other, Expr) else 0

  def __add__(self, other: Expr | float) -> Expr:
    return self._binary(other, _degree_add(self.degree, self._degree_of(other)))

  __radd__ = __add__

  def __sub__(self, other: Expr | float) -> Expr:
    return self._binary(other, _degree_add(self.degree, self._degree_of(other)))

  __rsub__ = __sub__

  def __neg__(self) -> Expr:
    return Expr(self.shape, degree=self.degree)

  def __mul__(self, other: Expr | float) -> Expr:
    return self._binary(other, _degree_mul(self.degree, self._degree_of(other)))

  __rmul__ = __mul__

  def __truediv__(self, other: Expr | float) -> Expr:
    other_degree = self._degree_of(other)
    return self._binary(other, self.degree if other_degree == 0 else None)

  def __pow__(self, power: int) -> Expr:
    return Expr(self.shape, degree=None if self.degree is None else self.degree * power)

  def __matmul__(self, other: Expr) -> Expr:
    if len(self.shape) == 2 and len(other.shape) == 1 and self.shape[1] == other.shape[0]:
      shape: Shape = (self.shape[0],)
    elif len(self.shape) == 1 and len(other.shape) == 1 and self.shape == other.shape:
      shape = ()
    elif len(self.shape) == 1 and len(other.shape) == 2 and self.shape[0] == other.shape[0]:
      shape = (other.shape[1],)
    else:
      raise ValueError(f"cannot matmul {self.shape} @ {other.shape}")
    return Expr(shape, degree=_degree_mul(self.degree, other.degree))

  def __getitem__(self, index: int | slice) -> Expr:
    if not self.shape:
      raise ValueError("cannot index a scalar")
    if isinstance(index, slice):
      return Expr((len(range(*index.indices(self.shape[0]))), *self.shape[1:]), degree=self.degree)
    return Expr(self.shape[1:], degree=self.degree)

  def sum(self) -> Expr:
    return Expr((), degree=self.degree)

  def sin(self) -> Expr:
    return Expr(self.shape, degree=None if self.degree else 0)


def const(value: float | np.ndarray) -> Expr:
  return Expr(np.shape(value), degree=0)


def concat(parts: tuple[Expr, ...]) -> Expr:
  degree: int | None = 0
  for part in parts:
    degree = _degree_add(degree, part.degree)
  return Expr((sum(part.size for part in parts),), degree=degree)


@dataclass(frozen=True)
class Buffer:
  """A numeric tensor (``np.ndarray`` in the real thing)."""

  shape: Shape

  @property
  def size(self) -> int:
    return _size(self.shape)


########################################################################################################
# Structure declarations: Tree, L, G
########################################################################################################


class Tree[Symbolic, Numerical]:
  """A declared pytree of named tensors. ``Symbolic`` and ``Numerical`` are the same structure over
  ``Expr`` and ``Buffer``. ``names`` and ``shapes`` are flat, in C-signature order."""

  names: tuple[str, ...]
  decls: tuple[ShapeDecl, ...]

  @property
  def shapes(self) -> tuple[Shape, ...]:
    if any(d is Ellipsis for d in self.decls):
      raise TypeError(f"tree {self.names} has inferred shapes; resolve them by tracing first")
    return cast(tuple[Shape, ...], self.decls)

  @property
  def size(self) -> int:
    return len(self.names)

  def symbols(self, degree: int = 0) -> Symbolic:
    raise NotImplementedError

  def relabel(self, prefix: str) -> Tree[Symbolic, Numerical]:
    """Same structure, every name prefixed (``lam:``, ``fwd:``)."""
    raise NotImplementedError

  def with_shapes(self, shapes: tuple[Shape, ...]) -> Tree[Symbolic, Numerical]:
    """Same structure with every ``...`` replaced by the traced shape."""
    raise NotImplementedError

  def resolved(self, shapes: tuple[Shape, ...]) -> tuple[Shape, ...]:
    """Check traced ``shapes`` against the declaration; ``...`` accepts anything."""
    if len(shapes) != self.size:
      raise TypeError(f"declared {self.size} outputs {self.names}, body returned {len(shapes)}")
    for name, decl, shape in zip(self.names, self.decls, shapes, strict=True):
      if decl is not Ellipsis and decl != shape:
        raise TypeError(f"{name!r} declared with shape {decl}, traced shape {shape}")
    return shapes

  def index(self, name: str) -> int:
    if name not in self.names:
      raise ValueError(f"unknown name {name!r}; declared {self.names}")
    return self.names.index(name)

  def _check_unique(self) -> None:
    if len(set(self.names)) != len(self.names):
      raise ValueError(f"duplicate names in {self.names}")


class L(Tree[Expr, Buffer]):
  """One named tensor. The real implementation also takes a ``TensorType`` for dtype and ``diff``."""

  def __init__(self, name: str, shape: int | ShapeDecl, /) -> None:
    self.names = (name,)
    self.decls = (shape if isinstance(shape, EllipsisType) else _as_shape(shape),)

  def symbols(self, degree: int = 0) -> Expr:
    return Expr(self.shapes[0], self.names[0], degree)

  def relabel(self, prefix: str) -> L:
    return L(prefix + self.names[0], self.decls[0])

  def with_shapes(self, shapes: tuple[Shape, ...]) -> L:
    return L(self.names[0], shapes[0])


class _G(Tree[Any, Any]):
  """Runtime class behind ``G``; the static types live on ``G``'s overloads."""

  def __init__(self, *parts: Tree[Any, Any]) -> None:
    if not 2 <= len(parts) <= 8:
      raise TypeError(f"G takes 2 to 8 trees, got {len(parts)}; nest for more")
    self.parts = parts
    self.names = tuple(n for part in parts for n in part.names)
    self.decls = tuple(d for part in parts for d in part.decls)
    self._check_unique()

  def symbols(self, degree: int = 0) -> tuple[Any, ...]:
    return tuple(part.symbols(degree) for part in self.parts)

  def relabel(self, prefix: str) -> _G:
    return _G(*(part.relabel(prefix) for part in self.parts))

  def with_shapes(self, shapes: tuple[Shape, ...]) -> _G:
    out: list[Tree[Any, Any]] = []
    i = 0
    for part in self.parts:
      out.append(part.with_shapes(shapes[i : i + part.size]))
      i += part.size
    return _G(*out)


# fmt: off
@overload
def G[SA, NA, SB, NB](a: Tree[SA, NA], b: Tree[SB, NB], /) -> Tree[tuple[SA, SB], tuple[NA, NB]]: ...
@overload
def G[SA, NA, SB, NB, SC, NC](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], /) -> Tree[tuple[SA, SB, SC], tuple[NA, NB, NC]]: ...
@overload
def G[SA, NA, SB, NB, SC, NC, SD, ND](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], /) -> Tree[tuple[SA, SB, SC, SD], tuple[NA, NB, NC, ND]]: ...
@overload
def G[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], /) -> Tree[tuple[SA, SB, SC, SD, SE], tuple[NA, NB, NC, ND, NE]]: ...
@overload
def G[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], /) -> Tree[tuple[SA, SB, SC, SD, SE, SF], tuple[NA, NB, NC, ND, NE, NF]]: ...
@overload
def G[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], /) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG], tuple[NA, NB, NC, ND, NE, NF, NG]]: ...
@overload
def G[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG, SH, NH](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], h: Tree[SH, NH], /) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG, SH], tuple[NA, NB, NC, ND, NE, NF, NG, NH]]: ...
# fmt: on
def G(*parts: Tree[Any, Any]) -> Tree[Any, Any]:
  """Group trees side by side: ``G(L("x", 3), L("p", ()))`` is ``tuple[Expr, Expr]`` / ``tuple[Buffer, Buffer]``."""
  return _G(*parts)


########################################################################################################
# Utilities
########################################################################################################


def leaves(value: object, /) -> tuple[Expr | Buffer, ...]:
  """Flatten a pytree of values in declaration order."""
  if isinstance(value, (Expr, Buffer)):
    return (value,)
  if isinstance(value, tuple):
    return tuple(item for part in value for item in leaves(part))
  raise TypeError(f"expected a pytree of Expr or Buffer values, got {type(value).__name__}")


def shapes_of(value: object, /) -> tuple[Shape, ...]:
  return tuple(v.shape for v in leaves(value))


def unflatten[S, N](tree: Tree[S, N], values: tuple[Any, ...]) -> Any:
  """Rebuild ``tree``'s structure from flat values (the C side hands back flat buffers)."""
  if isinstance(tree, L):
    return values[0]
  assert isinstance(tree, _G)
  out: list[Any] = []
  i = 0
  for part in tree.parts:
    out.append(unflatten(part, values[i : i + part.size]))
    i += part.size
  return tuple(out)


def _same_structure(tree: Tree[Any, Any], value: object) -> bool:
  if isinstance(tree, L):
    return isinstance(value, (Expr, Buffer))
  assert isinstance(tree, _G)
  return isinstance(value, tuple) and len(value) == len(tree.parts) and all(_same_structure(t, v) for t, v in zip(tree.parts, value, strict=True))


def _check_structure(tree: Tree[Any, Any], value: object, what: str) -> None:
  if not _same_structure(tree, value):
    raise ValueError(f"{what}: value does not have the declared structure of {tree.names}")
  if shapes_of(value) != tree.shapes:
    raise ValueError(f"{what}: expected shapes {tree.shapes} for {tree.names}, got {shapes_of(value)}")


########################################################################################################
# Function
########################################################################################################


class Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A named graph from one declared input tree to one declared output tree.

  ``symbolic_call`` is today's ``Function.call`` (a CALL node in a larger graph); ``numerical_call``
  is today's ``__call__``. Both take the whole input tree as one positional argument.
  """

  def __init__(
    self,
    name: str,
    fn: Callable[[SymbolicInputs], SymbolicOutputs],
    inputs: Tree[SymbolicInputs, NumericalInputs],
    outputs: Tree[SymbolicOutputs, NumericalOutputs],
  ) -> None:
    self.name = name
    self._fn = fn
    self.inputs = inputs
    traced = fn(inputs.symbols(degree=1))
    self.output_shapes = outputs.resolved(shapes_of(traced))
    self.outputs = outputs.with_shapes(self.output_shapes)
    self.output_degrees = tuple(cast(Expr, v).degree for v in leaves(traced))

  @property
  def input_names(self) -> tuple[str, ...]:
    return self.inputs.names

  @property
  def input_shapes(self) -> tuple[Shape, ...]:
    return self.inputs.shapes

  @property
  def output_names(self) -> tuple[str, ...]:
    return self.outputs.names

  def symbolic_call(self, inputs: SymbolicInputs, /) -> SymbolicOutputs:
    _check_structure(self.inputs, inputs, f"{self.name}.symbolic_call")
    return self._fn(inputs)

  def numerical_call(self, inputs: NumericalInputs, /) -> NumericalOutputs:
    _check_structure(self.inputs, inputs, f"{self.name}.numerical_call")
    return cast(NumericalOutputs, unflatten(self.outputs, tuple(Buffer(s) for s in self.output_shapes)))

  def c_signature(self) -> str:
    """What the generated header spells: one pointer per leaf, in tree order."""
    ins = ", ".join(f"const double* {n}" for n in self.input_names)
    outs = ", ".join(f"double* {n}" for n in self.output_names)
    return f"void {self.name}({ins}, {outs})"


def function[SI, NI, SO, NO](
  inputs: Tree[SI, NI], outputs: Tree[SO, NO], /, *, name: str | None = None
) -> Callable[[Callable[[SI], SO]], Function[SI, NI, SO, NO]]:
  def decorate(fn: Callable[[SI], SO]) -> Function[SI, NI, SO, NO]:
    return Function(name or getattr(fn, "__name__", "fn"), fn, inputs, outputs)

  return decorate


########################################################################################################
# Derivatives: same inputs, one output; seeded modes pair the inputs with the new group
########################################################################################################


def _shape_of_output(fn: Function[Any, Any, Any, Any], of: str) -> Shape:
  return fn.output_shapes[fn.outputs.index(of)]


def _shape_of_input(fn: Function[Any, Any, Any, Any], wrt: str) -> Shape:
  return fn.input_shapes[fn.inputs.index(wrt)]


def _derived[SI, NI](fn: Function[SI, NI, Any, Any], name: str, shape: Shape, degree: int | None) -> Function[SI, NI, Expr, Buffer]:
  def body(inputs: SI) -> Expr:
    fn.symbolic_call(inputs)
    return Expr(shape, degree=degree)

  return Function(f"{fn.name}_{name}", body, fn.inputs, L(name, shape))


def _lower_degree(degree: int | None, by: int) -> int | None:
  return None if degree is None else max(degree - by, 0)


def gradient[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  """``d of / d wrt`` as a function of every input of ``fn``."""
  of_shape = _shape_of_output(fn, of)
  if of_shape != ():
    raise ValueError(f"gradient needs a scalar output, {of!r} has shape {of_shape}")
  degree = fn.output_degrees[fn.outputs.index(of)]
  return _derived(fn, f"grad_{of}_{wrt}", _shape_of_input(fn, wrt), _lower_degree(degree, 1))


def jacobian[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  degree = fn.output_degrees[fn.outputs.index(of)]
  return _derived(fn, f"jac_{of}_{wrt}", (*_shape_of_output(fn, of), *_shape_of_input(fn, wrt)), _lower_degree(degree, 1))


def hessian[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  of_shape = _shape_of_output(fn, of)
  if of_shape != ():
    raise ValueError(f"hessian needs a scalar output, {of!r} has shape {of_shape}")
  n = _shape_of_input(fn, wrt)
  degree = fn.output_degrees[fn.outputs.index(of)]
  return _derived(fn, f"hess_{of}_{wrt}_{wrt}", (*n, *n), _lower_degree(degree, 2))


def _seeded[SI, NI, SS, NS](
  fn: Function[SI, NI, Any, Any], seed: Tree[SS, NS], name: str, shape: Shape
) -> Function[tuple[SI, SS], tuple[NI, NS], Expr, Buffer]:
  def body(inputs: tuple[SI, SS]) -> Expr:
    fn.symbolic_call(inputs[0])
    return Expr(shape, degree=None)

  return Function(f"{fn.name}_{name}", body, G(fn.inputs, seed), L(name, shape))


def forward[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[tuple[SI, Expr], tuple[NI, Buffer], Expr, Buffer]:
  """``J(of, wrt) @ seed``, called as ``fwd.numerical_call((inputs, seed))``."""
  return _seeded(fn, L(f"fwd:{wrt}", _shape_of_input(fn, wrt)), f"fwd_{of}_{wrt}", _shape_of_output(fn, of))


def adjoint[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[tuple[SI, Expr], tuple[NI, Buffer], Expr, Buffer]:
  """``J(of, wrt).T @ lam``, called as ``adj.numerical_call((inputs, lam))``."""
  return _seeded(fn, L(f"lam:{of}", _shape_of_output(fn, of)), f"adj_{of}_{wrt}", _shape_of_input(fn, wrt))


def lagrangian_hessian[SI, NI, SO, NO](fn: Function[SI, NI, SO, NO], wrt: str, /) -> Function[tuple[SI, SO], tuple[NI, NO], Expr, Buffer]:
  """Hessian of ``sum_i lam_i . out_i``: the multipliers have the *output* tree of ``fn``."""
  n = _shape_of_input(fn, wrt)
  return _seeded(fn, fn.outputs.relabel("lam:"), f"hess_lagrangian_{wrt}", (*n, *n))


########################################################################################################
# Problems and solvers
########################################################################################################


@dataclass(frozen=True)
class Bounded:
  """``lo <= expr <= hi``; ``None`` is an open side. ``name`` labels the group in diagnostics."""

  expr: Expr
  lo: Expr | float | None = None
  hi: Expr | float | None = None
  name: str | None = None


def bounded(expr: Expr, lo: Expr | float | None = None, hi: Expr | float | None = None, *, name: str | None = None) -> Bounded:
  if lo is None and hi is None:
    raise ValueError("bounded needs at least one of lo / hi")
  return Bounded(expr, lo, hi, name)


@dataclass(frozen=True)
class ProblemSpec[SymbolicVars]:
  """What a problem body returns. Every field is an expression over the declared vars and params.
  ``lb``/``ub`` have the variables' structure (a leaf may be a constant ``Expr``)."""

  minimize: Expr
  eq: tuple[Expr, ...] = ()
  ineq: tuple[Bounded, ...] = ()
  lb: SymbolicVars | None = None
  ub: SymbolicVars | None = None


@dataclass(frozen=True)
class Problem[SV, NV, SP, NP]:
  """A backend-free problem: the spec plus the two trees it was traced over."""

  name: str
  spec: ProblemSpec[SV]
  vars: Tree[SV, NV]
  params: Tree[SP, NP]

  @property
  def n_eq(self) -> int:
    return sum(e.size for e in self.spec.eq)

  @property
  def n_ineq(self) -> int:
    return sum(b.expr.size for b in self.spec.ineq)


def problem[SV, NV, SP, NP](
  *, vars: Tree[SV, NV], params: Tree[SP, NP], name: str | None = None
) -> Callable[[Callable[[SV, SP], ProblemSpec[SV]]], Problem[SV, NV, SP, NP]]:
  def decorate(fn: Callable[[SV, SP], ProblemSpec[SV]]) -> Problem[SV, NV, SP, NP]:
    spec = fn(vars.symbols(degree=1), params.symbols(degree=0))
    if spec.minimize.shape != ():
      raise TypeError(f"cost must be scalar, got shape {spec.minimize.shape}")
    for e in spec.eq:
      if len(e.shape) > 1:
        raise TypeError(f"equality constraints must be rank-1, got shape {e.shape}")
    for b in spec.ineq:
      if len(b.expr.shape) > 1:
        raise TypeError(f"inequality constraints must be rank-1, got shape {b.expr.shape}")
    for side, bound in (("lb", spec.lb), ("ub", spec.ub)):
      if bound is not None and len(leaves(bound)) != vars.size:
        raise TypeError(f"{side} must have the variables' structure {vars.names}")
    return Problem(name or getattr(fn, "__name__", "problem"), spec, vars, params)

  return decorate


BACKENDS: dict[str, tuple[Literal["qp", "nlp"], Literal["lower", "upper"]]] = {
  "piqp": ("qp", "upper"),
  "ipopt": ("nlp", "lower"),
  "sqp": ("nlp", "upper"),
}


class NotQuadratic(ValueError):
  pass


def _prove_quadratic(p: Problem[Any, Any, Any, Any]) -> None:
  """The QP gate: cost quadratic, constraints affine in the variables. Structural, so conservative."""
  if p.spec.minimize.degree is None or p.spec.minimize.degree > 2:
    raise NotQuadratic(f"{p.name}: cost has degree {p.spec.minimize.degree} in the variables, a QP backend needs at most 2")
  for i, e in enumerate(p.spec.eq):
    if e.degree is None or e.degree > 1:
      raise NotQuadratic(f"{p.name}: eq[{i}] has degree {e.degree}, a QP backend needs affine constraints")
  for b in p.spec.ineq:
    if b.expr.degree is None or b.expr.degree > 1:
      raise NotQuadratic(f"{p.name}: ineq {b.name or '?'} has degree {b.expr.degree}, a QP backend needs affine constraints")


def solver[SV, NV, SP, NP](
  p: Problem[SV, NV, SP, NP], backend: str, /, *, name: str | None = None, options: dict[str, Any] | None = None
) -> Function[tuple[SV, SV, Expr, Expr, SP], tuple[NV, NV, Buffer, Buffer, NP], tuple[SV, SV, Expr, Expr], tuple[NV, NV, Buffer, Buffer]]:
  """Build a solver for ``p``: a plain ``Function`` over ``(vars_init, lam_box0, lam_eq0, lam_ineq0, params)``."""
  if backend not in BACKENDS:
    raise ValueError(f"unknown backend {backend!r}; available {sorted(BACKENDS)}")
  kind, _triangle = BACKENDS[backend]
  if kind == "qp":
    _prove_quadratic(p)
  lam_box = p.vars.relabel("lam:")
  inputs = G(p.vars, lam_box, L("lam_eq", p.n_eq), L("lam_ineq", p.n_ineq), p.params)
  outputs = G(p.vars, lam_box, L("lam_eq", p.n_eq), L("lam_ineq", p.n_ineq))

  def body(args: tuple[SV, SV, Expr, Expr, SP]) -> tuple[SV, SV, Expr, Expr]:
    vars_init, lam_box0, lam_eq0, lam_ineq0, _params = args
    return vars_init, lam_box0, lam_eq0, lam_ineq0

  return Function(name or f"{p.name}_{backend}", body, inputs, outputs)


type QPData[T] = tuple[tuple[T, T], tuple[T, T], tuple[T, T, T]]


def qp_problem(n: int, n_eq: int, n_ineq: int) -> Problem[Expr, Buffer, QPData[Expr], QPData[Buffer]]:
  """The typed data form: ``min 0.5 x'Px + c'x  s.t.  A x = b,  g_lb <= G x <= g_ub``. Every block is a
  parameter, so a consumer updates the QP data per call; an absent block has size 0."""

  @problem(
    vars=L("x", n),
    params=G(G(L("P", (n, n)), L("c", n)), G(L("A", (n_eq, n)), L("b", n_eq)), G(L("G", (n_ineq, n)), L("g_lb", n_ineq), L("g_ub", n_ineq))),
    name="qp",
  )
  def qp(x: Expr, params: QPData[Expr]) -> ProblemSpec[Expr]:
    (P, c), (A, b), (Gm, g_lb, g_ub) = params
    return ProblemSpec(minimize=0.5 * (x @ P @ x) + c @ x, eq=(A @ x - b,), ineq=(bounded(Gm @ x, g_lb, g_ub, name="g"),))

  return qp


########################################################################################################
# Concrete definitions used by the tests
########################################################################################################


@function(L("x", 3), G(L("first", ...), L("second", 3)))
def duplicate(x: Expr) -> tuple[Expr, Expr]:
  return x, x


@function(G(L("x", 3), L("y", 3)), L("prod", ...))
def multiply(inputs: tuple[Expr, Expr]) -> Expr:
  return inputs[0] * inputs[1]


@function(L("x", 3), L("square", ...))
def square(x: Expr) -> Expr:
  return multiply.symbolic_call(duplicate.symbolic_call(x))


@function(G(L("x", 3), L("p", ())), L("f", ...))
def cost(inputs: tuple[Expr, Expr]) -> Expr:
  x, p = inputs
  return (x * x).sum() * p


# Five inputs grouped the way the problem thinks about them ...
@function(G(G(L("state", 4), L("u", 2)), G(L("pw", 10), L("physics", 3), L("dt", ()))), L("next", ...))
def step(inputs: tuple[tuple[Expr, Expr], tuple[Expr, Expr, Expr]]) -> Expr:
  (state, _u), (_pw, _physics, _dt) = inputs
  return state


# ... or flat, when there is no natural grouping. Same leaves, same C signature.
@function(G(L("state", 4), L("u", 2), L("pw", 10), L("physics", 3), L("dt", ())), L("next", ...))
def step_flat(inputs: tuple[Expr, Expr, Expr, Expr, Expr]) -> Expr:
  state, _u, _pw, _physics, _dt = inputs
  return state


# The form we would have liked, and cannot type: a literal `("u", 2)` is `tuple[str, int]` whatever
# classes exist, so mapping it to `tuple[Expr, Expr]` and its Buffer twin would need one overload per
# structure. The `L` wrapper is the smallest thing that carries the map.
#
#   @function(inputs=((("state", 4), ("u", 2)), (("pw", 10), ("physics", 3), ("dt", ()))), outputs=("next", ...))
#   def step(inputs: tuple[tuple[Expr, Expr], tuple[Expr, Expr, Expr]]) -> Expr: ...


@problem(vars=L("x", 3), params=L("scale", ()))
def quadratic(x: Expr, scale: Expr) -> ProblemSpec[Expr]:
  return ProblemSpec(minimize=(x * x).sum() * scale, lb=const(-1.0), ub=const(1.0))


# Multi-block variables, two constraint groups, box bounds in the variables' structure. The body
# binds `vs`/`sc`, not `u`/`s`/`scale`: declared names are external, body names are local.
@problem(vars=G(L("u", 2), L("s", 1)), params=G(L("x", 4), L("u_ref", 2)))
def filter_problem(vs: tuple[Expr, Expr], ps: tuple[Expr, Expr]) -> ProblemSpec[tuple[Expr, Expr]]:
  u, s = vs
  x, u_ref = ps
  barrier = x[:2] @ u + x[2:].sum()  # affine in u, parametrised by x
  return ProblemSpec(
    minimize=0.5 * ((u - u_ref) * (u - u_ref)).sum() + 10.0 * s.sum(),
    eq=(u[0:1] - u[1:2],),
    ineq=(bounded(barrier + s, lo=0.0, name="cbf"), bounded(u, lo=-1.0, hi=1.0, name="u_box")),
    lb=(const(np.full(2, -1.0)), const(np.zeros(1))),
  )


@problem(vars=L("x", 2), params=L("w", ()))
def rosenbrock(x: Expr, w: Expr) -> ProblemSpec[Expr]:
  return ProblemSpec(minimize=(1 - x[0]) ** 2 + w * (x[1] - x[0] ** 2) ** 2, ineq=(bounded(x[0].sin(), hi=0.5, name="wave"),))


qp3 = qp_problem(3, 1, 2)

grad_f_x = gradient(cost, "f", "x")
hess_f_x = hessian(cost, "f", "x")
jac_square_x = jacobian(square, "square", "x")
fwd_f_x = forward(cost, "f", "x")
adj_square_x = adjoint(square, "square", "x")
hess_l = lagrangian_hessian(duplicate, "x")

quadratic_ipopt = solver(quadratic, "ipopt")
quadratic_piqp = solver(quadratic, "piqp", name="quadratic_fast")
filter_sqp = solver(filter_problem, "sqp")
filter_piqp = solver(filter_problem, "piqp")
rosenbrock_ipopt = solver(rosenbrock, "ipopt")
qp3_piqp = solver(qp3, "piqp")
qp3_ipopt = solver(qp3, "ipopt")

########################################################################################################
# Behavioral tests: `uv run arity.py`
########################################################################################################


def _raises(exc: type[BaseException], thunk: Callable[[], object], contains: str = "") -> None:
  try:
    thunk()
  except exc as e:
    assert contains in str(e), f"expected {contains!r} in {e!s}"
  else:
    raise AssertionError(f"expected {exc.__name__}")


# --- declarations ---


def test_leaf_shapes() -> None:
  assert L("x", 3).shapes == ((3,),) and L("P", (2, 2)).shapes == ((2, 2),) and L("s", ()).shapes == ((),)
  _raises(TypeError, lambda: L("f", ...).shapes, "inferred")


def test_group_widths_and_nesting() -> None:
  assert G(L("a", 1), L("b", 1)).names == ("a", "b")
  eight = G(*(L(f"x{i}", 1) for i in range(8)))
  assert eight.size == 8
  _raises(TypeError, lambda: _G(L("a", 1)), "2 to 8")
  _raises(TypeError, lambda: _G(*(L(f"x{i}", 1) for i in range(9))), "nest for more")
  nested = G(G(L("a", 1), L("b", 1)), L("c", 1))
  assert nested.names == ("a", "b", "c") and nested.size == 3


def test_duplicate_names_rejected_within_a_tree() -> None:
  _raises(ValueError, lambda: G(L("x", 3), L("x", 3)), "duplicate")
  _raises(ValueError, lambda: G(G(L("x", 3), L("y", 3)), L("x", 1)), "duplicate")


def test_symbols_follow_the_structure() -> None:
  syms = G(G(L("a", 1), L("b", 2)), L("c", ())).symbols()
  assert isinstance(syms, tuple) and isinstance(syms[0], tuple)
  assert syms[0][1].name == "b" and syms[0][1].shape == (2,) and syms[1].shape == ()


def test_relabel_and_with_shapes_keep_structure() -> None:
  t = G(L("u", 2), L("s", ...))
  assert t.relabel("lam:").names == ("lam:u", "lam:s")
  assert t.with_shapes(((2,), (1,))).shapes == ((2,), (1,))
  assert isinstance(t.with_shapes(((2,), (1,))), _G)


# --- functions ---


def test_names_and_shapes_are_declared_not_inferred() -> None:
  assert duplicate.input_names == ("x",) and duplicate.output_names == ("first", "second")
  assert duplicate.output_shapes == ((3,), (3,))  # `first` was `...`, traced to (3,)
  assert multiply.output_shapes == ((3,),)


def test_grouping_is_not_a_signature() -> None:
  assert step.input_names == step_flat.input_names == ("state", "u", "pw", "physics", "dt")
  assert step.c_signature() == step_flat.c_signature().replace("step_flat", "step")


def test_calls_return_the_declared_structure() -> None:
  out = duplicate.numerical_call(Buffer((3,)))
  assert isinstance(out, tuple) and len(out) == 2 and out[0].shape == (3,)
  assert step.numerical_call(((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(())))).shape == (4,)


def test_composition_through_symbolic_call() -> None:
  assert square.output_shapes == ((3,),)
  assert jac_square_x.output_shapes == ((3, 3),)


def test_wrong_shape_at_call_is_a_runtime_error() -> None:
  _raises(ValueError, lambda: duplicate.numerical_call(Buffer((4,))), "expected shapes")
  _raises(ValueError, lambda: multiply.symbolic_call((Expr((3,)), Expr((2,)))), "expected shapes")


def test_wrong_structure_at_call_is_a_runtime_error() -> None:
  _raises(ValueError, lambda: step.numerical_call(cast(Any, (Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(())))), "structure")
  _raises(
    ValueError, lambda: step_flat.numerical_call(cast(Any, ((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(()))))), "structure"
  )


def test_output_count_and_shape_are_checked_at_the_decorator() -> None:
  _raises(TypeError, lambda: function(L("x", 3), L("y", 2))(lambda x: x), "declared with shape (2,)")
  _raises(TypeError, lambda: function(L("x", 3), G(L("a", ...), L("b", ...)))(cast(Any, lambda x: x)), "declared 2 outputs")


def test_body_names_are_independent_of_declared_names() -> None:
  assert filter_problem.vars.names == ("u", "s") and filter_problem.params.names == ("x", "u_ref")


# --- derivatives ---


def test_derivatives_keep_the_source_inputs() -> None:
  assert grad_f_x.input_names == ("x", "p") and grad_f_x.output_names == ("grad_f_x",) and grad_f_x.output_shapes == ((3,),)
  assert hess_f_x.output_names == ("hess_f_x_x",) and hess_f_x.output_shapes == ((3, 3),)
  assert grad_f_x.c_signature() == "void cost_grad_f_x(const double* x, const double* p, double* grad_f_x)"


def test_seeded_modes_pair_inputs_with_the_new_group() -> None:
  assert fwd_f_x.input_names == ("x", "p", "fwd:x") and fwd_f_x.output_shapes == ((),)
  assert adj_square_x.input_names == ("x", "lam:square") and adj_square_x.output_shapes == ((3,),)
  assert hess_l.input_names == ("x", "lam:first", "lam:second") and hess_l.output_shapes == ((3, 3),)
  assert fwd_f_x.numerical_call(((Buffer((3,)), Buffer(())), Buffer((3,)))).shape == ()


def test_unknown_of_or_wrt_fails_at_build_time_naming_the_choices() -> None:
  _raises(ValueError, lambda: gradient(cost, "f", "z"), "declared ('x', 'p')")
  _raises(ValueError, lambda: gradient(cost, "h", "x"), "declared ('f',)")
  _raises(ValueError, lambda: forward(cost, "f", "z"), "unknown name 'z'")
  _raises(ValueError, lambda: lagrangian_hessian(duplicate, "y"), "unknown name 'y'")


def test_gradient_and_hessian_need_a_scalar() -> None:
  _raises(ValueError, lambda: gradient(duplicate, "first", "x"), "scalar")
  _raises(ValueError, lambda: hessian(duplicate, "first", "x"), "scalar")


# --- problems ---


def test_problem_carries_spec_and_trees() -> None:
  assert quadratic.name == "quadratic" and quadratic.vars.names == ("x",) and quadratic.params.names == ("scale",)
  assert quadratic.spec.minimize.shape == () and quadratic.n_eq == 0 and quadratic.n_ineq == 0
  assert filter_problem.n_eq == 1 and filter_problem.n_ineq == 3
  assert [b.name for b in filter_problem.spec.ineq] == ["cbf", "u_box"]


def test_problem_checks_the_spec_at_the_decorator() -> None:
  _raises(TypeError, lambda: problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x)), "scalar")
  _raises(
    TypeError, lambda: problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum(), lb=cast(Any, (x, x)))), "structure"
  )
  _raises(ValueError, lambda: bounded(Expr((2,))), "at least one")


def test_qp_problem_is_a_problem() -> None:
  assert qp3.vars.names == ("x",) and qp3.params.names == ("P", "c", "A", "b", "G", "g_lb", "g_ub")
  assert qp3.n_eq == 1 and qp3.n_ineq == 2
  empty = qp_problem(3, 0, 0)
  assert empty.params.shapes[2:4] == ((0, 3), (0,)) and empty.n_eq == 0 and empty.n_ineq == 0


# --- solvers ---


def test_solver_signature_is_fixed() -> None:
  assert quadratic_ipopt.input_names == ("x", "lam:x", "lam_eq", "lam_ineq", "scale")
  assert quadratic_ipopt.input_shapes == ((3,), (3,), (0,), (0,), ())
  assert quadratic_ipopt.output_names == ("x", "lam:x", "lam_eq", "lam_ineq")
  assert filter_sqp.input_names == ("u", "s", "lam:u", "lam:s", "lam_eq", "lam_ineq", "x", "u_ref")
  assert filter_sqp.input_shapes[4:6] == ((1,), (3,))


def test_solver_name_and_backend() -> None:
  assert quadratic_ipopt.name == "quadratic_ipopt" and quadratic_piqp.name == "quadratic_fast"
  _raises(ValueError, lambda: solver(quadratic, "osqp"), "unknown backend")


def test_solution_and_warm_start_have_the_variables_structure() -> None:
  x0 = (Buffer((2,)), Buffer((1,)))
  out = filter_sqp.numerical_call((x0, x0, Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,)))))
  (u, s), (lam_u, lam_s), lam_eq, lam_ineq = out
  assert u.shape == (2,) and s.shape == (1,) and lam_u.shape == (2,) and lam_s.shape == (1,) and lam_eq.shape == (1,) and lam_ineq.shape == (3,)


def test_qp_backend_is_gated_by_the_quadratic_proof() -> None:
  assert filter_piqp.name == "filter_problem_piqp"  # quadratic cost, affine constraints: accepted
  assert qp3_piqp.input_names[-7:] == qp3.params.names  # the data form is updatable per call
  _raises(NotQuadratic, lambda: solver(rosenbrock, "piqp"), "cost has degree 4")
  quartic_constraint = problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=(x * x).sum(), eq=(x * x * x,)))
  _raises(NotQuadratic, lambda: solver(quartic_constraint, "piqp"), "eq[0] has degree 3")
  wavy = problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=(x * x).sum(), ineq=(bounded(x.sin(), hi=1.0, name="w"),)))
  _raises(NotQuadratic, lambda: solver(wavy, "piqp"), "ineq w has degree None")


def test_nlp_backend_takes_any_spec() -> None:
  assert rosenbrock_ipopt.input_shapes == ((2,), (2,), (0,), (1,), ())
  assert qp3_ipopt.input_names == qp3_piqp.input_names


def test_degree_tracking_used_by_the_proof() -> None:
  x = L("x", 3).symbols(degree=1)
  p = L("p", (3, 3)).symbols(degree=0)
  assert (x @ p @ x).degree == 2 and (p @ x).degree == 1 and (x / p[0]).degree == 1
  assert (x / x[0]).degree is None and x.sin().degree is None and (x**3).degree == 3 and const(1.0).degree == 0


if __name__ == "__main__":
  tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
  for k, v in tests:
    v()
  print(f"{len(tests)} tests passed")

########################################################################################################
# Typing tests: `uv run ty check --error-on-warning arity.py`
########################################################################################################

if TYPE_CHECKING:
  # declarations: the count of a group is static, and only trees may be grouped
  L("x", "3")  # ty: ignore[invalid-argument-type]
  G(L("x", 3))  # ty: ignore[no-matching-overload]
  G(L("x", 3), L("y", 3), ("z", 3))  # ty: ignore[invalid-argument-type]
  assert_type(G(L("x", 3), L("p", ())), Tree[tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(G(G(L("a", 1), L("b", 1)), L("c", 1)), Tree[tuple[tuple[Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], Buffer]])

  # functions: structure, count and leaf kind are all checked, on both sides
  assert_type(duplicate, Function[Expr, Buffer, tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(duplicate.symbolic_call(Expr((3,))), tuple[Expr, Expr])
  assert_type(duplicate.numerical_call(Buffer((3,))), tuple[Buffer, Buffer])
  assert_type(multiply.numerical_call((Buffer((3,)), Buffer((3,)))), Buffer)
  assert_type(step.numerical_call(((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(())))), Buffer)
  assert_type(step_flat.numerical_call((Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(()))), Buffer)
  multiply.numerical_call((Buffer((3,)),))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call((Expr((3,)), Expr((3,))))  # ty: ignore[invalid-argument-type]
  multiply.symbolic_call((Buffer((3,)), Buffer((3,))))  # ty: ignore[invalid-argument-type]
  duplicate.symbolic_call((Expr((3,)),))  # ty: ignore[invalid-argument-type]
  step.numerical_call((Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(())))  # ty: ignore[invalid-argument-type]
  step_flat.numerical_call(((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(()))))  # ty: ignore[invalid-argument-type]

  # decorator and body must agree
  function(G(L("x", 3), L("y", 3)), L("z", ...))(lambda x: x)  # ty: ignore[invalid-argument-type]
  function(L("x", 3), G(L("a", ...), L("b", ...)))(lambda x: x)  # ty: ignore[invalid-argument-type]

  # composition preserves types
  assert_type(multiply.symbolic_call(duplicate.symbolic_call(Expr((3,)))), Expr)
  multiply.symbolic_call(square.symbolic_call(Expr((3,))))  # ty: ignore[invalid-argument-type]

  # derivatives: the source's input tree is preserved, seeded modes pair it with the new group
  assert_type(grad_f_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(hess_f_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(jac_square_x, Function[Expr, Buffer, Expr, Buffer])
  assert_type(grad_f_x.numerical_call((Buffer((3,)), Buffer(()))), Buffer)
  assert_type(fwd_f_x, Function[tuple[tuple[Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], Buffer], Expr, Buffer])
  assert_type(adj_square_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(hess_l, Function[tuple[Expr, tuple[Expr, Expr]], tuple[Buffer, tuple[Buffer, Buffer]], Expr, Buffer])
  assert_type(fwd_f_x.numerical_call(((Buffer((3,)), Buffer(())), Buffer((3,)))), Buffer)
  assert_type(hess_l.numerical_call((Buffer((3,)), (Buffer((3,)), Buffer((3,))))), Buffer)
  grad_f_x.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  fwd_f_x.numerical_call((Buffer((3,)), Buffer(()), Buffer((3,))))  # ty: ignore[invalid-argument-type]
  hess_l.numerical_call((Buffer((3,)), Buffer((6,))))  # ty: ignore[invalid-argument-type]

  # problems: the body's parameter types are the declared trees, the spec's bounds have the vars' structure
  assert_type(quadratic, Problem[Expr, Buffer, Expr, Buffer])
  assert_type(filter_problem, Problem[tuple[Expr, Expr], tuple[Buffer, Buffer], tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(qp3, Problem[Expr, Buffer, QPData[Expr], QPData[Buffer]])
  problem(vars=G(L("u", 2), L("s", 1)), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum()))  # ty: ignore[unresolved-attribute]
  problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum(), lb=(x, x)))  # ty: ignore[invalid-argument-type]

  # solvers are Functions with a fixed five-group input and four-group output
  assert_type(
    quadratic_ipopt,
    Function[
      tuple[Expr, Expr, Expr, Expr, Expr],
      tuple[Buffer, Buffer, Buffer, Buffer, Buffer],
      tuple[Expr, Expr, Expr, Expr],
      tuple[Buffer, Buffer, Buffer, Buffer],
    ],
  )
  assert_type(
    filter_sqp,
    Function[
      tuple[tuple[Expr, Expr], tuple[Expr, Expr], Expr, Expr, tuple[Expr, Expr]],
      tuple[tuple[Buffer, Buffer], tuple[Buffer, Buffer], Buffer, Buffer, tuple[Buffer, Buffer]],
      tuple[tuple[Expr, Expr], tuple[Expr, Expr], Expr, Expr],
      tuple[tuple[Buffer, Buffer], tuple[Buffer, Buffer], Buffer, Buffer],
    ],
  )
  assert_type(
    qp3_piqp.numerical_call(
      (
        Buffer((3,)),
        Buffer((3,)),
        Buffer((1,)),
        Buffer((2,)),
        ((Buffer((3, 3)), Buffer((3,))), (Buffer((1, 3)), Buffer((1,))), (Buffer((2, 3)), Buffer((2,)), Buffer((2,)))),
      )
    ),
    tuple[Buffer, Buffer, Buffer, Buffer],
  )
  quadratic_ipopt.numerical_call((Buffer((3,)), Buffer((3,)), Buffer((0,)), Buffer((0,))))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call(((Buffer((2,)),), (Buffer((2,)), Buffer((1,))), Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,)))))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call(((Expr((2,)), Expr((1,))), (Buffer((2,)), Buffer((1,))), Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,)))))  # ty: ignore[invalid-argument-type]
  # a solver nests in a larger graph like any Function
  assert_type(
    filter_sqp.symbolic_call(((Expr((2,)), Expr((1,))), (Expr((2,)), Expr((1,))), Expr((1,)), Expr((3,)), (Expr((4,)), Expr((2,))))),
    tuple[tuple[Expr, Expr], tuple[Expr, Expr], Expr, Expr],
  )
