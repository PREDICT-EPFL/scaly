"""Typed declarations for symbolic and numerical function pytrees."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import EllipsisType
from typing import Any, Literal, TypeGuard, cast, overload

import numpy as np
from scipy import sparse

from ..ir.expr import Expr, ExprOp
from ..ir.layout import DenseLayout, Layout, LeafLayout, SparseLayout, sparse_layout
from ..ir.types import TensorType, as_shape

type ShapeDecl = int | tuple[int, ...] | TensorType | None
type LeafDecl = TensorType | None
type Array = np.ndarray
type NumericalArray = Array | sparse.sparray
type _DenseNumerical = Array | tuple[_DenseNumerical, ...]
# declaring a type alias makes the inferred types of functions easier to read:
# Function[tuple[Expr, Expr, Expr], tuple[Array, Array, Array], Expr, Array]
# instead of
# Function[tuple[Expr, Expr, Expr], tuple[ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]], Expr, ndarray[tuple[Any, ...], dtype[Any]]]


@dataclass(frozen=True, slots=True)
class _PatternHole(TensorType):
  """A leaf declaration whose stored pattern is supplied by a binding."""

  shape_hole: bool = False


def _is_hole(decl: LeafDecl) -> bool:
  return decl is None or isinstance(decl, _PatternHole)


def _shape_hole(decl: LeafDecl) -> bool:
  return decl is None or isinstance(decl, _PatternHole) and decl.shape_hole


def _leaves(value: Any) -> list[Any]:
  """Every non-tuple atom of ``value``, without consulting a declared structure."""
  if isinstance(value, tuple):
    return [leaf for item in value for leaf in _leaves(item)]
  return [value]


class Tree[Symbolic, Numerical]:
  """A pytree declaration whose leaves are ``Expr`` symbolically and NumPy arrays numerically.

  A tree *is* its structure: only ``group`` introduces a tuple, so a one-leaf tree is the bare leaf
  and never a one-element tuple. See ``arg`` for what that means at a call site.
  """

  names: tuple[str, ...]
  decls: tuple[LeafDecl, ...]
  layouts: tuple[LeafLayout, ...]
  _layout_decls: tuple[Layout, ...]

  def __setattr__(self, name: str, value: Any) -> None:
    if name in self.__dict__:
      raise AttributeError(f"tree attribute {name!r} is read-only")
    object.__setattr__(self, name, value)

  def __delattr__(self, name: str) -> None:
    raise AttributeError(f"tree attribute {name!r} is read-only")

  @property
  def has_holes(self) -> bool:
    """Whether any leaf needs binding."""
    return any(_is_hole(decl) for decl in self.decls)

  @property
  def shapes(self) -> tuple[tuple[int, ...], ...]:
    """The leaf shapes in C-signature order."""
    if any(_shape_hole(decl) for decl in self.decls):
      raise TypeError(f"tree {self.names} has inferred shapes; resolve them by tracing first")
    return tuple(cast(TensorType, decl).shape for decl in self.decls)

  @property
  def types(self) -> tuple[TensorType, ...]:
    """The leaf tensor types in C-signature order."""
    if self.has_holes:
      raise TypeError(f"tree {self.names} has shape or pattern holes; resolve them by tracing first")
    return cast(tuple[TensorType, ...], self.decls)

  @property
  def size(self) -> int:
    """The number of leaves."""
    return len(self.names)

  def symbols(self, *, diff: bool | None = None) -> Symbolic:
    """Create named input expressions with this tree's structure."""
    raise NotImplementedError

  def relabel(self, prefix: str) -> Tree[Symbolic, Numerical]:
    """Return the same structure with ``prefix`` added to every leaf name."""
    raise NotImplementedError

  def with_types(self, types: tuple[TensorType, ...]) -> Tree[Symbolic, Numerical]:
    """Return the same structure with resolved leaf types."""
    raise NotImplementedError

  def resolved(self, traced: tuple[TensorType, ...]) -> tuple[TensorType, ...]:
    """Check traced output types against this declaration and resolve inferred shapes."""
    if len(traced) != self.size:
      raise TypeError(f"declared {self.size} outputs {self.names}, body returned {len(traced)}")
    for name, decl, actual in zip(self.names, self.decls, traced, strict=True):
      if decl is not None and (
        not _shape_hole(decl)
        and decl.shape != actual.shape
        or decl.dtype != actual.dtype
        or not isinstance(decl, _PatternHole)
        and decl.pattern is not actual.pattern
      ):
        raise TypeError(f"{name!r} declared with type {decl}, traced type {actual}")
    return traced

  def index(self, name: str) -> int:
    """Return the flat index for ``name``, or raise with the declared choices."""
    if name not in self.names:
      raise ValueError(f"unknown name {name!r}; declared {self.names}")
    return self.names.index(name)

  def is_symbolic(self, value: Symbolic | Numerical, /) -> TypeGuard[Symbolic]:
    """Whether ``value`` has at least one leaf and every leaf is an ``Expr``.

    This is the leaf-kind half of ``Function.__call__``'s dispatch. It deliberately ignores
    structure so that a wrongly-shaped tree is reported by ``flatten_symbolic`` against the
    declared names instead of being rejected here as a kind mismatch.
    """
    leaves = _leaves(value)
    return bool(leaves) and all(isinstance(leaf, Expr) for leaf in leaves)

  def is_numerical(self, value: Symbolic | Numerical, /) -> TypeGuard[Numerical]:
    """Whether no leaf of ``value`` is an ``Expr``.

    The numerical side is the fallback: array-likes are coerced by ``flatten_numerical``, so a
    leaf only has to *not* be symbolic. Values with no leaves land here and are reported as a
    structure error rather than as a mixed call.
    """
    return not any(isinstance(leaf, Expr) for leaf in _leaves(value))

  def flatten_symbolic(self, value: Symbolic, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    """Validate and flatten a symbolic value."""
    raise NotImplementedError

  def flatten_numerical(self, value: Numerical, what: str) -> tuple[Array, ...]:
    """Validate and flatten a numerical value."""
    raise NotImplementedError

  def unflatten(self, values: tuple[Any, ...]) -> Any:
    """Rebuild this tree's structure from flat values."""
    raise NotImplementedError

  def _check_unique(self) -> None:
    if len(set(self.names)) != len(self.names):
      raise ValueError(f"duplicate names in {self.names}")


class _Leaf(Tree[Expr, Array]):
  """Declare one named tensor.

  The declared name is external metadata and need not match the local name used by a decorated
  function body. Pass a ``TensorType`` to set dtype or differentiability explicitly.
  ``pattern=...`` leaves the stored pattern to a call or an explicit ``TensorType`` binding.

  **A single leaf is passed and returned unpacked.** An ``arg`` input tree takes the tensor itself,
  not ``(tensor,)``, and an ``arg`` output tree returns the tensor itself, not a one-element tuple.
  Do not destructure a single-leaf result: ``(y,) = fn(x)`` does not raise, it iterates the
  returned tensor along its first axis exactly as NumPy would.
  """

  def __init__(self, name: str, shape: ShapeDecl = None, /, *, layout: Layout = None, pattern: EllipsisType | None = None) -> None:
    if not isinstance(name, str) or not name:
      raise ValueError("arg needs a non-empty name")
    self.names = (name,)
    if shape is None:
      decl = None
    elif isinstance(shape, TensorType):
      decl = shape
    else:
      decl = TensorType(as_shape(shape))
    if pattern is not None:
      if pattern is not Ellipsis:
        raise TypeError("a pattern hole is declared with pattern=...")
      if decl is not None and decl.pattern is not None:
        raise ValueError("a pattern hole cannot replace a concrete stored pattern")
      decl = _PatternHole(shape_hole=True) if decl is None else _PatternHole(decl.shape, decl.dtype, decl.diff)
    self.decls = (decl,)
    self._layout_decls = (layout,)
    if isinstance(decl, _PatternHole):
      if layout not in (None, "csc", "csr"):
        raise ValueError("a pattern hole needs a 'csc' or 'csr' layout")
      self.layouts = (None,)
    elif decl is None or decl.pattern is None:
      if layout is None or layout == "row-major":
        self.layouts = (None,)
      elif isinstance(layout, DenseLayout) or layout == "column-major":
        self.layouts = (None if decl is not None and len(decl.shape) < 2 else DenseLayout(None if decl is None else decl.shape),)
      else:
        raise ValueError("dense layout must be 'row-major' or 'column-major'")
    else:
      self.layouts = (sparse_layout(decl.pattern, "csc" if layout is None else layout),)

  def symbols(self, *, diff: bool | None = None) -> Expr:
    type_ = self.types[0]
    if diff is not None:
      type_ = replace(type_, diff=diff)
    return Expr(ExprOp.INPUT, type=type_, name=self.names[0])

  def relabel(self, prefix: str) -> _Leaf:
    return _Leaf(prefix + self.names[0], self.decls[0], layout=self.layouts[0] or self._layout_decls[0])

  def with_types(self, types: tuple[TensorType, ...]) -> _Leaf:
    if len(types) != 1:
      raise ValueError(f"arg expects one resolved type, got {len(types)}")
    layout = self.layouts[0] or self._layout_decls[0]
    if isinstance(self.decls[0], _PatternHole) and types[0].pattern is None:
      layout = None
    return _Leaf(self.names[0], types[0], layout=layout)

  def flatten_symbolic(self, value: Expr, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    if not isinstance(value, Expr):
      raise ValueError(f"{what}: expected an Expr for {self.names[0]!r}, got {type(value).__name__}")
    decl = self.decls[0]
    if decl is not None and not _shape_hole(decl) and value.shape != decl.shape and not (allow_scalar and value.shape == ()):
      raise ValueError(f"{what}: expected shape {decl.shape} for {self.names[0]!r}, got {value.shape}")
    if decl is not None and value.type.dtype != decl.dtype:
      raise ValueError(f"{what}: expected dtype {decl.dtype} for {self.names[0]!r}, got {value.type.dtype}")
    if decl is not None and not isinstance(decl, _PatternHole) and value.type.pattern is not decl.pattern:
      raise ValueError(f"{what}: different stored pattern for {self.names[0]!r}")
    return (value,)

  def flatten_numerical(self, value: NumericalArray, what: str) -> tuple[Array, ...]:
    if isinstance(value, Expr):
      raise ValueError(f"{what}: expected a numerical value for {self.names[0]!r}, got Expr")
    dtype = self.types[0].dtype
    layout = self.layouts[0]
    if isinstance(layout, SparseLayout):
      return (layout.numerical_values(value, self.shapes[0], dtype.numpy()),)
    if sparse.issparse(value):
      raise ValueError(f"{what}: sparse argument for dense leaf {self.names[0]!r}")
    array = np.asarray(value)
    if not np.can_cast(array.dtype, dtype.numpy(), "same_kind"):
      raise TypeError(f"{what}: cannot cast {array.dtype} to {dtype} for {self.names[0]!r}")
    array = array.astype(dtype.numpy(), copy=False)
    if array.shape != self.shapes[0]:
      raise ValueError(f"{what}: expected shape {self.shapes[0]} for {self.names[0]!r}, got {array.shape}")
    return (layout.numerical_values(array, self.shapes[0], dtype.numpy()) if layout is not None else np.require(array, requirements="C"),)

  def unflatten(self, values: tuple[Any, ...]) -> Any:
    if len(values) != 1:
      raise ValueError(f"arg expects one flat value, got {len(values)}")
    return values[0]


@overload
def arg(name: str, shape: int | tuple[int, ...], /) -> Tree[Expr, Array]: ...


@overload
def arg(name: str, shape: ShapeDecl = None, /, *, layout: Literal["row-major", "column-major"]) -> Tree[Expr, Array]: ...


@overload
def arg(name: str, shape: ShapeDecl = None, /, *, layout: Layout = None, pattern: EllipsisType | None = None) -> Tree[Expr, NumericalArray]: ...


def arg(name: str, shape: ShapeDecl = None, /, *, layout: Layout = None, pattern: EllipsisType | None = None) -> Tree[Expr, Any]:
  """Declare one named tensor, optionally leaving its shape to a call or trace.

  A leaf is passed and returned as a bare value. Use ``group`` to declare tuple structure.
  A ``TensorType`` sets the dtype and differentiability explicitly, and a stored pattern for a
  sparse matrix. Dense leaves use row-major buffers by default. ``layout="column-major"`` orders
  each matrix's values by column, keeping leading batch axes in row-major order. A sparse leaf's
  ``layout`` is ``"csc"`` (the default), ``"csr"``, or the coordinate order of a
  ``SparsityPattern``. ``pattern=...`` leaves the pattern to the call, with a ``"csc"`` or
  ``"csr"`` layout.
  """
  return _Leaf(name, shape, layout=layout, pattern=pattern)


class _G(Tree[Any, Any]):
  def __init__(self, parts: tuple[Tree[Any, Any], ...], *, public: bool = True) -> None:
    if public and not 1 <= len(parts) <= 8:
      raise TypeError(f"group takes 1 to 8 trees, got {len(parts)}; nest for more")
    self.parts = parts
    self.names = tuple(name for part in parts for name in part.names)
    self.decls = tuple(decl for part in parts for decl in part.decls)
    self.layouts = tuple(layout for part in parts for layout in part.layouts)
    self._layout_decls = tuple(layout for part in parts for layout in part._layout_decls)
    self._check_unique()

  def symbols(self, *, diff: bool | None = None) -> tuple[Any, ...]:
    return tuple(part.symbols(diff=diff) for part in self.parts)

  def relabel(self, prefix: str) -> _G:
    return _G(tuple(part.relabel(prefix) for part in self.parts), public=False)

  def with_types(self, types: tuple[TensorType, ...]) -> _G:
    out: list[Tree[Any, Any]] = []
    offset = 0
    for part in self.parts:
      out.append(part.with_types(types[offset : offset + part.size]))
      offset += part.size
    return _G(tuple(out), public=False)

  def flatten_symbolic(self, value: Any, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    if not isinstance(value, tuple) or len(value) != len(self.parts):
      raise ValueError(f"{what}: value does not have the declared structure of {self.names}")
    return tuple(expr for part, item in zip(self.parts, value, strict=True) for expr in part.flatten_symbolic(item, what, allow_scalar=allow_scalar))

  def flatten_numerical(self, value: Any, what: str) -> tuple[Array, ...]:
    if not isinstance(value, tuple) or len(value) != len(self.parts):
      raise ValueError(f"{what}: value does not have the declared structure of {self.names}")
    return tuple(array for part, item in zip(self.parts, value, strict=True) for array in part.flatten_numerical(item, what))

  def unflatten(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
    out: list[Any] = []
    offset = 0
    for part in self.parts:
      out.append(part.unflatten(values[offset : offset + part.size]))
      offset += part.size
    return tuple(out)


@overload
def group[SA, NA](a: Tree[SA, NA], /) -> Tree[tuple[SA], tuple[NA]]: ...


@overload
def group[SA, NA, SB, NB](a: Tree[SA, NA], b: Tree[SB, NB], /) -> Tree[tuple[SA, SB], tuple[NA, NB]]: ...


@overload
def group[SA, NA, SB, NB, SC, NC](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], /) -> Tree[tuple[SA, SB, SC], tuple[NA, NB, NC]]: ...


@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND](
  a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], /
) -> Tree[tuple[SA, SB, SC, SD], tuple[NA, NB, NC, ND]]: ...


@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE](
  a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], /
) -> Tree[tuple[SA, SB, SC, SD, SE], tuple[NA, NB, NC, ND, NE]]: ...


@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF](
  a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], /
) -> Tree[tuple[SA, SB, SC, SD, SE, SF], tuple[NA, NB, NC, ND, NE, NF]]: ...


@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG](
  a: Tree[SA, NA],
  b: Tree[SB, NB],
  c: Tree[SC, NC],
  d: Tree[SD, ND],
  e: Tree[SE, NE],
  f: Tree[SF, NF],
  g: Tree[SG, NG],
  /,
) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG], tuple[NA, NB, NC, ND, NE, NF, NG]]: ...


@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG, SH, NH](
  a: Tree[SA, NA],
  b: Tree[SB, NB],
  c: Tree[SC, NC],
  d: Tree[SD, ND],
  e: Tree[SE, NE],
  f: Tree[SF, NF],
  g: Tree[SG, NG],
  h: Tree[SH, NH],
  /,
) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG, SH], tuple[NA, NB, NC, ND, NE, NF, NG, NH]]: ...


@overload
def group[S, N](a: Tree[S, N], /, *parts: Tree[S, N]) -> Tree[tuple[S, ...], tuple[N, ...]]: ...


@overload
def group(a: Tree[Any, Any], /, *parts: Tree[Any, Any]) -> Tree[tuple[Any, ...], tuple[Any, ...]]: ...


def group(*parts: Tree[Any, Any]) -> Tree[Any, Any]:
  """Group one to eight trees side by side. Nest groups for greater widths."""
  return _G(parts)


def _flat_leaves(names: tuple[str, ...], types: tuple[LeafDecl, ...], layouts: tuple[LeafLayout, ...] | None) -> tuple[Tree[Any, Any], ...]:
  if len(names) != len(types):
    raise ValueError(f"expected {len(types)} names, got {len(names)}")
  return tuple(arg(name, type_, layout=layout) for name, type_, layout in zip(names, types, layouts or (None,) * len(types), strict=True))


def flat_tree(names: tuple[str, ...], types: tuple[LeafDecl, ...], *, layouts: tuple[LeafLayout, ...] | None = None) -> Tree[Any, Any]:
  """Build the private flat tree used by dynamic ``Function.factory`` results."""
  leaves = _flat_leaves(names, types, layouts)
  return leaves[0] if len(leaves) == 1 else _G(leaves, public=False)


def inferred_outputs(value: Any, name: str) -> Tree[Any, Any]:
  """Read output nesting and tensor types from a symbolic trace."""
  leaves = _leaves(value)
  if any(not isinstance(leaf, Expr) for leaf in leaves):
    raise TypeError("function body must return a pytree of Expr values")
  names = iter((name,) if len(leaves) == 1 else (f"out{i}" for i in range(len(leaves))))

  def build(item: Any) -> Tree[Any, Any]:
    if isinstance(item, tuple):
      return _G(tuple(build(part) for part in item), public=False)
    return arg(next(names), item.type)

  return build(value)


def parameter_list(trees: tuple[Tree[Any, Any], ...], /) -> _G:
  """Build one tree per positional parameter, including zero parameters."""
  return _G(trees, public=False)


def append_parameter(params: Tree[Any, Any], tree: Tree[Any, Any], /) -> _G:
  """Append one parameter while preserving each existing parameter's nesting."""
  return parameter_list((*cast(_G, params).parts, tree))


def flat_parameters(names: tuple[str, ...], types: tuple[TensorType, ...], *, layouts: tuple[LeafLayout, ...] | None = None) -> _G:
  """Build a parameter list with one tensor per parameter."""
  return parameter_list(_flat_leaves(names, types, layouts))
