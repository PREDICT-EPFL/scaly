"""Typed declarations for symbolic and numerical function pytrees."""

from __future__ import annotations

from typing import Any, TypeGuard, cast, overload

import numpy as np

from ..ir.expr import Expr, ExprOp
from ..ir.types import TensorType, as_shape, dtypes

type ShapeDecl = int | tuple[int, ...] | TensorType | None
type LeafDecl = TensorType | None
type Array = np.ndarray
# declaring a type alias makes the inferred types of functions easier to read:
# Function[tuple[Expr, Expr, Expr], tuple[Array, Array, Array], Expr, Array]
# instead of
# Function[tuple[Expr, Expr, Expr], tuple[ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]], Expr, ndarray[tuple[Any, ...], dtype[Any]]]


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

  def __setattr__(self, name: str, value: Any) -> None:
    if name in self.__dict__:
      raise AttributeError(f"tree attribute {name!r} is read-only")
    object.__setattr__(self, name, value)

  def __delattr__(self, name: str) -> None:
    raise AttributeError(f"tree attribute {name!r} is read-only")

  @property
  def has_holes(self) -> bool:
    """Whether any leaf shape needs binding."""
    return any(decl is None for decl in self.decls)

  @property
  def shapes(self) -> tuple[tuple[int, ...], ...]:
    """The leaf shapes in C-signature order."""
    if any(decl is None for decl in self.decls):
      raise TypeError(f"tree {self.names} has inferred shapes; resolve them by tracing first")
    return tuple(cast(TensorType, decl).shape for decl in self.decls)

  @property
  def types(self) -> tuple[TensorType, ...]:
    """The leaf tensor types in C-signature order."""
    if any(decl is None for decl in self.decls):
      raise TypeError(f"tree {self.names} has inferred shapes; resolve them by tracing first")
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
      if decl is not None and (decl.shape != actual.shape or decl.dtype != actual.dtype):
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

  **A single leaf is passed and returned unpacked.** An ``arg`` input tree takes the tensor itself,
  not ``(tensor,)``, and an ``arg`` output tree returns the tensor itself, not a one-element tuple.
  Do not destructure a single-leaf result: ``(y,) = fn(x)`` does not raise, it iterates the
  returned tensor along its first axis exactly as NumPy would.
  """

  def __init__(self, name: str, shape: ShapeDecl = None, /) -> None:
    if not isinstance(name, str) or not name:
      raise ValueError("arg needs a non-empty name")
    self.names = (name,)
    if shape is None:
      self.decls = (None,)
    elif isinstance(shape, TensorType):
      self.decls = (shape,)
    else:
      self.decls = (TensorType(as_shape(shape)),)

  def symbols(self, *, diff: bool | None = None) -> Expr:
    type_ = self.types[0]
    if diff is not None:
      type_ = TensorType(type_.shape, type_.dtype, diff)
    return Expr(ExprOp.INPUT, type=type_, name=self.names[0])

  def relabel(self, prefix: str) -> _Leaf:
    return _Leaf(prefix + self.names[0], self.decls[0])

  def with_types(self, types: tuple[TensorType, ...]) -> _Leaf:
    if len(types) != 1:
      raise ValueError(f"arg expects one resolved type, got {len(types)}")
    return _Leaf(self.names[0], types[0])

  def flatten_symbolic(self, value: Expr, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    if not isinstance(value, Expr):
      raise ValueError(f"{what}: expected an Expr for {self.names[0]!r}, got {type(value).__name__}")
    decl = self.decls[0]
    if decl is not None and value.shape != decl.shape and not (allow_scalar and value.shape == ()):
      raise ValueError(f"{what}: expected shape {decl.shape} for {self.names[0]!r}, got {value.shape}")
    if decl is not None and value.type.dtype != decl.dtype:
      raise ValueError(f"{what}: expected dtype {decl.dtype} for {self.names[0]!r}, got {value.type.dtype}")
    return (value,)

  def flatten_numerical(self, value: Array, what: str) -> tuple[Array, ...]:
    if isinstance(value, Expr):
      raise ValueError(f"{what}: expected a numerical value for {self.names[0]!r}, got Expr")
    dtype = self.types[0].dtype
    if dtype != dtypes.float64:
      raise NotImplementedError(f"{what}: input leaf {self.names[0]!r} has dtype {dtype}; only float64 leaves are supported")
    array = np.asarray(value, dtype=dtype.numpy())
    if array.shape != self.shapes[0]:
      raise ValueError(f"{what}: expected shape {self.shapes[0]} for {self.names[0]!r}, got {array.shape}")
    return (np.require(array, requirements="C"),)

  def unflatten(self, values: tuple[Any, ...]) -> Any:
    if len(values) != 1:
      raise ValueError(f"arg expects one flat value, got {len(values)}")
    return values[0]


def arg(name: str, shape: ShapeDecl = None, /) -> Tree[Expr, Array]:
  """Declare one named tensor, optionally leaving its shape to a call or trace.

  A leaf is passed and returned as a bare value. Use ``group`` to declare tuple structure.
  A ``TensorType`` sets the dtype and differentiability explicitly.
  """
  return _Leaf(name, shape)


class _G(Tree[Any, Any]):
  def __init__(self, parts: tuple[Tree[Any, Any], ...], *, public: bool = True) -> None:
    if public and not 1 <= len(parts) <= 8:
      raise TypeError(f"group takes 1 to 8 trees, got {len(parts)}; nest for more")
    self.parts = parts
    self.names = tuple(name for part in parts for name in part.names)
    self.decls = tuple(decl for part in parts for decl in part.decls)
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


def group(*parts: Tree[Any, Any]) -> Tree[Any, Any]:
  """Group one to eight trees side by side. Nest groups for greater widths."""
  return _G(parts)


def flat_tree(names: tuple[str, ...], types: tuple[LeafDecl, ...]) -> Tree[Any, Any]:
  """Build the private flat tree used by dynamic ``Function.factory`` results."""
  if len(names) != len(types):
    raise ValueError(f"expected {len(types)} names, got {len(names)}")
  if len(names) == 1:
    return arg(names[0], types[0])
  return _G(tuple(arg(name, type_) for name, type_ in zip(names, types, strict=True)), public=False)


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


def flat_parameters(names: tuple[str, ...], types: tuple[TensorType, ...]) -> _G:
  """Build a parameter list with one tensor per parameter."""
  if len(names) != len(types):
    raise ValueError(f"expected {len(types)} names, got {len(names)}")
  return parameter_list(tuple(arg(name, type_) for name, type_ in zip(names, types, strict=True)))
