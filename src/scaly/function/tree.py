"""Typed declarations for symbolic and numerical function pytrees."""

from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from ..ir.expr import Expr, ExprOp
from ..ir.types import DType, SparsityType, TensorType, as_dtype, as_shape, dtypes

if TYPE_CHECKING:
  from typing_extensions import TypeVar

  # A tree spec that is a shape or a name is one leaf, so its type variables default to the leaf types.
  SA = TypeVar("SA", default=Expr)
  NA = TypeVar("NA", default=np.ndarray)
  SB = TypeVar("SB", default=Expr)
  NB = TypeVar("NB", default=np.ndarray)
  SC = TypeVar("SC", default=Expr)
  NC = TypeVar("NC", default=np.ndarray)
  SD = TypeVar("SD", default=Expr)
  ND = TypeVar("ND", default=np.ndarray)
  SE = TypeVar("SE", default=Expr)
  NE = TypeVar("NE", default=np.ndarray)
  SF = TypeVar("SF", default=Expr)
  NF = TypeVar("NF", default=np.ndarray)
  SG = TypeVar("SG", default=Expr)
  NG = TypeVar("NG", default=np.ndarray)
  SH = TypeVar("SH", default=Expr)
  NH = TypeVar("NH", default=np.ndarray)
  SO = TypeVar("SO", default=Expr)
  NO = TypeVar("NO", default=np.ndarray)


type ShapeDecl = int | np.integer | tuple[int, ...] | EllipsisType | TensorType
type LeafDecl = TensorType | EllipsisType
type Spec = ShapeDecl | str
"""A tree spec that is not a ``Tree``: a shape (an unnamed leaf) or a name (a named leaf with a shape hole)."""


class SymbolicValue:
  """Base for library values that stand in a tree for one ``Expr`` leaf, as ``SparseMatrix`` does for
  its values: a call whose leaves are all ``Expr`` or ``SymbolicValue`` is symbolic."""

  __slots__ = ()


def _is_symbolic_leaf(leaf: Any) -> bool:
  return isinstance(leaf, (Expr, SymbolicValue))


def _leaves(value: Any) -> list[Any]:
  """Every non-tuple atom of ``value``, without consulting a declared structure."""
  if isinstance(value, tuple):
    return [leaf for item in value for leaf in _leaves(item)]
  return [value]


def is_symbolic_call(args: tuple[Any, ...], what: str) -> bool:
  """Whether a call's arguments take the symbolic path: every leaf an ``Expr`` or ``SymbolicValue``.

  Structure is not consulted, so a wrongly shaped argument is reported by ``flatten_*`` against the
  declared names instead of as a kind mismatch. No leaves at all is numerical: ``f()`` evaluates.
  """
  symbolic = numerical = False
  for leaf in _leaves(args):
    if _is_symbolic_leaf(leaf):
      symbolic = True
    else:
      numerical = True
  if symbolic and numerical:
    raise TypeError(
      f"{what}: inputs mix Expr and numerical leaves; pass all-Expr leaves for a symbolic call "
      f"or all-numerical leaves for an evaluation, wrapping constants in scaly.const if needed"
    )
  return symbolic


class Tree[Symbolic, Numerical]:
  """A pytree declaration whose leaves are ``Expr`` symbolically and NumPy arrays numerically.

  A tree *is* its structure: only ``G`` introduces a tuple, so a one-leaf tree is the bare leaf
  and never a one-element tuple. See ``L`` for what that means at a call site.
  """

  names: tuple[str, ...]
  decls: tuple[LeafDecl, ...]

  @property
  def shapes(self) -> tuple[tuple[int, ...], ...]:
    """The leaf shapes in C-signature order."""
    if any(decl is Ellipsis for decl in self.decls):
      raise TypeError(f"tree {self.names} has inferred shapes; resolve them by tracing first")
    return tuple(cast(TensorType, decl).shape for decl in self.decls)

  @property
  def types(self) -> tuple[TensorType, ...]:
    """The leaf tensor types in C-signature order."""
    if any(decl is Ellipsis for decl in self.decls):
      raise TypeError(f"tree {self.names} has inferred shapes; resolve them by tracing first")
    return cast(tuple[TensorType, ...], self.decls)

  @property
  def size(self) -> int:
    """The number of leaves."""
    return len(self.names)

  def symbols(self, *, diff: bool | None = None) -> Symbolic:
    """Create named input expressions with this tree's structure."""
    raise NotImplementedError

  def named(self, base: str) -> Tree[Symbolic, Numerical]:
    """Name the unnamed leaves after ``base``: a leaf is ``base``, a group's parts ``base_0``, ``base_1``, ..."""
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
      if decl is not Ellipsis and (decl.shape != actual.shape or decl.dtype != actual.dtype):
        raise TypeError(f"{name!r} declared with type {decl}, traced type {actual}")
    return traced

  def infer(self, value: Symbolic) -> Tree[Symbolic, Numerical]:
    """Return this declaration with what it leaves to tracing, such as a sparse output's pattern,
    taken from the traced ``value``. Shapes are resolved separately, by ``with_types``."""
    return self

  @property
  def sparsities(self) -> tuple[SparsityType | None, ...]:
    """The pattern each leaf's values are stored in, ``None`` for a dense leaf, in C-signature order."""
    return (None,) * self.size

  def index(self, name: str) -> int:
    """Return the flat index for ``name``, or raise with the declared choices."""
    if name not in self.names:
      raise ValueError(f"unknown name {name!r}; declared {self.names}")
    return self.names.index(name)

  def flatten_symbolic(self, value: Symbolic, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    """Validate and flatten a symbolic value."""
    raise NotImplementedError

  def flatten_numerical(self, value: Numerical, what: str) -> tuple[np.ndarray, ...]:
    """Validate and flatten a numerical value."""
    raise NotImplementedError

  def unflatten(self, values: tuple[Any, ...]) -> Any:
    """Rebuild this tree's structure from flat values."""
    raise NotImplementedError

  def _check_unique(self) -> None:
    named = [name for name in self.names if name]
    if len(set(named)) != len(named):
      raise ValueError(f"duplicate names in {self.names}")


def _unnamed(name: str) -> ValueError:
  return ValueError(f"an unnamed leaf takes its name from sc.function's parameter; name it here, as in sc.L('x', ...) (declared {name!r})")


def _leaf_decl(shape: Any, dtype: DType | str | None, diff: bool | None) -> LeafDecl:
  if isinstance(shape, TensorType):
    if dtype is not None or diff is not None:
      raise TypeError("a TensorType carries its own dtype and diff; pass them to it instead")
    return shape
  if shape is Ellipsis:
    if dtype is not None or diff is not None:
      raise TypeError("dtype= and diff= need a shape")
    return Ellipsis
  if isinstance(shape, np.integer):
    shape = int(shape)
  if isinstance(shape, bool) or not isinstance(shape, (int, tuple)):
    raise TypeError(f"a leaf shape is an int, a tuple of ints, ... or a TensorType, got {shape!r}; use () for a scalar")
  return TensorType(as_shape(shape), as_dtype(dtype) if dtype is not None else dtypes.float64, diff=True if diff is None else diff)


class L(Tree[Expr, np.ndarray]):
  """Declare one named tensor.

  The declared name is external metadata and need not match the local name used by a decorated
  function body. Pass a ``TensorType`` to set dtype or differentiability explicitly.

  **A single leaf is passed and returned unpacked.** An ``L`` input tree takes the tensor itself,
  not ``(tensor,)``, and an ``L`` output tree returns the tensor itself, not a one-element tuple.
  Do not destructure a single-leaf result: ``(y,) = fn(x)`` does not raise, it iterates the
  returned tensor along its first axis exactly as NumPy would.
  """

  @overload
  def __init__(self, name: str, shape: ShapeDecl = ..., /, *, dtype: DType | str | None = None, diff: bool | None = None) -> None: ...

  @overload
  def __init__(self, shape: ShapeDecl = ..., /, *, dtype: DType | str | None = None, diff: bool | None = None) -> None: ...

  def __init__(self, first: str | ShapeDecl = ..., shape: ShapeDecl = ..., /, *, dtype: DType | str | None = None, diff: bool | None = None) -> None:
    if isinstance(first, str):
      if not first:
        raise ValueError("L needs a non-empty name")
      name = first
    elif shape is not Ellipsis:
      raise TypeError("L takes a name and a shape, or a shape alone")
    else:
      name, shape = "", first
    self.names = (name,)
    self.decls = (_leaf_decl(shape, dtype, diff),)

  def symbols(self, *, diff: bool | None = None) -> Expr:
    if not self.names[0]:
      raise _unnamed(self.names[0])
    type_ = self.types[0]
    if diff is not None:
      type_ = TensorType(type_.shape, type_.dtype, type_.sparsity, diff)
    return Expr(ExprOp.INPUT, type=type_, name=self.names[0])

  def named(self, base: str) -> L:
    return self if self.names[0] else L(base, self.decls[0])

  def relabel(self, prefix: str) -> L:
    return L(prefix + self.names[0], self.decls[0])

  def with_types(self, types: tuple[TensorType, ...]) -> L:
    if len(types) != 1:
      raise ValueError(f"L expects one resolved type, got {len(types)}")
    return L(self.names[0], types[0])

  def flatten_symbolic(self, value: Expr, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    if not isinstance(value, Expr):
      raise ValueError(f"{what}: expected an Expr for {self.names[0]!r}, got {type(value).__name__}")
    decl = self.decls[0]
    if decl is not Ellipsis and value.shape != decl.shape and not (allow_scalar and value.shape == ()):
      raise ValueError(f"{what}: expected shape {decl.shape} for {self.names[0]!r}, got {value.shape}")
    return (value,)

  def flatten_numerical(self, value: np.ndarray, what: str) -> tuple[np.ndarray, ...]:
    decl = self.decls[0]
    # The common case, an array already of the declared shape, dtype and layout, as is.
    if (
      type(value) is np.ndarray
      and decl is not Ellipsis
      and value.shape == decl.shape
      and value.dtype == decl.dtype.numpy()
      and value.flags.c_contiguous
    ):
      return (value,)
    if isinstance(value, Expr):
      raise ValueError(f"{what}: expected a numerical value for {self.names[0]!r}, got Expr")
    array = np.asarray(value, dtype=self.types[0].dtype.numpy())
    if array.shape != self.shapes[0]:
      raise ValueError(f"{what}: expected shape {self.shapes[0]} for {self.names[0]!r}, got {array.shape}")
    return (np.require(array, requirements="C"),)

  def unflatten(self, values: tuple[Any, ...]) -> Any:
    if len(values) != 1:
      raise ValueError(f"L expects one flat value, got {len(values)}")
    return values[0]


class _G(Tree[Any, Any]):
  def __init__(self, parts: tuple[Tree[Any, Any], ...], *, public: bool = True) -> None:
    if public and not 2 <= len(parts) <= 8:
      raise TypeError(f"G takes 2 to 8 trees, got {len(parts)}; nest for more")
    self.parts = parts
    self.names = tuple(name for part in parts for name in part.names)
    self.decls = tuple(decl for part in parts for decl in part.decls)
    self._check_unique()

  def symbols(self, *, diff: bool | None = None) -> tuple[Any, ...]:
    return tuple(part.symbols(diff=diff) for part in self.parts)

  def named(self, base: str) -> _G:
    return _G(tuple(part.named(f"{base}_{i}") for i, part in enumerate(self.parts)), public=False)

  def relabel(self, prefix: str) -> _G:
    return _G(tuple(part.relabel(prefix) for part in self.parts), public=False)

  def with_types(self, types: tuple[TensorType, ...]) -> _G:
    out: list[Tree[Any, Any]] = []
    offset = 0
    for part in self.parts:
      out.append(part.with_types(types[offset : offset + part.size]))
      offset += part.size
    return _G(tuple(out), public=False)

  def infer(self, value: Any) -> _G:
    return _G(tuple(part.infer(item) for part, item in zip(self.parts, value, strict=True)), public=False)

  @property
  def sparsities(self) -> tuple[SparsityType | None, ...]:
    return tuple(sp for part in self.parts for sp in part.sparsities)

  def flatten_symbolic(self, value: Any, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    if not isinstance(value, tuple) or len(value) != len(self.parts):
      raise ValueError(f"{what}: value does not have the declared structure of {self.names}")
    return tuple(expr for part, item in zip(self.parts, value, strict=True) for expr in part.flatten_symbolic(item, what, allow_scalar=allow_scalar))

  def flatten_numerical(self, value: Any, what: str) -> tuple[np.ndarray, ...]:
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


# fmt: off
@overload
def G(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, /) -> Tree[tuple[SA, SB], tuple[NA, NB]]: ...
@overload
def G(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, /) -> Tree[tuple[SA, SB, SC], tuple[NA, NB, NC]]: ...
@overload
def G(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, /) -> Tree[tuple[SA, SB, SC, SD], tuple[NA, NB, NC, ND]]: ...
@overload
def G(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, /) -> Tree[tuple[SA, SB, SC, SD, SE], tuple[NA, NB, NC, ND, NE]]: ...
@overload
def G(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, /) -> Tree[tuple[SA, SB, SC, SD, SE, SF], tuple[NA, NB, NC, ND, NE, NF]]: ...
@overload
def G(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, /) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG], tuple[NA, NB, NC, ND, NE, NF, NG]]: ...
@overload
def G(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, h: Tree[SH, NH] | Spec, /) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG, SH], tuple[NA, NB, NC, ND, NE, NF, NG, NH]]: ...
# fmt: on
def G(*parts: Tree[Any, Any] | Spec) -> Tree[Any, Any]:
  """Group two to eight trees side by side; nest groups for greater widths. A part may be a tree spec."""
  return _G(tuple(as_tree(part) for part in parts))


def as_tree(spec: Tree[Any, Any] | Spec) -> Tree[Any, Any]:
  """A tree spec as a tree: a ``Tree`` is itself, a name a named leaf with a shape hole, a shape an unnamed leaf."""
  return spec if isinstance(spec, Tree) else L(spec)


def param_list(*slots: Tree[Any, Any]) -> _G:
  """The input tree of a Function: one slot per parameter, which a call passes as separate arguments."""
  return _G(slots, public=False)


def flat_tree(names: tuple[str, ...], types: tuple[TensorType, ...]) -> Tree[Any, Any]:
  """Build the private flat tree used by dynamic ``Function.factory`` results."""
  if len(names) != len(types):
    raise ValueError(f"expected {len(types)} names, got {len(names)}")
  if len(names) == 1:
    return L(names[0], types[0])
  return _G(tuple(L(name, type_) for name, type_ in zip(names, types, strict=True)), public=False)
