"""Structure declarations: ``Tree``, ``arg`` and ``group``, and the flatten/unflatten utilities over them.

A leaf declaration is a shape or ``...``, a hole. Holes are how output shapes are left to the trace
and how a declaration leaves input shapes to the call site; ``with_shapes`` fills them and ``resolved``
checks concrete shapes against the declaration. A function's inputs are a parameter list, a group
built by ``parameter_list`` with one tree per parameter; ``group`` is only ever structure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, TypeGuard, cast, overload

from .expr import Buffer, Expr, Shape, ShapeDecl, as_shape


class Tree[Symbolic, Numerical]:
  """A declared pytree of named tensors. ``Symbolic`` and ``Numerical`` are the same structure over
  ``Expr`` and ``Buffer``. ``names`` and ``decls`` are flat, in C-signature order."""

  names: tuple[str, ...]
  decls: tuple[ShapeDecl, ...]

  @property
  def shapes(self) -> tuple[Shape, ...]:
    if self.has_holes:
      raise TypeError(f"tree {self.names} has unresolved shapes")
    return cast(tuple[Shape, ...], self.decls)

  @property
  def size(self) -> int:
    return len(self.names)

  @property
  def has_holes(self) -> bool:
    return any(d is Ellipsis for d in self.decls)

  def symbols(self, degree: int = 0) -> Symbolic:
    raise NotImplementedError

  def relabel(self, prefix: str) -> Tree[Symbolic, Numerical]:
    """Same structure, every name prefixed (``lam:``, ``fwd:``); holes stay holes."""
    raise NotImplementedError

  def with_shapes(self, shapes: tuple[ShapeDecl, ...]) -> Tree[Symbolic, Numerical]:
    """Same structure with every decl replaced by the given shape or hole."""
    raise NotImplementedError

  def resolved(self, shapes: tuple[Shape, ...]) -> tuple[Shape, ...]:
    """Check ``shapes`` against the declaration; a hole accepts anything."""
    if len(shapes) != self.size:
      raise TypeError(f"declared {self.size} leaves {self.names}, got {len(shapes)}")
    for name, decl, shape in zip(self.names, self.decls, shapes, strict=True):
      if decl is not Ellipsis and decl != shape:
        raise TypeError(f"{name!r} declared with shape {decl}, got {shape}")
    return shapes

  def index(self, name: str) -> int:
    if name not in self.names:
      raise ValueError(f"unknown name {name!r}; declared {self.names}")
    return self.names.index(name)

  def is_symbolic(self, value: Symbolic | Numerical, /) -> TypeGuard[Symbolic]:
    """Whether ``value`` has at least one leaf and every leaf is an ``Expr``: the leaf-kind half of
    ``__call__``'s dispatch. Structure is left to the call it dispatches to, which reports it
    against the declared names."""
    got = leaves(value)
    return bool(got) and all(isinstance(v, Expr) for v in got)

  def is_numerical(self, value: Symbolic | Numerical, /) -> TypeGuard[Numerical]:
    """Whether no leaf of ``value`` is an ``Expr``. Values with no leaves land here and are reported
    as a structure error rather than as a mixed call."""
    return not any(isinstance(v, Expr) for v in leaves(value))

  def _check_unique(self) -> None:
    if len(set(self.names)) != len(self.names):
      raise ValueError(f"duplicate names in {self.names}")


class _Leaf(Tree[Expr, Buffer]):
  """Runtime class behind ``arg``. The real library's ``L`` also takes a ``TensorType`` for dtype and ``diff`` and
  requires the shape argument today; holes need it to be optional, as here."""

  def __init__(self, name: str, shape: int | ShapeDecl = ..., /) -> None:
    self.names = (name,)
    self.decls = (shape if isinstance(shape, EllipsisType) else as_shape(shape),)

  def symbols(self, degree: int = 0) -> Expr:
    return Expr(self.shapes[0], self.names[0], degree)

  def relabel(self, prefix: str) -> _Leaf:
    return _Leaf(prefix + self.names[0], self.decls[0])

  def with_shapes(self, shapes: tuple[ShapeDecl, ...]) -> _Leaf:
    return _Leaf(self.names[0], shapes[0])


def arg(name: str, shape: int | ShapeDecl = ..., /) -> Tree[Expr, Buffer]:
  """Declare one named tensor, an ``Expr`` in the body and a ``Buffer`` in a numerical call. Without
  a shape the leaf is a hole, bound per call on an input and traced on an output."""
  return _Leaf(name, shape)


class _Group(Tree[Any, Any]):
  """Runtime class behind ``group``; the static types live on ``group``'s overloads."""

  def __init__(self, parts: tuple[Tree[Any, Any], ...], *, public: bool = True) -> None:
    if public and not 1 <= len(parts) <= 8:
      raise TypeError(f"group takes 1 to 8 trees, got {len(parts)}; nest for more")
    self.parts = parts
    self.names = tuple(n for part in parts for n in part.names)
    self.decls = tuple(d for part in parts for d in part.decls)
    self._check_unique()

  def symbols(self, degree: int = 0) -> tuple[Any, ...]:
    return tuple(part.symbols(degree) for part in self.parts)

  def relabel(self, prefix: str) -> _Group:
    return _Group(tuple(part.relabel(prefix) for part in self.parts), public=False)

  def with_shapes(self, shapes: tuple[ShapeDecl, ...]) -> _Group:
    out: list[Tree[Any, Any]] = []
    i = 0
    for part in self.parts:
      out.append(part.with_shapes(shapes[i : i + part.size]))
      i += part.size
    return _Group(tuple(out), public=False)


# The one ladder in the design: deriving the Buffer structure from the Expr structure needs a
# type-level map, which Python lacks, so each width is spelled out (README, "Why a wrapper class").
# fmt: off
@overload
def group[SA, NA](a: Tree[SA, NA], /) -> Tree[tuple[SA], tuple[NA]]: ...
@overload
def group[SA, NA, SB, NB](a: Tree[SA, NA], b: Tree[SB, NB], /) -> Tree[tuple[SA, SB], tuple[NA, NB]]: ...
@overload
def group[SA, NA, SB, NB, SC, NC](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], /) -> Tree[tuple[SA, SB, SC], tuple[NA, NB, NC]]: ...
@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], /) -> Tree[tuple[SA, SB, SC, SD], tuple[NA, NB, NC, ND]]: ...
@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], /) -> Tree[tuple[SA, SB, SC, SD, SE], tuple[NA, NB, NC, ND, NE]]: ...
@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], /) -> Tree[tuple[SA, SB, SC, SD, SE, SF], tuple[NA, NB, NC, ND, NE, NF]]: ...
@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], /) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG], tuple[NA, NB, NC, ND, NE, NF, NG]]: ...
@overload
def group[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG, SH, NH](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], h: Tree[SH, NH], /) -> Tree[tuple[SA, SB, SC, SD, SE, SF, SG, SH], tuple[NA, NB, NC, ND, NE, NF, NG, NH]]: ...
# fmt: on
def group(*parts: Tree[Any, Any]) -> Tree[Any, Any]:
  """Group trees side by side: ``group(arg("x", 3), arg("p", ()))`` is ``tuple[Expr, Expr]`` / ``tuple[Buffer, Buffer]``.
  Groups are never normalized: ``group(arg("x"))`` is a one-element tuple, not a leaf."""
  return _Group(parts)


def parameter_list(trees: tuple[Tree[Any, Any], ...], /) -> _Group:
  """One tree per parameter, any number of them, zero included."""
  return _Group(trees, public=False)


def append_parameter(params: Tree[Any, Any], tree: Tree[Any, Any], /) -> _Group:
  """``params`` with ``tree`` as one more parameter, which is how seeded modes extend a signature."""
  return _Group((*cast(_Group, params).parts, tree), public=False)


def leaves(value: object, /) -> tuple[Expr | Buffer, ...]:
  """Flatten a pytree of values in declaration order; tuples are structure, everything else a leaf."""
  if isinstance(value, (Expr, Buffer)):
    return (value,)
  if isinstance(value, tuple):
    return tuple(item for part in value for item in leaves(part))
  raise TypeError(f"expected a pytree of Expr or Buffer values, got {type(value).__name__}")


def shapes_of(value: object, /) -> tuple[Shape, ...]:
  return tuple(v.shape for v in leaves(value))


@dataclass(frozen=True)
class LeafKey:
  """One leaf of a skeleton. Marked, so a scalar leaf is never mistaken for an empty group."""

  shape: Shape


def skeleton(value: object, /) -> Any:
  """Nested ``LeafKey``s: the structure and shapes of a call, which is what binds an instance when
  the declaration does not fix the structure."""
  if isinstance(value, tuple):
    return tuple(skeleton(v) for v in value)
  return LeafKey(leaves(value)[0].shape)


def skeleton_of(tree: Tree[Any, Any], shapes: tuple[Shape, ...], /) -> Any:
  return unflatten(tree, tuple(LeafKey(s) for s in shapes))


def skeleton_shapes(skel: Any, /) -> tuple[Shape, ...]:
  if isinstance(skel, tuple):
    return tuple(s for part in skel for s in skeleton_shapes(part))
  return (cast(LeafKey, skel).shape,)


def map_skeleton(skel: Any, f: Callable[[Shape], Shape], /) -> Any:
  if isinstance(skel, tuple):
    return tuple(map_skeleton(part, f) for part in skel)
  return LeafKey(f(cast(LeafKey, skel).shape))


def unflatten[S, N](tree: Tree[S, N], values: tuple[Any, ...]) -> Any:
  """Rebuild ``tree``'s structure from flat values (the C side hands back flat buffers)."""
  if isinstance(tree, _Leaf):
    return values[0]
  assert isinstance(tree, _Group)
  out: list[Any] = []
  i = 0
  for part in tree.parts:
    out.append(unflatten(part, values[i : i + part.size]))
    i += part.size
  return tuple(out)


def same_structure(tree: Tree[Any, Any], value: object) -> bool:
  """Whether ``value``, a pytree of values or a skeleton, nests like ``tree``, ignoring shapes."""
  if isinstance(tree, _Leaf):
    return not isinstance(value, tuple)
  assert isinstance(tree, _Group)
  return isinstance(value, tuple) and len(value) == len(tree.parts) and all(same_structure(t, v) for t, v in zip(tree.parts, value, strict=True))


def check_structure(tree: Tree[Any, Any], value: object, what: str) -> None:
  if not same_structure(tree, value):
    raise ValueError(f"{what}: value does not have the declared structure of {tree.names}")
  if shapes_of(value) != tree.shapes:
    raise ValueError(f"{what}: expected shapes {tree.shapes} for {tree.names}, got {shapes_of(value)}")


def inferred_tree(skel: Any, names: Iterator[str], /) -> Tree[Any, Any]:
  """A declaration read off a skeleton, with the given leaf names. This is how the bare mode reads its
  inputs; ``flat_tree`` is the real library's analogue."""
  if isinstance(skel, tuple):
    return _Group(tuple(inferred_tree(v, names) for v in skel), public=False)
  return arg(next(names), cast(LeafKey, skel).shape)


def inferred_outputs(value: object, name: str, /) -> Tree[Any, Any]:
  """The output tree read off a traced value: one leaf is named ``name``, several ``out0``, ``out1``..."""
  skel = skeleton(value)
  n = len(skeleton_shapes(skel))
  return inferred_tree(skel, iter((name,) if n == 1 else (f"out{i}" for i in range(n))))
