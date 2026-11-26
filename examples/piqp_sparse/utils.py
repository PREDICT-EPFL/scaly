"""What the other modules share: PIQP's infinity, C++-style extrema, masked norms, and named groups of values for Scaly loops and branches."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

import scaly as sc

# PIQP_INF: a bound at or beyond it is absent.
INF = 1e30

F64 = sc.dtypes.float64
BOOL = sc.dtypes.bool_
I32 = sc.dtypes.int32
I64 = sc.dtypes.int64


def const_bool(value: bool) -> sc.Expr:
  return sc.const(np.array(value), dtype=BOOL)


def const_int(value: int) -> sc.Expr:
  """An int32 constant, the type of the status and of every counter."""
  return sc.const(np.array(value, dtype=np.int32), dtype=I32)


def index(values: Any) -> sc.Expr:
  """An int64 table read by ``sc.gather`` or ``sc.segment_sum``."""
  return sc.const(np.asarray(values, dtype=np.int64), dtype=I64)


def zeros(size: int) -> sc.Expr:
  return sc.const(np.zeros(size))


def to_float(mask: sc.Expr) -> sc.Expr:
  return sc.cast(mask, F64)


def _expr(x: Any) -> sc.Expr:
  return x if isinstance(x, sc.Expr) else sc.const(float(x))


def std_max(a: Any, b: Any) -> sc.Expr:
  """``std::max(a, b)``, which returns ``a`` when either is NaN, unlike ``sc.maximum``."""
  a, b = _expr(a), _expr(b)
  return sc.where(a < b, b, a)


def std_min(a: Any, b: Any) -> sc.Expr:
  """``std::min(a, b)``, which returns ``a`` when either is NaN."""
  a, b = _expr(a), _expr(b)
  return sc.where(b < a, b, a)


def inf_norm(x: sc.Expr) -> sc.Expr:
  """Eigen's ``lpNorm<Infinity>``, zero for an empty vector."""
  return sc.const(0.0) if x.size == 0 else sc.norm_inf(x)


def masked_max(x: sc.Expr, mask: sc.Expr) -> sc.Expr:
  """The largest entry of ``x`` where ``mask`` holds, ``-inf`` when none does."""
  return sc.const(-np.inf) if x.size == 0 else sc.where(mask, x, -np.inf).max()


def masked_min(x: sc.Expr, mask: sc.Expr) -> sc.Expr:
  """The smallest entry of ``x`` where ``mask`` holds, ``inf`` when none does."""
  return sc.const(np.inf) if x.size == 0 else sc.where(mask, x, np.inf).min()


def masked_inf_norm(x: sc.Expr, mask: sc.Expr) -> sc.Expr:
  """The infinity norm of the entries of ``x`` where ``mask`` holds."""
  return sc.const(0.0) if x.size == 0 else sc.where(mask, x.abs(), 0.0).max()


def all_true(mask: sc.Expr) -> sc.Expr:
  return const_bool(True) if mask.size == 0 else (to_float(mask).min() > 0.5)


def any_true(mask: sc.Expr) -> sc.Expr:
  return const_bool(False) if mask.size == 0 else (to_float(mask).max() > 0.5)


_names = itertools.count()


def unique_name(stem: str) -> str:
  """A Function name no other generated Function shares."""
  return f"{stem}_{next(_names)}"


@dataclass(frozen=True)
class Layout:
  """Named values as one Scaly tree: ``sc.group`` takes at most eight parts, so wider layouts nest."""

  names: tuple[str, ...]
  types: tuple[sc.TensorType, ...]
  # Names in one Function's inputs and outputs must differ, so two layouts that share a name,
  # such as the data's c and the scaling's c, declare theirs under different prefixes.
  prefix: str = ""

  @classmethod
  def of(cls, values: Mapping[str, sc.Expr], prefix: str = "") -> Layout:
    return cls(tuple(values), tuple(v.type for v in values.values()), prefix)

  def tree(self) -> Any:
    # A sparse value crosses a loop or branch in the graph's own row-major order.
    declare = lambda name, type_: sc.arg(self.prefix + name, type_, layout="csr" if type_.pattern is not None else None)
    return _nest([declare(name, type_) for name, type_ in zip(self.names, self.types, strict=True)], sc.group)

  def pack(self, values: Mapping[str, Any]) -> Any:
    return _nest([values[name] for name in self.names], lambda *parts: parts)

  def unpack(self, tree: Any) -> dict[str, sc.Expr]:
    return dict(zip(self.names, _flatten(tree), strict=True))


def _nest(parts: list[Any], group: Callable[..., Any]) -> Any:
  while len(parts) > 8:
    parts = [group(*parts[i : i + 8]) if len(parts[i : i + 8]) > 1 else parts[i] for i in range(0, len(parts), 8)]
  return group(*parts)


def _flatten(tree: Any) -> list[sc.Expr]:
  if isinstance(tree, tuple):
    return [leaf for part in tree for leaf in _flatten(part)]
  return [tree]


def branch(
  selector: sc.Expr,
  operands: Mapping[str, sc.Expr],
  if_false: Callable[[dict[str, sc.Expr]], Mapping[str, sc.Expr]],
  if_true: Callable[[dict[str, sc.Expr]], Mapping[str, sc.Expr]],
  outputs: Layout,
  name: str,
) -> dict[str, sc.Expr]:
  """``sc.cond`` over named operands: only the selected function runs in the generated code."""
  inputs = Layout.of(operands, "in_")
  outputs = Layout(outputs.names, outputs.types, "out_")

  def make(fn: Callable[[dict[str, sc.Expr]], Mapping[str, sc.Expr]], stem: str) -> sc.Function:
    @sc.function(inputs.tree(), outputs=outputs.tree(), name=unique_name(stem))
    def traced(tree: Any) -> Any:
      return outputs.pack(fn(inputs.unpack(tree)))

    return traced

  result = sc.cond(selector, (make(if_false, f"{name}_no"), make(if_true, f"{name}_yes")), inputs.pack(operands))
  return outputs.unpack(result)


def loop(
  keep_going: Callable[[dict[str, sc.Expr], dict[str, sc.Expr]], sc.Expr],
  step: Callable[[dict[str, sc.Expr], dict[str, sc.Expr]], Mapping[str, sc.Expr]],
  state: Mapping[str, sc.Expr],
  params: Mapping[str, sc.Expr],
  max_iter: int,
  name: str,
) -> dict[str, sc.Expr]:
  """``sc.while_loop`` over a named state, with named parameters every step reads whole."""
  carry, fixed = Layout.of(state, "carry_"), Layout.of(params, "param_")
  go = sc.arg("go", sc.TensorType((), BOOL))
  if params:

    @sc.function(carry.tree(), fixed.tree(), outputs=go, name=unique_name(f"{name}_cond"))
    def cond(tree: Any, param_tree: Any) -> sc.Expr:
      return keep_going(carry.unpack(tree), fixed.unpack(param_tree))

    @sc.function(carry.tree(), fixed.tree(), outputs=carry.tree(), name=unique_name(f"{name}_body"))
    def body(tree: Any, param_tree: Any) -> Any:
      return carry.pack(step(carry.unpack(tree), fixed.unpack(param_tree)))

    final, _ = sc.while_loop(cond, body, (carry.pack(state),), max_iter=max_iter, params=(fixed.pack(params),))
  else:

    @sc.function(carry.tree(), outputs=go, name=unique_name(f"{name}_cond"))
    def cond_alone(tree: Any) -> sc.Expr:
      return keep_going(carry.unpack(tree), {})

    @sc.function(carry.tree(), outputs=carry.tree(), name=unique_name(f"{name}_body"))
    def body_alone(tree: Any) -> Any:
      return carry.pack(step(carry.unpack(tree), {}))

    final, _ = sc.while_loop(cond_alone, body_alone, (carry.pack(state),), max_iter=max_iter)
  return carry.unpack(final)


class Kernel:
  """A sub-function generated once and called wherever it is used, as a C function is, rather than traced into each caller.

  ``fn`` maps named inputs to named outputs. The first call traces it for its inputs' names and
  types. Every later call must pass the same ones and calls the same Function, which the
  generated C defines once.
  """

  def __init__(self, fn: Callable[[dict[str, sc.Expr]], Mapping[str, sc.Expr]], name: str) -> None:
    self._fn, self._name = fn, name
    self._function: Any = None
    self._inputs: Layout | None = None
    self._outputs: Layout | None = None

  def __call__(self, inputs: Mapping[str, sc.Expr]) -> dict[str, sc.Expr]:
    # A constant is declared nondifferentiable; the declaration leaves that out, so that a
    # constant and a computed value of one type call the same Function.
    declare = lambda type_: sc.TensorType(type_.shape, type_.dtype, pattern=type_.pattern)  # noqa: E731
    layout = Layout(tuple(inputs), tuple(declare(v.type) for v in inputs.values()), "in_")
    if self._function is None:
      self._inputs = layout
      self._outputs = self._output_layout(layout, inputs)
      outputs = self._outputs

      def body(tree: Any) -> Any:
        return outputs.pack(self._fn(layout.unpack(tree)))

      self._function = sc.function(layout.tree(), outputs=outputs.tree(), name=unique_name(self._name))(body)
    elif layout != self._inputs:
      raise ValueError(f"{self._name}: called with other inputs than at its first call")
    assert self._outputs is not None
    return self._outputs.unpack(self._function(layout.pack(inputs)))

  def _output_layout(self, layout: Layout, inputs: Mapping[str, sc.Expr]) -> Layout:
    """The outputs' names and types, read from a first trace.

    Declared outputs keep a sparse value in the graph's row-major order, as ``Layout.tree`` does
    for inputs. Left to inference, a sparse output would take the default CSC order, and every
    call would permute its values on the way out and back.
    """
    names: list[str] = []

    def body(tree: Any) -> Any:
      out = self._fn(layout.unpack(tree))
      names.extend(out)
      return _nest([out[name] for name in names], lambda *parts: parts)

    probe = sc.function(layout.tree(), name=unique_name(f"{self._name}_probe"))(body)
    leaves = _flatten(probe(layout.pack(inputs)))
    return Layout(tuple(names), tuple(sc.TensorType(v.type.shape, v.type.dtype, pattern=v.type.pattern) for v in leaves), "out_")
