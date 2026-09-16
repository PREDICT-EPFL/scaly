"""``Function``: a graph with fully resolved leaf shapes, plus the derivative wrappers and ``vmap`` over it.

Every body here is a stand-in. The real ``Function.__init__`` traces the body as ``@sc.function``
does today; ``symbolic_call`` is a ``CALL`` node and ``numerical_call`` goes through the JIT. The
derivative wrappers are shape bookkeeping here and the existing ``function/api.py`` wrappers there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .expr import Buffer, Expr, Shape
from .trees import G, L, Tree, check_structure, leaves, shapes_of, unflatten


class Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A named graph from one declared input tree to one declared output tree. Both calls take the
  whole input tree as one positional argument."""

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
    check_structure(self.inputs, inputs, f"{self.name}.symbolic_call")
    return self._fn(inputs)

  def numerical_call(self, inputs: NumericalInputs, /) -> NumericalOutputs:
    check_structure(self.inputs, inputs, f"{self.name}.numerical_call")
    return cast(NumericalOutputs, unflatten(self.outputs, tuple(Buffer(s) for s in self.output_shapes)))

  def c_signature(self) -> str:
    """What the generated header spells: one pointer per leaf, in tree order."""
    ins = ", ".join(f"const double* {n}" for n in self.input_names)
    outs = ", ".join(f"double* {n}" for n in self.output_names)
    return f"void {self.name}({ins}, {outs})"


def function[SI, NI, SO, NO](
  inputs: Tree[SI, NI], outputs: Tree[SO, NO], /, *, name: str | None = None
) -> Callable[[Callable[[SI], SO]], Function[SI, NI, SO, NO]]:
  """Trace a callable over declared input and output trees, now."""

  def decorate(fn: Callable[[SI], SO]) -> Function[SI, NI, SO, NO]:
    return Function(name or getattr(fn, "__name__", "fn"), fn, inputs, outputs)

  return decorate


# --- derivatives: same inputs, one output; seeded modes pair the inputs with the new group ---


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


# --- vmap: a candidate, not a settled design (README, "Open items") ---


def vmap[SI, NI, SO, NO](callee: Function[SI, NI, SO, NO], length: int, /) -> Function[SI, NI, SO, NO]:
  """Map ``callee`` over a leading axis of ``length``: every leaf of both trees gains that axis and
  the structure is preserved, so the result is typed by the callee's trees. The real ``sc.vmap``
  slices flat outer tensors and returns a bare ``Expr``; this is the shape of what could replace it.
  A template callee is instantiated explicitly first: the per-iteration shape is vmap's input."""
  batched_in = callee.inputs.with_shapes(tuple((length, *s) for s in callee.input_shapes))
  batched_out = callee.outputs.with_shapes(tuple((length, *s) for s in callee.output_shapes))

  def body(inputs: SI) -> SO:
    return cast(SO, unflatten(batched_out, tuple(Expr(s, degree=None) for s in batched_out.shapes)))

  return Function(f"{callee.name}_vmap{length}", body, batched_in, batched_out)
