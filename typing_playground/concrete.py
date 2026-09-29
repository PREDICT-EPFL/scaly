"""``ConcreteFunction``: one traced instance with every leaf shape resolved, and the transforms over one.

Users never build one. A ``Function`` owns its instances and hands them out through ``instances``
and ``instantiate``; everything shape-dependent (``input_shapes``, ``c_signature``) lives here. The
transforms are what ``function.py`` lifts over a ``Function``, so they are typed loosely and the
public wrappers carry the types.

Every body here is a stand-in. The real ``ConcreteFunction.__init__`` traces the body as
``@sc.function`` does today; ``symbolic_call`` is a ``CALL`` node and ``numerical_call`` goes through
the JIT. The derivative transforms are shape bookkeeping here and the existing ``function/api.py``
wrappers there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, overload

from .expr import Buffer, Expr, Shape
from .trees import L, Tree, append_parameter, check_structure, leaves, shapes_of, unflatten


class ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A named graph from a parameter list with resolved shapes to one output tree. The body and both
  calls take one argument per parameter; the input type parameters are those lists, as tuples."""

  def __init__(
    self,
    name: str,
    fn: Callable[..., SymbolicOutputs],
    inputs: Tree[SymbolicInputs, NumericalInputs],
    outputs: Tree[SymbolicOutputs, NumericalOutputs],
  ) -> None:
    self.name = name
    self._fn = fn
    self.inputs = inputs
    traced = fn(*cast(tuple[Any, ...], self.inputs.symbols(degree=1)))
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

  # The numerical overload comes first on purpose, as in `src/scaly/function/model.py`: a bare
  # function has an `Any` numerical side, both overloads then match, and the first one wins.
  # Evaluation is the reading that untyped code wants, and a typed symbolic call still resolves
  # exactly because `Expr` is not assignable to the numerical leaf type.
  @overload
  def __call__[*Ns](self: ConcreteFunction[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs: ...
  @overload
  def __call__[*Ss](self: ConcreteFunction[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs: ...
  def __call__(self, *args: Any) -> SymbolicOutputs | NumericalOutputs:
    """Call with one argument per parameter, dispatching on the leaves: all-``Expr`` builds a call
    node, all-numerical evaluates. A call mixing the two is an error; wrap the numerical leaves in
    ``const`` to make a symbolic call explicit."""
    inputs = cast(SymbolicInputs | NumericalInputs, args)
    if self.inputs.is_symbolic(inputs):
      return self._symbolic(inputs)
    if self.inputs.is_numerical(inputs):
      return self._numerical(inputs)
    raise TypeError(
      f"{self.name}: arguments mix Expr and numerical leaves; pass all-Expr leaves for a symbolic call "
      "or all-numerical leaves for an evaluation, wrapping constants in const if needed"
    )

  def symbolic_call[*Ss](self: ConcreteFunction[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs:
    return self._symbolic(args)

  def numerical_call[*Ns](self: ConcreteFunction[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs:
    return self._numerical(args)

  def _symbolic(self, args: SymbolicInputs) -> SymbolicOutputs:
    check_structure(self.inputs, args, f"{self.name}.symbolic_call")
    return self._fn(*cast(tuple[Any, ...], args))

  def _numerical(self, args: NumericalInputs) -> NumericalOutputs:
    check_structure(self.inputs, args, f"{self.name}.numerical_call")
    return cast(NumericalOutputs, unflatten(self.outputs, tuple(Buffer(s) for s in self.output_shapes)))

  def c_signature(self) -> str:
    """What the generated header spells: one pointer per leaf, in tree order."""
    ins = ", ".join(f"const double* {n}" for n in self.input_names)
    outs = ", ".join(f"double* {n}" for n in self.output_names)
    return f"void {self.name}({ins}, {outs})"


type Instance = ConcreteFunction[Any, Any, Any, Any]

# --- derivatives: same inputs, one output; seeded modes append the seed as one more parameter ---


def _shape_of_output(fn: Instance, of: str) -> Shape:
  return fn.output_shapes[fn.outputs.index(of)]


def _shape_of_input(fn: Instance, wrt: str) -> Shape:
  return fn.input_shapes[fn.inputs.index(wrt)]


def _derived(fn: Instance, name: str, shape: Shape, degree: int | None) -> Instance:
  def body(*args: Any) -> Expr:
    fn.symbolic_call(*args)
    return Expr(shape, degree=degree)

  return ConcreteFunction(f"{fn.name}_{name}", body, fn.inputs, L(name, shape))


def _lower_degree(degree: int | None, by: int) -> int | None:
  return None if degree is None else max(degree - by, 0)


def gradient(fn: Instance, of: str, wrt: str) -> Instance:
  """``d of / d wrt`` as a function of every input of ``fn``."""
  of_shape = _shape_of_output(fn, of)
  if of_shape != ():
    raise ValueError(f"gradient needs a scalar output, {of!r} has shape {of_shape}")
  degree = fn.output_degrees[fn.outputs.index(of)]
  return _derived(fn, f"grad_{of}_{wrt}", _shape_of_input(fn, wrt), _lower_degree(degree, 1))


def jacobian(fn: Instance, of: str, wrt: str) -> Instance:
  degree = fn.output_degrees[fn.outputs.index(of)]
  return _derived(fn, f"jac_{of}_{wrt}", (*_shape_of_output(fn, of), *_shape_of_input(fn, wrt)), _lower_degree(degree, 1))


def hessian(fn: Instance, of: str, wrt: str) -> Instance:
  of_shape = _shape_of_output(fn, of)
  if of_shape != ():
    raise ValueError(f"hessian needs a scalar output, {of!r} has shape {of_shape}")
  n = _shape_of_input(fn, wrt)
  degree = fn.output_degrees[fn.outputs.index(of)]
  return _derived(fn, f"hess_{of}_{wrt}_{wrt}", (*n, *n), _lower_degree(degree, 2))


def _seeded(fn: Instance, seed: Tree[Any, Any], name: str, shape: Shape) -> Instance:
  def body(*args: Any) -> Expr:
    fn.symbolic_call(*args[:-1])
    return Expr(shape, degree=None)

  return ConcreteFunction(f"{fn.name}_{name}", body, append_parameter(fn.inputs, seed), L(name, shape))


def forward(fn: Instance, of: str, wrt: str) -> Instance:
  """``J(of, wrt) @ seed``, called as ``fwd.numerical_call(*inputs, seed)``."""
  return _seeded(fn, L(f"fwd:{wrt}", _shape_of_input(fn, wrt)), f"fwd_{of}_{wrt}", _shape_of_output(fn, of))


def adjoint(fn: Instance, of: str, wrt: str) -> Instance:
  """``J(of, wrt).T @ lam``, called as ``adj.numerical_call(*inputs, lam)``."""
  return _seeded(fn, L(f"lam:{of}", _shape_of_output(fn, of)), f"adj_{of}_{wrt}", _shape_of_input(fn, wrt))


def lagrangian_hessian(fn: Instance, wrt: str) -> Instance:
  """Hessian of ``sum_i lam_i . out_i``: the multipliers are one more parameter shaped as the *output* tree of ``fn``."""
  n = _shape_of_input(fn, wrt)
  return _seeded(fn, fn.outputs.relabel("lam:"), f"hess_lagrangian_{wrt}", (*n, *n))


def vmap(fn: Instance, length: int) -> Instance:
  """``fn`` over a leading axis of ``length``: every leaf of both trees gains that axis."""
  batched_in = fn.inputs.with_shapes(tuple((length, *s) for s in fn.input_shapes))
  batched_out = fn.outputs.with_shapes(tuple((length, *s) for s in fn.output_shapes))

  def body(*args: Any) -> Any:
    return unflatten(batched_out, tuple(Expr(s, degree=None) for s in batched_out.shapes))

  return ConcreteFunction(f"{fn.name}_vmap{length}", body, batched_in, batched_out)
