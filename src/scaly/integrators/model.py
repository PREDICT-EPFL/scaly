"""The model contract: a ``Function`` ``f(x, ...) -> xdot``, and the discrete-time maps built over its signature."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..function.model import ConcreteFunction, Function
from ..function.sugar import scan
from ..function.tree import L, param_list
from ..ir.expr import Expr

UNROLL_STEPS = 4
"""The most substeps a discrete map unrolls; more run as a ``scan``, a loop in the generated C."""

type Rhs = Callable[[Expr], Expr]
"""The model's right-hand side at a state, every other argument held fixed over the step."""

type Step = Callable[[Expr, tuple[Expr, ...], Expr | float], Expr]
"""One step of a method: ``(x, others, h) -> x_next``, ``others`` the model's other leaves, flat."""

type Method = Callable[[ConcreteFunction[Any, Any, Any, Any], str, float | None], Step]
"""A method prepared for one model instance: ``(model, name, h) -> step``, ``h`` the substep size,
or ``None`` when it is an input and reaches the step as an ``Expr``. What a method builds once per
map (a stage Function and its derivative rules) it builds here, named from ``name``."""


def model_rhs(model: ConcreteFunction[Any, Any, Any, Any], others: tuple[Expr, ...]) -> Rhs:
  """The model called at a state, its other leaves ``others``: one call node per evaluation."""
  return lambda xs: model(*model.input_tree.unflatten((xs, *others)))


def check_model(f: ConcreteFunction[Any, Any, Any, Any], what: str) -> None:
  """Refuse a Function that is not a continuous-time model: its first parameter one vector leaf, the
  state, and its one output the state's derivative, shaped like it."""
  if not f.inputs or len(f.input_tree.parts[0].names) != 1:
    raise ValueError(f"{what}: the model's first parameter must be the state, one vector, got {f.input_names}")
  x = f.inputs[0]
  if len(x.shape) != 1 or not x.type.dtype.is_floating:
    raise ValueError(f"{what}: the state {f.input_names[0]!r} must be a float vector, got {x.type.dtype}{x.shape}")
  if len(f.outputs) != 1 or f.outputs[0].shape != x.shape:
    got = ", ".join(f"{n}{o.shape}" for n, o in zip(f.output_names, f.outputs, strict=True))
    raise ValueError(f"{what}: the model must have one output, the derivative of {f.input_names[0]!r} shaped {x.shape}; got {got}")


def parameter_names(f: ConcreteFunction[Any, Any, Any, Any]) -> list[str]:
  """The names a call of ``f`` binds keywords to: the body's parameters, else the leading leaf of each slot."""
  if f._signature is not None:
    return list(f._signature.parameters)
  return [part.names[0] if len(part.names) == 1 else f"arg{i}" for i, part in enumerate(f.input_tree.parts)]


def discrete_map(f: Function[Any, Any, Any, Any], label: str, dt: float | None, steps: int, name: str | None, method: Method) -> Any:
  """``F(x, ...) -> xnext``: ``steps`` applications of ``method``'s step over an interval ``dt``, with the
  model's own parameters, every one after the state held fixed over the interval, and a trailing
  ``dt`` parameter when ``dt`` is ``None``. A template model gives a template, one map per instance.

  The name defaults to ``{f.name}_{label}``. Substeps beyond ``UNROLL_STEPS`` run as a ``scan``."""
  if not isinstance(f, Function):
    raise TypeError(f"{label}: the model must be an sc.Function, got {type(f).__name__}")
  if dt is not None and not float(dt) > 0:
    raise ValueError(f"{label}: dt must be positive, or None to make it an input; got {dt}")
  if int(steps) != steps or steps < 1:
    raise ValueError(f"{label}: steps must be a positive integer, got {steps}")

  def build(model: ConcreteFunction[Any, Any, Any, Any], fname: str) -> ConcreteFunction[Any, Any, Any, Any]:
    check_model(model, fname)
    if dt is None and "dt" in model.input_names:
      raise ValueError(f"{fname}: the model already has an input named 'dt'; give the interval as a number, or rename the input")
    tree = model.input_tree
    state = model.inputs[0]
    step = method(model, fname, None if dt is None else float(dt) / steps)

    def body(*args: Any) -> Expr:
      *model_args, h = args if dt is None else (*args, float(dt))
      x, *others = tree.flatten_symbolic(tuple(model_args), fname)
      h = h / steps
      if steps <= UNROLL_STEPS:
        for _ in range(steps):
          x = step(x, tuple(others), h)
        return x
      leaves = [*others, *([h] if isinstance(h, Expr) else [])]
      sub = _substep(f"{fname}_substep", state, model.inputs[1:], isinstance(h, Expr), step, h)
      return scan(sub, x, [(leaf.reshape((leaf.size,)), 0, 0) for leaf in leaves], length=steps)[0]

    names = parameter_names(model) + (["dt"] if dt is None else [])
    setattr(body, "__signature__", inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in names]))
    slots = param_list(*tree.parts, *([L("dt", ())] if dt is None else []))
    return ConcreteFunction(fname, body, slots, L(f"{model.input_names[0]}next", state.type))

  default = f"{f.name}_{label}"
  if f.is_concrete:
    return build(f.concrete, name or default)
  return f.lift(
    lambda inst: build(inst, f"{inst.name}_{label}" if name is None else f"{name}__{inst.tokens}"), name or default, ("dt",) if dt is None else ()
  )


def _substep(name: str, state: Expr, others: tuple[Expr, ...], runtime_h: bool, step: Step, h: Expr | float) -> ConcreteFunction[Any, Any, Any, Any]:
  """One substep as a Function over flat leaves, the carry first, for ``scan``."""
  leaves = [L("x", state.type), *(L(f"a{i}", o.type) for i, o in enumerate(others)), *([L("h", ())] if runtime_h else [])]

  def body(xs: Expr, *rest: Expr) -> Expr:
    o, hs = (rest[:-1], rest[-1]) if runtime_h else (rest, h)
    return step(xs, tuple(o), hs)

  return ConcreteFunction(name, body, param_list(*leaves), L("xnext", state.type))
