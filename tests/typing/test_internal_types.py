"""Static checks for compiled handles, derivative caches, and tree transformations."""

from typing import TYPE_CHECKING, Any, assert_type

from scaly.ad.calls import _call_jvp_many_function
from scaly.ad.forward import _pushforward
from scaly.ad.helpers import HelperKey
from scaly.codegen.jit import CompiledFunction
from scaly.function.concrete import ConcreteFunction, _Memo


if TYPE_CHECKING:

  def check_internal_types(function: ConcreteFunction[Any, Any, Any, Any]) -> None:
    memo = _Memo()
    assert_type(memo.compiled, CompiledFunction | None)
    memo.compiled = "not compiled"  # ty: ignore[invalid-assignment]
    assert_type(
      _call_jvp_many_function(function, 0, (0,), 1, (None,), pushforward=_pushforward),
      tuple[ConcreteFunction[Any, Any, Any, Any], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    )
    _call_jvp_many_function("not a function", 0, (0,), 1, (None,), pushforward=_pushforward)  # ty: ignore[invalid-argument-type]
    key = HelperKey("forward", (0,), (0,), function._effective_lowering)
    memo.helpers[key] = (function, (0,), (0,), (0,))
    memo.helpers[key] = (function, "not indices", (0,), (0,))  # ty: ignore[invalid-assignment]
    memo.helpers["not a key"] = (function, (0,), (0,), (0,))  # ty: ignore[invalid-assignment]
