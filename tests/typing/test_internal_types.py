"""Static checks for compiled handles, derivative caches, and tree transformations."""

from typing import TYPE_CHECKING, Any, assert_type

from scaly.ad.calls import _call_jvp_function
from scaly.ad.forward import _jvp
from scaly.ad.helpers import HelperKey
from scaly.codegen.jit import CompiledFunction
from scaly.function.concrete import ConcreteFunction, _Memo
from typing_playground.trees import LeafKey, Skeleton, map_skeleton, skeleton_shapes


if TYPE_CHECKING:

  def check_internal_types(function: ConcreteFunction[Any, Any, Any, Any]) -> None:
    memo = _Memo()
    assert_type(memo.compiled, CompiledFunction | None)
    memo.compiled = "not compiled"  # ty: ignore[invalid-assignment]
    assert_type(
      _call_jvp_function(function, 0, (0,), pushforward=_jvp), tuple[ConcreteFunction[Any, Any, Any, Any], tuple[int, ...], tuple[int, ...]]
    )
    _call_jvp_function("not a function", 0, (0,), pushforward=_jvp)  # ty: ignore[invalid-argument-type]
    key = HelperKey("forward", (0,), (0,), function._effective_lowering)
    memo.helpers[key] = (function, (0,), (0,))
    memo.helpers[key] = (function, "not indices", (0,))  # ty: ignore[invalid-assignment]
    memo.helpers["not a key"] = (function, (0,), (0,))  # ty: ignore[invalid-assignment]
    assert_type(skeleton_shapes((LeafKey((2,)), LeafKey(()))), tuple[tuple[int, ...], ...])
    assert_type(map_skeleton(LeafKey((2,)), lambda shape: (3, *shape)), Skeleton)
    skeleton_shapes("not a skeleton")  # ty: ignore[invalid-argument-type]
