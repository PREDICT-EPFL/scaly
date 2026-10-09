"""Lower expression graphs into verified, optimized programs with per-operation rules."""

from .ctx import LoweringError, lower_function, main_proc
from . import elementwise, movement, reduce, contraction, calls, gather, effects  # noqa: F401 -- registers the lowering rules

__all__ = ["LoweringError", "lower_function", "main_proc"]
