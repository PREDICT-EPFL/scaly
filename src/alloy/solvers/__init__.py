"""QP and NLP solver bindings.

This package wires vendored PIQP and IPOPT shared libraries into Alloy via ctypes
and exposes the user-facing builders :func:`qp` and :func:`nlp`. Both builders
return a callable :class:`SolverFunction` whose oracle is an ordinary Alloy
``Function`` (so it goes through the existing JIT path) and whose backend is an
opaque solver call from the outside.
"""

from __future__ import annotations

from .nlp import nlp
from .qp import qp
from .solver_function import SolverDescriptor, SolverFunction, SolverStatus

__all__ = ["SolverDescriptor", "SolverFunction", "SolverStatus", "nlp", "qp"]
