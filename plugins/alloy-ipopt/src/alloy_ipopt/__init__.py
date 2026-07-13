from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from alloy.solvers import SolverDescriptor, SolverStatus


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


class _Backend:
  name = "ipopt"
  kind = "nlp"
  protocol_version = 1
  lib_stem = "ipopt"
  link_flags = ("-lipopt",)
  header = "coin-or/IpStdCInterface.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def run(self, descriptor: SolverDescriptor, inputs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], SolverStatus]:
    from ._ipopt import _nlp_backend

    return _nlp_backend(descriptor, inputs)


BACKEND = _Backend()
