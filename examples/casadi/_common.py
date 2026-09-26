"""What every pair in this directory shares: the result printer and the quiet IPOPT options.

Each example exists twice, ``<name>_casadi.py`` and ``<name>_scaly.py``. Both define
``build(verbose=False)``, which does all the modelling and returns ``run``, a function of no
arguments that does the numerical work and returns a dict of NumPy arrays. ``compare.py`` times the
two halves separately and checks that the dicts agree key by key; running a file on its own prints
its results.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Callable, Mapping

import numpy as np

Run = Callable[[], dict[str, np.ndarray]]


def as_arrays(out: Mapping[str, object]) -> dict[str, np.ndarray]:
  """CasADi ``DM``s, floats and lists to flat float arrays, so both sides compare the same way."""
  return {k: np.asarray(_full(v), dtype=float).reshape(-1) for k, v in out.items()}


def _full(value: object) -> object:
  full = getattr(value, "full", None)
  return full() if callable(full) else value


def show(out: Mapping[str, np.ndarray]) -> None:
  with np.printoptions(precision=6, linewidth=120, threshold=12, edgeitems=4):
    for key, value in out.items():
      print(f"{key:>14}: {value if value.size != 1 else value.item():}")


def casadi_jit(force: bool = False, expand: bool = True) -> dict[str, object]:
  """CasADi's just-in-time compilation of the oracles, with the flags Scaly's JIT uses, when
  ``CASADI_JIT=1`` (``compare.py`` sets it for its third column) or ``force``; else nothing.

  ``expand`` (an ``nlpsol`` option) turns MX graphs into SX first, the usual companion of ``jit``:
  code generated from an unexpanded MX Hessian can take minutes to compile.
  """
  if not force and os.environ.get("CASADI_JIT") != "1":
    return {}
  native = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
  flags = ["-O2", "-ftree-vectorize", native, "-fno-math-errno"]
  return {"jit": True, "compiler": "shell", "jit_options": {"flags": flags}} | ({"expand": True} if expand else {})


def casadi_ipopt_options(verbose: bool, **ipopt: object) -> dict[str, object]:
  """``nlpsol`` options: IPOPT's own under ``ipopt.``, CasADi's timing table off unless verbose."""
  return {"print_time": verbose, **{f"ipopt.{k}": v for k, v in {"print_level": 5 if verbose else 0, "sb": "yes", **ipopt}.items()}}


def scaly_ipopt_options(verbose: bool, **ipopt: object) -> dict[str, object]:
  """The same IPOPT options for ``sc.solver(..., "ipopt", options=...)``, which passes them straight through."""
  return {"print_level": 5 if verbose else 0, "sb": "yes", **ipopt}
