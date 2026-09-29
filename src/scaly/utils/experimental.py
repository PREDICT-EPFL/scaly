"""The warning a module of scaly-experimental gives on import: it makes no stability promise."""

from __future__ import annotations

import warnings


class ExperimentalWarning(UserWarning):
  """Given on importing a module of scaly-experimental (``scaly.nn``, ``scaly.geometry``, the ALTRO
  and SCvx methods), which may change or go without notice. Silence it with
  ``warnings.filterwarnings("ignore", category=sc.ExperimentalWarning)``."""


def warn_experimental(module: str) -> None:
  """Warn that ``module``, being imported, is experimental."""
  warnings.warn(
    f"{module} is experimental (scaly-experimental): it may change or be removed without notice",
    ExperimentalWarning,
    stacklevel=3,
  )


__all__ = ["ExperimentalWarning", "warn_experimental"]
