"""C identifier spelling shared by generated-name allocation and rendering."""

from __future__ import annotations

import re


def c_ident(name: str) -> str:
  """Sanitize an Alloy name, reserving the generated C ABI's parameter names."""
  ident = re.sub(r"\W", "_", name)
  if ident in ("w", "arg", "res", "iw", "mem"):
    ident += "_"
  return f"_{ident}" if ident[:1].isdigit() else ident
