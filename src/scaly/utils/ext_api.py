"""The version of the extension API (``scaly.ext``), and the check a package runs against it at import."""

from __future__ import annotations

EXT_API_VERSION = 1
"""Bumped by any incompatible change to what ``scaly.ext`` exports: the op registry and its rules and
traits, the lowering context, pass slots, option namespaces, the extern-callee protocol, output
adapters and the library-author Function API. It is part of every JIT cache key."""


def require_ext_api(version: int, package: str) -> None:
  """Refuse to load ``package``, written against extension API ``version``, into a Scaly that
  provides another."""
  if version != EXT_API_VERSION:
    raise ImportError(
      f"{package} is written against Scaly extension API {version}, but this Scaly provides {EXT_API_VERSION}; install matching versions"
    )
