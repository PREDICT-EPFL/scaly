"""C identifier spelling shared by generated-name allocation and rendering."""

from __future__ import annotations

import re

# Parameter names become C names by default, and a header is included from C or C++: a keyword of
# either language would not compile there.
_KEYWORDS = frozenset(
  """alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class compl concept
  const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default delete do double
  dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept
  not not_eq nullptr operator or or_eq private protected public register reinterpret_cast requires restrict return short signed
  sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename typeof
  union unsigned using virtual void volatile wchar_t while xor xor_eq""".split()
)


def c_ident(name: str) -> str:
  """Sanitize an Scaly name, reserving the generated C ABI's parameter names and the C and C++ keywords."""
  ident = re.sub(r"\W", "_", name)
  if ident in ("w", "arg", "res", "iw", "mem") or ident in _KEYWORDS:
    ident += "_"
  return f"_{ident}" if ident[:1].isdigit() else ident
