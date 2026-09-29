"""Generated-name allocation and ABI rendering use the same C spelling."""

import pytest

from scaly.codegen.abi import c_ident as abi_ident
from scaly.utils.names import c_ident


@pytest.mark.parametrize(
  "name,expected",
  [
    ("fwd:eq:z", "fwd_eq_z"),
    ("1value", "_1value"),
    ("w", "w_"),
    ("arg", "arg_"),
    ("_h0", "_h0"),
    ("new", "new_"),
    ("default", "default_"),
    ("int", "int_"),
    ("newer", "newer"),
  ],
)
def test_c_identifier_spelling(name: str, expected: str) -> None:
  assert c_ident(name) == expected
  assert abi_ident is c_ident
