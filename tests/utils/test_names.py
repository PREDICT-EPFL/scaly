"""Generated-name allocation and ABI rendering use the same C spelling."""

import pytest

from alloy.codegen.abi import c_ident as abi_ident
from alloy.utils.names import c_ident


@pytest.mark.parametrize(
  "name,expected",
  [("fwd:eq:z", "fwd_eq_z"), ("1value", "_1value"), ("w", "w_"), ("arg", "arg_"), ("_h0", "_h0")],
)
def test_c_identifier_spelling(name: str, expected: str) -> None:
  assert c_ident(name) == expected
  assert abi_ident is c_ident
