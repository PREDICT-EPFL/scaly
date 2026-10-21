"""The installed package advertises its inline type annotations."""

from importlib.resources import files


def test_package_includes_type_marker() -> None:
  assert files("scaly").joinpath("py.typed").is_file()
