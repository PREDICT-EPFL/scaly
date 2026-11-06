"""Paths to the vendored Fatrop C library and headers."""

from pathlib import Path


def include_dir() -> Path:
  """Return the directory containing Fatrop's C interface and BLASFEO headers."""
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  """Return the directory containing the Fatrop shared library."""
  return Path(__file__).resolve().parent / "lib"
