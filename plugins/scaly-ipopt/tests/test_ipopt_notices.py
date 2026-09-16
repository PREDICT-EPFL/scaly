"""Every dependency pinned in build_config.json ships its license texts in the wheel."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

import scaly_ipopt

PACKAGE = Path(scaly_ipopt.__file__).parent


@pytest.mark.solver("ipopt")
def test_license_directory_per_pinned_dependency():
  pinned = json.loads((PACKAGE / "build_config.json").read_text())
  licenses = PACKAGE / "licenses"
  notices = (licenses / "THIRD_PARTY_NOTICES.md").read_text()
  # Accelerate is an OS framework on macOS; only the Linux wheel bundles a BLAS.
  names = [n for n in pinned if n != "blas"] + (["openblas"] if platform.system() == "Linux" else [])
  for name in names:
    assert (licenses / name).is_dir() and any((licenses / name).iterdir()), f"no license text for {name}"
    assert f"| {name} |" in notices
