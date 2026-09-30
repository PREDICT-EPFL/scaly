"""Every dependency pinned in build_config.json ships its license texts in the wheel, at the pinned
version: the notices are written by the build, so they tell a stale library from a current one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scaly_piqp

PACKAGE = Path(scaly_piqp.__file__).parent


@pytest.mark.method("opt.piqp")
def test_license_directory_per_pinned_dependency():
  pinned = json.loads((PACKAGE / "build_config.json").read_text())
  licenses = PACKAGE / "licenses"
  notices = (licenses / "THIRD_PARTY_NOTICES.md").read_text()
  for name, pin in pinned.items():
    assert (licenses / name).is_dir() and any((licenses / name).iterdir()), f"no license text for {name}"
    assert f"| {name} | {pin['version']} |" in notices, (
      f"{name} was built at another version than {pin['version']}: uv sync --reinstall-package scaly-piqp"
    )
