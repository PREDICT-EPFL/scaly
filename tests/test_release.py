"""The release and isolation scripts: the version rewrite, and what an isolated run of each
distribution installs, from where, and leaves out."""

from __future__ import annotations

import pytest

from scripts import distributions as dist
from scripts import isolation, release


def test_the_version_rewrite_touches_only_the_table_version() -> None:
  text = dist.TABLE.read_text()
  new = release.set_version(text, "9.8.7rc1")
  assert 'version = "9.8.7rc1"' in new and new.replace('version = "9.8.7rc1"', 'version = "0.1.0a1"', 1) == text.replace(
    f'version = "{dist.load().version}"', 'version = "0.1.0a1"', 1
  )
  for bad in ("1.2", "v1.2.3", "1.2.3.dev0"):
    with pytest.raises(ValueError, match="not a release version"):
      release.set_version(text, bad)


def test_an_isolated_run_installs_what_is_declared_and_nothing_else() -> None:
  table = dist.load()
  build, first_party, _ = isolation.plan("scaly-core", "head", table)
  assert build == first_party == ["scaly-core", "scaly-testing"]
  _, first_party, _ = isolation.plan("scaly-control", "head", table)
  assert first_party == ["scaly-control", "scaly-core", "scaly-numerics", "scaly-testing"]
  _, first_party, _ = isolation.plan("scaly-testing", "head", table)
  assert {"scaly-numerics", "scaly-control"} <= set(first_party)  # its extras, which its tests use
  _, _, index = isolation.plan("scaly-sqp", "head", table)
  assert any(req.startswith("casadi") for req in index)  # a plugin's third-party extra too


def test_an_isolated_run_leaves_out_what_another_distribution_owns() -> None:
  table = dist.load()
  assert isolation.ignored("scaly-control", table) == ["tests/ocp/test_altro.py", "tests/ocp/test_scvx.py"]
  assert isolation.ignored("scaly-core", table) == []


def test_a_plugin_against_a_range_end_takes_its_dependencies_from_the_index() -> None:
  table = dist.load()
  build, first_party, index = isolation.plan("scaly-piqp", "lowest", table)
  assert build == first_party == ["scaly-piqp"]
  assert any(req.startswith("scaly-numerics") for req in index) and "scaly-testing" in index
  with pytest.raises(SystemExit, match="releases with the others"):
    isolation.plan("scaly-control", "lowest", table)
