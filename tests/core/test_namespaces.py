"""The namespaces of ``scaly/__init__.py``: ``sc.<name>`` imports a namespace on first use, is the
module an import of it gives, names the distribution to install when it is missing, and the package
merges a ``scaly`` directory found elsewhere on the path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import scaly as sc


def test_a_missing_namespace_names_its_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delattr(sc, "ocp", raising=False)
  monkeypatch.setitem(sys.modules, "scaly.ocp", None)  # importing it now fails as if not installed
  with pytest.raises(AttributeError, match=r"scaly\.ocp needs scaly-control, which is not installed \(uv add scaly-control\)"):
    sc.ocp
  assert not hasattr(sc, "ocp")


def test_graph_views_name_scaly_tools_when_it_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delattr(sc, "viz", raising=False)
  monkeypatch.setitem(sys.modules, "scaly.viz", None)
  with pytest.raises(AttributeError, match=r"scaly\.viz needs scaly-tools"):
    sc.expr_graph


def test_a_namespace_is_the_module_its_import_gives() -> None:
  linalg = pytest.importorskip("scaly.linalg", reason="needs scaly-numerics")
  assert sc.linalg is linalg


def test_the_package_merges_portions_found_on_the_path(tmp_path: Path) -> None:
  """A ``scaly`` directory without ``__init__.py`` elsewhere on the path, as a distribution installed
  to another target leaves, adds its modules to the package (``pkgutil.extend_path``)."""
  (tmp_path / "scaly").mkdir()
  (tmp_path / "scaly" / "probe_portion.py").write_text("WHERE = 'portion'\n")
  env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")]))}
  code = "import scaly.probe_portion as probe; assert probe.WHERE == 'portion'"
  proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True)
  assert proc.returncode == 0, proc.stderr
