"""Exercise native producer admission, binary audits and publication rejection controls."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load(name):
  path = Path(__file__).with_name(name + ".py")
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_all_five_targets_and_single_target_selection():
  native = load("ipopt_native")
  assert len(native.matrix("all")["include"]) == 5
  for target in native.TARGETS:
    assert [row["target"] for row in native.matrix(target)["include"]] == [target]
  with pytest.raises(ValueError):
    native.matrix("untrusted")


@pytest.mark.parametrize("name", ["/tmp/build/libgfortran.so.5", "libmissing.so", "@rpath/libgfortran.5.dylib"])
def test_unresolved_imports_and_stale_paths_fail(name):
  audit = load("ipopt_native_audit")
  with pytest.raises(RuntimeError):
    audit.resolve_import("Linux", name, "scaly_ipopt/lib/libipopt.so", {}, [])


def test_linux_origin_sidecar_resolution():
  audit = load("ipopt_native_audit")
  files = {"scaly_ipopt.libs/libquadmath-private.so": {}}
  assert (
    audit.resolve_import("Linux", "libquadmath-private.so", "scaly_ipopt/lib/libgfortran.so.5", files, ["$ORIGIN/../../scaly_ipopt.libs"])
    == "scaly_ipopt.libs/libquadmath-private.so"
  )
  with pytest.raises(RuntimeError):
    audit.resolve_import("Linux", "libquadmath-private.so", "scaly_ipopt/lib/libgfortran.so.5", files, ["/original/build"])


def test_floor_gate_rejects_a_newer_glibc():
  audit = load("ipopt_native_audit")
  audit.check_versions(["GLIBC_2.27", "GLIBCXX_3.4.19"], {"GLIBC": (2, 27), "GLIBCXX": (3, 4, 19)})
  with pytest.raises(RuntimeError, match="GLIBC_2.28"):
    audit.check_versions(["GLIBC_2.28"], {"GLIBC": (2, 27)})


def test_delay_imports_are_audited():
  audit = load("ipopt_native_audit")

  class Entry:
    dll = b"missing-delayed.dll"

  class PE:
    DIRECTORY_ENTRY_IMPORT = []
    DIRECTORY_ENTRY_DELAY_IMPORT = [Entry()]

  assert audit.pe_imports(PE()) == ["missing-delayed.dll"]
  with pytest.raises(RuntimeError):
    for name in audit.pe_imports(PE()):
      audit.resolve_import("Windows", name, "scaly_ipopt/lib/ipopt.dll", {}, [])


def test_builder_reads_repaired_pe_and_delay_tables(monkeypatch):
  native = load("ipopt_native")

  class PE:
    DIRECTORY_ENTRY_IMPORT = [SimpleNamespace(dll=b"private-fortran.dll")]
    DIRECTORY_ENTRY_DELAY_IMPORT = [SimpleNamespace(dll=b"private-delayed.dll")]

    def __enter__(self):
      return self

    def __exit__(self, *args):
      pass

  monkeypatch.setitem(sys.modules, "pefile", SimpleNamespace(PE=lambda path: PE()))
  assert native.hook()._windows_deps(Path("repaired.dll")) == ["private-fortran.dll", "private-delayed.dll"]


def test_publication_rejects_pr_partial_health_unreviewed_and_wrong_workflow():
  publish = load("ipopt_native_publish")
  good = dict(
    status="completed",
    conclusion="success",
    head_branch="main",
    event="workflow_dispatch",
    path=".github/workflows/ipopt-native.yml",
    head_sha="a" * 40,
    run_attempt=1,
  )
  publish.admit_run(good, "a" * 40)
  for change in (
    dict(head_branch="trial"),
    dict(event="pull_request"),
    dict(event="schedule"),
    dict(conclusion="failure"),
    dict(path="other.yml"),
    dict(head_sha="b" * 40),
  ):
    with pytest.raises(RuntimeError):
      publish.admit_run(good | change, "a" * 40)
  with pytest.raises(RuntimeError):
    publish.admit_reports({"windows-x86_64": {}}, "a" * 40, 1, 1)


def test_publication_detects_changed_bytes(tmp_path):
  publish = load("ipopt_native_publish")
  path = tmp_path / "candidate.tar.gz"
  path.write_bytes(b"reviewed")
  report = {"assets": {path.name: publish.sha256(path)}}
  publish.verify_assets(tmp_path, report)
  path.write_bytes(b"replaced")
  with pytest.raises(RuntimeError):
    publish.verify_assets(tmp_path, report)


def test_recipe_identity_ignores_wrapper_and_pin_edits(tmp_path):
  native = load("ipopt_native")
  root = Path(__file__).resolve().parents[2]
  source = (root / "plugins/scaly-ipopt/hatch_build.py").read_text(encoding="utf-8")
  assert native.recipe(root) == native.recipe(root, source + "\n# wrapper-only edit\n")
  assert native.recipe(root) != native.recipe(root, source.replace("DYNAMIC_ARCH=1", "DYNAMIC_ARCH=0"))


def test_numerical_comparison_detects_changed_iteration_and_solution():
  native = load("ipopt_native")
  base = {"solve": {"status": 0, "iter": 7, "x": [5.0, 5.0], "obj": 50.0}}
  native.compare(base, json.loads(json.dumps(base)))
  for changed in ({"iter": 8}, {"x": [6.0, 4.0]}, {"status": 1}):
    with pytest.raises(RuntimeError):
      native.compare(base, {"solve": base["solve"] | changed})
