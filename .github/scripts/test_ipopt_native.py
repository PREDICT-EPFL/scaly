"""Exercise native producer admission, binary audits and publication rejection controls."""

import importlib.util
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

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
  allowed = {"GLIBC": ["2.27"], "GLIBCXX": ["3.4.19"]}
  audit.check_versions(["GLIBC_2.27", "GLIBCXX_3.4.19"], allowed)
  with pytest.raises(RuntimeError, match="GLIBC_2.28"):
    audit.check_versions(["GLIBC_2.28"], allowed)


@pytest.mark.parametrize("minimum", ["15.0", "15.1"])
def test_macos_floor_ignores_linker_tool_version_and_rejects_newer_os(tmp_path, monkeypatch, minimum):
  audit = load("ipopt_native_audit")
  commands = f"Load command 1\n      cmd LC_BUILD_VERSION\n    minos {minimum}\n      sdk 26.0\n   ntools 1\n     tool 3\n  version 1231.3\n"
  monkeypatch.setattr(audit.subprocess, "check_output", lambda command, **kwargs: "arm64\n" if command[0] == "lipo" else commands)
  signatures = []
  monkeypatch.setattr(audit.subprocess, "run", lambda command, **kwargs: signatures.append(command))
  if minimum == "15.0":
    assert audit.macho(tmp_path / "libipopt.dylib", "arm64")["minima"] == [(15, 0)]
    assert signatures[0][:3] == ["codesign", "--verify", "--strict"]
  else:
    with pytest.raises(RuntimeError, match="floor exceeded"):
      audit.macho(tmp_path / "libipopt.dylib", "arm64")
    assert not signatures


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


def test_windows_floor_includes_system_function_requirements():
  audit = load("ipopt_native_audit")
  library = "api-ms-win-crt-runtime-l1-1-0.dll"
  reference = {library: ["memcpy"]}
  audit.check_windows_requirements(reference, reference)
  with pytest.raises(RuntimeError, match="system API requirements changed"):
    audit.check_windows_requirements({library: ["new_os_function"]}, reference)


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


@pytest.mark.parametrize("corrupt_download", [False, True])
def test_publication_uploads_and_verifies_every_asset_before_publishing(tmp_path, monkeypatch, corrupt_download):
  publish = load("ipopt_native_publish")
  reviewed = "a" * 40
  files = {"candidate.tar.gz": b"validated native bytes", "candidate.manifest.json": b"validated manifest"}
  for name, data in files.items():
    (tmp_path / name).write_bytes(data)
  plan = {
    "revision": "scaly-ipopt-native-v1",
    "candidate_run": 1,
    "run_attempt": 1,
    "reviewed_sha": reviewed,
    "assets": {name: publish.sha256(tmp_path / name) for name in files},
  }
  (tmp_path / "publication-plan.json").write_text(json.dumps(plan))
  monkeypatch.setattr(publish, "require_maintainer", lambda: "PREDICT-EPFL/scaly")

  def api(path):
    if "/actions/runs/" in path:
      return dict(
        status="completed",
        conclusion="success",
        head_branch="main",
        event="push",
        path=".github/workflows/ipopt-native.yml",
        head_sha=reviewed,
        run_attempt=1,
      )
    if path.endswith("commits/main"):
      return {"sha": reviewed}
    raise AssertionError(path)

  monkeypatch.setattr(publish, "api", api)
  monkeypatch.setattr(publish.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=b"HTTP 404"))
  events, uploaded = [], {}

  def gh(*command):
    action = command[2]
    events.append(action)
    if action == "create":
      assert "--draft" in command and "--latest=false" in command
    elif action == "upload":
      for path in command[4 : command.index("-R")]:
        uploaded[Path(path).name] = Path(path).read_bytes()
    elif action == "download":
      directory = Path(command[command.index("--dir") + 1])
      for name, data in uploaded.items():
        (directory / name).write_bytes(b"changed" if corrupt_download else data)
    elif action == "edit":
      assert "--draft=false" in command and "--latest=false" in command
      assert uploaded == files | {"publication-plan.json": (tmp_path / "publication-plan.json").read_bytes()}

  monkeypatch.setattr(publish, "run", gh)
  if corrupt_download:
    with pytest.raises(RuntimeError, match="Draft download differs"):
      publish.publish(SimpleNamespace(directory=tmp_path))
    assert events == ["create", "upload", "download"]
  else:
    publish.publish(SimpleNamespace(directory=tmp_path))
    assert events == ["create", "upload", "download", "edit"]


def test_recipe_identity_ignores_wrapper_and_pin_edits(tmp_path):
  native = load("ipopt_native")
  root = Path(__file__).resolve().parents[2]
  source = (root / "plugins/scaly-ipopt/hatch_build.py").read_text(encoding="utf-8")
  assert native.recipe(root) == native.recipe(root, source + "\n# wrapper-only edit\n")
  assert native.recipe(root) == native.recipe(root, source.replace("class BuildHook(BuildHookInterface)", "class WrapperHook(BuildHookInterface)"))
  assert native.recipe(root) != native.recipe(root, source.replace("DYNAMIC_ARCH=1", "DYNAMIC_ARCH=0"))


def test_numerical_comparison_detects_changed_iteration_and_solution():
  native = load("ipopt_native")
  base = {"solve": {"status": 0, "iter": 7, "x": [5.0, 5.0], "obj": 50.0}}
  native.compare(base, json.loads(json.dumps(base)))
  for changed in ({"iter": 8}, {"x": [6.0, 4.0]}, {"status": 1}):
    with pytest.raises(RuntimeError):
      native.compare(base, {"solve": base["solve"] | changed})


def test_host_thread_capacity_is_reported_without_changing_policy():
  native = load("ipopt_native")
  manifest = {"configuration": {"threading": {"effective": {"Makefile.conf": "CORE=PRESCOTT\nNUM_CORES=20\n"}}}}
  control = {"blas": {"common_code_target": "PRESCOTT", "compiled_max_threads": 4}}
  result = native.configuration_comparison(manifest, control)
  assert result["compiled_maximum_differs"] and result["candidate_compiled_max_threads"] == 20
  assert result["control_compiled_max_threads"] == 4 and result["thread_policy"] == "unchanged"
  assert manifest["configuration"]["threading"]["effective"]["Makefile.conf"].endswith("NUM_CORES=20\n")


@pytest.mark.parametrize("corruption", ["none", "hash", "path"])
def test_retained_controls_remain_hash_bound_and_extract_safely(tmp_path, corruption):
  native = load("ipopt_native")
  output = io.BytesIO()
  with zipfile.ZipFile(output, "w") as archive:
    archive.writestr("scaly_ipopt-control.whl", b"source bytes" if corruption != "hash" else b"changed")
    archive.writestr(("../" if corruption == "path" else "") + "scaly_piqp-control.whl", b"qp bytes")
  control = {"wheel": "scaly_ipopt-control.whl", "wheel_sha256": hashlib.sha256(b"source bytes").hexdigest()}
  baseline, wheels = tmp_path / "baseline", tmp_path / "wheels"
  if corruption == "none":
    native.extract_controls(output.getvalue(), control, baseline, wheels)
    assert (baseline / control["wheel"]).read_bytes() == b"source bytes"
  else:
    with pytest.raises(RuntimeError):
      native.extract_controls(output.getvalue(), control, baseline, wheels)
    assert not baseline.exists() and not wheels.exists()
