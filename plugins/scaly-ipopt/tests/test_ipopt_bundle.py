"""Native archives preserve repaired layouts and reject invalid local inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def bundle_hook():
  spec = importlib.util.spec_from_file_location("ipopt_bundle", Path(__file__).parents[1] / "hatch_build.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture(params=["windows-x86_64", "linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-arm64"])
def native_layout(request):
  return json.loads((Path(__file__).parent / "fixtures" / "bundles" / f"{request.param}.json").read_text())


def archive_fixture(path, layout, *, change=None, extra=None):
  manifest = dict(layout)
  files = {name: name.encode() for name in manifest.pop("fixture_files")}
  manifest["files"] = {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "mode": 0o644} for name, data in files.items()}
  if change:
    change(manifest, files)
  with tarfile.open(path, "w:gz") as archive:
    for name, data in {"manifest.json": json.dumps(manifest).encode(), **files}.items():
      info = tarfile.TarInfo(name)
      info.size = len(data)
      info.mode = int(manifest["files"][name]["mode"]) if name in manifest["files"] else 0o644
      archive.addfile(info, io.BytesIO(data))
    if extra:
      archive.addfile(extra)
  return manifest


def test_archive_roundtrip_all_targets(bundle_hook, native_layout, tmp_path, monkeypatch):
  archive = tmp_path / "native.tar.gz"
  manifest = archive_fixture(archive, native_layout)
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: manifest["target"])
  out = tmp_path / "out"
  result = bundle_hook._read_bundle(archive.read_bytes(), out)
  assert result == manifest
  for name, entry in manifest["files"].items():
    assert hashlib.sha256((out / name).read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("failure", ["target", "hash", "missing", "closure", "header", "license", "import-library", "tag", "extra"])
def test_invalid_bundle_rejected(bundle_hook, tmp_path, failure, monkeypatch):
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/windows-x86_64.json").read_text())
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: layout["target"])

  def change(manifest, files):
    if failure == "target":
      manifest["target"] = dict(manifest["target"], arch="arm64")
    elif failure == "hash":
      files[next(iter(files))] += b"corrupt"
    elif failure == "missing":
      files.pop(next(iter(files)))
    elif failure == "closure":
      manifest["dependencies"][next(iter(manifest["dependencies"]))]["bundled"].append("scaly_ipopt/lib/missing.dll")
    elif failure in {"header", "license", "import-library"}:
      name = {
        "header": "scaly_ipopt/include/coin-or/IpStdCInterface.h",
        "license": "scaly_ipopt/licenses/ipopt/LICENSE",
        "import-library": "scaly_ipopt/lib/ipopt.lib",
      }[failure]
      files.pop(name)
      manifest["files"].pop(name)
    elif failure == "tag":
      manifest["target"]["platform_tag"] = "macosx_15_0_arm64"
    elif failure == "extra":
      files["scaly_ipopt/wrapper.py"] = b"unwanted wrapper"

  archive = tmp_path / "bad.tar.gz"
  archive_fixture(archive, layout, change=change)
  with pytest.raises(RuntimeError):
    bundle_hook._read_bundle(archive.read_bytes(), tmp_path / "out")
  assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
  "name", ["../escape", "/absolute", "C:/absolute", "scaly_ipopt/../escape", "scaly_ipopt\\lib\\evil", "scaly_ipopt/lib/aux.txt"]
)
def test_unsafe_archive_paths(bundle_hook, tmp_path, name):
  archive = tmp_path / "unsafe.tar.gz"
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/windows-x86_64.json").read_text())
  archive_fixture(archive, layout, extra=tarfile.TarInfo(name))
  with pytest.raises(RuntimeError, match="[Uu]nsafe|[Ii]nvalid"):
    bundle_hook._read_bundle(archive.read_bytes())


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_unsafe_archive_links(bundle_hook, tmp_path, kind):
  archive = tmp_path / "unsafe.tar.gz"
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/windows-x86_64.json").read_text())
  info = tarfile.TarInfo("scaly_ipopt/lib/escape")
  info.type = kind
  info.linkname = "../../../outside"
  archive_fixture(archive, layout, extra=info)
  with pytest.raises(RuntimeError):
    bundle_hook._read_bundle(archive.read_bytes())


def test_override_content_identity_and_install(bundle_hook, tmp_path, monkeypatch):
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/linux-x86_64.json").read_text())
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: layout["target"])
  monkeypatch.setattr(bundle_hook, "_missing_ipopt_tools", lambda: pytest.fail("archive install requested native tools"))
  monkeypatch.setattr(bundle_hook, "_cache_root", lambda: tmp_path / "cache")
  monkeypatch.setenv("SCALY_BUILD_SOLVERS", "required")
  archive = tmp_path / "native.tar.gz"
  monkeypatch.setenv("SCALY_IPOPT_BUNDLE", str(archive))
  root = tmp_path / "plugin"
  root.mkdir()
  hook = SimpleNamespace(root=str(root), target_name="wheel", app=SimpleNamespace(display_info=lambda message: None))
  keys = []
  for content in (b"first", b"replacement"):

    def change(manifest, files):
      name = "scaly_ipopt/lib/libipopt.so"
      files[name] = content
      manifest["files"][name] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content), "mode": 0o755}

    manifest = archive_fixture(archive, layout, change=change)
    data = {}
    bundle_hook.BuildHook.initialize(hook, "editable", data)
    keys.append((root / ".build_key").read_text())
    assert data["tag"] == "py3-none-manylinux_2_27_x86_64"
    for name, entry in manifest["files"].items():
      assert hashlib.sha256((root / "src" / name).read_bytes()).hexdigest() == entry["sha256"]
  assert keys[0] != keys[1]
  assert len(list((tmp_path / "cache").glob("ipopt-*"))) == 2
  entry = tmp_path / "cache" / f"ipopt-{keys[1]}"
  (entry / "scaly_ipopt/lib/libipopt.so").write_bytes(b"corrupt cache")
  with pytest.raises(RuntimeError, match="Invalid native cache entry") as error:
    bundle_hook.BuildHook.initialize(hook, "editable", {})
  assert str(entry) in str(error.value)
  archive.unlink()
  with pytest.raises(RuntimeError, match="Cannot read SCALY_IPOPT_BUNDLE"):
    bundle_hook.BuildHook.initialize(hook, "editable", {})


def test_sdist_ignores_bundle_and_toolchain(bundle_hook, tmp_path, monkeypatch):
  monkeypatch.setenv("SCALY_IPOPT_BUNDLE", str(tmp_path / "does-not-exist"))
  hook = SimpleNamespace(target_name="sdist", app=SimpleNamespace(display_info=lambda message: None))
  bundle_hook.BuildHook.initialize(hook, "standard", {})


def test_recipe_identity_excludes_wrapper_and_consumer_pins(bundle_hook, tmp_path):
  root = Path(__file__).parents[1]
  shutil.copytree(root / "licenses", tmp_path / "licenses")
  config_path = tmp_path / "src/scaly_ipopt/build_config.json"
  config_path.parent.mkdir(parents=True)
  shutil.copy2(root / "src/scaly_ipopt/build_config.json", config_path)
  source = (root / "hatch_build.py").read_text()
  original = bundle_hook._recipe_hash(tmp_path, source)
  config = json.loads(config_path.read_text())
  config["bundles"] = {"release": "fixture", "sha256": "a" * 64}
  config_path.write_text(json.dumps(config))
  assert bundle_hook._recipe_hash(tmp_path, source + "\n# wrapper-only change\n") == original
  assert bundle_hook._recipe_hash(tmp_path, source.replace('if self.target_name == "sdist":', 'if self.target_name == "another-target":')) == original
  assert bundle_hook._recipe_hash(tmp_path, source.replace("TARGET=PRESCOTT", "TARGET=HASWELL")) != original
  config["ipopt"]["version"] = "different"
  config_path.write_text(json.dumps(config))
  assert bundle_hook._recipe_hash(tmp_path, source) != original


def test_cached_bundle_corruption_is_rejected(bundle_hook, tmp_path, monkeypatch):
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/windows-x86_64.json").read_text())
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: layout["target"])
  archive = tmp_path / "native.tar.gz"
  manifest = archive_fixture(archive, layout)
  payload = tmp_path / "payload"
  bundle_hook._read_bundle(archive.read_bytes(), payload)
  library = payload / "scaly_ipopt/lib/libscaly_ipopt-3.dll"
  library.write_bytes(b"changed")
  with pytest.raises(RuntimeError, match="Corrupt cached"):
    bundle_hook._verify_payload(payload, manifest)


def test_corrupt_gzip_trailer_is_rejected(bundle_hook, tmp_path):
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/windows-x86_64.json").read_text())
  archive = tmp_path / "native.tar.gz"
  archive_fixture(archive, layout)
  content = bytearray(archive.read_bytes())
  content[-8] ^= 1
  with pytest.raises(RuntimeError, match="Invalid native bundle"):
    bundle_hook._read_bundle(bytes(content))


def test_safe_relative_symlink(bundle_hook, tmp_path, monkeypatch):
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/linux-aarch64.json").read_text())
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: layout["target"])
  archive = tmp_path / "links.tar.gz"
  manifest = archive_fixture(archive, layout)
  name = "scaly_ipopt/lib/libipopt.so.3"
  link = "libipopt.so"
  manifest["files"][name] = {"link": link, "size": len(link), "sha256": hashlib.sha256(link.encode()).hexdigest(), "mode": 0o777}
  with tarfile.open(archive, "r:gz") as original:
    contents = {}
    for info in original:
      if info.name != name:
        stream = original.extractfile(info)
        assert stream is not None
        contents[info.name] = stream.read()
  contents["manifest.json"] = json.dumps(manifest).encode()
  with tarfile.open(archive, "w:gz") as result:
    for path, data in contents.items():
      info = tarfile.TarInfo(path)
      info.size = len(data)
      result.addfile(info, io.BytesIO(data))
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.mode = tarfile.SYMTYPE, link, 0o777
    result.addfile(info)
  assert bundle_hook._read_bundle(archive.read_bytes()) == manifest
  if os.name != "nt":
    bundle_hook._read_bundle(archive.read_bytes(), tmp_path / "out")
    assert (tmp_path / "out" / name).is_symlink()
    bundle_hook._verify_payload(tmp_path / "out", manifest)


@pytest.mark.parametrize("repaired_sibling", [False, True])
def test_hatch_wheel_and_sdist_preserve_native_boundary(bundle_hook, tmp_path, repaired_sibling):
  system = platform.system()
  arch = bundle_hook._host_target()["arch"]
  name = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[system] + "-" + arch
  layout = json.loads((Path(__file__).parent / f"fixtures/bundles/{name}.json").read_text())
  if repaired_sibling:
    sibling = "scaly_ipopt.libs/extra.dll"
    layout["fixture_files"].append(sibling)
    layout["dependencies"][sibling] = {"bundled": [], "system": []}
  archive = tmp_path / "native.tar.gz"
  manifest = archive_fixture(archive, layout)
  plugin = tmp_path / "plugin"
  root = Path(__file__).parents[1]
  shutil.copytree(root, plugin, ignore=shutil.ignore_patterns("lib", "include", "licenses", "native_manifest.json", ".build_key", "__pycache__"))
  # Copy source license inputs separately; generated package licenses stay excluded.
  shutil.copytree(root / "licenses", plugin / "licenses")
  env = dict(os.environ, SCALY_IPOPT_BUNDLE=str(archive), SCALY_SOLVER_CACHE=str(tmp_path / "cache"), SCALY_BUILD_SOLVERS="required")
  subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "wheel"], cwd=plugin, env=env, check=True, capture_output=True)
  with zipfile.ZipFile(next((plugin / "dist").glob("*.whl"))) as wheel:
    for path, entry in manifest["files"].items():
      assert hashlib.sha256(wheel.read(path)).hexdigest() == entry["sha256"]
    metadata = wheel.read(next(name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL"))).decode()
    assert "Tag: py3-none-" + manifest["target"]["platform_tag"] in metadata
    assert "scaly_ipopt/native_manifest.json" in wheel.namelist()
  env["SCALY_IPOPT_BUNDLE"] = str(tmp_path / "absent.tar.gz")
  subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "sdist"], cwd=plugin, env=env, check=True, capture_output=True)
  with tarfile.open(next((plugin / "dist").glob("*.tar.gz"))) as sdist:
    names = [name.split("/", 1)[1] for name in sdist.getnames()]
    assert "hatch_build.py" in names
    assert "src/scaly_ipopt/build_config.json" in names
    assert not any(
      name.startswith(("src/scaly_ipopt.libs/", "src/scaly_ipopt/lib/", "src/scaly_ipopt/include/", "src/scaly_ipopt/licenses/"))
      or name.endswith("native_manifest.json")
      for name in names
    )


@pytest.mark.parametrize("fixture,expected", [("otool-build-version.txt", (10, 14)), ("otool-version-min-macosx.txt", (10, 9))])
def test_macos_minimum_reads_only_deployment_versions(bundle_hook, fixture, expected):
  commands = (Path(__file__).parent / "fixtures" / fixture).read_text()
  assert bundle_hook._macos_minimum(commands) == expected


@pytest.mark.parametrize("layout_name", ["windows-x86_64", "macos-x86_64", "macos-arm64"])
def test_producer_retains_work_and_freezes_complete_payload(bundle_hook, tmp_path, monkeypatch, layout_name):
  fixtures = Path(__file__).parent / "fixtures"
  layout = json.loads((fixtures / f"bundles/{layout_name}.json").read_text())
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: dict(layout["target"]))
  monkeypatch.setattr(bundle_hook, "_missing_ipopt_tools", lambda: [])
  monkeypatch.setattr(bundle_hook, "_recipe_hash", lambda root: "0" * 64)
  expected = archive_fixture(tmp_path / "fixture.tar.gz", layout)
  if layout_name.startswith("macos"):
    commands = (fixtures / "otool-build-version.txt").read_text() + (fixtures / "otool-version-min-macosx.txt").read_text()
    monkeypatch.setattr(bundle_hook.subprocess, "check_output", lambda *args, **kwargs: commands)

  def build(hook, src, lib, include, licenses, vendored):
    (src / "retained-source").write_text("native tools can rebuild here")
    for name in expected["files"]:
      path = lib.parents[1] / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(name.encode())
      expected["files"][name]["mode"] = path.stat().st_mode & 0o777

  monkeypatch.setattr(bundle_hook, "_build_ipopt_stack", build)

  def manifest(root, work, payload, target):
    assert target == layout["target"]
    return expected

  monkeypatch.setattr(bundle_hook, "_bundle_manifest", manifest)
  work = tmp_path / "work"
  output = tmp_path / "native.tar.gz"
  bundle_hook._produce_bundle(work, output)
  assert bundle_hook._read_bundle(output.read_bytes()) == expected
  assert (work / "src/retained-source").is_file()
  assert output.with_name(output.name + ".sha256").read_text().startswith(hashlib.sha256(output.read_bytes()).hexdigest())
  assert json.loads(output.with_name(output.name + ".manifest.json").read_text()) == expected
  # The command reuses sources, but regenerates the payload before freezing it.
  (work / "payload/scaly_ipopt/lib/obsolete.dll").write_bytes(b"stale")
  bundle_hook._produce_bundle(work, output)
  assert not (work / "payload/scaly_ipopt/lib/obsolete.dll").exists()
