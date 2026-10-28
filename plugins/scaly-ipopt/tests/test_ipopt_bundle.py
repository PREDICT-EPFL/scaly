"""Native archives preserve repaired layouts and reject invalid local inputs."""

from __future__ import annotations

import hashlib
import gzip
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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
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


@contextmanager
def bundle_mirror(content):
  requests = []

  class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      requests.append(self.path)
      self.send_response(200)
      self.end_headers()
      self.wfile.write(content)

    def log_message(self, format, *args):
      pass

  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  thread = Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    yield f"http://127.0.0.1:{server.server_port}", requests
  finally:
    server.shutdown()
    server.server_close()
    thread.join()


def fixture_pins(content):
  targets = {}
  for path in (Path(__file__).parent / "fixtures/bundles").glob("*.json"):
    layout = json.loads(path.read_text())
    targets[path.stem] = {
      "asset": f"scaly-ipopt-native-{path.stem}.tar.gz",
      "sha256": hashlib.sha256(content).hexdigest(),
      "platform_tag": layout["target"]["platform_tag"],
    }
  return {"repository": "PREDICT-EPFL/scaly", "release": "test-only", "targets": targets}


def test_pinned_download_cache_offline_all_targets(bundle_hook, native_layout, tmp_path, monkeypatch):
  archive = tmp_path / "native.tar.gz"
  expected = archive_fixture(archive, native_layout)
  content = archive.read_bytes()
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: expected["target"])
  monkeypatch.setattr(bundle_hook.platform, "system", lambda: expected["target"]["system"])
  monkeypatch.setattr(bundle_hook.platform, "libc_ver", lambda: ("glibc", "2.35"))
  monkeypatch.setattr(bundle_hook, "_cache_root", lambda: tmp_path / "cache")
  pins = fixture_pins(content)
  with bundle_mirror(content) as (mirror, requests):
    monkeypatch.setattr(bundle_hook, "_BUNDLE_RELEASE_URL", mirror + "/{repository}/releases/download/{release}/{asset}")
    with ThreadPoolExecutor(max_workers=4) as pool:
      results = list(pool.map(lambda _: bundle_hook._official_bundle(pins), range(4)))
    assert len(requests) == 1
  bundle, manifest, identity = bundle_hook._official_bundle(pins)
  assert bundle == content and manifest == expected
  assert all(result == (bundle, manifest, identity) for result in results)
  target = next(name for name, pin in pins["targets"].items() if pin["platform_tag"] == expected["target"]["platform_tag"])
  assert identity == {"source": "official", "repository": pins["repository"], "release": pins["release"], **pins["targets"][target]}
  cached = next((tmp_path / "cache").glob("ipopt-download-*/bundle.tar.gz"))
  cached.write_bytes(content + b"corrupt")
  with pytest.raises(RuntimeError, match="SHA-256.*remove corrupt cache"):
    bundle_hook._official_bundle(pins)


@pytest.mark.parametrize("failure", ["checksum", "manifest", "tag", "pins", "musl", "floor", "missing-target", "unavailable"])
def test_official_input_failures(bundle_hook, tmp_path, monkeypatch, failure):
  layout = json.loads((Path(__file__).parent / "fixtures/bundles/linux-x86_64.json").read_text())
  archive = tmp_path / "native.tar.gz"
  archive_fixture(archive, layout)
  content = archive.read_bytes()
  pins = fixture_pins(content)
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: layout["target"])
  monkeypatch.setattr(bundle_hook.platform, "system", lambda: "Linux")
  monkeypatch.setattr(bundle_hook.platform, "libc_ver", lambda: ("glibc", "2.35"))
  monkeypatch.setattr(bundle_hook, "_cache_root", lambda: tmp_path / "cache")
  if failure == "checksum":
    content += gzip.compress(b"")
  elif failure == "manifest":
    content = b"not an archive"
    pins = fixture_pins(content)
  elif failure == "tag":
    pins["targets"]["linux-x86_64"]["platform_tag"] = "manylinux_2_28_x86_64"
  elif failure == "pins":
    with pytest.raises(RuntimeError, match="No official Ipopt bundle pins configured"):
      bundle_hook._official_bundle(None)
    pins["targets"]["linux-x86_64"]["sha256"] = "not-a-hash"
  elif failure == "musl":
    monkeypatch.setattr(bundle_hook.platform, "libc_ver", lambda: ("musl", "1.2.5"))
  elif failure == "floor":
    layout["target"]["floor"] = "glibc 2.26"
  elif failure == "missing-target":
    layout["target"]["arch"] = "riscv64"
  with bundle_mirror(content) as (mirror, requests):
    monkeypatch.setattr(bundle_hook, "_BUNDLE_RELEASE_URL", mirror + "/{repository}/{release}/{asset}")
    if failure == "unavailable":
      monkeypatch.setattr(bundle_hook, "_BUNDLE_RELEASE_URL", "http://127.0.0.1:1/{asset}")
    with pytest.raises(RuntimeError) as error:
      bundle_hook._official_bundle(pins)
    if failure == "checksum":
      assert "check the release asset and checked-in SHA-256 pin" in str(error.value)
      assert "remove corrupt cache" not in str(error.value)
    if failure in {"pins", "musl", "missing-target"}:
      assert not requests
  assert not list((tmp_path / "cache").glob("ipopt-download-*/bundle.tar.gz"))


@pytest.mark.parametrize("version,mode", [("standard", "auto"), ("editable", "required"), ("editable", "auto"), ("editable", "skip")])
def test_pinned_modes_never_compile(bundle_hook, tmp_path, monkeypatch, version, mode):
  monkeypatch.delenv("SCALY_IPOPT_BUNDLE", raising=False)
  monkeypatch.setenv("SCALY_BUILD_SOLVERS", mode)
  monkeypatch.setitem(bundle_hook._BUILD_CONFIG, "native_bundle", fixture_pins(b"missing"))
  monkeypatch.setattr(bundle_hook, "_BUNDLE_RELEASE_URL", "http://127.0.0.1:1/{asset}")
  monkeypatch.setattr(bundle_hook, "_cache_root", lambda: tmp_path / "cache")
  monkeypatch.setattr(bundle_hook, "_missing_ipopt_tools", lambda: pytest.fail("consumer requested a compiler"))
  monkeypatch.setattr(bundle_hook, "_build_ipopt_stack", lambda *args: pytest.fail("consumer invoked producer"))
  messages = []
  hook = SimpleNamespace(root=str(tmp_path), target_name="wheel", app=SimpleNamespace(display_info=messages.append))
  if mode == "required" or version == "standard":
    with pytest.raises(RuntimeError, match="[Bb]undle"):
      bundle_hook.BuildHook.initialize(hook, version, {})
  else:
    bundle_hook.BuildHook.initialize(hook, version, {})
    assert messages and "Skipping" in messages[-1]


def test_pinned_install_and_override_precedence(bundle_hook, native_layout, tmp_path, monkeypatch):
  archive = tmp_path / "native.tar.gz"
  manifest = archive_fixture(archive, native_layout)
  content = archive.read_bytes()
  pins = fixture_pins(content)
  monkeypatch.setitem(bundle_hook._BUILD_CONFIG, "native_bundle", pins)
  monkeypatch.setattr(bundle_hook, "_host_target", lambda: manifest["target"])
  monkeypatch.setattr(bundle_hook.platform, "system", lambda: manifest["target"]["system"])
  monkeypatch.setattr(bundle_hook.platform, "libc_ver", lambda: ("glibc", "2.35"))
  monkeypatch.setattr(bundle_hook, "_wheel_tag", lambda: "py3-none-" + manifest["target"]["platform_tag"])
  monkeypatch.setattr(bundle_hook, "_missing_ipopt_tools", lambda: pytest.fail("consumer requested tools"))
  monkeypatch.setattr(bundle_hook, "_cache_root", lambda: tmp_path / "cache")
  monkeypatch.setenv("SCALY_BUILD_SOLVERS", "required")
  monkeypatch.delenv("SCALY_IPOPT_BUNDLE", raising=False)
  hook = SimpleNamespace(root=str(tmp_path), target_name="wheel", app=SimpleNamespace(display_info=lambda message: None))
  with bundle_mirror(content) as (mirror, requests):
    monkeypatch.setattr(bundle_hook, "_BUNDLE_RELEASE_URL", mirror + "/{asset}")
    data = {}
    bundle_hook.BuildHook.initialize(hook, "editable", data)
    assert len(requests) == 1
  package = tmp_path / "src/scaly_ipopt"
  assert data["tag"] == "py3-none-" + manifest["target"]["platform_tag"]
  assert json.loads((package / "native_bundle.json").read_text())["release"] == "test-only"
  bundle_hook._verify_payload(package.parent, manifest)
  (package / "build_config.json").write_text(json.dumps({"native_bundle": pins}))
  script = Path(__file__).parents[3] / ".github/scripts/ipopt_bundle_verify.py"
  spec = importlib.util.spec_from_file_location("consumer_verify", script)
  assert spec is not None and spec.loader is not None
  verifier = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(verifier)
  assert verifier.verify(package)["release"] == "test-only"
  library = package / "lib" / bundle_hook._shared_lib_name(manifest["target"]["system"], "ipopt")
  original = library.read_bytes()
  library.write_bytes(b"changed after packaging")
  with pytest.raises(RuntimeError, match="native bytes differ"):
    verifier.verify(package)
  library.write_bytes(original)
  identity_path = package / "native_bundle.json"
  identity = json.loads(identity_path.read_text())
  identity_path.write_text(json.dumps({**identity, "asset": "not-a-pinned-asset.tar.gz"}))
  with pytest.raises(RuntimeError, match="not an official release pin"):
    verifier.verify(package)
  monkeypatch.setenv("SCALY_IPOPT_BUNDLE", str(archive))
  bundle_hook.BuildHook.initialize(hook, "editable", {})
  assert json.loads((package / "native_bundle.json").read_text()) == {"source": "override", "sha256": hashlib.sha256(content).hexdigest()}
  with pytest.raises(RuntimeError, match="not an official release pin"):
    verifier.verify(package)


def test_hatch_pins_checkout_and_sdist_use_same_offline_payload(bundle_hook, tmp_path):
  root = Path(__file__).parents[1]
  target = bundle_hook._host_target()
  name = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[target["system"]] + "-" + target["arch"]
  layout = json.loads((Path(__file__).parent / f"fixtures/bundles/{name}.json").read_text())
  archive = tmp_path / "native.tar.gz"
  manifest = archive_fixture(archive, layout)
  content = archive.read_bytes()
  plugin = tmp_path / "checkout"
  shutil.copytree(
    root,
    plugin,
    ignore=shutil.ignore_patterns("lib", "include", "licenses", "native_manifest.json", "native_bundle.json", ".build_key", "__pycache__", "dist"),
  )
  shutil.copytree(root / "licenses", plugin / "licenses")
  config = plugin / "src/scaly_ipopt/build_config.json"
  pins = fixture_pins(content)
  config.write_text(json.dumps({**json.loads(config.read_text()), "native_bundle": pins}))
  env = dict(os.environ, SCALY_BUILD_SOLVERS="required", SCALY_SOLVER_CACHE=str(tmp_path / "cache"))
  env.pop("SCALY_IPOPT_BUNDLE", None)
  with bundle_mirror(content) as (mirror, requests):
    hook_path = plugin / "hatch_build.py"
    hook_path.write_text(hook_path.read_text().replace("https://github.com/{repository}/releases/download/{release}/{asset}", mirror + "/{asset}"))
    subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "sdist"], cwd=plugin, env=env, check=True, capture_output=True)
    assert not requests
    with tarfile.open(next((plugin / "dist").glob("*.tar.gz"))) as sdist:
      assert not any(name.endswith(("native_manifest.json", "native_bundle.json")) for name in sdist.getnames())
      sdist.extractall(tmp_path / "sdist", filter="data")
    subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "wheel"], cwd=plugin, env=env, check=True, capture_output=True)
    assert len(requests) == 1
  unpacked = next((tmp_path / "sdist").iterdir())
  subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "wheel"], cwd=unpacked, env=env, check=True, capture_output=True)
  assert json.loads((unpacked / "src/scaly_ipopt/build_config.json").read_text())["native_bundle"] == pins
  for directory in (plugin, unpacked):
    with zipfile.ZipFile(next((directory / "dist").glob("*.whl"))) as wheel:
      for name, entry in manifest["files"].items():
        assert hashlib.sha256(wheel.read(name)).hexdigest() == entry["sha256"]
      identity = json.loads(wheel.read("scaly_ipopt/native_bundle.json"))
      assert identity["source"] == "official" and identity["sha256"] == hashlib.sha256(content).hexdigest()


def test_editable_auto_failure_removes_stale_payload(bundle_hook, tmp_path, monkeypatch):
  monkeypatch.setenv("SCALY_BUILD_SOLVERS", "auto")
  monkeypatch.delenv("SCALY_IPOPT_BUNDLE", raising=False)
  monkeypatch.setitem(bundle_hook._BUILD_CONFIG, "native_bundle", fixture_pins(b"missing"))
  monkeypatch.setattr(bundle_hook, "_BUNDLE_RELEASE_URL", "http://127.0.0.1:1/{asset}")
  monkeypatch.setattr(bundle_hook, "_cache_root", lambda: tmp_path / "cache")
  package = tmp_path / "src/scaly_ipopt"
  (package / "lib").mkdir(parents=True)
  (package / "lib/stale.dll").write_bytes(b"previous selection")
  hook = SimpleNamespace(root=str(tmp_path), target_name="wheel", app=SimpleNamespace(display_info=lambda message: None))
  bundle_hook.BuildHook.initialize(hook, "editable", {})
  assert not (package / "lib").exists()


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
  config["native_bundle"] = fixture_pins(b"test-only")
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
