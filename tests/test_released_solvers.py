"""Solver hooks reuse verified released payloads without a compiler or a warm cache."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
import tarfile
import urllib.error
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from packaging.tags import sys_tags

ROOT = Path(__file__).parents[1]


def _wheel(package, digest):
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as wheel:
    wheel.writestr(f"{package}/native-inputs.sha256", digest)
    for directory in ("lib", "include", "licenses"):
      wheel.writestr(f"{package}/{directory}/sentinel", b"released payload")
    wheel.writestr(f"{package}.libs/runtime.so", b"repaired runtime")
    wheel.writestr(f"{package}/__init__.py", "raise RuntimeError('old wrapper must not be copied')")
  return buffer.getvalue()


@pytest.mark.parametrize("failure", [None, "hash", "platform", "offline", "sha256", "truncated", "http"])
def test_release_lookup(tmp_path, monkeypatch, failure):
  helper = runpy.run_path(str(ROOT / "plugins/solver_build.py"))
  digest = "a" * 64
  payload = _wheel("scaly_ipopt", "b" * 64 if failure == "hash" else digest)
  tag = "py3-none-impossible" if failure == "platform" else str(next(t for t in sys_tags() if t.interpreter == "py3"))
  metadata = {
    "urls": [
      {
        "filename": f"scaly_ipopt-1.0-{tag}.whl",
        "packagetype": "bdist_wheel",
        "url": "https://files/wheel",
        "digests": {"sha256": "bad" if failure == "sha256" else hashlib.sha256(payload).hexdigest()},
      }
    ]
  }

  def fetch(url, **kwargs):
    if failure == "offline":
      raise urllib.error.URLError("offline")

    class TruncatedResponse(io.BytesIO):
      def read(self, *args):
        if failure == "truncated":
          raise http.client.IncompleteRead(b"")
        raise http.client.HTTPException("invalid HTTP response")

    if not url.endswith("/json") and failure in {"truncated", "http"}:
      return TruncatedResponse()
    return io.BytesIO(json.dumps(metadata).encode() if url.endswith("/json") else payload)

  monkeypatch.setattr("urllib.request.urlopen", fetch)
  target = tmp_path / "entry"
  assert helper["download_release"]("scaly_ipopt", digest, target) == (failure is None)
  if failure is None:
    assert (target / "lib/sentinel").read_bytes() == b"released payload"
    assert (target / "wheel-libs/runtime.so").read_bytes() == b"repaired runtime"
    assert not (target / "__init__.py").exists()
  else:
    assert not target.exists()


@pytest.mark.parametrize("solver", ["ipopt", "piqp"])
def test_build_wheel_from_local_release(tmp_path, monkeypatch, solver):
  root = tmp_path / f"scaly-{solver}"
  shutil.copytree(ROOT / f"plugins/scaly-{solver}", root, ignore=shutil.ignore_patterns("lib", "include", "__pycache__", ".build_key", "dist"))
  shutil.copy2(ROOT / "plugins/solver_build.py", root / "solver_build.py")
  shutil.copy2(ROOT / "plugins/solver_build.py", tmp_path / "solver_build.py")
  helper = runpy.run_path(str(root / "solver_build.py"))
  package = f"scaly_{solver}"
  digest = helper["native_hash"](root, package)
  index = tmp_path / "index"
  index.mkdir()
  tag = str(next(t for t in sys_tags() if t.interpreter == "py3"))
  filename = f"{package}-1.0-{tag}.whl"
  payload = _wheel(package, digest)
  (index / filename).write_bytes(payload)
  handler = partial(SimpleHTTPRequestHandler, directory=str(index))
  server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  url = f"http://127.0.0.1:{server.server_port}"
  metadata = {
    "urls": [
      {"filename": filename, "packagetype": "bdist_wheel", "url": f"{url}/{filename}", "digests": {"sha256": hashlib.sha256(payload).hexdigest()}}
    ]
  }
  endpoint = index / f"scaly-{solver}"
  endpoint.mkdir()
  (endpoint / "json").write_text(json.dumps(metadata))
  env = {
    **os.environ,
    "SCALY_SOLVER_INDEX": url,
    "SCALY_SOLVER_CACHE": str(tmp_path / "cache"),
    "SCALY_BUILD_SOLVERS": "required",
    "CC": "missing-cc",
    "CXX": "missing-cxx",
    "FC": "missing-fc",
  }
  try:
    sdist = subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "sdist"], cwd=root, env=env, capture_output=True, text=True)
    assert sdist.returncode == 0, sdist.stdout + sdist.stderr
    extracted = tmp_path / "extracted"
    with tarfile.open(next((root / "dist").glob("*.tar.gz"))) as archive:
      archive.extractall(extracted, filter="data")
    root = next(extracted.iterdir())
    result = subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "wheel"], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
  finally:
    server.shutdown()
    thread.join()
    server.server_close()
  with zipfile.ZipFile(next((root / "dist").glob("*.whl"))) as wheel:
    assert wheel.read(f"{package}/lib/sentinel") == b"released payload"
    assert wheel.read(f"{package}/native-inputs.sha256").decode().strip() == digest
    assert wheel.read(f"{package}.libs/runtime.so") == b"repaired runtime"
    assert b"old wrapper" not in wheel.read(f"{package}/__init__.py")


@pytest.mark.parametrize("solver", ["ipopt", "piqp"])
def test_preparation_order_and_source_override(tmp_path, monkeypatch, solver):
  hook = runpy.run_path(str(ROOT / f"plugins/scaly-{solver}/hatch_build.py"))
  prepare = hook["_prepare_native"]
  shared = prepare.__globals__["_SHARED"]
  monkeypatch.setenv("SCALY_SOLVER_CACHE", str(tmp_path / "cache"))
  monkeypatch.setenv("SCALY_BUILD_SOLVERS", "required")
  monkeypatch.setitem(prepare.__globals__, "_cache_key", lambda root: "local-key")
  monkeypatch.setitem(shared, "native_hash", lambda *args: "digest")
  monkeypatch.setitem(shared, "download_release", lambda *args: pytest.fail("local cache must precede network"))
  local = tmp_path / "cache" / f"{solver}-local-key"
  local.mkdir(parents=True)
  assert prepare(tmp_path)[0] == local
  monkeypatch.setenv("SCALY_BUILD_SOLVERS", "source")
  monkeypatch.setitem(shared, "find_local", lambda *args: pytest.fail("source must bypass caches"))
  assert prepare(tmp_path)[0] == local
  assert hook["_solver_build_mode"]() == "source"


@pytest.mark.parametrize("solver", ["ipopt", "piqp"])
def test_restored_local_build_without_compiler(tmp_path, monkeypatch, solver):
  helper = runpy.run_path(str(ROOT / "plugins/solver_build.py"))
  entry = tmp_path / f"{solver}-compiler-key"
  entry.mkdir()
  saved = {"hash": "digest", "platform": ["Windows", "AMD64"], "env": {"FC": ""}, "tools": {"FC": "gfortran: GCC 16"}}
  (entry / "native-build.json").write_text(json.dumps(saved))
  for name in ("lib", "include", "licenses"):
    (entry / name).mkdir()
  context = {**saved, "tools": {"FC": "gfortran: none"}}
  assert helper["find_local"](tmp_path, solver, "digest", context) == entry
  context["tools"]["FC"] = "gfortran: GCC 17"
  assert helper["find_local"](tmp_path, solver, "digest", context) == entry
  saved["env"] = {"FC": "gfortran"}
  (entry / "native-build.json").write_text(json.dumps(saved))
  context["env"] = {"FC": "gfortran"}
  assert helper["find_local"](tmp_path, solver, "digest", context) is None
  context["tools"]["FC"] = "gfortran: GCC 16"
  assert helper["find_local"](tmp_path, solver, "digest", context) == entry


@pytest.mark.parametrize("solver", ["ipopt", "piqp"])
def test_native_hash_tracks_recipe_inputs_only(tmp_path, solver):
  root = tmp_path / f"scaly-{solver}"
  shutil.copytree(ROOT / f"plugins/scaly-{solver}", root, ignore=shutil.ignore_patterns("lib", "include", "__pycache__", "dist"))
  helper = tmp_path / "solver_build.py"
  shutil.copy2(ROOT / "plugins/solver_build.py", helper)
  native_hash = runpy.run_path(str(helper))["native_hash"]
  package = f"scaly_{solver}"
  original = native_hash(root, package)
  for path in (helper, root / "hatch_build.py"):
    path.write_text(path.read_text() + "\n# lookup-only change\n")
    assert native_hash(root, package) == original
  inputs = [root / "native_build.py", root / "src" / package / "build_config.json", next(p for p in (root / "licenses").rglob("*") if p.is_file())]
  for path in inputs:
    contents = path.read_bytes()
    path.write_bytes(contents + b"\n# changed native input\n")
    assert native_hash(root, package) != original
    path.write_bytes(contents)
