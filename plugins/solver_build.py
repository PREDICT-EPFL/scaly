"""Reuse released solver payloads and identify native inputs for the build hooks."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

OUTPUTS = ("lib", "include", "licenses")
STAMP = "native-inputs.sha256"


def native_hash(root: Path, package: str) -> str:
  digest = hashlib.sha256()
  inputs = [root / "native_build.py", root / "src" / package / "build_config.json", *(root / "licenses").rglob("*")]
  for path in sorted(p for p in inputs if p.is_file()):
    digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
  return digest.hexdigest()


def release_key(package: str, digest: str) -> str:
  return f"{package}-release-{next(sys_tags()).platform}-{digest}"


def download_release(package: str, digest: str, destination: Path) -> bool:
  """Verify the newest release's wheel before copying its native payload, or return a cache miss."""
  index = os.environ.get("SCALY_SOLVER_INDEX", "https://pypi.org/pypi").rstrip("/")
  try:
    with urllib.request.urlopen(f"{index}/{package.replace('_', '-')}/json", timeout=15) as response:
      metadata = json.load(response)
    supported = {tag: rank for rank, tag in enumerate(sys_tags())}
    candidates = []
    for item in metadata["urls"]:
      if item.get("packagetype") != "bdist_wheel" or item.get("yanked"):
        continue
      _, _, _, tags = parse_wheel_filename(item["filename"])
      ranks = [supported[tag] for tag in tags if tag in supported]
      if ranks:
        candidates.append((min(ranks), item))
    for _, item in sorted(candidates, key=lambda pair: pair[0]):
      with urllib.request.urlopen(item["url"], timeout=60) as response:
        payload = response.read()
      if hashlib.sha256(payload).hexdigest() != item["digests"]["sha256"]:
        continue
      with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
        if wheel.read(f"{package}/{STAMP}").decode().strip() != digest:
          continue
        names = wheel.namelist()
        selected = []
        for name in names:
          parts = Path(name).parts
          if len(parts) >= 3 and parts[0] == package and parts[1] in OUTPUTS:
            if ".." in parts or "\\" in name:
              raise ValueError("unsafe wheel member")
            selected.append((name, Path(*parts[1:])))
          # auditwheel may relocate runtime dependencies beside the package.
          elif parts and parts[0] == f"{package}.libs":
            if ".." in parts or "\\" in name:
              raise ValueError("unsafe wheel member")
            selected.append((name, Path("wheel-libs", *parts[1:])))
        if not all(any(path.parts[0] == output for _, path in selected) for output in OUTPUTS):
          continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
          out = Path(temporary) / "out"
          out.mkdir()
          for name, relative in selected:
            target = out / relative
            if name.endswith("/"):
              target.mkdir(parents=True, exist_ok=True)
            else:
              target.parent.mkdir(parents=True, exist_ok=True)
              target.write_bytes(wheel.read(name))
              target.chmod((wheel.getinfo(name).external_attr >> 16) & 0o777 or 0o644)
          out.rename(destination)
        return True
  except (OSError, ValueError, KeyError, zipfile.BadZipFile, urllib.error.URLError, http.client.HTTPException):
    return False
  return False


def find_local(cache: Path, package: str, digest: str, context: dict) -> Path | None:
  """Find a completed build with matching inputs, checking the compilers available on this machine."""
  for entry in sorted(cache.glob(f"{package}-*"), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
      saved = json.loads((entry / "native-build.json").read_text())
      if saved["hash"] != digest or saved["platform"] != context["platform"] or saved["env"] != context["env"]:
        continue
      if any(not value.endswith(": none") and value != saved["tools"][name] for name, value in context["tools"].items()):
        continue
      if all((entry / name).is_dir() for name in OUTPUTS):
        return entry
    except (OSError, ValueError, KeyError):
      continue
  return None
