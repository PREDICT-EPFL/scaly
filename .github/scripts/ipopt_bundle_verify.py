"""Check that an installed consumer wheel preserves its pinned native payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify(package: Path) -> dict:
  config = json.loads((package / "build_config.json").read_text())["native_bundle"]
  identity = json.loads((package / "native_bundle.json").read_text())
  pin = next(pin for pin in config["targets"].values() if pin["asset"] == identity["asset"])
  expected = {"source": "official", "release": config["release"], "repository": config["repository"], **pin}
  if identity != expected:
    raise RuntimeError("Installed Ipopt bundle identity differs from release pins")
  manifest = json.loads((package / "native_manifest.json").read_text())
  if manifest["target"]["platform_tag"] != pin["platform_tag"]:
    raise RuntimeError("Installed Ipopt platform differs from release pins")
  for name, entry in manifest["files"].items():
    path = package.parent / name
    if "link" in entry:
      if path.read_bytes() != (path.parent / entry["link"]).read_bytes():
        raise RuntimeError(f"Installed Ipopt symlink contents differ: {name}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"] or path.stat().st_size != entry["size"]:
      raise RuntimeError(f"Installed Ipopt native bytes differ: {name}")
  return identity


if __name__ == "__main__":
  import scaly_ipopt

  print(json.dumps(verify(Path(scaly_ipopt.__file__).resolve().parent), indent=2))
