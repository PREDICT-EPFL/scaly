"""Select native candidates and record their immutable producer inputs and acceptance results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
  "windows-x86_64": {"os": "windows-latest", "control": "win_amd64"},
  "linux-x86_64": {
    "os": "ubuntu-latest",
    "control": "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
    "image": "quay.io/pypa/manylinux_2_28_x86_64@sha256:53390351aeb4688114b02c36a23b3e6ce1166ee9b7afc5df1a4f776354fc764c",
  },
  "linux-aarch64": {
    "os": "ubuntu-24.04-arm",
    "control": "manylinux_2_27_aarch64.manylinux_2_28_aarch64",
    "image": "quay.io/pypa/manylinux_2_28_aarch64@sha256:ad74e53b713f3b07d8c889c526dc0c6500da9827b45e38739570875fef52e28f",
  },
  "macos-x86_64": {"os": "macos-15-intel", "control": "macosx_15_0_x86_64"},
  "macos-arm64": {"os": "macos-15", "control": "macosx_15_0_arm64"},
}
BASELINE_RUN = 39988283717


def run(command, **kwargs):
  print("+", subprocess.list2cmdline(list(map(str, command))), flush=True)
  return subprocess.run(list(map(str, command)), check=True, **kwargs)


def sha256(path):
  return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def write(path, value):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def hook(root=ROOT):
  spec = importlib.util.spec_from_file_location("native_hook", root / "plugins/scaly-ipopt/hatch_build.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def recipe(root, source=None):
  return hook()._recipe_hash(root / "plugins/scaly-ipopt", source)


def matrix(target):
  if target != "all" and target not in TARGETS:
    raise ValueError(f"Unknown target {target}")
  return {"include": [{"target": name, **settings} for name, settings in TARGETS.items() if target in ("all", name)]}


def changed(base):
  names = subprocess.check_output(["git", "diff", "--name-only", base, "HEAD"], text=True).splitlines()
  if any(name.startswith((".github/scripts/", ".github/workflows/ipopt-native")) for name in names):
    return True
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", base, "plugins/scaly-ipopt"], text=True).splitlines()
    for name in paths:
      if name.endswith(("hatch_build.py", "build_config.json")) or name.startswith("plugins/scaly-ipopt/licenses/"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "show", f"{base}:{name}"]))
    return recipe(ROOT) != recipe(root)


def compare(baseline, candidate):
  if baseline.keys() != candidate.keys():
    raise RuntimeError("Numerical case inventory changed")
  for name, expected in baseline.items():
    actual = candidate[name]
    for key in ("status", "iter"):
      if expected[key] != actual[key]:
        raise RuntimeError(f"Investigate {name}: {key} differs: {expected[key]} -> {actual[key]}")
    for key in ("x", "obj"):
      a = expected[key] if isinstance(expected[key], list) else [expected[key]]
      b = actual[key] if isinstance(actual[key], list) else [actual[key]]
      if len(a) != len(b) or not all(math.isclose(x, y, abs_tol=1e-6, rel_tol=1e-8) for x, y in zip(a, b, strict=True)):
        raise RuntimeError(f"Investigate {name}: numerical {key} differs")


def configuration_comparison(manifest, control):
  effective = manifest["configuration"]["threading"]["effective"]
  if not isinstance(effective, dict):
    return {"blas": "Accelerate", "active_threads": "unmeasured"}
  settings = "\n".join(effective.values())
  cores = re.search(r"^NUM_CORES\s*=\s*(\d+)", settings, re.MULTILINE)
  core = re.search(r"^CORE\s*=\s*(\S+)", settings, re.MULTILINE)
  if not cores or not core or core[1] != control["blas"]["common_code_target"]:
    raise RuntimeError("Missing effective OpenBLAS configuration or changed common-code CPU baseline")
  if re.search(r"(?:INTERFACE64\s*=\s*1|#define\s+(?:OPENBLAS_)?USE64BITINT\b)", settings):
    raise RuntimeError("OpenBLAS changed its 32-bit integer interface")
  maximum = int(cores[1])
  return {
    "blas": "OpenBLAS",
    "candidate_compiled_max_threads": maximum,
    "control_compiled_max_threads": control["blas"]["compiled_max_threads"],
    "compiled_maximum_differs": maximum != control["blas"]["compiled_max_threads"],
    "common_code_target": core[1],
    "active_threads": "unmeasured",
    "thread_policy": "unchanged",
  }


def build(target):
  archive = ROOT / "candidate" / f"scaly-ipopt-native-{target}.tar.gz"
  work = ROOT / "build/native"
  command = ["uv", "run", "plugins/scaly-ipopt/hatch_build.py", "bundle", "--work-dir", work, "--output", archive]
  image = TARGETS[target].get("image")
  identity = None
  if image:
    run(["docker", "pull", image])
    identity = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    # The image is the one pinned by the existing cibuildwheel 4.2.1 source controls.
    run(
      [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{ROOT}:/project",
        "-w",
        "/project",
        image,
        "bash",
        "-ec",
        "/opt/python/cp312-cp312/bin/python -m pip install uv; "
        "export PATH=/opt/python/cp312-cp312/bin:$PATH; "
        "uv run plugins/scaly-ipopt/hatch_build.py bundle --work-dir /project/build/native "
        f"--output /project/candidate/{archive.name} --platform-tag manylinux_2_27_{target.removeprefix('linux-')}; "
        "gcc -O2 -DSCALY_STANDALONE .github/scripts/ipopt_native_consumer.c "
        "-Ibuild/native/payload/scaly_ipopt/include/coin-or -Lbuild/native/payload/scaly_ipopt/lib -lipopt "
        f"-Wl,-rpath,'$ORIGIN/scaly_ipopt/lib' -o candidate/{target}.floor-probe",
      ]
    )
  else:
    run(command)
  manifest = json.loads(archive.with_name(archive.name + ".manifest.json").read_text())
  control = json.loads((ROOT / "internal/notes/ipopt_native_controls" / (TARGETS[target]["control"] + ".json")).read_text())
  provenance = {
    "schema_version": 1,
    "target": target,
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "run_id": int(os.environ["GITHUB_RUN_ID"]),
    "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
    "run_url": f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
    "event": os.environ["GITHUB_EVENT_NAME"],
    "runner": {name: os.environ.get(name) for name in ("ImageOS", "ImageVersion", "RUNNER_ARCH", "RUNNER_OS")},
    "linux_image": {"reference": image, "id": identity["Id"], "digests": identity["RepoDigests"]} if identity else None,
    "archive_sha256": sha256(archive),
    "manifest_sha256": sha256(archive.with_name(archive.name + ".manifest.json")),
    "recipe_sha256": manifest["recipe_sha256"],
    "configuration": manifest["configuration"],
    "source_control": control,
    "configuration_comparison": configuration_comparison(manifest, control),
  }
  write(archive.with_name(archive.name + ".provenance.json"), provenance)


def main():
  parser = argparse.ArgumentParser(__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  select = sub.add_parser("select")
  select.add_argument("--target", default="all")
  select.add_argument("--base")
  produce = sub.add_parser("build")
  produce.add_argument("--target", choices=TARGETS, required=True)
  args = parser.parse_args()
  if args.command == "build":
    build(args.target)
  else:
    selected = not args.base or changed(args.base)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
      output.write("build=" + str(selected).lower() + "\n")
      output.write("matrix=" + json.dumps(matrix(args.target)) + "\n")
      output.write("commit=" + subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() + "\n")


if __name__ == "__main__":
  main()
