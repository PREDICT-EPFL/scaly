"""Validate each candidate on a fresh runner using installed wheels and the existing solver checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ipopt_native import ROOT, TARGETS, compare, configuration_comparison, hook, run, sha256, write
from ipopt_native_audit import audit
from ipopt_native_publish import GATES

SCRIPTS = ROOT / ".github/scripts"


def clean_env(python, cache):
  keep = ("SystemRoot", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP", "HOME", "TMPDIR")
  env = {key: os.environ[key] for key in keep if key in os.environ}
  system = os.environ.get("SystemRoot", "C:/Windows")
  paths = [str(python.parent), system + "/System32", system] if sys.platform == "win32" else [str(python.parent), "/usr/bin", "/bin"]
  return env | {"PATH": os.pathsep.join(paths), "SCALY_CACHE_DIR": str(cache), "SCALY_REQUIRE_SOLVERS": "ipopt,piqp", "UV_NO_SYNC": "1"}


def environment(path, wheels, constraints=None):
  run(["uv", "venv", "--python", "3.12", path])
  python = path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
  command = ["uv", "pip", "install", "--python", python, "--only-binary", ":all:", *wheels, "casadi", "psutil"]
  if constraints:
    command += ["--constraint", constraints]
  run(command)
  return python


def site(python):
  return Path(subprocess.check_output([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())


def verify_installed(root, manifest):
  for name, entry in manifest["files"].items():
    path = root / name
    if "link" in entry:
      # Wheels dereference archive file links while preserving the target bytes.
      target = (path.parent / entry["link"]).resolve()
      if not path.is_file() or path.read_bytes() != target.read_bytes():
        raise RuntimeError(f"Installed symlink target differs: {name}")
    elif sha256(path) != entry["sha256"]:
      raise RuntimeError(f"Installed bytes differ: {name}")


def consumer(python, installed, work, language, env):
  if sys.platform == "win32":
    query = "from scaly.codegen.toolchain import find_c_compiler; import json; c=find_c_compiler(); assert c.source=='zig-bin',c; print(json.dumps(c.command))"
    compiler = json.loads(subprocess.check_output([str(python), "-I", "-c", query], env=env, text=True))
    if language == "cxx":
      compiler[-1] = "c++"
  else:
    compiler = ["c++" if language == "cxx" else "cc"]
  source = work / ("consumer.cpp" if language == "cxx" else "consumer.c")
  shutil.copy2(SCRIPTS / "ipopt_native_consumer.c", source)
  output = work / (language + (".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"))
  library = installed / "scaly_ipopt/lib"
  flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
  if sys.platform != "win32":
    flags += ["-fPIC", f"-Wl,-rpath,{library}"]
  command = [*compiler, "-O2", *flags, source, f"-I{installed / 'scaly_ipopt/include/coin-or'}", f"-L{library}", "-lipopt", "-o", output]
  version = subprocess.check_output([*compiler, "--version"], env=env, text=True)
  run(command, env=env, cwd=work)
  return output, {"command": list(map(str, command)), "identity": version}


def probe(python, installed, manifest_path, output, env, work, *args):
  run(
    [python, "-I", SCRIPTS / "ipopt_native_probe.py", "--root", installed, "--manifest", manifest_path, "--output", output, *args], env=env, cwd=work
  )
  return json.loads(output.read_text())


def negative(root, manifest, work, target, report):
  isolated = work / "negative"
  shutil.copytree(root / "scaly_ipopt/lib", isolated / "scaly_ipopt/lib")
  if (root / "scaly_ipopt.libs").exists():
    shutil.copytree(root / "scaly_ipopt.libs", isolated / "scaly_ipopt.libs")
  runtime = next((isolated / name) for name in manifest["files"] if "gfortran" in name and "/lib/" in name)
  library = next(name for name in manifest["dependencies"] if Path(name).name in {"libscaly_ipopt-3.dll", "libipopt.so", "libipopt.dylib"})
  if target.startswith("linux"):
    # An actual glibc 2.27 runtime with only allowed system libraries. No Fortran runtime,
    # compiler, build prefix, Python package or host search path enters this container.
    context = work / "floor-image"
    context.mkdir()
    (context / "Dockerfile").write_text(
      "FROM ubuntu:18.04\nRUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgcc1 zlib1g && rm -rf /var/lib/apt/lists/*\n"
    )
    image = "scaly-ipopt-floor"
    run(["docker", "build", "-t", image, context])
    runtime_identity = subprocess.check_output(["docker", "run", "--rm", "--network=none", image, "getconf", "GNU_LIBC_VERSION"], text=True).strip()
    if runtime_identity != "glibc 2.27":
      raise RuntimeError(f"Wrong compatibility-floor runtime: {runtime_identity}")
    executable = ROOT / "candidate" / (target + ".floor-probe")
    shutil.copy2(executable, isolated / "floor-probe")
    (isolated / "floor-probe").chmod(0o755)
    command = ["docker", "run", "--rm", "--network=none", "-v", f"{isolated}:/payload:ro", "-w", "/payload", image, "./floor-probe"]
    positive = subprocess.run(command, text=True, capture_output=True, check=True)
    runtime.rename(runtime.with_name(runtime.name + ".hidden"))
    failed = subprocess.run(command, text=True, capture_output=True)
    report["floor_runtime"] = {
      "glibc": runtime_identity,
      "image": json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]["Id"],
      "positive": positive.stdout,
      "probe_sha256": sha256(executable),
    }
  else:
    python = Path(sys.executable)
    env = clean_env(python, work / "negative-jit")
    prelude = "import ctypes,os; " + (f"guard=os.add_dll_directory({str(isolated / 'scaly_ipopt/lib')!r}); " if sys.platform == "win32" else "")
    command = [str(python), "-I", "-c", prelude + f"ctypes.CDLL({str(isolated / library)!r})"]
    positive = subprocess.run(command, env=env, cwd=isolated, text=True, capture_output=True, check=True)
    runtime.rename(runtime.with_name(runtime.name + ".hidden"))
    failed = subprocess.run(command, env=env, cwd=isolated, text=True, capture_output=True)
  if failed.returncode == 0 or not any(
    value in failed.stderr.lower() for value in ("gfortran", "could not find module", "cannot open shared object", "library not loaded")
  ):
    raise RuntimeError(f"Runtime-removal control did not fail for its missing dependency: {failed.returncode}: {failed.stdout} {failed.stderr}")
  report["negative_control"] = {
    "runtime": runtime.name,
    "positive_returncode": positive.returncode,
    "negative_returncode": failed.returncode,
    "stderr": failed.stderr,
  }


def validate(target):
  if (ROOT / "build/native").exists() or Path("/project/build/native").exists():
    raise RuntimeError("Acceptance must run on a fresh runner without the producer prefix")
  output = ROOT / "validation"
  output.mkdir(exist_ok=True)
  payload = ROOT / "build/audit-payload"
  archive = ROOT / "candidate" / f"scaly-ipopt-native-{target}.tar.gz"
  manifest = hook()._read_bundle(archive.read_bytes(), payload)
  detached = archive.with_name(archive.name + ".manifest.json")
  if manifest != json.loads(detached.read_text()):
    raise RuntimeError("Detached manifest differs from the frozen archive")
  if archive.with_name(archive.name + ".sha256").read_text().split()[0] != sha256(archive):
    raise RuntimeError("Archive checksum differs")
  provenance = json.loads(archive.with_name(archive.name + ".provenance.json").read_text())
  control_path = ROOT / "internal/notes/ipopt_native_controls" / (TARGETS[target]["control"] + ".json")
  control = json.loads(control_path.read_text())
  baseline = next((ROOT / "baseline").glob("*.whl"))
  if sha256(baseline) != control["wheel_sha256"]:
    raise RuntimeError("Retained baseline wheel hash differs from the recorded source control")
  report = {
    "target": target,
    "commit": provenance["commit"],
    "run_id": provenance["run_id"],
    "run_attempt": provenance["run_attempt"],
    "platform_tag": manifest["target"]["platform_tag"],
    "gates": {},
    "assets": {},
    "source_control": {"run": control["source_run"], "wheel_sha256": control["wheel_sha256"]},
  }
  report["gates"]["archive"] = True
  write(output / f"{target}.audit.json", audit(payload, manifest))
  report["gates"]["static_audit"] = True
  # Keep extraction and environments out of retained evidence. Only the report and logs ship.
  work = ROOT / "build/acceptance"
  work.mkdir(parents=True)
  wheels = ROOT / "wheels"
  for package in (ROOT, ROOT / "plugins/scaly-sqp"):
    run(["uv", "build", package, "--wheel", "--out-dir", wheels], env=dict(os.environ, SCALY_BUILD_SOLVERS="skip"))
  run(
    ["uv", "build", ROOT / "plugins/scaly-ipopt", "--wheel", "--out-dir", wheels],
    env=dict(os.environ, SCALY_BUILD_SOLVERS="required", SCALY_IPOPT_BUNDLE=str(archive)),
  )
  all_wheels = list(wheels.glob("*.whl"))
  run(["uv", "pip", "install", "--reinstall", "--no-deps", *all_wheels])
  candidate_python = environment(work / "candidate", all_wheels)
  candidate_site = site(candidate_python)
  verify_installed(candidate_site, manifest)
  report["gates"]["installed_bytes"] = True
  freeze = subprocess.check_output(["uv", "pip", "freeze", "--python", str(candidate_python)], text=True)
  constraints = work / "constraints.txt"
  constraints.write_text("\n".join(line for line in freeze.splitlines() if not line.startswith("scaly")) + "\n")
  report["python_dependencies"] = freeze
  baseline_python = environment(work / "baseline", [p for p in all_wheels if not p.name.startswith("scaly_ipopt-")] + [baseline], constraints)
  baseline_site = site(baseline_python)
  verify_installed(baseline_site, control)
  baseline_manifest = work / "baseline-manifest.json"
  write(baseline_manifest, control)
  candidate_manifest = candidate_site / "scaly_ipopt/native_manifest.json"
  results = {}
  for label, python, installed, metadata in (
    ("candidate", candidate_python, candidate_site, candidate_manifest),
    ("baseline", baseline_python, baseline_site, baseline_manifest),
  ):
    env = clean_env(python, work / (label + "-jit"))
    results[label] = {}
    for language in ("c", "cxx"):
      directory = work / f"{label}-{language}"
      directory.mkdir()
      binary, compiler = consumer(python, installed, directory, language, env)
      result = probe(python, installed, metadata, output / f"{target}.{label}.{language}.json", env, directory, "--consumer", binary)
      results[label][language] = result["cases"]
      report[f"{label}_{language}_compiler"] = compiler
    for order in ("scaly", "others"):
      result = probe(python, installed, metadata, output / f"{target}.{label}.{order}.json", env, work, "--order", order)
      results[label][order] = result["cases"]
  for case in results["baseline"]:
    compare(results["baseline"][case], results["candidate"][case])
  report["comparison"] = {
    "statuses_iterations_and_solutions": "matched",
    "cases": results,
    "identical_native_bytes": all(manifest["files"].get(name, {}).get("sha256") == entry["sha256"] for name, entry in control["files"].items()),
    "control_blas": control["blas"],
    "candidate_configuration": manifest["configuration"],
    "effective_configuration_comparison": configuration_comparison(manifest, control),
  }
  for gate in ("c", "cxx", "direct_nested_stats", "sparse_mumps", "coexistence_both_orders", "source_comparison"):
    report["gates"][gate] = True
  negative(payload, manifest, work, target, report)
  report["gates"]["negative_runtime"] = True
  clean = clean_env(Path(sys.executable), work / "test-jit")
  run(
    [
      sys.executable,
      "-m",
      "pytest",
      "plugins/scaly-ipopt/tests/test_nlp_ipopt.py",
      "tests/solvers",
      "tests/benchmarks/test_casadi_ipopt.py",
      "-n",
      "2",
      "-q",
    ],
    env=clean,
  )
  report["gates"]["solver_tests"] = True
  run([candidate_python, "-I", SCRIPTS / "solve_smoke.py"], env=clean_env(candidate_python, work / "smoke-jit"), cwd=work)
  if sys.platform == "win32":
    run(["uv", "run", "--no-project", SCRIPTS / "windows_install.py", wheels, "3.12"])
  report["gates"]["installed_smoke"] = True
  for problem, oracle in (("chain", "scaly"), ("race_cars", "casadi")):
    run(
      [
        sys.executable,
        "benchmarks/run.py",
        "closed-loop",
        "--smoke",
        "--problem",
        problem,
        "--solver",
        "ipopt",
        "--oracle",
        oracle,
        "--out-dir",
        work / f"closed-loop-{problem}",
      ],
      env=clean,
    )
  report["gates"]["closed_loop"] = True
  assert report["gates"].keys() == GATES and all(report["gates"].values())
  report["assets"] = {path.name: sha256(path) for path in archive.parent.glob(archive.name + "*")}
  # All detailed acceptance evidence is hash-bound to the target report and published with it.
  for path in output.glob(f"{target}.*.json"):
    report["assets"][path.name] = sha256(path)
  write(output / f"{target}.validation.json", report)
  shutil.rmtree(payload)


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("--target", choices=TARGETS, required=True)
  args = parser.parse_args()
  validate(args.target)


if __name__ == "__main__":
  main()
