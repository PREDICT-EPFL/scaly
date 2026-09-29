"""Release the lockstep distributions: one version for all of them, their wheels, and the release check.

    uv run scripts/release.py 0.2.0          # set 0.2.0, regenerate the manifests, build, check
    uv run scripts/release.py 0.2.0 --tag    # ... and tag v0.2.0 once the check passes
    uv run scripts/release.py --check-only   # build and check the version in distributions.toml

Setting a version rewrites ``version`` in ``distributions.toml``, so every lockstep manifest the
generator writes pins its first-party dependencies to it, and relocks. The build puts the wheels
and sdists of every lockstep distribution and solver plugin into ``dist/``. The release check
installs ``scaly[experimental,solvers]`` from those wheels alone into a clean environment, with the
third-party packages the examples declare, and runs the conformance suites and the example runner
there, every solver plugin required to load. Nothing is uploaded: ``uv publish dist/*`` does that.
CI's release job runs ``--check-only`` on a version tag.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a file: `scripts` is a package of the root
from scripts import distributions as dist  # noqa: E402
from scripts import isolation  # noqa: E402

ROOT = dist.ROOT
DIST_DIR = ROOT / "dist"
CHECKS = ["tests/conformance", "tests/integration/test_example_runner.py", "tests/integration/test_examples_lint.py"]
"""What the release check runs in the clean environment."""
_VERSION = re.compile(r'(?m)^version = "[^"]*"$')
_SEMVER = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+)?$")


def set_version(table_text: str, version: str) -> str:
  """``distributions.toml``'s text with its top-level ``version`` set to ``version``."""
  if not _SEMVER.match(version):
    raise ValueError(f"not a release version: {version!r} (X.Y.Z, optionally aN, bN or rcN)")
  if len(_VERSION.findall(table_text)) != 1:
    raise ValueError("distributions.toml must hold exactly one top-level version line")
  return _VERSION.sub(f'version = "{version}"', table_text)


def example_requirements() -> list[str]:
  """The third-party packages the examples declare, which the release check installs so that no
  example skips for want of one."""
  from scaly.testing.examples import parse, requirements

  table = dist.load()
  names: set[str] = set()
  for path in sorted([*(ROOT / "examples").rglob("*.py"), *(ROOT / "examples").rglob("*.ipynb")]):
    if "case_studies" in path.parts or ".ipynb_checkpoints" in path.parts:
      continue
    reqs = requirements(path)
    names |= {parse(r)[0] for r in (reqs.dependencies if reqs else ()) if parse(r)[0] not in table.distributions}
  return sorted(names)


def build(out: Path = DIST_DIR) -> list[str]:
  """Build the wheel and sdist of every lockstep distribution and solver plugin into ``out``."""
  table = dist.load()
  names = sorted(table.distributions)
  out.mkdir(parents=True, exist_ok=True)
  for name in names:
    isolation.run(["uv", "build", "--package", name, "--out-dir", out, "--quiet"])
  return names


def check(wheels: Path = DIST_DIR) -> int:
  """Install ``scaly[experimental,solvers]`` from ``wheels`` alone into a clean environment and run
  the release checks there; returns pytest's exit code."""
  table = dist.load()
  first_party = [f"scaly[experimental,solvers]=={table.version}", f"scaly-testing=={table.version}"]
  everything = sorted(table.distributions)
  index = [*isolation.third_party_requirements(everything, table), *isolation.test_tools(), *example_requirements()]
  with tempfile.TemporaryDirectory(prefix="scaly-release-") as tmp:
    python = isolation.make_env(Path(tmp) / "env", first_party, wheels, index=index)
    return isolation.run_tests(python, CHECKS, ["-n", "auto"], require_methods=True)


def tag(version: str) -> None:
  status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
  if status.strip():
    raise SystemExit("the working tree has changes: commit the release first, then tag it")
  isolation.run(["git", "tag", "-a", f"v{version}", "-m", f"scaly {version}"])


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("version", nargs="?", help="the version to release, X.Y.Z")
  parser.add_argument("--check-only", action="store_true", help="build and check the version distributions.toml holds")
  parser.add_argument("--tag", action="store_true", help="tag vX.Y.Z once the check passes")
  args = parser.parse_args(argv)
  if bool(args.version) == args.check_only:
    parser.error("give a version, or --check-only")
  if args.version:
    table_path = dist.TABLE
    table_path.write_text(set_version(table_path.read_text(), args.version))
    isolation.run([sys.executable, ROOT / "scripts" / "distributions.py"])
    isolation.run(["uv", "lock", "--quiet"])
  build()
  code = check()
  if code != 0:
    print(f"the release check failed (pytest exit {code}); nothing tagged", file=sys.stderr)
    return code
  if args.tag:
    tag(dist.load().version)
  print(f"ready: {DIST_DIR.relative_to(ROOT)}/ holds the wheels; `uv publish {DIST_DIR.relative_to(ROOT)}/*` uploads them")
  return 0


if __name__ == "__main__":
  sys.exit(main())
