"""Select and record completed release checks with their immutable GitHub artifacts."""

import argparse
import hashlib
import io
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

DIST_NAMES = {"dist-pure", "dist-linux-x86_64", "dist-linux-arm64", "dist-macos-arm64", "dist-windows-x86_64"}


def gh(*args):
  return subprocess.check_output(["gh", *args])


def listing(path, key):
  pages = json.loads(gh("api", "--paginate", "--slurp", path))
  return [item for page in pages for item in page[key]]


def artifacts_for(repo, run_id):
  return {a["id"]: a for a in listing(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", "artifacts")}


def check_testpypi(packages, directory):
  published = {}
  for name, version in packages.items():
    with urllib.request.urlopen(f"https://test.pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
      published.update({f["filename"]: f["digests"]["sha256"] for f in json.load(response)["urls"]})
  for path in directory.iterdir():
    if published.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
      raise ValueError(f"TestPyPI file differs from the selected release file: {path.name}")


def record_validation(repo, sha, run_id, attempt, kind, packages):
  names = {"release-dist"} if kind == "testpypi" else DIST_NAMES
  artifacts = [a for a in artifacts_for(repo, run_id).values() if a["name"] in names and not a["expired"]]
  if {a["name"] for a in artifacts} != names or len(artifacts) != len(names):
    raise ValueError("validation requires all distribution artifacts to be available")
  return {
    "sha": sha,
    "run_id": run_id,
    "run_attempt": attempt,
    "kind": kind,
    "artifacts": sorted(a["id"] for a in artifacts),
    "packages": packages if kind == "testpypi" else {},
  }


def select_source(repo, sha, current_run, packages):
  runs = {}
  for workflow in ("release.yml", "nightly.yml"):
    candidates = listing(f"repos/{repo}/actions/workflows/{workflow}/runs?branch=main&head_sha={sha}&status=completed&per_page=100", "workflow_runs")
    runs[workflow] = [
      r
      for r in candidates
      if r["id"] != current_run
      and r["head_sha"] == sha
      and r["head_branch"] == "main"
      and r["status"] == "completed"
      and r["event"] in ("schedule", "workflow_dispatch")
      and (workflow == "release.yml" or r["conclusion"] == "success")
    ]
  artifacts = {}
  nightly_runs = sorted(runs["release.yml"] + runs["nightly.yml"], key=lambda r: r["created_at"], reverse=True)
  for kind, candidates in (("testpypi", runs["release.yml"]), ("nightly", nightly_runs)):
    for run in candidates:
      run_id = run["id"]
      if run_id not in artifacts:
        artifacts[run_id] = artifacts_for(repo, run_id)
      available = artifacts[run_id]
      proofs = [a for a in available.values() if a["name"] == f"{kind}-validated" and not a["expired"]]
      for proof in proofs:
        try:
          with zipfile.ZipFile(io.BytesIO(gh("api", f"repos/{repo}/actions/artifacts/{proof['id']}/zip"))) as archive:
            record = json.loads(archive.read("validation.json"))
          if any(record.get(k) != v for k, v in {"sha": sha, "run_id": run_id, "run_attempt": run["run_attempt"], "kind": kind}.items()):
            continue
          selected = [available[i] for i in record["artifacts"]]
          names = {"release-dist"} if kind == "testpypi" else DIST_NAMES
          if len(selected) != len(names) or {a["name"] for a in selected} != names or any(a["expired"] for a in selected):
            continue
          if kind == "testpypi" and any(record["packages"].get(name) != version for name, version in packages.items()):
            continue
        except (KeyError, TypeError, ValueError, AttributeError, zipfile.BadZipFile):
          continue
        return {
          "artifact-run": str(run_id),
          "artifact-ids": ",".join(str(a["id"]) for a in selected),
          "reused": "true",
          "testpypi-verified": str(kind == "testpypi").lower(),
        }
  return {"artifact-run": str(current_run), "artifact-ids": "", "reused": "false", "testpypi-verified": "false"}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  sub.add_parser("select")
  sub.add_parser("check-testpypi")
  sub.add_parser("record").add_argument("kind", choices=("nightly", "testpypi"))
  args = parser.parse_args()
  packages = {p["name"]: p["version"] for p in json.loads(os.environ.get("PACKAGES", "[]"))}
  if args.command == "check-testpypi":
    check_testpypi(packages, Path("release"))
    return
  repo, sha, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"], int(os.environ["GITHUB_RUN_ID"])
  if args.command == "record":
    record = record_validation(repo, sha, run_id, int(os.environ["GITHUB_RUN_ATTEMPT"]), args.kind, packages)
    Path("validation.json").write_text(json.dumps(record) + "\n")
  else:
    source = select_source(repo, sha, run_id, packages)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
      output.writelines(f"{name}={value}\n" for name, value in source.items())
    if source["reused"] == "true":
      stage = "TestPyPI verification" if source["testpypi-verified"] == "true" else "full nightly validation"
      message = f"Reusing {stage} and its files from https://github.com/{repo}/actions/runs/{source['artifact-run']}."
    else:
      message = "Building and validating this release commit in the current run."
    print(message)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
      with open(summary, "a") as output:
        output.write(message + "\n")


if __name__ == "__main__":
  main()
