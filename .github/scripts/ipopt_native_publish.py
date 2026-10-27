"""Preflight reviewed five-target runs and publish their exact bytes only on explicit dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

TARGETS = {"windows-x86_64", "linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-arm64"}
GATES = {
  "archive",
  "static_audit",
  "installed_bytes",
  "c",
  "cxx",
  "direct_nested_stats",
  "sparse_mumps",
  "coexistence_both_orders",
  "negative_runtime",
  "source_comparison",
  "solver_tests",
  "closed_loop",
  "installed_smoke",
}


def sha256(path):
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def api(path):
  return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def run(*command):
  subprocess.run(list(map(str, command)), check=True)


def admit_run(run, reviewed_sha):
  if not (
    run["status"] == "completed"
    and run["conclusion"] == "success"
    and run["head_branch"] == "main"
    and run["event"] in {"push", "workflow_dispatch"}
    and run["path"] == ".github/workflows/ipopt-native.yml"
    and run["head_sha"] == reviewed_sha
  ):
    raise RuntimeError("Publication requires a successful full producer run of the reviewed main commit; PR, trial and monthly runs cannot publish")


def admit_reports(reports, sha, run_id, attempt):
  if reports.keys() != TARGETS:
    raise RuntimeError("Publication requires exactly five validated targets")
  for target, report in reports.items():
    if (report["target"], report["commit"], report["run_id"], report["run_attempt"]) != (target, sha, run_id, attempt):
      raise RuntimeError("Mixed commits, runs or attempts in candidate reports")
    if report["gates"].keys() != GATES or not all(value is True for value in report["gates"].values()):
      raise RuntimeError(f"Incomplete acceptance: {target}")


def verify_assets(directory, report):
  for name, digest in report["assets"].items():
    if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(directory / name) != digest:
      raise RuntimeError(f"Changed or unsafe asset: {name}")


def summarize(directory, target, output):
  reports = {value["target"]: value for path in directory.glob("*.validation.json") if (value := json.loads(path.read_text()))}
  expected = TARGETS if target == "all" else {target}
  if reports.keys() != expected:
    raise RuntimeError("Missing or unexpected target report")
  first = next(iter(reports.values()))
  if target == "all":
    admit_reports(reports, first["commit"], first["run_id"], first["run_attempt"])
  result = {
    "kind": "full" if target == "all" else "trial",
    "commit": first["commit"],
    "run_id": first["run_id"],
    "run_attempt": first["run_attempt"],
    "reports": reports,
    "report_hashes": {path.name: sha256(path) for path in directory.glob("*.validation.json")},
  }
  output.write_text(json.dumps(result, indent=2) + "\n")


def preflight(args):
  repo = os.environ["GITHUB_REPOSITORY"]
  if repo != "PREDICT-EPFL/scaly" or os.environ["GITHUB_REF"] != "refs/heads/main":
    raise RuntimeError("Publication workflow must run on this repository's main branch")
  for actor in {os.environ["GITHUB_ACTOR"], os.environ.get("GITHUB_TRIGGERING_ACTOR", os.environ["GITHUB_ACTOR"])}:
    if api(f"repos/{repo}/collaborators/{actor}/permission")["permission"] not in {"admin", "maintain"}:
      raise RuntimeError("Only a repository maintainer can admit an official native revision")
  if not re.fullmatch(r"scaly-ipopt-native-v[1-9][0-9]*", args.revision):
    raise RuntimeError("Use a new separate native release revision")
  if not re.fullmatch(r"[0-9a-f]{40}", args.reviewed_sha):
    raise RuntimeError("Supply the full reviewed main commit")
  source = api(f"repos/{repo}/actions/runs/{args.candidate_run}")
  admit_run(source, args.reviewed_sha)
  if source["head_repository"]["full_name"] != repo:
    raise RuntimeError("Foreign repository output")
  main = api(f"repos/{repo}/commits/main")["sha"]
  if main != args.reviewed_sha:
    raise RuntimeError("Generate and review candidates at current main before publication")
  pulls = api(f"repos/{repo}/commits/{args.reviewed_sha}/pulls")
  approved = False
  for pull in pulls:
    if pull["merged_at"] and pull["base"]["ref"] == "main":
      reviews = api(f"repos/{repo}/pulls/{pull['number']}/reviews?per_page=100")
      latest = {review["user"]["login"]: review["state"] for review in reviews if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}}
      approved |= "APPROVED" in latest.values() and "CHANGES_REQUESTED" not in latest.values()
  if not approved:
    raise RuntimeError("The candidate main commit needs a merged, approved main-branch pull request")
  directory = args.directory
  directory.mkdir(parents=True, exist_ok=True)
  for name in [
    "ipopt-accepted",
    *(f"ipopt-candidate-{target}" for target in sorted(TARGETS)),
    *(f"ipopt-validation-{target}" for target in sorted(TARGETS)),
  ]:
    run("gh", "run", "download", args.candidate_run, "-R", repo, "--name", name, "--dir", directory)
  accepted = json.loads((directory / "accepted.json").read_text())
  if accepted["kind"] != "full":
    raise RuntimeError("A single-target trial cannot publish")
  admit_reports(accepted["reports"], args.reviewed_sha, args.candidate_run, source["run_attempt"])
  verify_assets(directory, {"assets": accepted["report_hashes"]})
  pins = {"release": args.revision, "repository": repo, "targets": {}}
  assets = {"accepted.json": sha256(directory / "accepted.json")}
  for target, report in accepted["reports"].items():
    verify_assets(directory, report)
    assets.update(report["assets"])
    name = f"scaly-ipopt-native-{target}.tar.gz"
    provenance = json.loads((directory / (name + ".provenance.json")).read_text())
    if (provenance["commit"], provenance["run_id"], provenance["run_attempt"], provenance["archive_sha256"]) != (
      args.reviewed_sha,
      args.candidate_run,
      source["run_attempt"],
      report["assets"][name],
    ):
      raise RuntimeError("Producer provenance differs from validated output")
    run(
      "gh",
      "attestation",
      "verify",
      directory / name,
      "--repo",
      repo,
      "--signer-workflow",
      "PREDICT-EPFL/scaly/.github/workflows/ipopt-native.yml",
      "--source-ref",
      "refs/heads/main",
      "--source-digest",
      args.reviewed_sha,
    )
    pins["targets"][target] = {"asset": name, "sha256": report["assets"][name], "platform_tag": report["platform_tag"]}
  (directory / "consumer-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
  assets.update(accepted["report_hashes"])
  assets["consumer-pins.json"] = sha256(directory / "consumer-pins.json")
  plan = {
    "revision": args.revision,
    "candidate_run": args.candidate_run,
    "run_attempt": source["run_attempt"],
    "reviewed_sha": args.reviewed_sha,
    "assets": assets,
    "pins": pins,
  }
  (directory / "publication-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
  print(json.dumps(plan, indent=2))
  return plan


def publish(args):
  repo = os.environ["GITHUB_REPOSITORY"]
  plan = json.loads((args.directory / "publication-plan.json").read_text())
  verify_assets(args.directory, plan)
  rules = api(f"repos/{repo}/environments/ipopt-native-publication")["protection_rules"]
  if not any(rule["type"] == "required_reviewers" and rule.get("reviewers") and rule.get("prevent_self_review") for rule in rules):
    raise RuntimeError("The publication environment must require an independent reviewer")
  # Never edit a previous revision or resume an ambiguous partially uploaded release.
  exists = subprocess.run(["gh", "api", f"repos/{repo}/releases/tags/{plan['revision']}"], capture_output=True)
  tag = subprocess.run(["gh", "api", f"repos/{repo}/git/ref/tags/{plan['revision']}"], capture_output=True)
  if exists.returncode == 0 or tag.returncode == 0:
    raise RuntimeError("Revision already exists; keep it and use a new revision")
  if b"404" not in exists.stderr or b"404" not in tag.stderr:
    raise RuntimeError("Cannot establish that release and tag are absent")
  notes = args.directory / "release-notes.md"
  notes.write_text(
    f"Native Ipopt revision from reviewed candidate run {plan['candidate_run']}, attempt {plan['run_attempt']}.\n\n"
    "The attached manifests, producer provenance, acceptance reports and consumer-pins.json identify the validated bytes.\n"
  )
  revision = plan["revision"]
  run(
    "gh",
    "release",
    "create",
    revision,
    "-R",
    repo,
    "--draft",
    "--target",
    plan["reviewed_sha"],
    "--latest=false",
    "--title",
    revision,
    "--notes-file",
    notes,
  )
  paths = [args.directory / name for name in sorted(plan["assets"])] + [args.directory / "publication-plan.json"]
  run("gh", "release", "upload", revision, *paths, "-R", repo)
  with tempfile.TemporaryDirectory() as temporary:
    run("gh", "release", "download", revision, "-R", repo, "--dir", temporary)
    remote = Path(temporary)
    if {path.name for path in remote.iterdir()} != {path.name for path in paths}:
      raise RuntimeError("Draft asset inventory differs; leave the draft unpublished")
    for path in paths:
      if sha256(path) != sha256(remote / path.name):
        raise RuntimeError("Draft download differs; leave the draft unpublished")
  run("gh", "release", "edit", revision, "-R", repo, "--draft=false", "--latest=false")


def main():
  parser = argparse.ArgumentParser(__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  summary = sub.add_parser("summarize")
  summary.add_argument("--directory", type=Path, required=True)
  summary.add_argument("--target", required=True)
  summary.add_argument("--output", type=Path, required=True)
  check = sub.add_parser("preflight")
  check.add_argument("--candidate-run", type=int, required=True)
  check.add_argument("--reviewed-sha", required=True)
  check.add_argument("--revision", required=True)
  check.add_argument("--directory", type=Path, required=True)
  release = sub.add_parser("publish")
  release.add_argument("--directory", type=Path, required=True)
  args = parser.parse_args()
  if args.command == "summarize":
    summarize(args.directory, args.target, args.output)
  elif args.command == "preflight":
    preflight(args)
  else:
    publish(args)


if __name__ == "__main__":
  main()
