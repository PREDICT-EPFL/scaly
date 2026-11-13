"""Derive nightly Python entries and workflow failure labels from their source records."""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("record", choices=("python", "labels"))
  parser.add_argument("--alert-label", default="ci:nightly-failing")
  args = parser.parse_args()
  if args.record == "python":
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    versions = [m[1] for c in project["classifiers"] if (m := re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", c))]
    print(json.dumps([{"platform": "linux-x86_64", "runner": "ubuntu-latest", "python": v, "label": v} for v in versions]))
  else:
    labels = ["source:maintainer", "kind:bug", "area:tooling", args.alert_label]
    if any(name.startswith("benchmarks / ") for name in sys.stdin.read().splitlines()):
      labels.append("area:benchmarks")
    print(",".join(labels))


if __name__ == "__main__":
  main()
