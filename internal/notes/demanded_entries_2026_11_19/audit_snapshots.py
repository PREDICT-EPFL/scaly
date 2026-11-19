"""Compare the generated C corpus to its base and record every changed hunk."""

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

CORPUS = Path("tests/baseline/c")


def audit(base):
  before = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", base, "--", str(CORPUS)], text=True).splitlines())
  after = {p.as_posix() for p in CORPUS.iterdir() if p.is_file()}
  rows = []
  for name in sorted(before | after):
    old = subprocess.check_output(["git", "show", f"{base}:{name}"]) if name in before else b""
    new = Path(name).read_bytes() if name in after else b""
    rows.append({
      "file": name,
      "before_bytes": len(old),
      "after_bytes": len(new),
      "before_sha256": hashlib.sha256(old).hexdigest(),
      "after_sha256": hashlib.sha256(new).hexdigest(),
      "changed": old != new,
      "diff": list(difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(), f"base/{name}", name, lineterm="")),
    })
  return rows


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("base")
  args = parser.parse_args()
  print(json.dumps({"base": args.base, "files": audit(args.base)}, indent=2))
