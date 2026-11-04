"""Temporary diagnostic for #252: why a restored zig cache is rebuilt on Windows. Removed before review."""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scaly.codegen.toolchain import find_c_compiler


def snapshot(root: Path) -> dict[str, float]:
  return {str(p.relative_to(root)): p.stat().st_mtime for p in root.rglob("*") if p.is_file()} if root.exists() else {}


def size(root: Path) -> str:
  files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
  return f"{root}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB"


compiler = find_c_compiler()
assert compiler is not None
zig = compiler.command[0]
root = Path(os.environ["ZIG_GLOBAL_CACHE_DIR"])
local = Path(os.environ.get("LOCALAPPDATA", "")) / "zig"
print(subprocess.run([zig, "env"], capture_output=True, text=True).stdout)
print(size(root))
print(size(local))
if sys.argv[1:] == ["after"]:
  sys.exit()
before = snapshot(root)
manifests = [name for name in sorted(before) if Path(name).parts[0] == "h" and Path(name).name != "timestamp"]
print(f"{len(manifests)} manifests restored")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
  Path(tmp, "probe.c").write_text("double probe(double x) { return x * 2.0; }\n")
  start = time.perf_counter()
  subprocess.run([zig, "cc", "-O2", "-march=native", "-shared", "probe.c", "-o", "probe.dll"], cwd=tmp, check=True)
  print(f"probe compile: {time.perf_counter() - start:.1f} s")
after = snapshot(root)
for name in sorted(after):
  if name not in before:
    print("new:", name)
  elif after[name] != before[name]:
    print("rewritten:", name)
for name in sorted(after):
  if Path(name).parts[0] == "h" and Path(name).name != "timestamp" and (name not in before or after[name] != before[name]):
    lines = [m.decode() for m in re.findall(rb"[\x20-\x7e]{6,}", (root / name).read_bytes())]
    print(f"--- {name} ({len(lines)} strings)")
    print("\n".join(lines[:12]))
for name in manifests[:8]:
  lines = [m.decode() for m in re.findall(rb"[\x20-\x7e]{6,}", (root / name).read_bytes())]
  print(f"--- restored {name} ({len(lines)} strings)")
  print("\n".join(lines[:6]))
print(size(local))
