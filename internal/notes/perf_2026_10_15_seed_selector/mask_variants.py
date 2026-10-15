"""Rewrite the npmpc selector masks of a generated kernel as double (A), uint8 (B) or bit (C) masks."""
import re, shutil, subprocess, sys
from pathlib import Path

src, out_root = Path(sys.argv[1]), Path(sys.argv[2])
c_file = next(src.glob("*.c"))
text = c_file.read_text()
tables = {m[1]: [float(x) for x in m[2].split(",")] for m in re.finditer(r"static const double (k\d+)\[\d+\] = \{([^}]*)\};", text)}
masks = {k: v for k, v in tables.items() if len(v) >= 500 and set(v) <= {0.0, 1.0}}
use = re.compile(r"\((k\d+)\[(\w+)\] \* (\w+\[\w+\])\)")

def variant(name: str) -> str:
  t = text
  for k, v in masks.items():
    decl = re.search(rf"static const double {k}\[\d+\] = \{{[^}}]*\}};", t).group(0)
    if name in ("B", "D"):
      t = t.replace(decl, f"static const uint8_t {k}[{len(v)}] = {{{', '.join(str(int(x)) for x in v)}}};")
    elif name == "C":
      packed = [sum(int(v[8 * b + j]) << j for j in range(8) if 8 * b + j < len(v)) for b in range((len(v) + 7) // 8)]
      t = t.replace(decl, f"static const uint8_t {k}[{len(packed)}] = {{{', '.join(map(str, packed))}}};")
  def sub(m):
    k, i, x = m.group(1), m.group(2), m.group(3)
    if k not in masks or name == "A":
      return m.group(0)
    if name == "D":
      return f"((double){k}[{i}] * {x})"
    bit = f"{k}[{i}]" if name == "B" else f"(({k}[({i}) >> 3] >> (({i}) & 7)) & 1)"
    return f"({bit} ? {x} : 0.0)"
  return use.sub(sub, t)

commands = [l[2:] for l in (src / "compile.log").read_text().splitlines() if l.startswith("$ ")]
for name in sys.argv[3].split(","):
  d = out_root / name
  if d.exists(): shutil.rmtree(d)
  shutil.copytree(src, d, ignore=shutil.ignore_patterns("kernel.o", "wrapper.o", "benchmark"))
  (d / c_file.name).write_text(variant(name))
  for cmd in commands:
    subprocess.run(cmd.replace(str(src), str(d)), shell=True, cwd=d, check=True, capture_output=True)
  size = subprocess.run(["size", "-A", str(d / "kernel.o")], capture_output=True, text=True).stdout
  rodata = sum(int(l.split()[1]) for l in size.splitlines() if l.startswith(".rodata"))
  times = []
  for _ in range(3):
    run = subprocess.run([str(d / "benchmark"), "--benchmark_min_time=0.5s"], cwd=d, capture_output=True, text=True)
    assert "correctness ok" in run.stdout, run.stdout + run.stderr
    times.append(float(next(l for l in run.stdout.splitlines() if l.startswith("BM_")).split()[1]) / 1e3)
  print(f"{src.name} {name}: masks {len(masks)}, rodata {rodata // 1000} kB, eval µs {sorted(times)}")
