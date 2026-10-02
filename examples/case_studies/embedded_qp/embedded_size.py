"""The embedded leg, sizes only: Scaly's generated PIQP and QOCOGEN compiled for a Cortex-M7.

    uv run --with qoco --with qocogen --with cvxpy examples/case_studies/embedded_qp/embedded_size.py --out examples/case_studies/embedded_qp/results/embedded.json

No board is attached, so this reports what the plan's embedded leg can say without one: the flash
(`.text` + `.rodata` + `.data`) and the static RAM (`.data` + `.bss`) of each generated solver built with
clang for `thumbv7em-none-eabihf -mcpu=cortex-m7 -Os -ffreestanding`, plus the workspace each needs at
run time (Scaly: the `w` array its header declares; QOCOGEN: `sizeof(Workspace)`, which its caller
allocates). Stub `math.h` and `stdio.h` declare what the code calls, as a board's libc would; QOCOGEN's
iteration printers, which its `DISABLE_PRINTING` does not remove, are subtracted from its flash.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "baseline"))
TARGET = ["--target=thumbv7em-none-eabihf", "-mcpu=cortex-m7", "-mfloat-abi=hard", "-Os", "-ffreestanding", "-ffunction-sections"]
MATH_H = """#ifndef STUB_MATH_H
#define STUB_MATH_H
double sqrt(double); double fabs(double); double fmax(double, double); double fmin(double, double);
#define isfinite(x) __builtin_isfinite(x)
#define isnan(x) __builtin_isnan(x)
#define INFINITY __builtin_inf()
#define NAN __builtin_nan("")
#endif
"""


def sizes(objects: list[Path]) -> dict:
  out = subprocess.run(["llvm-size", "-A", *map(str, objects)], capture_output=True, text=True, check=True).stdout
  total: dict[str, int] = {}
  for line in out.splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[0].startswith(".") and parts[1].isdigit():
      name = parts[0].split(".")[1] if parts[0].count(".") > 1 else parts[0][1:]
      total[name] = total.get(name, 0) + int(parts[1])
  flash = sum(v for k, v in total.items() if k in ("text", "rodata", "data"))
  ram = sum(v for k, v in total.items() if k in ("data", "bss"))
  return {"sections": total, "flash_bytes": flash, "static_ram_bytes": ram}


STDIO_H = "#ifndef STUB_STDIO_H\n#define STUB_STDIO_H\nint printf(const char*, ...);\n#endif\n"
PRINTERS = ("print_header", "print_footer", "log_iter")


def compile_objects(sources: list[Path], work: Path, include: list[Path], extra: list[str] | None = None) -> list[Path]:
  (work / "stub").mkdir(exist_ok=True)
  (work / "stub" / "math.h").write_text(MATH_H)
  (work / "stub" / "stdio.h").write_text(STDIO_H)
  objects = []
  for src in sources:
    obj = work / (src.stem + ".o")
    subprocess.run(["clang", *TARGET, *(extra or []), f"-I{work / 'stub'}", *(f"-I{i}" for i in include), "-c", str(src), "-o", str(obj)], check=True, capture_output=True, text=True)
    objects.append(obj)
  return objects


def scaly(horizon: int, work: Path) -> dict:
  import scaly as sc
  from scaly.codegen import render_c_module

  from problem import problem

  fun = sc.opt.solver(problem(horizon), sc.opt.IPM(sparse=True, options={"eps_abs": 1e-7, "eps_rel": 1e-7}), name=f"masses_T{horizon}_piqp")
  module = render_c_module(fun)
  src = work / "scaly_piqp.c"
  src.write_text(module.body)
  return {**sizes(compile_objects([src], work, [])), "workspace_bytes": 8 * int(module.workspace_size), "source_bytes": len(module.body.encode())}


def qocogen(horizon: int, work: Path) -> dict:
  from problem import instances
  from run_qoco import qocogen_build

  solver, info = qocogen_build(horizon, instances([horizon], 1)[(horizon, 0)], work / "gen")
  sources = [Path(s) for s in info["sources"]]
  objects = compile_objects(sources, work, [solver])
  probe = work / "probe.c"
  probe.write_text('#include "workspace.h"\nunsigned long workspace_bytes = sizeof(Workspace);\n')
  subprocess.run(["clang", *TARGET, f"-I{solver}", f"-I{work / 'stub'}", "-S", "-o", str(work / "probe.s"), str(probe)], check=True, capture_output=True)
  lines = (work / "probe.s").read_text().splitlines()
  at = next(i for i, line in enumerate(lines) if line.startswith("workspace_bytes:"))
  ws = int(next(line.split()[-1] for line in lines[at + 1 :] if line.strip().startswith((".long", ".word", ".quad"))), 0)
  # QOCOGEN's DISABLE_PRINTING does not remove the calls to its printers, so it is built as generated
  # and the printers' own code is taken off its flash: Scaly's code prints nothing.
  nm = subprocess.run(["llvm-nm", "-S", *map(str, objects)], capture_output=True, text=True, check=True).stdout
  printers = sum(int(parts[1], 16) for parts in (line.split() for line in nm.splitlines()) if len(parts) == 4 and parts[3] in PRINTERS)
  out = sizes(objects)
  out["flash_bytes"] -= printers
  return {**out, "printer_bytes": printers, "workspace_bytes": ws, "source_bytes": info["source_bytes"], "t_generate": info["t_generate"]}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--horizons", type=int, nargs="+", default=[8, 20, 32, 56])
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  rows = []
  for horizon in args.horizons:
    with tempfile.TemporaryDirectory(prefix="embedded-size-") as tmp:
      work = Path(tmp)
      (work / "s").mkdir()
      (work / "q").mkdir()
      row = {"horizon": horizon, "scaly": scaly(horizon, work / "s"), "qocogen": qocogen(horizon, work / "q")}
    rows.append(row)
    print(horizon, {k: {kk: vv for kk, vv in v.items() if kk != "sections"} for k, v in row.items() if isinstance(v, dict)}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
