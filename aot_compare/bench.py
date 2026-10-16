"""Generate each backend's npmpc Hessian at each horizon, build it into the gbench harness, and time the binaries as interleaved fresh processes.

Run from this directory with `uv run bench.py`. Everything lands in build/: third_party/ once, then
for each horizon N, inputs/ and one directory per backend holding exactly the generated files its
binary is built from, with the harness build in bin/.
"""

import argparse
import csv
import importlib
import shutil
from pathlib import Path

import harness

BUILD = Path(__file__).parent / "build"
BACKENDS = ("scaly", "xla", "iree")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--sizes", type=lambda s: [int(n) for n in s.split(",")], default=[12, 50, 200], help="horizons, comma-separated")
  parser.add_argument("--reps", type=int, default=3, help="fresh processes per binary; the best is reported")
  parser.add_argument("--min-time", default="0.5s", help="Google Benchmark's minimum time per process")
  parser.add_argument("--backends", type=lambda s: s.split(","), default=list(BACKENDS), help="backends, comma-separated; xla needs `--group xla`")
  args = parser.parse_args()
  # Imported only when selected, so that the others run without TensorFlow installed.
  lowerings = {name: importlib.import_module(f"lower_{name}").lower for name in args.backends}

  binaries: dict[tuple[int, str], Path] = {}
  for N in args.sizes:
    inputs = harness.write_inputs(BUILD / f"N{N}" / "inputs", N)
    for name, lower in lowerings.items():
      out = BUILD / f"N{N}" / name
      shutil.rmtree(out, ignore_errors=True)
      out.mkdir(parents=True)
      print(f"N={N}: lowering and building {name}", flush=True)
      binaries[N, name] = harness.build(lower(N, out), N, inputs)

  times: dict[tuple[int, str], list[float]] = {key: [] for key in binaries}
  for rep in range(args.reps):
    for key, binary in binaries.items():
      times[key].append(harness.run(binary, args.min_time))
      print(f"rep {rep + 1}/{args.reps} N={key[0]} {key[1]}: {times[key][-1]:.2f} us", flush=True)

  with open(BUILD / "results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "backend", "best_us", *(f"rep{i + 1}_us" for i in range(args.reps))])
    for (N, name), values in times.items():
      writer.writerow([N, name, f"{min(values):.2f}", *(f"{v:.2f}" for v in values)])
  print(f"\nLower-triangle Lagrangian Hessian, gbench CPU time in us, best of {args.reps} processes (ratio to Scaly)")
  print(f"{'N':>5}" + "".join(f"{name:>18}" for name in args.backends))
  for N in args.sizes:
    scaly = min(times[N, "scaly"])
    cells = [f"{min(times[N, name]):9.1f}" + ("" if name == "scaly" else f" ({min(times[N, name]) / scaly:.2f}x)") for name in args.backends]
    print(f"{N:>5}" + "".join(f"{cell:>18}" for cell in cells))


if __name__ == "__main__":
  main()
