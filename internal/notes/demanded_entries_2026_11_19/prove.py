"""Run the demanded-entries tests against main and against one-rule perturbations of this branch.

Usage: uv run prove.py MAIN_SRC SCRATCH, with MAIN_SRC an extracted ``src`` tree of main."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TESTS = "tests/passes/lowering/test_demand.py"
LOWERING = Path("src/scaly/passes/lowering")
PERTURBATIONS = {
  "never narrow a stored node": ("ctx.py", "    if any(arg.type.pattern is not None for arg in node.args):\n      return False\n    op", "    if True:\n      return False\n    op"),
  "read a transpose through its own axes": ("demand.py", "tuple(axes.index(dim) for dim in range(len(axes)))", "axes"),
  "demand a scatter's values that miss the entries": ("demand.py", "lanes = np.flatnonzero(np.isin(targets, entries))", "lanes = np.flatnonzero(~np.isin(targets, entries))"),
  "demand the wrong columns of a product": ("demand.py", "np.arange(inner) * width + columns[:, None] % width", "np.arange(inner) * width + columns[:, None] // width"),
  "evaluate a computed part inside a selected concat": ("fusion.py", "      kinds = dict.fromkeys(kinds, _MANY)\n", "      pass\n"),
  "narrow through any table": ("demand.py", "return 3 * len(entries) <= 2 * size or", "return True or"),
  "read an unselected part at entry zero": ("ctx.py", "p.const_int(0 if computed is None or not len(computed) else int(computed[0]))", "p.const_int(0)"),
  "skip the zeros under a narrowed sum of scatters": ("gather.py", "  if zeros is None or len(zeros):", "  if zeros is None:"),
  "distribute a repeating gather": ("fusion.py", "_ONCE if len(np.unique(indices)) == len(indices) else _MANY)]", "_ONCE)]"),
}


def run(src: Path) -> dict[str, str]:
  """Each test's outcome."""
  command = ["uv", "run", "pytest", "-q", "-n", "2", "-p", "no:cacheprovider", "-rA", TESTS]
  result = subprocess.run(command, env={**os.environ, "PYTHONPATH": f"{src}:."}, capture_output=True, text=True)
  outcomes = {}
  for line in result.stdout.splitlines():
    for state in ("PASSED", "FAILED"):
      if line.startswith(state + " "):
        outcomes[line.split(" ", 1)[1].split(" - ")[0].split("::", 1)[1]] = state.lower()
  return outcomes


def main() -> None:
  main_src, scratch = Path(sys.argv[1]), Path(sys.argv[2])
  report = {"main": run(main_src), "branch": run(Path("src"))}
  for name, (module, old, new) in PERTURBATIONS.items():
    copy = scratch / name.replace(" ", "_").replace("'", "")
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree("src", copy / "src")
    path = copy / "src" / "scaly" / "passes" / "lowering" / module
    text = path.read_text()
    assert text.count(old) == 1, name
    path.write_text(text.replace(old, new))
    report[name] = run(copy / "src")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
