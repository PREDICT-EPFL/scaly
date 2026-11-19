"""Run the new lowering gates on the base and against one disabled rule at a time."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FUSION_TEST = "tests/passes/lowering/test_fusion.py::"
SEEDS = FUSION_TEST + "test_folded_matmul_seeds_keep_their_logical_layout"
SEED_MAPS = FUSION_TEST + "test_seed_pointwise_maps_share_the_active_pairs"
EPILOGUE = FUSION_TEST + "test_seed_contraction_applies_the_activation_before_storing"
KKT = "tests/passes/lowering/test_kkt_assembly.py::test_kkt_assembly_is_one_loop"
PERTURBATIONS = (
  ("materialize the input transpose", " or k == _STRIDED and op == ExprOp.TRANSPOSE", "", SEEDS, "assert not _privates"),
  ("materialize the contraction output before copying it", "result[node.id] = root, np.argsort(slots), tuple(chain)", "pass", SEEDS, "assert not _privates"),
  ("disable seed-map agreement", "not (seed_pairs and node.op in ELEMENTWISE)", "True", SEED_MAPS, "assert not any"),
  ("store the product before the activation", "if seed_pairs:\n    for view in nodes:", "if False:\n    for view in nodes:", EPILOGUE, "assert (3, 4, 1) not in shapes"),
  (
    "allocate every seed plane",
    "epilogues[result] = replace(plan, packed_consumer=consumer.id)",
    "pass",
    EPILOGUE,
    "assert (3, 4) not in shapes",
  ),
  (
    "materialize the KKT block concatenation",
    "op == ExprOp.CONCAT\n      and dense",
    "False\n      and dense",
    KKT,
    "assert 7 == 1",
  ),
)


def run(src, *tests):
  env = {**os.environ, "PYTHONPATH": f"{src.resolve()}:{Path.cwd()}"}
  return subprocess.run(
    [sys.executable, "-m", "pytest", "-n", "2", "-q", "--tb=short", *tests], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
  )


def record(label, result, exit_code, *assertions):
  row = {"case": label, "exit_code": result.returncode, "output": result.stdout}
  if result.returncode != exit_code or any(expected not in result.stdout for expected in assertions):
    raise RuntimeError(row)
  return row


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("base_src", type=Path)
  parser.add_argument("scratch", type=Path)
  args = parser.parse_args()
  tests = SEEDS, SEED_MAPS, EPILOGUE, KKT
  rows = [record("base", run(args.base_src, *tests), 1, "5 failed", "assert 7 == 1", "assert (3, 4, 1) not in shapes")]
  rows.append(record("layer 4", run(Path("src"), *tests), 0, "5 passed"))
  shutil.copytree("src", args.scratch, ignore=shutil.ignore_patterns("__pycache__"))
  path = args.scratch / "scaly/passes/lowering/fusion.py"
  source = path.read_text()
  for label, old, new, test, expected in PERTURBATIONS:
    assert source.count(old) == 1, label
    path.write_text(source.replace(old, new))
    rows.append(record(label, run(args.scratch, test), 1, expected))
  path.write_text(source)
  print(json.dumps(rows, indent=2))


if __name__ == "__main__":
  main()
