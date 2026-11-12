"""Audit the saved study and compare compiler metrics with paired retiming noise estimates."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import numpy as np
from scipy.stats import t


def read(path):
  with path.open() as fp:
    return list(csv.DictReader(fp))


def write(path, rows):
  with path.open("w") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)


def mean_cv(values):
  mean = st.mean(values)
  return mean, st.stdev(values) / mean if len(values) > 1 and mean else None


def independent_interval(before, after):
  a, b = np.log(before), np.log(after)
  ratio = math.exp(float(b.mean() - a.mean()))
  if min(len(a), len(b)) < 2:
    return ratio, None, None
  va, vb = float(a.var(ddof=1)) / len(a), float(b.var(ddof=1)) / len(b)
  if va + vb == 0:
    return ratio, ratio, ratio
  df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
  error = float(t.ppf(0.975, df)) * math.sqrt(va + vb)
  return ratio, ratio * math.exp(-error), ratio * math.exp(error)


def paired_interval(before, after):
  logs = [math.log(b / a) for a, b in zip(before, after, strict=True)]
  ratio = math.exp(st.mean(logs))
  if len(logs) < 2:
    return ratio, None, None
  error = float(t.ppf(0.975, len(logs) - 1)) * st.stdev(logs) / math.sqrt(len(logs))
  return ratio, ratio * math.exp(-error), ratio * math.exp(error)


def main(root):
  manifest = json.loads((root / "study.json").read_text())
  assert (root / "COMPLETE").exists() and (root / "exit-code").read_text().strip() == "0"
  assert all(c["returncode"] == 0 for c in manifest["commands"])
  assert manifest["native_libraries"]["before"] == manifest["native_libraries"]["after"]
  frozen = root / "frozen"
  for name, digest in json.loads((frozen / "frozen.json").read_text()).items():
    assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest, name
  audit = {"sweep": {}, "episodes": {}, "native_libraries_identical": True, "frozen_files_unchanged": True}
  cells = defaultdict(lambda: defaultdict(list))
  raw = []
  artifacts = defaultdict(lambda: defaultdict(list))
  for side in ("before", "after"):
    outcomes = Counter()
    for category in ("sweep", "controls"):
      for path in sorted((root / side / category).glob("*/*.csv")):
        if path.name.endswith(".summary.csv"):
          continue
        provenance = json.loads(Path(str(path) + ".provenance.json").read_text())
        assert provenance["git_commit"] == manifest["revisions"][side] and not provenance["git_dirty"]
        assert provenance["vector_libm"] == "glibc" and not provenance["cpu_settings"]["boost_enabled"]
        assert all(p["scaling_governor"] == "performance" for p in provenance["cpu_settings"]["policies"].values())
        for row in read(path):
          key = row["workload"], int(row["size"]), row["backend"]
          cells[key][side].append(row)
          outcomes[row["compile_status"] + "/" + row["runtime_status"]] += 1
          raw.append({"side": side, **row})
          if row["runtime_status"] == "ok":
            axis = "M" if key[0].startswith("chain") else "W" if key[0].startswith("npmpc_decoder") else "C" if key[0] == "unbumpercars" else "N"
            directory = path.parent / f"repeat_{row['repetition']}" / key[0] / f"{key[2]}_{axis}{key[1]}"
            assert "correctness ok" in (directory / "benchmark.json").read_text(), directory
            artifacts[key][side].append(
              {
                "kernel_sha256": hashlib.sha256((directory / "kernel.o").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(next(directory.glob("*.c")).read_bytes()).hexdigest(),
                "samples": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("sample_*.bin"))},
                "reference": str(directory / "expected_dense.bin"),
              }
            )
    audit["sweep"][side] = dict(outcomes)
  write(root / "all-sweep.csv", raw)
  for key, sides in artifacts.items():
    assert all(item["samples"] == sides["before"][0]["samples"] for items in sides.values() for item in items), key
    reference = np.fromfile(sides["before"][0]["reference"], dtype=np.float64)
    for items in sides.values():
      for item in items:
        other = np.fromfile(item["reference"], dtype=np.float64)
        np.testing.assert_allclose(other, reference, rtol=1e-12, atol=1e-12)
        item["reference_max_abs_diff"] = float(np.max(np.abs(other - reference)))
  audit["all_samples_identical"] = True
  audit["control_objects_identical"] = all(
    sides["before"][0]["kernel_sha256"] == sides["after"][0]["kernel_sha256"] for key, sides in artifacts.items() if key[2] != "scaly"
  )
  (root / "artifact-audit.json").write_text(json.dumps({"/".join(map(str, key)): sides for key, sides in artifacts.items()}, indent=2) + "\n")
  retiming = defaultdict(dict)
  assert (root / "retiming-noise-check/COMPLETE").exists()
  for path in (root / "retiming.csv", root / "retiming-noise-check.csv"):
    for row in read(path):
      key = row["workload"], int(row["size"]), row["backend"]
      index = row["side"], int(row["build"]), int(row["repetition"])
      assert index not in retiming[key]
      retiming[key][index] = float(row["runtime_ns"])
  for path in (root / "retiming-noise-check").glob("*.json"):
    assert "correctness ok" in path.read_text(), path
  results = []
  for key, sides in sorted(cells.items()):
    for metric in (
      "runtime_ns",
      "kernel_compile_ms",
      "source_bytes",
      "dispatch_arithmetic",
      "build_ms",
      "render_ms",
      "executable_bytes",
      "workspace",
    ):
      values = {side: [float(row[metric]) for row in sides[side] if row[metric]] for side in ("before", "after")}
      if not all(values.values()):
        continue
      original = values
      method = (
        "exact" if metric in ("source_bytes", "dispatch_arithmetic", "executable_bytes", "workspace") else "Welch log interval over fresh builds"
      )
      if metric == "runtime_ns" and key in retiming:
        times = retiming[key]
        builds = sorted({build for side, build, repetition in times})
        repetitions = sorted({repetition for side, build, repetition in times})
        values = {
          side: [st.geometric_mean([times[side, build, repetition] for repetition in repetitions]) for build in builds]
          for side in ("before", "after")
        }
        if len(builds) > 1:
          ratio, low, high = paired_interval(values["before"], values["after"])
          method = f"paired log interval over 3 builds, {len(repetitions)} alternating timings each"
        else:
          a, b = [[times[side, 1, repetition] for repetition in repetitions] for side in ("before", "after")]
          ratio, low, high = paired_interval(a, b)
          method = f"paired log interval over {len(repetitions)} alternating timings of one control build"
      elif method == "exact":
        assert all(len(set(v)) == 1 for v in values.values())
        ratio = values["after"][0] / values["before"][0] if values["before"][0] else None
        low = high = ratio
      else:
        ratio, low, high = independent_interval(values["before"], values["after"])
      before, before_cv = mean_cv(values["before"])
      after, after_cv = mean_cv(values["after"])
      if metric == "runtime_ns":
        before, after = st.geometric_mean(values["before"]), st.geometric_mean(values["after"])
      status = "larger" if low is not None and low > 1 else "smaller" if high is not None and high < 1 else "unresolved"
      if method == "exact" and before == after:
        status = "unchanged"
      results.append(
        dict(
          workload=key[0],
          size=key[1],
          backend=key[2],
          metric=metric,
          before=before,
          after=after,
          ratio=ratio,
          low95=low,
          high95=high,
          before_cv=before_cv,
          after_cv=after_cv,
          original_before_cv=mean_cv(original["before"])[1],
          original_after_cv=mean_cv(original["after"])[1],
          status=status,
          method=method,
        )
      )
  control_band = math.exp(max(abs(math.log(row["ratio"])) for row in results if row["backend"] != "scaly" and row["metric"] == "runtime_ns"))
  audit["observed_control_ratio_band"] = [1 / control_band, control_band]
  for row in results:
    row["interval_status"] = row["status"]
    if row["backend"] == "scaly" and row["metric"] == "runtime_ns":
      row["status"] = "larger" if row["low95"] > control_band else "smaller" if row["high95"] < 1 / control_band else "unresolved"
  write(root / "comparison.csv", results)
  write(
    root / "changes.csv",
    [
      row
      for row in results
      if row["backend"] == "scaly"
      and row["metric"] in {"runtime_ns", "kernel_compile_ms", "source_bytes", "dispatch_arithmetic"}
      and row["status"] in {"larger", "smaller"}
    ],
  )
  episode_rows = []
  episodes = defaultdict(dict)
  for side in ("before", "after"):
    paths = sorted((root / side / "closed-loop").glob("**/telemetry.csv"))
    audit["episodes"][side] = {"episodes": len(paths), "steps": 0, "failed_steps": 0}
    for path in paths:
      data = read(path)
      problem, pair = path.parent.parent.name, path.parent.name
      solver, provider = pair.split("+")
      episodes[problem, solver, provider][side] = (path.parent, data)
      audit["episodes"][side]["steps"] += len(data)
      audit["episodes"][side]["failed_steps"] += sum(row["success"] != "True" for row in data)
  for key, sides in sorted(episodes.items()):
    a_path, a = sides["before"]
    b_path, b = sides["after"]
    assert [r["step"] for r in a] == [r["step"] for r in b]
    counts = ("iterations", "n_eval_f", "n_eval_g", "n_eval_grad_f", "n_eval_jac_g", "n_eval_h")
    mismatches = {
      count: sum(x[count] != y[count] for x, y in zip(a, b, strict=True)) if all(row.get(count) for row in (*a, *b)) else None for count in counts
    }
    trajectory = {}
    with np.load(a_path / "rollout.npz") as av, np.load(b_path / "rollout.npz") as bv:
      for field in ("state", "applied" if key[0] == "unbumpercars" else "control"):
        assert av[field].shape == bv[field].shape
        trajectory[field + "_max_abs_diff"] = float(np.max(np.abs(av[field] - bv[field])))
    for metric in ("solver_time_ms", "fe_time_ms"):
      before = st.mean(float(r[metric]) for r in a)
      after = st.mean(float(r[metric]) for r in b)
      ratios = [float(y[metric]) / float(x[metric]) for x, y in zip(a, b, strict=True) if float(x[metric]) > 0]
      episode_rows.append(
        dict(
          problem=key[0],
          solver=key[1],
          provider=key[2],
          metric=metric,
          before=before,
          after=after,
          ratio=after / before,
          paired_median=st.median(ratios),
          paired_p10=float(np.percentile(ratios, 10)),
          paired_p90=float(np.percentile(ratios, 90)),
          steps=len(a),
          **mismatches,
          **trajectory,
        )
      )
  write(root / "closed-loop-comparison.csv", episode_rows)
  audit["retimed_calls"] = sum(len(v) for v in retiming.values())
  (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
  print(json.dumps(audit, indent=2))
  print("Scaly runtime / compile / operations:")
  for row in results:
    if row["backend"] == "scaly" and row["metric"] in ("runtime_ns", "kernel_compile_ms", "dispatch_arithmetic"):
      print(
        f"{row['workload']:20} {row['size']:3} {row['metric']:20} {row['before']:12.2f} {row['after']:12.2f} {row['ratio']:7.3f} [{row['low95']:.3f}, {row['high95']:.3f}] {row['status']}"
      )
  print("Closed-loop Scaly:")
  for row in episode_rows:
    if row["provider"] == "scaly":
      print(row)
  print("Control runtime ratios:")
  for row in results:
    if row["backend"] != "scaly" and row["metric"] == "runtime_ns":
      print(row["workload"], row["size"], row["ratio"], row["low95"], row["high95"])
  print("Large retiming variation:")
  for key, times in retiming.items():
    for side in ("before", "after"):
      cv = mean_cv([value for (s, _, _), value in times.items() if s == side])[1]
      if cv and cv > 0.05:
        print(
          key,
          side,
          cv,
          "source variants",
          len({r["source_sha256"] for r in artifacts[key][side]}),
          "object variants",
          len({r["kernel_sha256"] for r in artifacts[key][side]}),
        )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("root", type=Path)
  main(parser.parse_args().root)
