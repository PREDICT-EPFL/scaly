"""Frequency-policy gates and aggregation of independent timing processes."""

import csv
from pathlib import Path

import pytest

from bench.harness import provenance
from bench.harness.sweep import summarize_runs


def test_headline_settings_reject_powersave_and_wrong_boost(monkeypatch):
  settings = {"policies": {"policy0": {"scaling_governor": "powersave"}}, "boost_enabled": True}
  monkeypatch.setattr(provenance, "cpu_settings", lambda: settings)
  with pytest.raises(RuntimeError, match="performance governor"):
    provenance.require_headline_settings(True)
  settings["policies"]["policy0"]["scaling_governor"] = "performance"
  with pytest.raises(RuntimeError, match="boost disabled"):
    provenance.require_headline_settings(False)
  assert provenance.require_headline_settings(True) == settings
  settings["policies"] = {}
  with pytest.raises(RuntimeError, match="Linux cpufreq controls"):
    provenance.require_headline_settings(True)


def test_cpu_settings_reads_linux_policies_and_boost(tmp_path):
  policy = tmp_path / "cpufreq/policy0"
  policy.mkdir(parents=True)
  (policy / "scaling_governor").write_text("performance\n")
  (tmp_path / "cpufreq/boost").write_text("0\n")
  settings = provenance.cpu_settings(tmp_path)
  assert settings["policies"] == {"policy0": {"scaling_governor": "performance"}}
  assert settings["boost_enabled"] is False


def test_dispersion_uses_successful_processes_and_keeps_failure_counts(tmp_path: Path):
  rows = [
    dict(workload="fixture", size=2, backend="scaly", runtime_status=status, runtime_ns=runtime)
    for status, runtime in [("ok", "10"), ("ok", "20"), ("correctness_fail", "1000")]
  ]
  out = tmp_path / "summary.csv"
  summarize_runs(rows, out)
  with out.open() as fp:
    summary = next(csv.DictReader(fp))
  assert summary["attempts"] == "3" and summary["successful"] == "2"
  assert float(summary["runtime_ns_mean"]) == 15
  assert float(summary["runtime_ns_stdev"]) == pytest.approx(50**0.5)
  assert float(summary["runtime_ns_cv"]) == pytest.approx(50**0.5 / 15)
  summarize_runs(rows[:1], out)
  with out.open() as fp:
    assert next(csv.DictReader(fp))["runtime_ns_stdev"] == ""


def test_closed_loop_cli_rotates_oracles_and_rejects_reused_caches(tmp_path, monkeypatch):
  import sys

  from bench import run
  from bench.harness import timing

  calls = []
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "run.py",
      "closed-loop",
      "--problem",
      "race_cars",
      "--solver",
      "ipopt,sqp",
      "--oracle",
      "scaly,casadi",
      "--repetitions",
      "3",
      "--out-dir",
      str(tmp_path),
    ],
  )
  monkeypatch.setattr(run.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
  monkeypatch.setattr(run, "collect", lambda *args: {})
  monkeypatch.setattr(run, "write", lambda *args: None)
  monkeypatch.setattr(timing, "summarize_modes", lambda *args: [])
  run.main()
  pairs = [(command[command.index("--solver") + 1], command[command.index("--oracle") + 1]) for command, _ in calls]
  assert len(pairs) == 12 and len(set(pairs[:4])) == 4
  assert pairs[4:8] == pairs[1:4] + pairs[:1] and pairs[8:] == pairs[2:4] + pairs[:2]
  caches = [kwargs["env"]["SCALY_CACHE_DIR"] for _, kwargs in calls]
  assert len(set(caches)) == 12 and all(Path(cache).is_dir() for cache in caches)
  with pytest.raises(SystemExit) as error:
    run.main()
  assert error.value.code == 2
  assert len(calls) == 12
  sys.argv.append("--overwrite")
  run.main()
  assert len(calls) == 24


def test_cpu_provenance_without_linux_interfaces(tmp_path, monkeypatch):
  monkeypatch.delattr(provenance.os, "sched_getaffinity", raising=False)
  settings = provenance.cpu_settings(tmp_path)
  assert settings == {"policies": {}, "boost_enabled": None, "affinity": None}
  monkeypatch.setattr(provenance, "cpu_settings", lambda: settings)
  with pytest.raises(RuntimeError, match="on macOS or unsupported hosts, run without --headline"):
    provenance.require_headline_settings(False)


def test_intel_boost_uses_the_inverse_no_turbo_flag(tmp_path):
  turbo = tmp_path / "intel_pstate/no_turbo"
  turbo.parent.mkdir()
  for disabled in (0, 1):
    turbo.write_text(str(disabled))
    assert provenance.cpu_settings(tmp_path)["boost_enabled"] is (not bool(disabled))
