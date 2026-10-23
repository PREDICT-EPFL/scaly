"""Frequency-policy gates and aggregation of independent timing processes."""

import csv
import os
from pathlib import Path

import pytest

from benchmarks.harness import provenance
from benchmarks.harness.sweep import summarize_runs


def test_vector_math_policy_matches_supported_compiler_flags(monkeypatch: pytest.MonkeyPatch) -> None:
  from benchmarks import harness
  from scaly.codegen.toolchain import BuildRecipe, Compiler

  monkeypatch.setattr(harness, "compiler_version", lambda *command: command[0])
  monkeypatch.setenv("SCALY_VECTOR_LIBM", "glibc")
  assert harness.math_flags("clang version 20") == (("-fveclib=libmvec",), ("-lmvec",))
  assert harness.math_flags("gcc version 13") == ((), ("-lmvec",))
  monkeypatch.setenv("SCALY_VECTOR_LIBM", "none")
  assert harness.math_flags("clang version 20") == ((), ())
  assert harness.math_flags("gcc version 13") == ((), ())
  monkeypatch.delenv("SCALY_VECTOR_LIBM")
  monkeypatch.setattr(harness, "find_c_compiler", lambda: Compiler(("gcc",), "CC"))
  monkeypatch.setattr(harness, "native_recipe", lambda compiler: BuildRecipe(vector_libm="none"))
  harness.configure_math_policy()
  assert harness.vector_libm() == "none"
  monkeypatch.setenv("SCALY_VECTOR_LIBM", "glibc")
  with pytest.raises(ValueError, match="glibc x86-64 host"):
    harness.configure_math_policy()


def test_vector_math_policy_rejects_unequal_clang_jit_flags(monkeypatch: pytest.MonkeyPatch) -> None:
  from benchmarks import harness
  from scaly.codegen.toolchain import BuildRecipe, Compiler

  monkeypatch.setattr(harness, "find_c_compiler", lambda: Compiler(("clang",), "SCALY_CC"))
  monkeypatch.setattr(harness, "native_recipe", lambda compiler: BuildRecipe(vector_libm="glibc"))
  monkeypatch.setattr(harness, "compiler_version", lambda *command: "clang version 20")
  # configure_math_policy writes os.environ directly, which monkeypatch would not undo
  monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "SCALY_VECTOR_LIBM"})
  with pytest.raises(ValueError, match="Scaly JIT does not pass"):
    harness.configure_math_policy()
  harness.configure_math_policy(measured_jit=False)
  monkeypatch.setenv("SCALY_VECTOR_LIBM", "none")
  harness.configure_math_policy()
  assert harness.vector_libm() == "none"


@pytest.mark.parametrize("command", [("clang",), ("zig", "cc")])
def test_smoke_accepts_clang_while_measured_closed_loops_keep_the_compiler_guard(monkeypatch: pytest.MonkeyPatch, command: tuple[str, ...]) -> None:
  import sys

  from benchmarks import harness, run
  from scaly.codegen.toolchain import BuildRecipe, Compiler

  monkeypatch.setattr(harness, "find_c_compiler", lambda: Compiler(command, "test"))
  monkeypatch.setattr(harness, "native_recipe", lambda compiler: BuildRecipe(vector_libm="glibc"))
  monkeypatch.setattr(harness, "compiler_version", lambda *args: "clang version 20")
  monkeypatch.setattr(os, "environ", {**os.environ, "SCALY_VECTOR_LIBM": "glibc"})
  monkeypatch.setattr(run, "smoke", lambda args: True)
  monkeypatch.setattr(sys, "argv", ["run.py", "smoke"])
  with pytest.raises(SystemExit) as result:
    run.main()
  assert result.value.code == 0
  monkeypatch.setattr(sys, "argv", ["run.py", "closed-loop", "--oracle", "scaly,casadi"])
  with pytest.raises(ValueError, match="Scaly JIT does not pass"):
    run.main()


def test_sweep_render_uses_the_selected_math_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  import scaly as sc
  from benchmarks.harness.sweep import _render_scaly

  @sc.function(sc.arg("x", 4), outputs=sc.arg("y", 4), name="math_policy")
  def kernel(x: sc.Expr) -> sc.Expr:
    return x.sin()

  for policy in ("glibc", "none"):
    monkeypatch.setenv("SCALY_VECTOR_LIBM", policy)
    module, _ = _render_scaly(kernel, "math_policy", tmp_path)
    assert module.recipe.vector_libm == policy
    assert ("-lmvec" in module.link_flags) == (policy == "glibc")


def test_headline_settings_reject_powersave_and_wrong_boost(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_cpu_settings_reads_linux_policies_and_boost(tmp_path: Path) -> None:
  policy = tmp_path / "cpufreq/policy0"
  policy.mkdir(parents=True)
  (policy / "scaling_governor").write_text("performance\n")
  (tmp_path / "cpufreq/boost").write_text("0\n")
  settings = provenance.cpu_settings(tmp_path)
  assert settings["policies"] == {"policy0": {"scaling_governor": "performance"}}
  assert settings["boost_enabled"] is False


def test_dispersion_uses_successful_processes_and_keeps_failure_counts(tmp_path: Path) -> None:
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


def test_closed_loop_cli_rotates_oracles_and_rejects_reused_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  import sys

  from benchmarks import run
  from benchmarks.harness import timing

  calls = []
  monkeypatch.setattr(run, "configure_math_policy", lambda **kwargs: None)
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


def test_closed_loop_cli_runs_one_explicit_repetition_in_a_fresh_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  import sys

  from benchmarks import run
  from benchmarks.harness import timing

  calls = []
  monkeypatch.setattr(run, "configure_math_policy", lambda **kwargs: None)
  argv = ["run.py", "closed-loop", "--problem", "chain", "--solver", "ipopt", "--oracle", "scaly", "--repetitions", "1", "--out-dir", str(tmp_path)]
  monkeypatch.setattr(sys, "argv", argv)
  monkeypatch.setattr(run.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
  monkeypatch.setattr(run, "collect", lambda *args: {})
  monkeypatch.setattr(run, "write", lambda *args: None)
  monkeypatch.setattr(timing, "summarize_modes", lambda *args: [])
  run.main()
  ((command, kwargs),) = calls
  assert command[command.index("--out-dir") + 1] == str(tmp_path / "repeat_1")
  assert kwargs["env"]["SCALY_CACHE_DIR"] == str(tmp_path / "repeat_1" / "cache" / "chain" / "ipopt+scaly")
  # the child episode adds interpreted/ itself, so the parent must not
  interpreted = tmp_path / "interpreted-run"
  argv[argv.index("chain")] = "race_cars"
  argv[argv.index("scaly")] = "casadi"
  argv[-1] = str(interpreted)
  argv.append("--casadi-interpreted")
  run.main()
  command = calls[-1][0]
  assert command[command.index("--out-dir") + 1] == str(interpreted / "repeat_1")


def test_cpu_provenance_without_linux_interfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delattr(provenance.os, "sched_getaffinity", raising=False)
  settings = provenance.cpu_settings(tmp_path)
  assert settings == {"policies": {}, "boost_enabled": None, "affinity": None}
  monkeypatch.setattr(provenance, "cpu_settings", lambda: settings)
  with pytest.raises(RuntimeError, match="on macOS or unsupported hosts, run without --headline"):
    provenance.require_headline_settings(False)


def test_intel_boost_uses_the_inverse_no_turbo_flag(tmp_path: Path) -> None:
  turbo = tmp_path / "intel_pstate/no_turbo"
  turbo.parent.mkdir()
  for disabled in (0, 1):
    turbo.write_text(str(disabled))
    assert provenance.cpu_settings(tmp_path)["boost_enabled"] is (not bool(disabled))


def test_smoke_skips_a_comparison_backend_over_the_compile_budget(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
  from benchmarks import run

  timeout = {"compile_status": "timeout", "runtime_status": "skipped", "runtime_ns": "", "note": "compile > 180s (kernel_compile_ms)"}
  ok = {"compile_status": "ok", "runtime_status": "ok", "runtime_ns": 1.0, "note": ""}
  info = {"nnz": 1, "w_size": 0, "n_rows": 2, "n_cols": 2}
  monkeypatch.setattr(run, "SMOKE_CELLS", {"npmpc_jac": (("scaly", 6), ("casadi_sx", 6))})
  monkeypatch.setattr(run, "run_cell", lambda workload, size, backend, *a, **k: (dict(ok if backend == "scaly" else timeout), info))
  run._benchmark_smoke(full=True)
  assert "backend=casadi_sx: skipped (compile > 180s" in capsys.readouterr().out
  monkeypatch.setattr(run, "run_cell", lambda *a, **k: (dict(timeout), info))
  with pytest.raises(RuntimeError, match="npmpc_jac scaly size=6 smoke failed"):
    run._benchmark_smoke(full=True)


def test_problem_checks_compile_at_O0(monkeypatch: pytest.MonkeyPatch) -> None:
  from types import SimpleNamespace

  from benchmarks import run
  from scaly.codegen.jit import opt_flag

  flags = []
  monkeypatch.setenv("SCALY_CC_OPT", "-O2")
  monkeypatch.setattr(run, "QUICK_CHECKS", {"toy": "gate"})
  monkeypatch.setattr(run.importlib, "import_module", lambda name: SimpleNamespace(CHECKS={"gate": (lambda: flags.append(opt_flag()), False, False)}))
  run._problem_checks("toy", full=True)
  assert flags == ["-O0"]


def test_windows_smoke_reports_unsupported_groups_and_runs_problems(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
  from concurrent.futures import Future
  from types import SimpleNamespace
  from benchmarks import run

  submitted = []

  class Pool:
    def __init__(self, **kwargs):
      pass

    def __enter__(self):
      return self

    def __exit__(self, *args):
      pass

    def submit(self, task):
      submitted.append(task)
      future = Future()
      future.set_result(None)
      return future

  monkeypatch.setattr(run.sys, "platform", "win32")
  monkeypatch.setattr(run, "ProcessPoolExecutor", Pool)
  monkeypatch.setattr(run, "_solver_call_smoke", lambda **kw: pytest.fail("unsupported C++ solver driver ran"))
  assert run.smoke(SimpleNamespace(select=None, skip=None, full=False))
  assert [task.args[0] for task in submitted] == list(run.QUICK_CHECKS)
  output = capsys.readouterr().out
  assert "smoke benchmarks: skipped" in output and "size" in output and "nm" in output and "C++" in output
  assert "smoke solver_call: skipped" in output and "IPOPT" in output


@pytest.mark.parametrize("full", [False, True])
def test_windows_problem_smoke_keeps_numerical_checks_and_reports_ipopt(
  full: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  from types import SimpleNamespace
  from benchmarks import run

  called = []
  monkeypatch.setattr(run.sys, "platform", "win32")
  monkeypatch.setattr(run, "QUICK_CHECKS", {"toy": "solver"})
  monkeypatch.setattr(run, "WINDOWS_QUICK_CHECKS", {"toy": "numerical"})
  monkeypatch.setattr(run, "solver_loadable", lambda name: False)
  monkeypatch.delenv("SCALY_REQUIRE_SOLVERS", raising=False)
  monkeypatch.setattr(
    run.importlib,
    "import_module",
    lambda name: SimpleNamespace(
      CHECKS={
        "solver": (lambda: pytest.fail("IPOPT ran"), True, False),
        "numerical": (lambda: called.append("numerical"), False, False),
      }
    ),
  )
  run._problem_checks("toy", full)
  assert called == ["numerical"]
  output = capsys.readouterr().out
  assert "solver: skipped: IPOPT" in output
  assert "numerical: ok" in output
