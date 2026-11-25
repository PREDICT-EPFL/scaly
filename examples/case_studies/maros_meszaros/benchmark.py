"""Measure the generated PIQP example against PIQP on Maros–Mészáros problems and tabulate both in Markdown.

    uv run examples/case_studies/maros_meszaros/benchmark.py [--sparse] [--jobs N] [--work DIR] [--core C] [--out FILE] [NAME ...]

Runs the 48 small problems, qpbenchmark's 76 sparse ones with ``--sparse``, or the named ones, in
three passes, each resumable: a problem whose result file exists is not run again.

1. Ordering and build, ``--jobs`` problems at a time, each in its own process.
   - The pattern of ``Psi = P + A^T A + G^T G`` (upper triangle), its AMD ordering as scaly's
     ``linalg.ldl`` computes it, PIQP's ``Eigen::AMDOrdering`` (``piqp_amd.cpp``, built against
     Eigen 3.4), and the nonzeros of ``L`` under each.
   - The example's setup and solve Functions, exported with ``scaly.codegen.write_module`` with the
     recipe the JIT uses on this machine, compiled to objects with the JIT's flags and linked with
     ``driver.c``. Generation is ``write_module``'s time; the build is generation plus compilation.
     A build stops at ``GENERATE_S`` or ``COMPILE_S`` seconds, at ``MEMORY`` bytes, or when the C
     exceeds ``SOURCE_BYTES``; the table records why.
2. Timing, one process at a time pinned to ``--core``: ``driver.c`` calls the generated setup and
   solve, with nothing but the generated C timed. PIQP, the ``scaly-piqp`` plugin's ``libpiqpc``
   with ``kkt_solver = sparse_ldlt_cond`` (every constraint eliminated, as the example), runs
   through ``piqp_sparse.reference`` and is timed by its own timers (``compute_timings``). Each
   takes the median over repeated calls, at least 3 and up to ``BUDGET_S`` seconds.
3. Report: statuses, iterations, the residuals of both solutions computed here from the problem
   (primal, dual, duality gap, in the infinity norm), the objectives' relative difference, times,
   object sizes and build times.

The timings are of this machine; the benchmark protocol in ``internal/notes/benchmark_protocol.md``
governs any published comparison.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

import maros_meszaros as mm  # noqa: E402

GENERATE_S = 1800
COMPILE_S = 1800
MEMORY = 6 << 30
SOURCE_BYTES = 256 << 20
PIQP_S = 1800
BUDGET_S = 2.0


def _limit_memory() -> None:
  # The benchmark runs on Linux (taskset, size); the platform test keeps resource out of other platforms' type checks.
  if sys.platform != "win32":
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))


def _child(args: list[str], timeout: float, log: Path, core: int | None = None) -> int | str:
  """Run this file with ``args`` in a memory-limited child; its exit code, or ``"timeout"``."""
  command = [sys.executable, str(Path(__file__).resolve()), *args]
  if core is not None:
    command = ["taskset", "-c", str(core), *command]
  with log.open("a") as out:
    try:
      return subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=timeout, preexec_fn=_limit_memory).returncode
    except subprocess.TimeoutExpired:
      return "timeout"


def _write(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=1) + "\n")


# Pass 1: ordering and build.


def _psi(name: str) -> tuple[Any, Any, Any]:
  from piqp_sparse.data import Structure
  from piqp_sparse.kkt import KKT

  q = mm.load(name)
  structure, *_ = Structure.from_matrices(q["P"], q["A"], q["G"])
  return q, structure, KKT(structure)


def _nnz_l(n: int, rows: np.ndarray, cols: np.ndarray, perm: np.ndarray) -> int:
  """The nonzeros of ``L`` strictly below the diagonal for the upper pattern ``(rows, cols)`` ordered by ``perm[new] = old``."""
  import scaly as sc
  from scaly.linalg import symbolic

  inverse = np.argsort(perm)
  i, j = inverse[rows], inverse[cols]
  i, j = np.minimum(i, j), np.maximum(i, j)
  order = np.lexsort((j, i))
  pattern, _ = sc.SparsityPattern((n, n), rows=tuple(i[order].tolist()), cols=tuple(j[order].tolist())).canonicalize()
  return symbolic.ldl(pattern, "natural").nnz_l


def order(name: str, out: Path) -> None:
  """nnz(Psi), and nnz(L) under scaly's AMD and under PIQP's."""
  from scaly import linalg

  _, s, k = _psi(name)
  rows, cols = k._upper
  row: dict[str, Any] = {"n": s.n, "p": s.p, "m": s.m, "nnz_P": s.P.nnz, "nnz_A": s.A.nnz, "nnz_G": s.G.nnz, "nnz_psi": len(rows)}
  _write(out, row)
  # PIQP's ordering, from the upper triangle in CSC
  csc = np.lexsort((rows, cols))
  colptr = np.searchsorted(cols[csc], np.arange(s.n + 1)).astype(np.int32)
  payload = np.array([s.n, len(rows)], dtype=np.int32).tobytes() + colptr.tobytes() + rows[csc].astype(np.int32).tobytes()
  t = time.perf_counter()
  piqp_perm = np.frombuffer(subprocess.run([str(_piqp_amd())], input=payload, capture_output=True, check=True).stdout, dtype=np.int32)
  row["piqp_amd_s"] = round(time.perf_counter() - t, 3)
  row["nnz_L_piqp_amd"] = _nnz_l(s.n, rows, cols, piqp_perm.astype(np.int64))
  _write(out, row)
  t = time.perf_counter()
  perm = linalg.permutation(k.pattern, ordering="amd")
  row["scaly_amd_s"] = round(time.perf_counter() - t, 3)
  row["nnz_L_scaly_amd"] = _nnz_l(s.n, rows, cols, np.asarray(perm, dtype=np.int64))
  row["same_ordering"] = bool(np.array_equal(perm, piqp_perm))
  _write(out, row)


def _piqp_amd() -> Path:
  """``piqp_amd.cpp`` built against the Eigen 3.4 headers that CasADi installs, once."""
  binary = Path(os.environ.get("SCALY_PIQP_AMD", HERE / "build" / "piqp_amd"))
  if not binary.exists():
    import casadi

    eigen = Path(casadi.__file__).parent / "include" / "eigen3"
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["c++", "-O2", "-std=c++17", f"-I{eigen}", str(HERE / "piqp_amd.cpp"), "-o", str(binary)], check=True)
  return binary


def _leaf_bytes(leaves: Any) -> list[int]:
  return [int(e.type.storage_size) * e.type.dtype.numpy().itemsize for e in leaves]


def build(name: str, work: Path) -> None:
  """Export, compile and link one problem's generated setup and solve; writes ``build.json``."""
  from scipy import sparse

  from piqp_sparse import Settings
  from piqp_sparse import solver as solver_mod
  from piqp_sparse.results import INFO_FIELDS
  from piqp_sparse.variables import Variables
  from scaly.codegen import write_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.toolchain import find_c_compiler, native_recipe, no_gather_flags, vector_libm_flags
  from scaly.function.model import as_concrete

  q, s, _ = _psi(name)
  record: dict[str, Any] = {"stage": "generate"}
  _write(work / "build.json", record)
  n = s.n
  _, P_values, A_values, G_values = type(s).from_matrices(q["P"], sparse.csc_array(q["A"]), sparse.csc_array(q["G"]))
  vector = solver_mod._vector
  raw = [P_values, vector(q["c"], n, 0.0, "c"), A_values, vector(q["b"], s.p, 0.0, "b"), G_values]
  raw += [vector(q["h_l"], s.m, -np.inf, "h_l"), vector(q["h_u"], s.m, np.inf, "h_u"), vector(q["x_l"], n, -np.inf, "x_l"), vector(q["x_u"], n, np.inf, "x_u")]
  method = solver_mod._method(s, Settings())
  compiler = find_c_compiler()
  assert compiler is not None, "no C compiler"
  cc = compiler.command
  recipe = native_recipe(cc)
  options = {"cpu": recipe.cpu, "lanes": recipe.lanes, "dialect": recipe.dialect, "vector_libm": recipe.vector_libm}
  record["recipe"] = options

  modules = {}
  for part, fun in (("setup", method.setup_function), ("solve", method.solve_function)):
    t = time.perf_counter()
    modules[part] = write_module(fun, work / part, cpu=recipe.cpu, lanes=recipe.lanes, dialect=recipe.dialect, vector_libm=recipe.vector_libm)
    record[f"generate_{part}_s"] = round(time.perf_counter() - t, 3)
    record[f"source_{part}_bytes"] = (work / part / modules[part].source_name).stat().st_size
    _write(work / "build.json", record)
  if sum(record[f"source_{part}_bytes"] for part in modules) > SOURCE_BYTES:
    record["stage"] = "source too large"
    _write(work / "build.json", record)
    return

  setup, solve = (as_concrete(modules[p].fun) for p in ("setup", "solve"))
  assert _leaf_bytes(setup.outputs) == _leaf_bytes(solve.inputs), "the setup's outputs are not the solve's inputs"
  record["meta"] = {"inputs": _leaf_bytes(setup.inputs), "middle": _leaf_bytes(setup.outputs), "outputs": _leaf_bytes(solve.outputs)}
  names = [*Variables._fields, "status", *INFO_FIELDS]
  assert len(names) == len(solve.outputs)
  record["outputs"] = [[name, e.type.dtype.numpy().str, int(e.type.storage_size)] for name, e in zip(names, solve.outputs, strict=True)]
  meta = [len(setup.inputs), *record["meta"]["inputs"], len(setup.outputs), *record["meta"]["middle"], len(solve.outputs), *record["meta"]["outputs"]]
  meta += [modules["setup"].workspace_size, modules["solve"].workspace_size]
  (work / "meta.txt").write_text(" ".join(map(str, meta)) + "\n")
  (work / "inputs.bin").write_bytes(b"".join(np.ascontiguousarray(v, dtype=np.float64).tobytes() for v in raw))

  record["stage"] = "compile"
  _write(work / "build.json", record)
  flags = ["-O2", *recipe.cpu_flags, "-fno-math-errno", *no_gather_flags(cc), *vector_libm_flags(cc, recipe.vector_libm)]
  record["compile_command"] = shlex.join([*cc, *flags, "-c"])
  objects = []
  for part, module in modules.items():
    obj = work / f"{part}.o"
    t = time.perf_counter()
    subprocess.run([*cc, *flags, "-c", str(work / part / module.source_name), f"-I{work / part}", "-o", str(obj)], check=True, timeout=COMPILE_S)
    record[f"compile_{part}_s"] = round(time.perf_counter() - t, 3)
    sizes = subprocess.run(["size", str(obj)], capture_output=True, text=True, check=True).stdout.splitlines()[1].split()
    record[f"object_{part}_bytes"] = int(sizes[0]) + int(sizes[1])  # text (code and constant tables) + data
    objects.append(str(obj))
    _write(work / "build.json", record)
  symbols = {part: c_ident(module.fun.name) for part, module in modules.items()}
  link = [*cc, *flags, str(HERE / "driver.c"), *objects, f"-DSETUP={symbols['setup']}_abi", f"-DSOLVE={symbols['solve']}_abi"]
  link += [*(("-lmvec",) if recipe.vector_libm == "glibc" else ()), "-lm", "-o", str(work / "run")]
  subprocess.run(link, check=True)
  record["stage"] = "built"
  _write(work / "build.json", record)


# Pass 2: timing.


def piqp(name: str, work: Path) -> None:
  """PIQP on one problem, timed by its own timers; writes ``piqp.json`` and ``piqp.npz``."""
  from piqp_sparse import reference

  problem = mm.load(name)
  runs, start = [], time.perf_counter()
  while len(runs) < 3 or (time.perf_counter() - start < BUDGET_S and len(runs) < 1000):
    result = reference.solve(**problem, compute_timings=True)
    runs.append((result.info["setup_time"], result.info["solve_time"], result.info["run_time"]))
  setup_s, solve_s, run_s = (float(np.median([r[i] for r in runs])) for i in range(3))
  info = result.info
  _write(
    work / "piqp.json",
    {"status": int(result.status), "iter": int(info["iter"]), "setup_us": setup_s * 1e6, "solve_us": solve_s * 1e6, "run_us": run_s * 1e6, "reps": len(runs)},
  )
  np.savez(work / "piqp.npz", allow_pickle=False, **result.variables)


def _scaly_outputs(work: Path) -> dict[str, np.ndarray]:
  record = json.loads((work / "build.json").read_text())
  data, offset, out = (work / "outputs.bin").read_bytes(), 0, {}
  for name, dtype, size in record["outputs"]:
    width = np.dtype(dtype).itemsize * int(size)
    out[name] = np.frombuffer(data[offset : offset + width], dtype=dtype).copy()
    offset += width
  return out


# Pass 3: report.


def residuals(q: dict[str, Any], v: dict[str, np.ndarray]) -> dict[str, float]:
  """Primal and dual residuals and duality gap of a solution of ``q``, in the infinity norm, as PIQP defines them."""
  x, y, z_l, z_u, z_bl, z_bu = (np.asarray(v[k], dtype=float) for k in ("x", "y", "z_l", "z_u", "z_bl", "z_bu"))
  P, A, G = q["P"], q["A"], q["G"]
  h_l, h_u, x_l, x_u = (np.asarray(q[k], dtype=float) for k in ("h_l", "h_u", "x_l", "x_u"))
  finite = lambda b: np.isfinite(b)  # noqa: E731
  Gx = G @ x
  violation = [np.abs(A @ x - q["b"]), np.maximum(Gx - h_u, 0), np.maximum(h_l - Gx, 0), np.maximum(x - x_u, 0), np.maximum(x_l - x, 0)]
  primal = max((float(np.max(r, initial=0.0)) for r in violation))
  z_l, z_u = np.where(finite(h_l), z_l, 0.0), np.where(finite(h_u), z_u, 0.0)
  z_bl, z_bu = np.where(finite(x_l), z_bl, 0.0), np.where(finite(x_u), z_bu, 0.0)
  Px = P @ x
  dual = float(np.max(np.abs(Px + q["c"] + A.T @ y + G.T @ (z_u - z_l) + z_bu - z_bl), initial=0.0))
  zero_inf = lambda b: np.where(finite(b), b, 0.0)  # noqa: E731
  gap = x @ Px + q["c"] @ x + q["b"] @ y + zero_inf(h_u) @ z_u - zero_inf(h_l) @ z_l + zero_inf(x_u) @ z_bu - zero_inf(x_l) @ z_bl
  objective = 0.5 * x @ Px + q["c"] @ x
  return {"primal_res": primal, "dual_res": dual, "duality_gap": abs(float(gap)), "objective": float(objective)}


def _fmt(value: Any, kind: str = "g") -> str:
  if value is None:
    return "–"
  if kind == "e":
    return f"{value:.1e}"
  if kind == "ms":
    return f"{value / 1e3:.3g}"
  if kind == "kB":
    return f"{value / 1e3:,.0f}"
  if kind == "s":
    return f"{value:.1f}"
  return str(value)


def _status(code: int) -> str:
  from piqp_sparse.results import Status

  try:
    return Status(code).name.lower()
  except ValueError:
    return str(code)


def report(names: list[str], work: Path, out: Path) -> None:
  library = Path(__import__("scaly_piqp").__file__).parent / "lib" / "libpiqpc.so"
  sizes = subprocess.run(["size", str(library)], capture_output=True, text=True, check=True).stdout.splitlines()[1].split()
  piqp_bytes = int(sizes[0]) + int(sizes[1])
  header = [
    "Problem", "n", "nnz(Ψ)", "nnz(L) scaly AMD", "nnz(L) PIQP AMD",
    "Status scaly", "Status PIQP", "Iter scaly", "Iter PIQP",
    "Primal res scaly", "Primal res PIQP", "Dual res scaly", "Dual res PIQP", "Gap scaly", "Gap PIQP", "Obj. rel. diff",
    "Setup ms scaly", "Setup ms PIQP", "Solve ms scaly", "Solve ms PIQP",
    "Binary kB scaly", "Binary kB PIQP", "Build s scaly (gen + cc)",
  ]  # fmt: skip
  lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["--:"] * (len(header) - 1)) + "|"]
  rows = []
  for name in names:
    d = work / name
    o = _load(d / "order.json")
    b = _load(d / "build.json")
    t = _load(d / "timing.json")
    p = _load(d / "piqp.json")
    q: dict[str, Any] | None = None
    res_s = res_p = {}
    scaly_status = scaly_iter = None
    if t and (d / "outputs.bin").exists():
      v = _scaly_outputs(d)
      q = q or mm.load(name)
      res_s = residuals(q, v)
      scaly_status, scaly_iter = _status(int(v["status"][0])), int(v["iter"][0])
    if p and (d / "piqp.npz").exists():
      q = q or mm.load(name)
      with np.load(d / "piqp.npz") as z:
        res_p = residuals(q, dict(z))
    rel = abs(res_s["objective"] - res_p["objective"]) / max(1.0, abs(res_p["objective"])) if res_s and res_p else None
    if b and b.get("stage") != "built":
      scaly_status = f"not built: {b.get('stage')}"
    elif not b:
      scaly_status = "not built"
    build_s = None
    if b and b.get("stage") == "built":
      gen = b["generate_setup_s"] + b["generate_solve_s"]
      cc = b["compile_setup_s"] + b["compile_solve_s"]
      build_s = f"{gen:.1f} + {cc:.1f}"
    piqp_status = _status(p["status"]) if p else (o or {}).get("piqp_failure", "–")
    row = [
      name, _fmt((o or {}).get("n")), _fmt((o or {}).get("nnz_psi")), _fmt((o or {}).get("nnz_L_scaly_amd")), _fmt((o or {}).get("nnz_L_piqp_amd")),
      scaly_status or "–", piqp_status, _fmt(scaly_iter), _fmt(p.get("iter") if p else None),
      _fmt(res_s.get("primal_res"), "e"), _fmt(res_p.get("primal_res"), "e"), _fmt(res_s.get("dual_res"), "e"), _fmt(res_p.get("dual_res"), "e"),
      _fmt(res_s.get("duality_gap"), "e"), _fmt(res_p.get("duality_gap"), "e"), _fmt(rel, "e"),
      _fmt(t.get("setup_median_us") if t else None, "ms"), _fmt(p.get("setup_us") if p else None, "ms"),
      _fmt(t.get("solve_median_us") if t else None, "ms"), _fmt(p.get("solve_us") if p else None, "ms"),
      _fmt(b["object_setup_bytes"] + b["object_solve_bytes"] if b and b.get("stage") == "built" else None, "kB"), _fmt(piqp_bytes, "kB"),
      build_s or "–",
    ]  # fmt: skip
    lines.append("| " + " | ".join(row) + " |")
    rows.append(dict(zip(header, row, strict=True)) | {"order": o, "build": b, "timing": t, "piqp": p, "res_scaly": res_s, "res_piqp": res_p})
  out.write_text("\n".join(lines) + "\n")
  _write(out.with_suffix(".json"), rows)


def _load(path: Path) -> dict[str, Any] | None:
  return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str]) -> None:
  args = list(argv)

  def option(flag: str, default: str) -> str:
    if flag in args:
      i = args.index(flag)
      value = args[i + 1]
      del args[i : i + 2]
      return value
    return default

  sparse_subset = "--sparse" in args
  if sparse_subset:
    args.remove("--sparse")
  jobs, core = int(option("--jobs", "2")), int(option("--core", "3"))
  work = Path(option("--work", str(HERE / "build" / ("sparse" if sparse_subset else "small"))))
  out = Path(option("--out", str(work / "results.md")))
  names = args or mm.problems(sparse_subset)
  work.mkdir(parents=True, exist_ok=True)
  _piqp_amd()

  def prepare(name: str) -> None:
    d = work / name
    d.mkdir(exist_ok=True)
    if not (d / "order.json").exists() or "same_ordering" not in json.loads((d / "order.json").read_text()):
      code = _child(["_order", name, str(d / "order.json")], GENERATE_S, d / "log.txt")
      if code != 0:
        o = _load(d / "order.json") or {}
        _write(d / "order.json", o | {"failure": f"ordering: {'out of memory or error' if code != 'timeout' else 'timeout'}", "same_ordering": None})
    b = _load(d / "build.json")
    if b is None or b.get("stage") not in ("built", "source too large") and "failed" not in b.get("stage", ""):
      code = _child(["_build", name, str(d)], GENERATE_S + 2 * COMPILE_S, d / "log.txt")
      b = _load(d / "build.json") or {"stage": "generate"}
      if b.get("stage") not in ("built", "source too large"):
        reason = "timeout" if code == "timeout" else "out of memory or error"
        _write(d / "build.json", b | {"stage": f"failed in {b.get('stage')} ({reason})"})
    print(f"{name}: {(_load(d / 'build.json') or {}).get('stage')}", flush=True)

  with ThreadPoolExecutor(jobs) as pool:
    list(pool.map(prepare, names))

  for name in names:
    d = work / name
    b = _load(d / "build.json")
    if b and b.get("stage") == "built" and not (d / "timing.json").exists():
      try:
        run = subprocess.run(["taskset", "-c", str(core), str(d / "run"), str(d), str(BUDGET_S)], capture_output=True, text=True, timeout=PIQP_S, check=True)
        _write(d / "timing.json", json.loads(run.stdout))
      except (subprocess.SubprocessError, json.JSONDecodeError) as error:
        _write(d / "timing.json", {"failure": str(error)[:200]})
    if not (d / "piqp.json").exists():
      code = _child(["_piqp", name, str(d)], PIQP_S, d / "log.txt", core)
      if code != 0:
        o = _load(d / "order.json") or {}
        _write(d / "order.json", o | {"piqp_failure": "timeout" if code == "timeout" else "out of memory or error"})
    print(f"{name}: timed", flush=True)

  report(names, work, out)
  print(f"wrote {out}")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "_order":
    order(sys.argv[2], Path(sys.argv[3]))
  elif len(sys.argv) > 1 and sys.argv[1] == "_build":
    build(sys.argv[2], Path(sys.argv[3]))
  elif len(sys.argv) > 1 and sys.argv[1] == "_piqp":
    piqp(sys.argv[2], Path(sys.argv[3]))
  else:
    main(sys.argv[1:])
