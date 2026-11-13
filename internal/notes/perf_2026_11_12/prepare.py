"""Freeze one benchmark package with boundary adapters for the two compiler revisions."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def replace(root, name, old, new):
  path = root / name
  text = path.read_text()
  assert text.count(old) == 1, (name, old)
  path.write_text(text.replace(old, new))


def main(root, out):
  out.mkdir(parents=True, exist_ok=False)
  archive = subprocess.check_output(["git", "archive", "5335c46d", "benchmarks", "docs/assets/benchmarks/sweep.summary.csv"], cwd=root)
  with tarfile.open(fileobj=io.BytesIO(archive)) as files:
    files.extractall(out, filter="data")
  replace(out, "benchmarks/run.py", "from scaly.codegen.external import compile_library, prepare_library", "")
  replace(
    out,
    "benchmarks/run.py",
    "    builds = tuple(prepare_library(library, compiler) for library in module._libraries)\n"
    "    library_objects.extend(obj for build in builds for obj in compile_library(build, compiler))\n"
    "    for flag in (*module.recipe.link_flags, *(flag for build in builds for flag in build.link_flags)):",
    '    if hasattr(module, "_libraries"):\n'
    "      from scaly.codegen.external import compile_library, prepare_library\n"
    "      builds = tuple(prepare_library(library, compiler) for library in module._libraries)\n"
    "      library_objects.extend(obj for build in builds for obj in compile_library(build, compiler))\n"
    "      link_flags = (*module.recipe.link_flags, *(flag for build in builds for flag in build.link_flags))\n"
    "    else:\n"
    "      from scaly.solvers.graph import solver_compile_flags\n"
    "      link_flags = solver_compile_flags(module.fun)\n"
    "    for flag in link_flags:",
  )
  replace(
    out,
    "benchmarks/harness/sweep.py",
    "from scaly.ir.expr import is_map, topo",
    "from scaly.ir.expr import topo\nfrom scaly.ir import expr as _expr\n\n"
    "def is_map(node):\n"
    '  return _expr.is_map(node) if hasattr(_expr, "is_map") else node.op == _expr.ExprOp.VMAP\n',
  )
  replace(
    out,
    "benchmarks/harness/sweep.py",
    "  assert function.outputs[0].type.pattern is sparsity.canonicalize()[0]",
    '  if hasattr(function, "output_sparsities"):\n'
    "    assert function.output_sparsities[0] == sparsity\n"
    "  else:\n"
    "    assert function.outputs[0].type.pattern is sparsity.canonicalize()[0]",
  )
  replace(
    out,
    "benchmarks/harness/provenance.py",
    "  command = (compiler,) if isinstance(compiler, str) else compiler",
    '  root = Path(os.environ["SCALY_STUDY_SOURCE"])\n  command = (compiler,) if isinstance(compiler, str) else compiler',
  )
  replace(
    out,
    "benchmarks/harness/sweep.py",
    "def _fresh_cell(workload, size, backend, out_dir, options):\n",
    "def _fresh_cell(workload, size, backend, out_dir, options):\n"
    '  cache = out_dir.parent / (out_dir.name + "_cache")\n'
    "  assert not cache.exists()\n"
    '  os.environ["SCALY_CACHE_DIR"] = str(cache)\n'
    '  os.environ["SCALY_CASADI_IPOPT_CACHE"] = str(cache / "casadi-ipopt")\n',
  )
  replace(
    out,
    "benchmarks/harness/casadi_ipopt.py",
    "    self._dll = load_library(library, isolated=True, library_dirs=tuple(str(path) for path in solver_paths().lib_dirs))",
    "    import inspect\n"
    '    extra = {"library_dirs": tuple(str(path) for path in solver_paths().lib_dirs)} if "library_dirs" in inspect.signature(load_library).parameters else {}\n'
    "    self._dll = load_library(library, isolated=True, **extra)",
  )
  replace(
    out,
    "benchmarks/problems/chain/checks.py",
    "    compact = sparse(zv, pv)\n"
    "    np.testing.assert_allclose(compact.toarray(), actual, rtol=1e-10, atol=1e-10)\n"
    "    assert compact.nnz < actual.size, (compact.nnz, actual.size)",
    "    compact = sparse(zv, pv)\n"
    '    if hasattr(compact, "toarray"):\n'
    "      dense = compact.toarray()\n"
    "    else:\n"
    "      pattern = sparse.sparsity()\n"
    "      dense = np.zeros(actual.shape)\n"
    "      dense[pattern.rows, pattern.cols] = np.asarray(compact).ravel()\n"
    "    np.testing.assert_allclose(dense, actual, rtol=1e-10, atol=1e-10)\n"
    "    assert sparse.sparsity().nnz < actual.size",
  )
  replace(
    out,
    "benchmarks/problems/chain/checks.py",
    '    return sc.values(sc.sparse_hessian(lam @ eq, z, triangle="lower"))',
    '    result = sc.sparse_hessian(lam @ eq, z, triangle="lower")\n    return sc.values(result) if isinstance(result, sc.Expr) else result.values',
  )
  replace(
    out,
    "benchmarks/problems/unbumpercars/checks.py",
    "        np.testing.assert_allclose(sparse.coo_array(values).toarray(), reference, rtol=1e-8, atol=1e-9)",
    "        if not sparse.issparse(values):\n"
    "          pattern = derivative.sparsity()\n"
    "          values = sparse.coo_array((np.asarray(values).ravel(), (pattern.rows, pattern.cols)), shape=pattern.shape)\n"
    "        np.testing.assert_allclose(sparse.coo_array(values).toarray(), reference, rtol=1e-8, atol=1e-9)",
  )
  for solver in ("ipopt", "piqp", "fatrop"):
    libraries = root / f"plugins/scaly-{solver}/src/scaly_{solver}/lib"
    shutil.copytree(libraries, out / "native" / solver)
  manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.rglob("*")) if path.is_file()}
  (out / "frozen.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"Frozen {len(manifest)} files in {out}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("root", type=Path)
  parser.add_argument("out", type=Path)
  args = parser.parse_args()
  main(args.root, args.out)
