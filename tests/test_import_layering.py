"""Import-layer discipline for ``src/scaly``.

Every module has an import layer (``docs/dev/codebase.md``): a module may import modules in its own import layer
or a lower import layer, never a higher one. Two dicts hold the exceptions. ``SEAM`` is the one sanctioned upward edge —
calling a ``Function`` JIT-compiles it. ``TOLERATED`` is the escape hatch for a violation being
carried deliberately through a refactor in progress, and is empty; an entry there is a decision to
make in the open, not a deferred import to leave lying around, and it is meant to go back to empty
before the refactor lands. The remaining graph must also be acyclic:
an import cycle that nobody recorded is what let the layout drift in the first place.

Two limits worth knowing, because they bound what a green run means. The acyclicity check runs on
the graph *minus* the recorded edges, so it says "the only cycles left are ones a recorded
violation creates" — such a cycle disappears when its `TOLERATED` entry is fixed, which is why it
is not also recorded separately. And this is a discipline over what the source says, not what the
interpreter does: imports are read with ``ast`` (function-local ones included — deferring an
import inside a function is the usual way a cycle gets hidden), while ``if TYPE_CHECKING:`` blocks, the implicit parent-package import,
and dynamic loading (``EntryPoint.load`` in ``solvers/registry.py``) are not counted.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "scaly"

# Import layer per module. Keys are today's module paths; the numbers are the target layout's, so a key
# is renamed when its file moves but its import layer only changes if the design changes.
IMPORT_LAYERS: dict[str, int] = {
  "scaly.utils": 0,
  "scaly.utils.env": 0,
  "scaly.utils.ext_api": 0,
  "scaly.utils.names": 0,
  "scaly.utils.options": 0,
  "scaly.utils.torch_state_dict": 0,
  "scaly.ir": 1,
  "scaly.ir.types": 1,
  "scaly.ir.expr": 1,
  "scaly.ir.match": 1,
  "scaly.ir.program": 1,
  "scaly.ir.program_spec": 1,
  "scaly.ir.expr_spec": 1,
  "scaly.ir.spec": 1,
  "scaly.ir.text": 1,
  "scaly.passes": 1,
  "scaly.passes.affine": 2,
  "scaly.passes.arith": 2,
  "scaly.passes.expr": 2,
  "scaly.ad.sparsity": 2,
  "scaly.solvers.stats": 2,
  "scaly.function": 3,
  "scaly.function.model": 3,
  "scaly.function.tree": 3,
  "scaly.function.extern": 3,
  "scaly.ad": 4,
  "scaly.ad.derivatives": 4,
  "scaly.ad.forward": 4,
  "scaly.ad.reverse": 4,
  "scaly.ad.sparse": 4,
  # Import layer 4, not the plan's 5: ``vmap`` needs a ``Function``, and ``ad`` needs ``vmap``.
  "scaly.function.sugar": 4,
  "scaly.function.api": 5,
  "scaly.function.factory": 5,
  "scaly.solvers": 5,
  "scaly.solvers._oracle": 5,
  "scaly.solvers.graph": 5,
  "scaly.solvers.nlp": 5,
  "scaly.solvers.problem": 5,
  "scaly.solvers.solver": 5,
  "scaly.solvers.paths": 5,
  "scaly.solvers.qp": 5,
  "scaly.solvers.registry": 5,
  "scaly.solvers.model": 5,
  "scaly.solvers.wrapper": 5,
  "scaly.solvers.ipm": 5,
  "scaly.solvers.ipm.structure": 5,
  "scaly.solvers.ipm.ruiz": 5,
  "scaly.solvers.ipm.kkt": 5,
  "scaly.solvers.ipm.algorithm": 5,
  "scaly.linalg": 5,
  "scaly.linalg.dense": 5,
  "scaly.linalg.sparse_factor": 5,
  "scaly.linalg.sparse": 5,
  "scaly.linalg.symbolic": 5,
  "scaly.integrators": 5,
  "scaly.integrators.explicit": 5,
  "scaly.integrators.implicit": 5,
  "scaly.integrators.linear": 5,
  "scaly.integrators.model": 5,
  "scaly.integrators.polynomial": 5,
  "scaly.integrators.tableau": 5,
  "scaly.integrators.transcription": 5,
  "scaly.interp": 5,
  "scaly.interp.constrained": 5,
  "scaly.interp.fit": 5,
  "scaly.interp.grid": 5,
  "scaly.interp.spline": 5,
  "scaly.mpc": 5,
  "scaly.mpc.controller": 5,
  "scaly.mpc.ocp": 5,
  "scaly.mpc.polytope": 5,
  "scaly.mpc.terminal": 5,
  "scaly.passes.program": 6,
  "scaly.passes.program._common": 6,
  "scaly.passes.program.combine_scatter_sums": 6,
  "scaly.passes.program.delinearize_loops": 6,
  "scaly.passes.program.fold_arith": 6,
  "scaly.passes.program.fuse_elementwise": 6,
  "scaly.passes.program.hoist_invariant": 6,
  "scaly.passes.program.pack_workspace": 6,
  "scaly.passes.program.unroll_unit_loops": 6,
  "scaly.passes.program.scalarize": 6,
  "scaly.passes.program.coalesce_stores": 6,
  "scaly.passes.program.prepare_scalar": 6,
  "scaly.passes.program.scheduling": 6,
  "scaly.passes.lowering": 6,
  "scaly.codegen.abi": 7,
  "scaly.codegen.adapter": 7,
  "scaly.codegen.jit": 7,
  "scaly.codegen.toolchain": 7,
  "scaly.codegen": 7,
  "scaly.codegen.__main__": 7,
  "scaly.codegen.aot": 7,
  "scaly.codegen.c": 7,
  "scaly.codegen.casadi": 7,
  "scaly.codegen.cpp": 7,
  "scaly.viz": 8,
  "scaly.viz.graph": 8,
  "scaly.viz.recording": 8,
  "scaly.viz.serve": 8,
  "scaly": 9,  # the curated public re-exports sit above everything they re-export
  "scaly.ext": 9,  # the extension API, collected from every layer
}

# The one upward import the architecture sanctions (docs/dev/codebase.md, "Import layers").
SEAM: dict[tuple[str, str], str] = {
  ("scaly.function.model", "scaly.codegen.jit"): "calling a Function JIT-compiles it",
}

# Violations the restructure has not reached yet. Shrinks every phase; empty when it is done.
TOLERATED: dict[tuple[str, str], str] = {}


def _modules() -> dict[str, Path]:
  out: dict[str, Path] = {}
  for path in sorted(SRC.rglob("*.py")):
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    if not all(part.isidentifier() for part in parts):
      continue  # not importable, such as an editor's checkpoint copy (.ipynb_checkpoints/c-checkpoint.py)
    out[".".join(parts)] = path
  return out


def _is_type_checking(test: ast.expr) -> bool:
  return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _import_statements(node: ast.AST) -> list[ast.Import | ast.ImportFrom]:
  found: list[ast.Import | ast.ImportFrom] = []
  for child in ast.iter_child_nodes(node):
    if isinstance(child, ast.If) and _is_type_checking(child.test):
      for alt in child.orelse:
        found += _import_statements(alt)
      continue
    if isinstance(child, ast.Import | ast.ImportFrom):
      found.append(child)
    found += _import_statements(child)
  return found


def _base_package(module: str, path: Path, level: int) -> str:
  parts = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
  return ".".join(parts[: len(parts) - (level - 1)])


def _edges() -> dict[tuple[str, str], list[int]]:
  modules = _modules()
  edges: dict[tuple[str, str], list[int]] = {}
  for name, path in modules.items():
    for stmt in _import_statements(ast.parse(path.read_text())):
      if isinstance(stmt, ast.Import):
        targets = [alias.name for alias in stmt.names]
      else:
        base = stmt.module if stmt.level == 0 else ".".join(filter(None, (_base_package(name, path, stmt.level), stmt.module)))
        # Charge the most specific module the statement names: ``from .solvers import stats``
        # is an edge to ``stats``, not to the package, which nobody is charged for.
        targets = [sub if (sub := f"{base}.{alias.name}") in modules else base for alias in stmt.names]
      for target in targets:
        if target in modules and target != name:
          edges.setdefault((name, target), []).append(stmt.lineno)
  return edges


def _cycle(edges: set[tuple[str, str]]) -> list[str] | None:
  graph: dict[str, list[str]] = {}
  for src, dst in sorted(edges):
    graph.setdefault(src, []).append(dst)
  done: set[str] = set()
  for start in sorted(graph):
    stack = [start]
    on_stack = {start}
    while stack:
      node = stack[-1]
      if node in done:
        stack.pop()
        on_stack.discard(node)
        continue
      pending = [n for n in graph.get(node, ()) if n not in done]
      if not pending:
        done.add(node)
        stack.pop()
        on_stack.discard(node)
        continue
      nxt = pending[0]
      if nxt in on_stack:
        return [*stack[stack.index(nxt) :], nxt]
      stack.append(nxt)
      on_stack.add(nxt)
  return None


@pytest.mark.parametrize("module", sorted(IMPORT_LAYERS))
def test_module_imports_standalone(module: str) -> None:
  """The runtime counterpart of the table above, and the half it cannot see: a module that is
  imported first must not deadlock on a half-initialized one. Package ``__init__`` execution is
  what makes this different from the static check — moving a file can break it."""
  proc = subprocess.run([sys.executable, "-c", f"import {module}"], check=False, capture_output=True, text=True)
  assert proc.returncode == 0, f"importing {module} first fails:\n{proc.stderr}"


def test_import_layer_table_covers_every_module() -> None:
  missing = sorted(set(_modules()) - set(IMPORT_LAYERS))
  extra = sorted(set(IMPORT_LAYERS) - set(_modules()))
  assert not missing, f"new modules need an import layer in IMPORT_LAYERS: {missing}"
  assert not extra, f"IMPORT_LAYERS names modules that no longer exist: {extra}"


def test_no_upward_imports() -> None:
  recorded = set(SEAM) | set(TOLERATED)
  bad = [
    f"{src} -> {dst} (import layer {IMPORT_LAYERS[src]} -> {IMPORT_LAYERS[dst]}, line {lines[0]})"
    for (src, dst), lines in _edges().items()
    if (src, dst) not in recorded and IMPORT_LAYERS[dst] > IMPORT_LAYERS[src]
  ]
  assert not bad, "imports from a higher import layer:\n  " + "\n  ".join(sorted(bad))


def test_no_import_cycles() -> None:
  edges = {edge for edge in _edges() if edge not in SEAM and edge not in TOLERATED}
  cycle = _cycle(edges)
  assert cycle is None, "import cycle outside the recorded exceptions: " + " -> ".join(cycle or ())


def test_recorded_exceptions_still_exist() -> None:
  edges = set(_edges())
  stale = sorted(f"{src} -> {dst}" for src, dst in (*SEAM, *TOLERATED) if (src, dst) not in edges)
  assert not stale, "these edges are gone; drop them from SEAM/TOLERATED:\n  " + "\n  ".join(stale)


def test_tolerated_violations_are_still_violations() -> None:
  """An entry that stopped being an upward edge would go on suppressing the cycle check for free."""
  legal = sorted(f"{src} -> {dst}" for src, dst in TOLERATED if IMPORT_LAYERS[dst] <= IMPORT_LAYERS[src])
  assert not legal, "these edges no longer break the import-layer rule; drop them from TOLERATED:\n  " + "\n  ".join(legal)


def test_each_seam_is_a_single_import() -> None:
  """The plan sanctions *one* named seam per pair, not a habit of reaching upward."""
  edges = _edges()
  scattered = sorted(f"{src} -> {dst} at lines {edges[src, dst]}" for src, dst in SEAM if len(edges[src, dst]) > 1)
  assert not scattered, "a sanctioned seam is one import statement:\n  " + "\n  ".join(scattered)


# The flat leaf seams under ``symbolic_call``/``numerical_call``. ``function/model.py`` defines and
# uses them; differentiation is the one sanctioned consumer, because it synthesizes callees from
# flat expression lists and calls them with that same list.
FLAT_SEAM_USERS = {"scaly.function.model", "scaly.ad.forward"}


def test_flat_call_seams_stay_inside_their_sanctioned_modules() -> None:
  """Everything else addresses a Function by its declared tree, which is what users write."""
  leaked = sorted(
    f"{name}:{i}"
    for name, path in _modules().items()
    if name not in FLAT_SEAM_USERS
    for i, line in enumerate(path.read_text().splitlines(), 1)
    if "_flat_symbolic_call" in line or "_flat_numerical_call" in line
  )
  assert not leaked, "the flat call seam leaked outside function/model.py and ad/forward.py:\n  " + "\n  ".join(leaked)


# The compiler's packages, and what none of them may name: solvers reach the compiler only through the
# extern-callee protocol (``function/extern.py``) and output adapters only through their registry
# (``codegen/adapter.py``). The adapters themselves still live under ``codegen/`` and are exempt.
CORE_PACKAGES = ("scaly.ir", "scaly.ad", "scaly.function", "scaly.passes", "scaly.codegen", "scaly.utils")
NOT_FROM_CORE = ("scaly.solvers", "scaly.codegen.cpp", "scaly.codegen.casadi")


def _named_modules(name: str, path: Path, stmt: ast.Import | ast.ImportFrom) -> list[str]:
  if isinstance(stmt, ast.Import):
    return [alias.name for alias in stmt.names]
  base = stmt.module if stmt.level == 0 else ".".join(filter(None, (_base_package(name, path, stmt.level), stmt.module)))
  return [base or "", *(f"{base}.{alias.name}" for alias in stmt.names)]


def test_core_names_no_solver_or_adapter() -> None:
  """Every import statement counts here, inside a function or under ``TYPE_CHECKING`` too: a core
  distribution must build and type-check without the solver package or the adapters installed."""
  bad = []
  for name, path in _modules().items():
    if not any(name == pkg or name.startswith(f"{pkg}.") for pkg in CORE_PACKAGES) or name in NOT_FROM_CORE:
      continue
    for stmt in ast.walk(ast.parse(path.read_text())):
      if isinstance(stmt, ast.Import | ast.ImportFrom):
        for target in _named_modules(name, path, stmt):
          if any(target == forbidden or target.startswith(f"{forbidden}.") for forbidden in NOT_FROM_CORE):
            bad.append(f"{name}:{stmt.lineno} imports {target}")
  assert not bad, "core modules name a solver or an output adapter:\n  " + "\n  ".join(sorted(set(bad)))
