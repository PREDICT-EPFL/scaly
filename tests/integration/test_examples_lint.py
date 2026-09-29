"""The examples use scaly as a user would: no ``sys.path`` edits, no underscore names from ``scaly``,
and requirements declared in full, a script's in its PEP 723 header and a notebook's in its
metadata, so the runner and a user's ``uv run`` install what it imports. The case studies, drivers
with baselines in other environments, keep their own arrangements."""

from __future__ import annotations

import io
import json
import re
import sys
import tokenize
import tomllib
from pathlib import Path

import pytest

from scaly.testing import examples

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
NAMESPACES = ("core", "linalg", "roots", "opt", "integrators", "interp", "ocp", "sets", "nn", "geometry", "export", "viz")
REQUIRES_PYTHON = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]

# Third-party modules an example may import and the distribution that provides each; numpy and
# scipy come with scaly, and a notebook's kernel brings IPython.
DISTRIBUTION = {"casadi": "casadi", "matplotlib": "matplotlib", "cycler": "matplotlib", "IPython": None}
PLUGIN_USE = {
  "scaly-piqp": (r"\bPIQP\s*\(|\bopt\.PIQP\b", r"[\"'](opt\.)?piqp[\"']"),
  "scaly-ipopt": (r"\bIPOPT\s*\(|\bopt\.IPOPT\b", r"[\"'](opt\.)?ipopt[\"']"),
  "scaly-sqp": (r"\bSQP\s*\(|\bopt\.SQP\b", r"[\"'](opt\.)?sqp[\"']"),
}
SYS_PATH_EDIT = re.compile(r"\bsys\.path\s*(\.\s*(insert|append|extend|remove)\b|\[[^\]]*\]\s*=[^=]|\+?=[^=])|\bsite\.addsitedir\b")
PRIVATE = re.compile(
  r"(?m)^\s*from\s+scaly(\.\w+)*\s+import\s+[^\n]*?(?<![\w.])_(?!_)\w*"  # from scaly.x import _y
  r"|\bscaly(\.\w+)*\._(?!_)\w*"  # scaly.x._y, as a module or an attribute
  r"|\bsc\._(?!_)\w*"  # sc._y
)


def _files() -> list[Path]:
  out = []
  for path in sorted([*EXAMPLES.rglob("*.py"), *EXAMPLES.rglob("*.ipynb")]):
    rel = path.relative_to(EXAMPLES).as_posix()
    if not rel.startswith(("case_studies/", "generated/", "gen/")) and ".ipynb_checkpoints" not in rel:
      out.append(path)
  return out


FILES = _files()


def _strip(text: str) -> str:
  """``text`` with comments and docstrings blanked, so prose that names a solver or a path is not code."""
  lines = text.splitlines(keepends=True)
  cuts = []
  prev = tokenize.NEWLINE
  for tok in tokenize.generate_tokens(io.StringIO(text).readline):
    if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and prev in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT)):
      cuts.append((tok.start, tok.end))
    if tok.type not in (tokenize.NL, tokenize.COMMENT):
      prev = tok.type
  for (r0, c0), (r1, c1) in reversed(cuts):
    if r0 == r1:
      lines[r0 - 1] = lines[r0 - 1][:c0] + " " * (c1 - c0) + lines[r0 - 1][c1:]
    else:
      lines[r0 - 1] = lines[r0 - 1][:c0] + "\n"
      for r in range(r0, r1 - 1):
        lines[r] = "\n"
      lines[r1 - 1] = " " * c1 + lines[r1 - 1][c1:]
  return "".join(lines)


def code(path: Path) -> str:
  """An example's code: a script's source, or a notebook's code cells without IPython magics."""
  if path.suffix == ".ipynb":
    cells = ["".join(c["source"]) for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    return "\n".join(_strip("\n".join(line for line in cell.splitlines() if not line.lstrip().startswith(("%", "!"))) + "\n") for cell in cells)
  return _strip(path.read_text())


def violations(source: str) -> list[str]:
  """What in ``source`` (comments and docstrings already out) an example may not do."""
  found = [f"edits sys.path: {m.group(0).strip()}" for m in SYS_PATH_EDIT.finditer(source)]
  found += [f"imports a private scaly name: {m.group(0).strip()}" for m in PRIVATE.finditer(source)]
  return found


def _top_imports(source: str) -> set[str]:
  return set(re.findall(r"(?m)^import ([\w.]+)", source)) | set(re.findall(r"(?m)^from ([\w.]+) import", source))


def needed(path: Path, *, seen: set[Path] | None = None) -> set[str]:
  """The distributions ``path`` needs beside scaly, as its code shows them: the solver plugins it
  names (by class; by method string unless it also drives CasADi, whose solvers share the names) and
  the third-party packages it and the modules beside it import at top level."""
  seen = set() if seen is None else seen
  seen.add(path)
  source = code(path)
  out = set()
  if len(seen) == 1:
    casadi = re.search(r"(?m)^\s*(import|from) casadi\b", source) is not None
    for dist, (by_class, by_string) in PLUGIN_USE.items():
      if re.search(by_class, source) or (not casadi and re.search(by_string, source)):
        out.add(dist)
  for module in _top_imports(source):
    top = module.split(".")[0]
    if top in sys.stdlib_module_names or top in ("numpy", "scipy", "scaly"):
      continue
    sibling = path.parent / (module.replace(".", "/") + ".py")
    if sibling.exists():
      if sibling not in seen:
        out |= needed(sibling, seen=seen)
      continue
    if top not in DISTRIBUTION:
      out.add(f"<unknown module {module}: add it to DISTRIBUTION>")
    elif dist := DISTRIBUTION[top]:
      out.add(dist)
  return out


def _is_example(path: Path) -> bool:
  if path.suffix == ".ipynb":
    return True
  rel = path.relative_to(EXAMPLES)
  gallery = len(rel.parts) == 2 and rel.parts[0] in NAMESPACES and path.name != "plotstyle.py"
  return gallery or '__name__ == "__main__"' in path.read_text()


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(EXAMPLES).as_posix())
def test_the_example_uses_scaly_as_a_user_would(path: Path) -> None:
  assert not violations(code(path))


@pytest.mark.parametrize("path", [p for p in FILES if _is_example(p)], ids=lambda p: p.relative_to(EXAMPLES).as_posix())
def test_the_example_declares_what_it_needs(path: Path) -> None:
  reqs = examples.requirements(path)
  assert reqs is not None, "no PEP 723 header" if path.suffix == ".py" else f"no '{examples.NOTEBOOK_KEY}' metadata"
  assert reqs.requires_python == REQUIRES_PYTHON
  declared = {examples.parse(req)[0] for req in reqs.dependencies}
  assert not needed(path) - declared, f"declare {sorted(needed(path) - declared)}"
  uses_scaly = re.search(r"(?m)^\s*(import|from) scaly\b", code(path)) is not None
  assert "scaly" in declared or not uses_scaly, "declare scaly"


def test_the_examples_live_in_namespace_folders() -> None:
  """An example sits in the folder of its namespace; the CasADi pairs, which span several, have their own."""
  folders = {p.relative_to(EXAMPLES).parts[0] for p in FILES}
  assert folders <= {*NAMESPACES, "casadi"}, folders - {*NAMESPACES, "casadi"}


def test_the_plot_style_is_one_file_in_many_places() -> None:
  """A notebook imports the shared plot style from beside it, with no path edit; the copies are one file."""
  copies = sorted(p for p in EXAMPLES.rglob("plotstyle.py") if ".ipynb_checkpoints" not in p.parts)
  importers = {p.parent for p in FILES if p.suffix == ".ipynb" and re.search(r"(?m)^\s*import plotstyle\b", code(p))}
  assert importers <= {p.parent for p in copies}, importers - {p.parent for p in copies}
  assert len({p.read_text() for p in copies}) == 1, [p.relative_to(EXAMPLES).as_posix() for p in copies]


@pytest.mark.parametrize(
  "source, what",
  [
    ("import sys\nsys.path.insert(0, 'x')\n", "edits sys.path"),
    ("sys.path[:0] = ['x']\n", "edits sys.path"),
    ("sys.path.append(str(HERE))\n", "edits sys.path"),
    ("sys.path += ['x']\n", "edits sys.path"),
    ("from scaly.roots import _newton_step\n", "private"),
    ("from scaly.linalg._ldl import thing\n", "private"),
    ("x = sc._backend\n", "private"),
    ("from scaly import linalg, _lazy\n", "private"),
  ],
)
def test_the_lint_catches(source: str, what: str) -> None:
  assert any(what in v for v in violations(source)), violations(source)


def test_the_lint_passes_what_is_allowed() -> None:
  assert not violations("import sys\nprint(sys.path[0])\nfrom scaly import __version__\nx = sc.opt.IPOPT()\nif sys.path == []: pass\n")


def test_needed_sees_plugins_and_packages(tmp_path: Path) -> None:
  (tmp_path / "helper.py").write_text("import matplotlib\n")
  script = tmp_path / "a.py"
  script.write_text('"""Solved by PIQP."""\nimport casadi\nimport helper\nimport scaly as sc\nm = sc.opt.IPOPT()\ns = ca.nlpsol("s", "ipopt", {})\n')
  assert needed(script) == {"casadi", "matplotlib", "scaly-ipopt"}
  script.write_text('import scaly as sc\nsolve = sc.opt.solver(p, "sqp")\nimport torch\n')
  assert needed(script) == {"scaly-sqp", "<unknown module torch: add it to DISTRIBUTION>"}
