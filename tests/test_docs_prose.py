"""Check punctuation in published prose and execute selected guide examples."""

import builtins
from pathlib import Path
import re

import numpy as np
import pytest

import scaly as sc

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
  ROOT / "README.md",
  ROOT / "docs/index.md",
  *sorted(path for directory in ("guide", "how_it_works", "benchmarks") for path in (ROOT / "docs" / directory).rglob("*.md")),
]
RUNNABLE = [
  pytest.param(ROOT / "docs/guide/linalg.md", None, id="docs/guide/linalg.md"),
  pytest.param(ROOT / "docs/guide/sparse_matrices.md", None, id="docs/guide/sparse_matrices.md"),
  pytest.param(ROOT / "docs/guide/codegen.md", "External C functions", id="external_c"),
]


def prose_violations(source: str) -> list[tuple[int, str]]:
  violations = []
  fence = None
  tick = None  # an inline code span that continues onto the next line of its paragraph
  for number, line in enumerate(source.splitlines(), 1):
    if not line.strip():
      tick = None
    delimiter = re.match(r"^\s*(`{3,}|~{3,})", line)
    if delimiter:
      marker = delimiter[1]
      if fence is None:
        fence = marker
      elif marker[0] == fence[0] and len(marker) >= len(fence):
        fence = None
      continue
    if fence is not None:
      continue
    if tick is not None:
      end = re.search(rf"(?<!`){tick}(?!`)", line)
      if end is None:
        continue
      line, tick = line[end.end() :], None
    line = re.sub(r"(`+).*?\1", "", line)
    opening = re.search(r"`+", line)
    if opening:
      line, tick = line[: opening.start()], opening[0]
    line = re.sub(r"https?://[^\s<>]+", "", line)
    if "—" in line:
      violations.append((number, "em dash"))
    if re.search(r"(?<!\d)–|–(?!\d)", line):
      violations.append((number, "en dash"))
    # Table cells contain prose too. HTML entities are punctuation escapes, not sentences.
    line = re.sub(r"&(?:#\d+|#x[\da-fA-F]+|\w+);", "", line)
    if ";" in line:
      violations.append((number, "semicolon"))
  return violations


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_published_prose(path: Path) -> None:
  relative = path.relative_to(ROOT).as_posix()
  violations = [f"{relative}:{line}: {kind}: end the sentence or use a comma" for line, kind in prose_violations(path.read_text(encoding="utf-8"))]
  assert not violations, "\n".join(violations)


@pytest.mark.parametrize("path,section", RUNNABLE)
def test_guide_examples_run(path: Path, section: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.chdir(tmp_path)
  namespace: dict[str, object] = {}
  source = path.read_text(encoding="utf-8")
  if section is not None:
    source = source.split(f"\n## {section}\n", 1)[1].split("\n## ", 1)[0]
  for code in re.findall(r"^```python\n(.*?)^```", source, re.MULTILINE | re.DOTALL):
    # A fence ending in "# SomeError: message" shows a failure: it must raise exactly that.
    error = re.search(r"^# (\w+Error): (.*)\Z", code.rstrip(), re.MULTILINE)
    if error is None:
      exec(code, namespace)
      continue
    with pytest.raises(getattr(builtins, error[1]), match=f"^{re.escape(error[2])}$"):
      exec(code, namespace)


@pytest.mark.parametrize(
  ("name", "expected_output", "input_dtypes", "output_dtypes"),
  [
    ("row_extrema", "(2, 1)\n[2. 5.]\n[2 2]\n", ("float64",), ("float64", "int64")),
    ("select_entries", "[ 7. -1.  3.]\n", ("float64", "int64"), ("float64",)),
  ],
  ids=["row_extrema", "select_entries"],
)
def test_functions_guide_integer_leaves(
  name: str, expected_output: str, input_dtypes: tuple[str, ...], output_dtypes: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
  source = (ROOT / "docs/guide/functions.md").read_text(encoding="utf-8")
  code = next(code for code in re.findall(r"^```python\n(.*?)^```", source, re.MULTILINE | re.DOTALL) if f"def {name}(" in code)
  namespace: dict[str, object] = {"np": np, "sc": sc}
  exec(code, namespace)
  assert capsys.readouterr().out == expected_output
  function = namespace[name]
  assert isinstance(function, sc.Function)
  instance = function.instantiate()
  assert tuple(expr.type.dtype.name for expr in instance.inputs) == input_dtypes
  assert tuple(expr.type.dtype.name for expr in instance.outputs) == output_dtypes
