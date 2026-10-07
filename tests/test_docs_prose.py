"""Check punctuation in published user prose."""

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
  ROOT / "README.md",
  ROOT / "docs/index.md",
  *sorted(path for directory in ("guide", "how_it_works", "benchmarks") for path in (ROOT / "docs" / directory).rglob("*.md")),
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
  violations = [f"{relative}:{line}: {kind}: end the sentence or use a comma" for line, kind in prose_violations(path.read_text())]
  assert not violations, "\n".join(violations)
