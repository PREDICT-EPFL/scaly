"""Prepare, fingerprint, compile and export the C libraries reached by generated code."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from ..function.extern import CLibrary, _Extern
from ..function.concrete import ConcreteFunction
from ..ir.expr import callees_of
from ..solvers.graph import solver_callees
from .toolchain import LinkedLibrary, cache_root, compiler_fingerprint, directory_fingerprint, resolve_library


def libraries_of(fun: ConcreteFunction) -> tuple[CLibrary, ...]:
  """The libraries in a graph, in the same callee order lowering follows, without lowering."""
  found: dict[CLibrary, None] = {}
  seen = set()
  pending = [fun]
  while pending:
    callee = pending.pop()
    if callee in seen:
      continue
    seen.add(callee)
    if isinstance(callee, _Extern):
      found[callee.library] = None
    else:
      children = [*solver_callees(callee), *(child for node in callee.nodes for child in callees_of(node))]
      pending.extend(reversed(children))
  return tuple(found)


def _write_files(library: CLibrary, directory: Path) -> None:
  for name, source in (*library.sources, *library.headers):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != source:
      descriptor, temporary = tempfile.mkstemp(dir=path.parent)
      try:
        with os.fdopen(descriptor, "w") as stream:
          stream.write(source)
        try:
          Path(temporary).replace(path)
        except PermissionError:
          if not path.exists():
            raise
      finally:
        Path(temporary).unlink(missing_ok=True)


def _compile_flags(library: CLibrary, directory: Path) -> tuple[str, ...]:
  return (
    "-I" + directory.as_posix(),
    *("-I" + Path(path).as_posix() for path in library.include_dirs),
    *(f"-D{name}" + (f"={value}" if value is not None else "") for name, value in library.defines),
    *library.compile_flags,
  )


@dataclass(frozen=True)
class LibraryBuild:
  library: CLibrary
  directory: Path
  key: str
  flags: tuple[str, ...]
  links: tuple[LinkedLibrary, ...]

  @property
  def link_flags(self) -> tuple[str, ...]:
    return tuple(flag for link in self.links for flag in link.flags)


def prepare_library(library: CLibrary, compiler: tuple[str, ...]) -> LibraryBuild:
  """Fingerprint source, expanded headers, flags, compiler and resolved linked libraries."""
  description = json.dumps({field: getattr(library, field) for field in library.__dataclass_fields__}, sort_keys=True)
  directory = cache_root() / "extern" / hashlib.sha256(description.encode()).hexdigest()
  directory.mkdir(parents=True, exist_ok=True)
  _write_files(library, directory)
  flags = _compile_flags(library, directory)
  links = tuple(resolve_library(name, compiler) for name in library.link)
  digest = hashlib.sha256(description.encode())
  digest.update(compiler_fingerprint(compiler).encode())
  for include in library.include_dirs:
    digest.update(directory_fingerprint(Path(include)))
  for link in links:
    digest.update(json.dumps((link.path, link.identity, link.flags)).encode())
  for name, _ in library.sources:
    expanded = subprocess.run(
      [*compiler, *flags, "-E", "-P", "-x", "c", (directory / name).as_posix()], cwd=tempfile.gettempdir(), capture_output=True, check=True, text=True
    ).stdout.encode()
    digest.update(len(expanded).to_bytes(8, "little"))
    digest.update(expanded)
  return LibraryBuild(library, directory, digest.hexdigest(), flags, links)


def compile_library(build: LibraryBuild, compiler: tuple[str, ...]) -> tuple[str, ...]:
  """Compile each source once into a content-addressed object, with library-local flags."""
  directory = cache_root() / "extern" / build.key
  directory.mkdir(parents=True, exist_ok=True)
  objects = []
  for index, (name, _) in enumerate(build.library.sources):
    target = directory / f"{index}.o"
    if not target.exists():
      descriptor, temporary = tempfile.mkstemp(suffix=".o", dir=directory)
      os.close(descriptor)
      try:
        subprocess.run(
          [*compiler, "-O2", "-fPIC", *build.flags, "-x", "c", "-c", (build.directory / name).as_posix(), "-o", temporary],
          cwd=tempfile.gettempdir(),
          capture_output=True,
          check=True,
          text=True,
        )
        Path(temporary).replace(target)
      finally:
        Path(temporary).unlink(missing_ok=True)
    objects.append(target.as_posix())
  return tuple(objects)


def export_libraries(libraries: tuple[CLibrary, ...], directory: Path) -> list[dict]:
  """Write owned sources and headers, retaining logical dependency names in a build manifest."""
  records = []
  for index, library in enumerate(libraries):
    relative = Path("libraries") / str(index)
    _write_files(library, directory / relative)
    records.append(
      {
        "name": library.name,
        "sources": [(relative / name).as_posix() for name, _ in library.sources],
        "headers": [(relative / name).as_posix() for name, _ in library.headers],
        "include_dirs": [relative.as_posix(), *library.include_dirs],
        "defines": dict(library.defines),
        "compile_flags": list(library.compile_flags),
        "link": list(library.link),
        "versions": dict(library.versions),
      }
    )
  return records
