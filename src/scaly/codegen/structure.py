"""The structural key of a Function: a digest of its graph, and of the code that would render it, taken without lowering anything, by which the JIT finds a library it already built."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import os
import struct
import sys
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..function import ConcreteFunction, Function
from ..function.extern import ExternCallee, ExternRenderCtx
from ..ir.expr import _RULE_KINDS, Expr, op_def, topo
from ..ir.types import TensorType
from ..utils.env import LOADED_AT_NS
from ..utils.names import c_ident


class Unkeyed(Exception):
  """The walk met a value it does not know how to digest. The Function then has no structural key
  and is rendered to find its library, as every Function was before there was one."""


# Packages whose code a version stands for: the interpreter's and NumPy's go into every key.
_VERSIONED = frozenset({"numpy", *sys.stdlib_module_names})


class _Walk:
  """One digest of a graph. Every node, type and Function is written once, in the order a
  depth-first walk from the root first meets it, and referred to afterwards by its number in that
  order, so two graphs built the same way give the same bytes whatever their objects' addresses."""

  def __init__(self) -> None:
    self.hash = hashlib.sha256()
    self.exprs: dict[int, int] = {}
    self.functions: dict[int, int] = {}
    self.types: dict[TensorType, int] = {}
    self.type_ids: dict[int, int] = {}
    self.ops: set[str] = set()
    self.packages: set[str] = set()

  def put(self, tag: bytes, payload: bytes = b"") -> None:
    # Tagged and length-prefixed: no two sequences of values share their bytes.
    self.hash.update(tag)
    self.hash.update(len(payload).to_bytes(8, "little"))
    self.hash.update(payload)

  def code_of(self, obj: Any) -> None:
    """Note the package that defines ``obj`` (a class, a rule): its source is part of the key."""
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str):
      raise Unkeyed(f"{obj!r} names no module")
    self.packages.add(module.partition(".")[0])

  def value(self, v: Any, owner: ConcreteFunction | None) -> None:
    """Write an attribute's value. ``owner`` is the Function whose graph holds it."""
    kind = type(v)
    if v is None:
      self.put(b"N")
    elif kind is bool:
      self.put(b"B", b"1" if v else b"0")
    elif kind is int:
      self.put(b"I", str(v).encode())
    elif kind is float:
      self.put(b"R", struct.pack("<d", v))  # by bits: -0.0 is not 0.0, and a NaN keeps its payload
    elif kind is str:
      self.put(b"S", v.encode())
    elif kind is bytes:
      self.put(b"Y", v)
    elif isinstance(v, np.ndarray):
      if v.dtype.hasobject:
        raise Unkeyed("an array of objects")
      self.put(b"A", f"{v.dtype.str}{v.shape}".encode())
      # Its bytes through the buffer, so that a large constant is not copied; the dtype and shape
      # above say how many there are.
      self.hash.update(np.ascontiguousarray(v))
    elif isinstance(v, np.generic):
      self.put(b"G", v.dtype.str.encode())
      self.put(b"g", v.tobytes())
    elif kind is tuple or kind is list:
      self.put(b"T" if kind is tuple else b"L", str(len(v)).encode())
      for item in v:
        self.value(item, owner)
    elif kind is dict:
      if not all(type(key) is str for key in v):
        raise Unkeyed("a dict with keys that are not strings")
      self.put(b"D", str(len(v)).encode())
      for key in sorted(v):
        self.put(b"K", key.encode())
        self.value(v[key], owner)
    elif kind is set or kind is frozenset:
      if not all(type(item) in (str, int) for item in v):
        raise Unkeyed("a set of anything but strings and integers")
      self.put(b"Z", str(len(v)).encode())
      for item in sorted(v, key=lambda item: (type(item).__name__, item)):
        self.value(item, owner)
    elif kind is slice or kind is range:
      self.put(b"C" if kind is slice else b"P")
      for item in (v.start, v.stop, v.step):
        self.value(item, owner)
    elif isinstance(v, Expr):
      self.nodes((v,), owner)
      self.put(b"E", str(self.exprs[id(v)]).encode())
    elif isinstance(v, Function):
      if not v.is_concrete:
        raise Unkeyed(f"the Function template {v.name!r}")
      self.put(b"F", str(self.function(v.concrete)).encode())
    elif isinstance(v, enum.Enum):
      self.code_of(kind)
      self.put(b"M", f"{kind.__module__}.{kind.__qualname__}.{v.name}".encode())
    elif isinstance(v, ExternCallee):
      # An extern body is digested with the Function that owns it (``extern``), by what it renders.
      if owner is None or owner.extern is not v:
        raise Unkeyed("an extern callee outside the Function it is the body of")
      self.put(b"X")
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
      self.record(v, owner)
    else:
      raise Unkeyed(f"a value of type {kind.__module__}.{kind.__qualname__}")

  def record(self, v: Any, owner: ConcreteFunction | None) -> None:
    """A dataclass instance (a device, a sparsity pattern) by its class and its fields, written
    out wherever it occurs: whether two places share one object is not part of the graph."""
    kind = type(v)
    self.code_of(kind)
    self.put(b"O", f"{kind.__module__}.{kind.__qualname__}".encode())
    for field in dataclasses.fields(v):
      if not field.compare:
        # A field its own class leaves out of equality may still shape the code; nothing says.
        raise Unkeyed(f"{kind.__qualname__}.{field.name} is not part of its value")
      self.put(b"K", field.name.encode())
      self.value(getattr(v, field.name), owner)

  def type(self, t: TensorType, owner: ConcreteFunction | None) -> int:
    """A node's type by number. Equal types are one, as they are to the graph itself: interning
    compares a node's type by value, so which of two equal objects a node holds is an accident."""
    serial = self.type_ids.get(id(t))
    if serial is None:
      serial = self.types.get(t)
      if serial is None:
        self.record(t, owner)
        serial = self.types[t] = len(self.types)
      self.type_ids[id(t)] = serial  # the node that holds ``t`` is alive for the walk
    return serial

  def nodes(self, roots: Iterable[Expr], owner: ConcreteFunction | None) -> None:
    exprs = self.exprs
    for node in topo(roots):
      if id(node) in exprs:
        continue
      self.ops.add(node.op)
      args = ",".join([str(exprs[id(arg)]) for arg in node.args])
      self.put(b"n", f"{node.op}|{args}|{self.type(node.type, owner)}|{node.name!r}|{node.lowering}".encode())
      if node.value is not None:
        self.value(node.value, owner)
      if node.attrs:
        self.put(b"@", str(len(node.attrs)).encode())
        for key in sorted(node.attrs):
          self.put(b"K", key.encode())
          self.value(node.attrs[key], owner)
      exprs[id(node)] = len(exprs)

  def function(self, fun: ConcreteFunction) -> int:
    serial = self.functions.get(id(fun))
    if serial is not None:
      return serial
    kind = type(fun)
    self.code_of(kind)
    self.put(b"f", f"{kind.__module__}.{kind.__qualname__}".encode())
    self.put(b"S", fun.name.encode())
    for part in (fun.input_names, fun.output_names, fun.device, fun.output_sparsities, fun.output_coloring_widths):
      self.value(part, fun)
    self.nodes(fun.inputs, fun)
    self.nodes(fun.outputs, fun)
    self.put(b"i", ",".join([str(self.exprs[id(e)]) for e in fun.inputs]).encode())
    self.put(b"u", ",".join([str(self.exprs[id(e)]) for e in fun.outputs]).encode())
    if fun.extern is not None:
      self.extern(fun)
    self.functions[id(fun)] = serial = len(self.functions)
    return serial

  def extern(self, fun: ConcreteFunction) -> None:
    """An extern body by everything code generation asks of it: the Functions its C calls, the
    sources it adds, the C it renders and what building that takes."""
    callee = fun.extern
    assert callee is not None
    self.code_of(type(callee))
    for dependency in callee.dependencies():
      self.put(b"d", str(self.function(dependency)).encode())
    for source in callee.extern_sources():
      self.put(b"x", source.raw_symbol.encode())
      self.put(b"x", source.source.encode())
      self.put(b"x", str(source.workspace_size).encode())
    symbol = c_ident(fun.name)
    self.put(b"r", "\n".join(callee.render(fun, ExternRenderCtx(symbol=symbol, raw_symbol=f"{symbol}_raw"))).encode())
    needs = callee.build_requirements(fun)
    for part in (needs.includes, needs.header_types, needs.source_blocks, needs.declarations, needs.libraries, needs.isolated, needs.versions):
      self.value(part, None)

  def rules(self) -> None:
    """Note the code behind every op the graph holds: its rules and the traits that are functions."""
    for op in sorted(self.ops):
      definition = op_def(op)
      for rule in (definition.numpy, *(getattr(definition, kind) for kind in _RULE_KINDS), *definition.traits.values()):
        for item in rule if isinstance(rule, tuple) else (rule,):
          if callable(item):
            self.code_of(item if hasattr(item, "__module__") else type(item))
          elif dataclasses.is_dataclass(item):
            self.code_of(type(item))


def graph_digest(fun: ConcreteFunction) -> tuple[str, frozenset[str]] | None:
  """A digest of everything rendering reads of ``fun``: its names, its graph (every node's op, type,
  name, value and attributes, with the Functions it calls taken the same way) and what its extern
  bodies render; with it, the packages whose code that rendering runs. None when the graph holds a
  value the walk does not know how to digest."""
  walk = _Walk()
  try:
    walk.function(fun)
    walk.rules()
  except Unkeyed:
    return None
  return walk.hash.hexdigest(), frozenset(walk.packages)


def _roots(package: str) -> list[str] | None:
  module = sys.modules.get(package)
  if module is None:
    return None
  paths = getattr(module, "__path__", None)
  if paths is not None:
    return sorted(str(path) for path in paths)
  file = getattr(module, "__file__", None)
  return None if file is None else [file]


def code_digest(packages: Iterable[str]) -> str | None:
  """A digest of the code of ``packages`` and of scaly as it stands on disk: every file's path and
  contents (for a file over a megabyte, a built library, its size and modification time), and the
  interpreter's and NumPy's versions.

  None when that cannot stand for the code this process runs: a file was written after scaly was
  loaded (a module imported before the edit runs the old code, one imported after it the new), or
  a package has no files to read."""
  digest = hashlib.sha256()
  digest.update(f"{sys.version}\0{np.__version__}\0".encode())
  for package in sorted({"scaly", *packages} - _VERSIONED):
    roots = _roots(package)
    if not roots:
      return None
    digest.update(f"{package}\0".encode())
    for root in roots:
      if not _scan(root, digest):
        return None
  return digest.hexdigest()


_READ_WHOLE = 1 << 20
# A file's digest by its path, size and modification time: read once per process, then only stat.
_CONTENTS: dict[tuple[str, int, int], bytes] = {}


def _scan(root: str, digest: Any) -> bool:
  """Add every file under ``root`` to ``digest``; False if one was written after scaly was loaded
  or cannot be read."""
  try:
    if os.path.isfile(root):
      entries = [(os.path.basename(root), root, os.stat(root))]
    else:
      entries = []
      stack = [root]
      while stack:
        directory = stack.pop()
        with os.scandir(directory) as found:
          for entry in found:
            if entry.is_dir(follow_symlinks=False):
              if entry.name != "__pycache__":
                stack.append(entry.path)
            else:
              entries.append((os.path.relpath(entry.path, root), entry.path, entry.stat()))
      if not entries:
        return False
    for name, path, stat in sorted(entries, key=lambda entry: entry[0]):
      if stat.st_mtime_ns >= LOADED_AT_NS:
        return False
      if stat.st_size > _READ_WHOLE:
        contents = f"{stat.st_size}\0{stat.st_mtime_ns}".encode()
      else:
        key = (path, stat.st_size, stat.st_mtime_ns)
        contents = _CONTENTS.get(key)
        if contents is None:
          with open(path, "rb") as file:
            contents = _CONTENTS[key] = hashlib.sha256(file.read()).digest()
      digest.update(f"{name}\0".encode())
      digest.update(contents)
  except OSError:
    return False
  return True


__all__ = ["Unkeyed", "code_digest", "graph_digest"]
