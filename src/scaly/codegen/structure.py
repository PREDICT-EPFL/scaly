"""The structural key of a Function: a digest of its graph, and of the code that would render it, taken without lowering anything, by which the JIT finds a library it already built."""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import os
import struct
import sys
import types
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..function import ConcreteFunction, Function
from ..function.extern import BuildRequirements, ExternCallee, ExternRenderCtx
from ..ir.expr import _RULE_KINDS, Expr, op_def, topo
from ..ir.types import DType, TensorType
from ..passes import lowering
from ..passes.program import pipeline
from ..utils import env
from ..utils.names import c_ident
from ..utils.options import default_options


class Unkeyed(Exception):
  """The walk met something it does not know how to digest. The Function then has no structural
  key and is rendered to find its library, as every Function was before there was one."""


# Packages whose code a version stands for: the interpreter's and NumPy's go into every key.
_VERSIONED = frozenset({"numpy", *sys.stdlib_module_names})
# The array element kinds whose bytes are their value: flags, integers, floats and complex numbers.
_PLAIN_KINDS = "biufc"


@dataclasses.dataclass(frozen=True, slots=True)
class GraphDigest:
  """What ``graph_digest`` found: the digest, the packages whose code the rendering would run, and
  what each extern body needs to build, in the order met."""

  digest: str
  packages: frozenset[str]
  requirements: tuple[BuildRequirements, ...]


class _Walk:
  """One digest of a graph. Every node, type and Function is written once, in the order a
  depth-first walk from the root first meets it, and referred to afterwards by its number in that
  order, so two graphs built the same way give the same bytes whatever their objects' addresses."""

  def __init__(self) -> None:
    self.hash = hashlib.sha256()
    self.exprs: dict[int, int] = {}
    self.functions: dict[int, int] = {}
    self.types: dict[tuple[Any, ...], int] = {}
    self.type_ids: dict[int, int] = {}
    self.dtypes: dict[int, tuple[Any, ...]] = {}
    self.ops: set[str] = set()
    self.packages: set[str] = set()
    self.requirements: list[BuildRequirements] = []
    # Everything numbered by its address stays alive until the walk ends: an extern body may build
    # the Functions it calls anew each time it is asked, and a freed one's address is reused.
    self.held: list[Any] = []

  def put(self, tag: bytes, payload: bytes = b"") -> None:
    # Tagged and length-prefixed: no two sequences of values share their bytes.
    self.hash.update(tag)
    self.hash.update(len(payload).to_bytes(8, "little"))
    self.hash.update(payload)

  def named(self, obj: Any) -> str:
    """The module and qualified name of a class or a function, its package noted: that package's
    source is part of the key. One defined inside a function has no name of its own (two such
    classes share one), so it gives no key."""
    module, name = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str) or "<locals>" in name:
      raise Unkeyed(f"{obj!r} has no module-level name")
    self.packages.add(module.partition(".")[0])
    return f"{module}.{name}"

  def value(self, v: Any, owner: ConcreteFunction | None) -> None:
    """Write an attribute's value. ``owner`` is the Function whose graph holds it."""
    kind = type(v)
    if v is None:
      self.put(b"N")
    elif kind is bool:
      self.put(b"B", b"1" if v else b"0")
    elif kind is int:
      self.put(b"I", hex(v).encode())
    elif kind is float:
      self.put(b"R", struct.pack("<d", v))  # by bits: -0.0 is not 0.0, and a NaN keeps its payload
    elif kind is str:
      self.put(b"S", v.encode("utf-8", "surrogatepass"))
    elif kind is bytes:
      self.put(b"Y", v)
    elif kind is np.ndarray:
      # A plain array of numbers: a subclass (a masked array) or a structured one holds more than its bytes.
      if v.dtype.kind not in _PLAIN_KINDS:
        raise Unkeyed(f"an array of {v.dtype}")
      self.put(b"A", f"{v.dtype.str}{v.shape}".encode())
      # Its bytes in C order, through the buffer so that a large constant is not copied; the dtype
      # and shape above say how many there are.
      self.hash.update(np.ascontiguousarray(v))
    elif isinstance(v, np.generic) and v.dtype.kind in _PLAIN_KINDS:
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
      for key, item in v.items():  # in its own order, which code reading it may follow
        self.put(b"K", key.encode("utf-8", "surrogatepass"))
        self.value(item, owner)
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
      self.put(b"M", f"{self.named(kind)}.{v.name}".encode())
    elif isinstance(v, ExternCallee):
      # An extern body is digested with the Function that owns it (``extern``), by what it renders.
      # Equal to the owner's, not the same object: equal bodies intern to one node, which holds
      # the body of the Function built first.
      if owner is None or owner.extern != v:
        raise Unkeyed("an extern callee outside the Function it is the body of")
      self.put(b"X")
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
      self.record(v, owner)
    else:
      raise Unkeyed(f"a value of type {kind.__module__}.{kind.__qualname__}")

  def record(self, v: Any, owner: ConcreteFunction | None) -> None:
    """A dataclass instance (a device, a sparsity pattern) by its class and its fields, written
    out wherever it occurs: whether two places share one object is not part of the graph."""
    self.put(b"O", self.named(type(v)).encode())
    for field in dataclasses.fields(v):
      if not field.compare:
        # A field its own class leaves out of equality may still shape the code; nothing says.
        raise Unkeyed(f"{type(v).__qualname__}.{field.name} is not part of its value")
      self.put(b"K", field.name.encode())
      self.value(getattr(v, field.name), owner)

  def type(self, t: TensorType, owner: ConcreteFunction | None) -> int:
    """A node's type by number. Types with the same fields are one, as they are to the graph:
    interning compares a node's type by value, so which of two equal objects a node holds is an
    accident of what the process built before. The fields are compared one by one, a dtype's
    included, since a dtype's own equality is its name alone."""
    serial = self.type_ids.get(id(t))
    if serial is None:
      dtype = self.dtypes.get(id(t.dtype))
      if dtype is None:
        # A dtype of another class has fields the walk cannot name: it stands alone.
        dtype = self.dtypes[id(t.dtype)] = dataclasses.astuple(t.dtype) if type(t.dtype) is DType else (object(),)
        self.held.append(t.dtype)
      pattern = None if t.sparsity is None else (type(t.sparsity), t.sparsity.shape, t.sparsity.rows, t.sparsity.cols)
      fields = (type(t), t.shape, dtype, pattern, t.diff)
      serial = self.types.get(fields)
      if serial is None:
        self.record(t, owner)
        serial = self.types[fields] = len(self.types)
      self.type_ids[id(t)] = serial
      self.held.append(t)
    return serial

  def nodes(self, roots: Iterable[Expr], owner: ConcreteFunction | None) -> None:
    exprs = self.exprs
    roots = tuple(roots)
    self.held.append(roots)
    for node in topo(roots):
      if id(node) in exprs:
        continue
      self.ops.add(node.op)
      args = ",".join([str(exprs[id(arg)]) for arg in node.args])
      self.put(b"n", f"{node.op}|{args}|{self.type(node.type, owner)}|{node.name!r}|{node.lowering}".encode("utf-8", "surrogatepass"))
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
    self.held.append(fun)
    self.put(b"f", self.named(type(fun)).encode())
    self.put(b"S", fun.name.encode("utf-8", "surrogatepass"))
    # What lowering and the C body read of a Function. Its output patterns and coloring widths go
    # to the header alone, which the JIT does not compile.
    for part in (fun.input_names, fun.output_names, fun.device):
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
    """An extern body by everything code generation asks of it for the C body and its build: the
    Functions its C calls, the sources it adds, the C it renders and what building that takes."""
    callee = fun.extern
    assert callee is not None
    self.put(b"x", self.named(type(callee)).encode())
    symbol = c_ident(fun.name)
    try:
      dependencies = callee.dependencies()
      sources = callee.extern_sources()
      rendered = callee.render(fun, ExternRenderCtx(symbol=symbol, raw_symbol=f"{symbol}_raw"))
      needs = callee.build_requirements(fun)
    except Exception as error:  # the callee's own: rendering raises it again, with its own account
      raise Unkeyed(f"the extern body of {fun.name!r} raised {error!r}") from error
    self.held.append(dependencies)
    for dependency in dependencies:
      self.put(b"d", str(self.function(dependency)).encode())
    for source in sources:
      self.put(b"s", source.raw_symbol.encode())
      self.put(b"s", source.source.encode())
      self.put(b"s", str(source.workspace_size).encode())
    self.put(b"r", "\n".join(rendered).encode())
    # The libraries reach the build through the link flags they resolve to, which the JIT's key holds.
    for part in (needs.includes, needs.source_blocks, needs.isolated, needs.versions):
      self.value(part, None)
    self.requirements.append(needs)

  def rule(self, fn: Any) -> None:
    """A rule, a trait or a pass that is a function: which one, by module and name. Its code is its
    package's files (``code_digest``); what a closure, a bound method or a partial application
    carries is in no file, so outside scaly, whose own are fixed by its code, such a rule gives
    no key."""
    if isinstance(fn, np.ufunc):
      self.packages.add("numpy")
      self.put(b"c", f"numpy.{fn.__name__}".encode())
      return
    if isinstance(fn, functools.partial):
      raise Unkeyed(f"the rule {fn!r} is a partial application")
    name = self.named(fn)
    bound = getattr(fn, "__self__", None)  # a builtin's is its module, which is no state
    if (getattr(fn, "__closure__", None) or not (bound is None or isinstance(bound, types.ModuleType))) and not name.startswith("scaly."):
      raise Unkeyed(f"the rule {name} carries state of its own")
    self.put(b"c", name.encode())

  def definitions(self) -> None:
    """The definition of every op the graph holds: its arity, its rules and its traits. A trait
    that is a value (the program op an elementwise op lowers to) is written; one that is a function
    is named, like a rule."""
    for op in sorted(self.ops):
      definition = op_def(op)
      self.put(b"q", f"{op}|{definition.arity}|{definition.differentiable}".encode())
      for kind in ("numpy", *_RULE_KINDS):
        rule = getattr(definition, kind)
        for item in rule if isinstance(rule, tuple) else (rule,):
          self.put(b"K", kind.encode())
          if item is None:
            self.put(b"N")
          elif callable(item):
            self.rule(item)
          else:
            self.put(b"c", self.named(type(item)).encode())  # a verifier's rule object: by its class
      for name in sorted(definition.traits):
        trait = definition.traits[name]
        self.put(b"K", name.encode())
        if callable(trait) and not isinstance(trait, (enum.Enum, type)):
          self.rule(trait)
        else:
          self.value(trait, None)

  def switches(self) -> None:
    """What shapes the lowering of every Function and is no part of any graph: the passes
    extensions inserted in the pipeline, and whether a loop's carry may be overwritten in place."""
    for name, fn in pipeline():
      self.put(b"p", name.encode())
      self.rule(fn)
    self.put(b"k", b"1" if lowering.DONATE_CARRIES else b"0")


def graph_digest(fun: ConcreteFunction) -> GraphDigest | None:
  """A digest of everything rendering reads of ``fun``: its names, its graph (every node's op, type,
  name, value and attributes, with the Functions it calls taken the same way), what its extern
  bodies render, the definitions of the ops it holds and the pass pipeline. None when the graph
  holds something the walk does not know how to digest, or is nested too deep for it."""
  walk = _Walk()
  try:
    with default_options():  # as rendering runs: an extern body's C must not follow an option in force
      walk.function(fun)
    walk.definitions()
    walk.switches()
  except (Unkeyed, RecursionError):
    return None
  return GraphDigest(walk.hash.hexdigest(), frozenset(walk.packages), tuple(walk.requirements))


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
  contents (for a file over a megabyte, a built library, its size and times), and the
  interpreter's and NumPy's versions. Hidden files and directories and bytecode are not code.

  None when that cannot stand for the code this process runs: a file was written or replaced
  after scaly was loaded (a module imported before the edit runs the old code, one imported after
  it the new), or a package has no files to read. A package is taken whole and alone: code in
  another package that a rule calls is not seen, so an extension keeps what shapes its C in the
  package that registers it."""
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
# A file's digest by its path, size and times: read once per process, then only stat.
_CONTENTS: dict[tuple[str, int, int, int], bytes] = {}
# A file system may keep times to two seconds: a file written that close before the load could have
# been written after it.
_CLOCK_SLACK_NS = 2_000_000_000


def _scan(root: str, digest: Any) -> bool:
  """Add every file under ``root`` to ``digest``; False if one changed after scaly was loaded or
  cannot be read."""
  try:
    if os.path.isfile(root):
      entries = [(os.path.basename(root), root, os.stat(root))]
    else:
      entries = []
      stack = [root]
      seen = set()
      while stack:
        directory = stack.pop()
        with os.scandir(directory) as found:
          for entry in found:
            if entry.name.startswith(".") or entry.name == "__pycache__":
              continue  # an editor's or the system's own, and bytecode
            if entry.is_symlink() and not os.path.exists(entry.path):
              continue  # a link to nothing is no code
            stat = entry.stat()
            if entry.is_dir():
              if (stat.st_dev, stat.st_ino) not in seen:  # a link back into the tree is walked once
                seen.add((stat.st_dev, stat.st_ino))
                stack.append(entry.path)
            else:
              entries.append((os.path.relpath(entry.path, root), entry.path, stat))
      if not entries:
        return False
    loaded = env.LOADED_AT_NS - _CLOCK_SLACK_NS
    for name, path, stat in sorted(entries, key=lambda entry: entry[0]):
      # The change time too: a copy that keeps the modification time of the file it replaces
      # (an archive, an installer) still moves that one.
      if max(stat.st_mtime_ns, stat.st_ctime_ns) >= loaded:
        return False
      if stat.st_size > _READ_WHOLE:
        contents = f"{stat.st_size}\0{stat.st_mtime_ns}".encode()
      else:
        key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        contents = _CONTENTS.get(key)
        if contents is None:
          with open(path, "rb") as file:
            contents = _CONTENTS[key] = hashlib.sha256(file.read()).digest()
      digest.update(f"{name}\0".encode())
      digest.update(contents)
  except OSError:
    return False
  return True


__all__ = ["GraphDigest", "Unkeyed", "code_digest", "graph_digest"]
