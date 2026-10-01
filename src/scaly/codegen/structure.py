"""The structural key of a Function: a digest of its graph, and of the code that would render it, taken without lowering anything, by which the JIT finds a library it already built."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib.util
import os
import struct
import sys
import sysconfig
import types
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..function import ConcreteFunction, Function
from ..function.extern import BuildRequirements, ExternCallee, ExternRenderCtx
from ..ir.expr import _RULE_KINDS, Expr, op_def, registered_ops, topo
from ..ir.types import DType, SparsityType, TensorType
from ..passes import lowering
from ..passes.program import pipeline
from ..utils import env
from ..utils.names import c_ident
from ..utils.options import default_options


class Unkeyed(Exception):
  """The walk met something it does not know how to digest. The Function then has no structural
  key and is rendered to find its library, as every Function was before there was one."""


_STDLIB = tuple(os.path.realpath(path) + os.sep for path in {sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")} if path)


def _versioned(package: str) -> bool:
  """Whether a version stands for ``package``'s code: NumPy itself, and the interpreter's own
  modules, told by where they live and not by their name, which a project's module may share."""
  module = sys.modules.get(package)
  if module is None:
    return False
  if module is np:
    return True
  file = getattr(module, "__file__", None)
  if file is None:
    return not hasattr(module, "__path__")  # built in or frozen; a namespace package has a path
  real = os.path.realpath(file)
  return real.startswith(_STDLIB) and "site-packages" not in real and "dist-packages" not in real


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
    # the Functions it calls anew each time it is asked, and a freed one's address is reused. The
    # graph holds the rest (a node its type and its callees), so the roots and those are enough.
    self.held: list[Any] = []

  def put(self, tag: bytes, payload: bytes = b"") -> None:
    # Tagged and length-prefixed: no two sequences of values share their bytes.
    self.hash.update(tag)
    self.hash.update(len(payload).to_bytes(8, "little"))
    self.hash.update(payload)

  def named(self, obj: Any) -> str:
    """The module and qualified name of a class or a function, its package noted: that package's
    source is part of the key. One defined inside a function has no name of its own (two such
    classes share one), so it gives no key. For a class or a builtin; a Python function is told by
    its code (``rule``)."""
    module, name = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str) or "<locals>" in name:
      raise Unkeyed(f"{obj!r} has no module-level name")
    # The name has to lead back to the object, as a pickle's does: a wrapper may carry the name
    # of what it wraps.
    found: Any = sys.modules.get(module)
    for part in name.split("."):
      found = getattr(found, part, None)
    if found is not obj:
      raise Unkeyed(f"{module}.{name} is not {obj!r}")
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
    included, since a dtype's own equality is its name alone, and written as plain integers,
    since a shape may hold NumPy's. A type of a subclass may hold more than the fields: no key."""
    serial = self.type_ids.get(id(t))
    if serial is None:
      if type(t) is not TensorType or type(t.dtype) is not DType or type(t.sparsity) not in (SparsityType, type(None)):
        raise Unkeyed(f"a type of a subclass, {type(t).__qualname__}")
      pattern = None
      if t.sparsity is not None:
        coordinates = np.asarray([t.sparsity.rows, t.sparsity.cols], dtype=np.int64).tobytes()
        pattern = (tuple(int(d) for d in t.sparsity.shape), coordinates)
      fields = (tuple(int(d) for d in t.shape), dataclasses.astuple(t.dtype), pattern, bool(t.diff))
      serial = self.types.get(fields)
      if serial is None:
        self.put(b"t", repr(fields[:2] + fields[3:]).encode())
        self.put(b"z", b"" if pattern is None else repr(pattern[0]).encode() + pattern[1])
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
    """A rule, a trait or a pass that is a function: which one. A Python function is told by its
    code, the module it was defined in and where: the name it carries may be another's (a
    decorator's wrapper takes the name of what it wraps). Its code is its package's files
    (``code_digest``), and its defaults are written, being values a file does not fix. What a
    closure, a bound method, a partial application or an object that is called carries is in no
    file, so those give no key. A rule that reads a global of its module that something sets at
    run time is not seen either: an extension keeps such state out of its rules."""
    if isinstance(fn, np.ufunc):
      if getattr(np, fn.__name__, None) is not fn:
        raise Unkeyed(f"the ufunc {fn.__name__} is not NumPy's")
      self.packages.add("numpy")
      self.put(b"c", f"numpy.{fn.__name__}".encode())
      return
    if not isinstance(fn, types.FunctionType):
      # A builtin, a class, or one of NumPy's dispatching functions: by a name that leads back to
      # it, which a method bound to an object's does not.
      self.put(b"c", self.named(fn).encode())
      return
    code, module = fn.__code__, fn.__globals__.get("__name__")
    defined = sys.modules.get(module) if isinstance(module, str) else None
    if defined is None or defined.__dict__ is not fn.__globals__ or fn.__closure__:
      raise Unkeyed(f"the rule {fn!r} is a closure, or not a function of a module's own")
    self.packages.add(module.partition(".")[0])
    self.put(b"c", f"{module}.{code.co_qualname}:{code.co_firstlineno}".encode())
    self.value(fn.__defaults__, None)
    self.value(fn.__kwdefaults__, None)

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
    extensions inserted in the pipeline, whether a loop's carry may be overwritten in place, and
    what fusion reads of the whole op registry, the program ops it does not duplicate."""
    for name, fn in pipeline():
      self.put(b"p", name.encode())
      self.rule(fn)
    self.put(b"k", b"1" if lowering.DONATE_CARRIES else b"0")
    traits = [op_def(op).traits for op in registered_ops()]
    for op in sorted(str(t["elementwise"]) for t in traits if t.get("expensive") and "elementwise" in t):
      self.put(b"e", op.encode())


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
  contents (for a file over a megabyte, a built library, its size and times), the bytecode
  Python keeps for a source where it is older than the source's last change, and the
  interpreter's, NumPy's and SciPy's versions. Hidden files and directories are not code.

  None when that cannot stand for the code this process runs: a file or a directory was written,
  replaced or relinked after the code was loaded or in the two seconds before, as fine as some
  file systems keep time (a module imported before the edit runs the old code, one imported after
  it the new), or a package has no files to read. Scaly's own code is loaded when scaly is;
  another package may have been imported at any time since the process started, and where the
  platform does not say when that was, it has no digest. A package is taken whole and alone: code
  in another package that a rule calls is not seen, so an extension keeps what shapes its C in the
  package that registers it."""
  import scipy

  digest = hashlib.sha256()
  digest.update(f"{sys.version}\0{np.__version__}\0{scipy.__version__}\0".encode())
  for package in sorted({"scaly", *packages}):
    if _versioned(package):
      continue
    roots = _roots(package)
    mark = env.LOADED_AT_NS if package == "scaly" else env.process_started_ns()
    if not roots or mark is None:
      return None
    digest.update(f"{package}\0".encode())
    for root in roots:
      if not _scan(root, digest, mark - _CLOCK_SLACK_NS):
        return None
  return digest.hexdigest()


_READ_WHOLE = 1 << 20
# A file's digest by its path, size and times: read once per process, then only stat.
_CONTENTS: dict[tuple[str, int, int, int], bytes] = {}
# A file system may keep times to two seconds: a file written that close before the code was loaded
# could have been written after it.
_CLOCK_SLACK_NS = 2_000_000_000


def _contents(path: str, stat: os.stat_result) -> bytes:
  key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
  found = _CONTENTS.get(key)
  if found is None:
    with open(path, "rb") as file:
      found = _CONTENTS[key] = hashlib.sha256(file.read()).digest()
  return found


def _accepted(cached: str, source: os.stat_result) -> bool:
  """Whether Python would load the bytecode at ``cached`` for a source with ``source``'s size and
  modification time without compiling it again: its header records both, or a hash of the source,
  which this does not check and so takes as accepted."""
  with open(cached, "rb") as file:
    header = file.read(16)
  if len(header) < 16:
    return False
  flags, recorded, size = struct.unpack("<III", header[4:])
  return bool(flags & 1) or (recorded, size) == (int(source.st_mtime) & 0xFFFFFFFF, source.st_size & 0xFFFFFFFF)


def _scan(root: str, digest: Any, mark: int) -> bool:
  """Add every file under ``root`` to ``digest``; False if a file, a link or a directory changed at
  ``mark`` or later, or something cannot be read."""

  def moved(stat: os.stat_result) -> bool:
    # The change time too: a copy that keeps the modification time of the file it replaces (an
    # archive, an installer) still moves that one.
    return max(stat.st_mtime_ns, stat.st_ctime_ns) >= mark

  try:
    if os.path.isfile(root):
      if moved(os.lstat(root)):
        return False
      entries = [(os.path.basename(root), root, os.stat(root))]
    else:
      entries = []
      stack = [root]
      top = os.stat(root)
      seen = {(top.st_dev, top.st_ino)}
      if moved(top) or moved(os.lstat(root)):
        return False
      while stack:
        directory = stack.pop()
        with os.scandir(directory) as found:
          for entry in found:
            if entry.name.startswith(".") or entry.name == "__pycache__":
              continue  # an editor's or the system's own, and bytecode, which goes with its source
            if entry.is_symlink() and not os.path.exists(entry.path):
              continue  # a link to nothing is no code
            stat = entry.stat()
            if entry.is_dir():
              # A directory moves when an entry is added, removed or renamed in it: a file deleted
              # or swapped in under a live process, or a link pointed elsewhere, leaves no other
              # trace.
              if moved(stat):
                return False
              if (stat.st_dev, stat.st_ino) not in seen:  # a link back into the tree is walked once
                seen.add((stat.st_dev, stat.st_ino))
                stack.append(entry.path)
            else:
              entries.append((os.path.relpath(entry.path, root), entry.path, stat))
      if not entries:
        return False
    for name, path, stat in sorted(entries, key=lambda entry: entry[0]):
      if moved(stat):
        return False
      digest.update(f"{name}\0".encode())
      if stat.st_size > _READ_WHOLE:
        digest.update(f"{stat.st_size}\0{stat.st_mtime_ns}\0{stat.st_ctime_ns}".encode())
        continue
      digest.update(_contents(path, stat))
      if name.endswith(".py"):
        # Python loads the bytecode it kept when the source's size and modification time are what
        # the bytecode recorded, so a source replaced with both kept runs as the old code.
        # Bytecode older than its source's last change that Python would still accept may be that
        # case, and says which code runs: it is digested too. Bytecode written since is the
        # source's own, and bytecode Python would compile again is not what runs.
        try:
          cached = importlib.util.cache_from_source(path)
          kept = os.stat(cached)
        except (FileNotFoundError, NotImplementedError, ValueError):
          continue
        if kept.st_mtime_ns < stat.st_ctime_ns and _accepted(cached, stat):
          digest.update(_contents(cached, kept))
  except OSError:
    return False
  return True


__all__ = ["GraphDigest", "Unkeyed", "code_digest", "graph_digest"]
