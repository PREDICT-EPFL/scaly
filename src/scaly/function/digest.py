"""Digests the JIT index is keyed on, taken without lowering: a Function's graph, and the installed source that renders it."""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib.util
import os
import struct
import sys
import weakref
from importlib.metadata import entry_points
from types import MappingProxyType
from typing import Any

import numpy as np

from ..ir.expr import Expr
from ..utils.env import LOADED_AT_NS
from .concrete import ConcreteFunction
from .tree import Tree


class Unkeyed(Exception):
  """The graph holds a value the digest does not know how to encode, so it has no digest."""


# The fields of a ConcreteFunction the digest reads, and the ones it leaves out: of the trees, which
# group the flat inputs and outputs, rendering reads only the names and the sparse layouts, and the
# memo is computed state.
_FIELDS = ("name", "inputs", "outputs", "output_sparsities", "output_coloring_widths", "device", "descriptor", "role", "rules")
_NOT_READ = ("input_tree", "output_tree", "_memo")
# What a tree holds: the digest reads the names and the layouts, while the declared types only reach
# the generated header, which the expression types already fix, the parts only group the leaves, and
# the declared layouts are what the frontend resolves into the layouts.
_TREE_READ = ("names", "layouts")
_TREE_NOT_READ = ("decls", "parts", "_layout_decls")

# A Function's own digest, with its callees in the order it meets them, and the digest of the whole
# graph it roots. A callee enters its caller's own digest by its place in that order alone.
_LOCAL: weakref.WeakKeyDictionary[ConcreteFunction, tuple[bytes, tuple[ConcreteFunction, ...]] | None] = weakref.WeakKeyDictionary()
_DIGESTS: weakref.WeakKeyDictionary[ConcreteFunction, str | None] = weakref.WeakKeyDictionary()


class _Walk:
  """One Function's own digest. Nodes and callees are numbered in the order the walk meets them,
  never by address."""

  def __init__(self) -> None:
    self.hash = hashlib.sha256()
    self.nodes: dict[int, int] = {}
    self.types: dict[int, bytes] = {}
    self.callees: dict[int, int] = {}
    self.called: list[ConcreteFunction] = []

  def put(self, tag: bytes, payload: bytes = b"") -> None:
    # Tagged and length-prefixed, so no two sequences of values share their bytes.
    self.hash.update(tag + len(payload).to_bytes(8, "little") + payload)

  def named(self, cls: type) -> bytes:
    """A class by a module-level name that leads back to it: a class defined inside a function shares
    its name with every other class defined there."""
    module, name = cls.__module__, cls.__qualname__
    found: Any = sys.modules.get(module)
    for part in name.split("."):
      found = getattr(found, part, None)
    if found is not cls:
      raise Unkeyed(f"{module}.{name} does not name {cls!r}")
    return f"{module}.{name}".encode()

  def value(self, v: Any) -> None:
    kind = type(v)
    if v is None or v is Ellipsis:
      self.put(b"N" if v is None else b".")
    elif kind is bool:
      self.put(b"B", b"1" if v else b"0")
    elif kind is int:
      self.put(b"I", hex(v).encode())
    elif kind is float:
      self.put(b"R", struct.pack("<d", v))  # by bits: -0.0 is not 0.0
    elif kind is str:
      self.put(b"S", v.encode("utf-8", "surrogatepass"))
    elif kind is np.ndarray or (isinstance(v, np.generic) and kind.__module__ == "numpy"):
      array = np.asarray(v)
      if array.dtype.kind not in "biufc":  # flags, integers, floats and complex numbers: their bytes are their value
        raise Unkeyed(f"an array of {array.dtype}")
      self.put(b"A" if kind is np.ndarray else b"G", f"{array.dtype.str}{array.shape}".encode())
      self.put(b"a", np.ascontiguousarray(array).tobytes())
    elif kind is tuple or kind is list:
      self.put(b"T" if kind is tuple else b"L", str(len(v)).encode())
      for item in v:
        self.value(item)
    elif kind is dict or kind is MappingProxyType:
      if not all(type(key) is str for key in v):
        raise Unkeyed("a mapping with keys that are not strings")
      self.put(b"D", str(len(v)).encode())
      for key, item in v.items():  # in its own order, which code reading it may follow
        self.put(b"K", key.encode())
        self.value(item)
    elif kind is slice:
      self.put(b"C")
      for item in (v.start, v.stop, v.step):
        self.value(item)
    elif kind is Expr:
      if id(v) not in self.nodes:
        self.graph((v,))
      self.put(b"E", str(self.nodes[id(v)]).encode())
    elif kind is ConcreteFunction:
      if id(v) not in self.callees:
        self.callees[id(v)] = len(self.called)
        self.called.append(v)
      self.put(b"F", str(self.callees[id(v)]).encode())
    elif isinstance(v, Tree):
      self.put(b"tree")
      self.value((v.names, v.decls, v.layouts))
    elif isinstance(v, enum.Enum):
      self.put(b"M", self.named(kind))
      self.value(v.value)
    elif dataclasses.is_dataclass(v) and not isinstance(v, type) and kind.__dataclass_params__.frozen:  # type: ignore[attr-defined]
      self.put(b"O", self.named(kind))
      for field in dataclasses.fields(v):
        if field.compare and field.init:  # a field its class keeps out of its value or computes is state, such as a cache
          self.put(b"K", field.name.encode())
          self.value(getattr(v, field.name))
    else:
      raise Unkeyed(f"a value of type {kind.__module__}.{kind.__qualname__}")

  def tensor_type(self, t: Any) -> bytes:
    """A node's type, encoded once per type object: a type holds a whole sparsity pattern."""
    found = self.types.get(id(t))
    if found is None:
      inner = _Walk()
      inner.value(t)
      found = self.types[id(t)] = inner.hash.digest()
    return found

  def graph(self, roots: tuple[Expr, ...]) -> None:
    # Number the outer graph before encoding attributes that may refer back into it.
    nodes = []
    stack = [(root, False) for root in reversed(roots)]
    while stack:
      node, visited = stack.pop()
      if id(node) in self.nodes:
        continue
      if visited:
        self.nodes[id(node)] = len(self.nodes)
        nodes.append(node)
      else:
        stack.append((node, True))
        stack.extend((arg, False) for arg in reversed(node.args))
    for node in nodes:
      args = ",".join(str(self.nodes[id(arg)]) for arg in node.args)
      self.put(b"n", f"{node.op}|{args}|{node.name!r}|{node.lowering}".encode("utf-8", "surrogatepass"))
      self.put(b"t", self.tensor_type(node.type))
      self.value(node.value)
      self.value(dict(sorted(node.attrs.items())))

  def function(self, fun: ConcreteFunction) -> None:
    self.graph(fun.inputs + fun.outputs)
    self.put(b"f", self.named(type(fun)))
    self.value((fun.input_names, fun.output_names, fun.input_tree.layouts, fun.output_tree.layouts))
    for name in _FIELDS:
      self.put(b"K", name.encode())
      self.value(getattr(fun, name))


def _local(fun: ConcreteFunction) -> tuple[bytes, tuple[ConcreteFunction, ...]] | None:
  if fun not in _LOCAL:
    walk = _Walk()
    try:
      walk.function(fun)
      _LOCAL[fun] = (walk.hash.digest(), tuple(walk.called))
    except Unkeyed:
      _LOCAL[fun] = None
  return _LOCAL[fun]


def graph_digest(fun: ConcreteFunction) -> str | None:
  """A digest of everything rendering reads of ``fun``: its names, every node's op, type, name,
  lowering hint, value and attributes, and every Function it reaches, each numbered in the order a
  walk from ``fun`` meets it, since rendering tells one Function called twice from two equal ones.
  Memoized per Function, which is immutable. None when the graph holds a value the digest cannot
  encode."""
  if fun not in _DIGESTS:
    numbers: dict[int, int] = {}
    reached: list[tuple[bytes, tuple[ConcreteFunction, ...]]] = []
    stack = [fun]
    while stack:
      top = stack.pop()
      if id(top) in numbers:
        continue
      local = _local(top)
      if local is None:
        _DIGESTS[fun] = None
        return None
      numbers[id(top)] = len(numbers)
      reached.append(local)
      stack.extend(reversed(local[1]))
    digest = hashlib.sha256()
    for own, called in reached:
      digest.update(own + ",".join(str(numbers[id(callee)]) for callee in called).encode() + b";")
    _DIGESTS[fun] = digest.hexdigest()
  return _DIGESTS[fun]


# A file system may keep times to two seconds, so a file written that close before scaly was loaded
# may have been written after it.
_CLOCK_SLACK_NS = 2_000_000_000
# A file over a megabyte, a built library, is taken by its size and times; its content is the solver
# fingerprint's.
_READ_WHOLE = 1 << 20
_CONTENTS: dict[tuple[str, int, int, int], bytes] = {}


if sys.platform == "win32":
  import ctypes
  from ctypes import wintypes

  class _FileBasicInfo(ctypes.Structure):
    _fields_ = [
      ("CreationTime", ctypes.c_int64),
      ("LastAccessTime", ctypes.c_int64),
      ("LastWriteTime", ctypes.c_int64),
      ("ChangeTime", ctypes.c_int64),
      ("FileAttributes", wintypes.DWORD),
    ]

  _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
  _create_file = _kernel32.CreateFileW
  _create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
  _create_file.restype = wintypes.HANDLE
  _basic_info = _kernel32.GetFileInformationByHandleEx
  _basic_info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.POINTER(_FileBasicInfo), wintypes.DWORD]
  _basic_info.restype = wintypes.BOOL
  _close_handle = _kernel32.CloseHandle
  _close_handle.argtypes = [wintypes.HANDLE]
  _close_handle.restype = wintypes.BOOL


def _source_stat(path: str, stat: os.stat_result, *, follow_symlinks: bool = True) -> os.stat_result:
  if sys.platform != "win32":
    return stat
  # FILE_READ_ATTRIBUTES, all sharing modes, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS
  # (directories too), and FILE_FLAG_OPEN_REPARSE_POINT when checking a link itself.
  flags = 0x02000000 | (0 if follow_symlinks else 0x00200000)
  handle = _create_file(path, 0x80, 7, None, 3, flags, None)
  if handle == wintypes.HANDLE(-1).value:
    raise ctypes.WinError(ctypes.get_last_error())
  try:
    info = _FileBasicInfo()
    if not _basic_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):  # FileBasicInfo
      raise ctypes.WinError(ctypes.get_last_error())
    if info.ChangeTime == 0:
      raise OSError("the file system supplies no change time")
    changed = (info.ChangeTime - 116444736000000000) * 100  # 100 ns since 1601 to Unix nanoseconds
    return os.stat_result(
      stat,
      {
        "st_atime": stat.st_atime,
        "st_atime_ns": stat.st_atime_ns,
        "st_mtime": stat.st_mtime,
        "st_mtime_ns": stat.st_mtime_ns,
        "st_ctime": changed / 1e9,
        "st_ctime_ns": changed,
      },
    )
  finally:
    _close_handle(handle)


@functools.cache
def _packages() -> tuple[tuple[str, tuple[str, ...]], ...]:
  """Scaly and every installed solver plugin, the packages whose code renders a Function, with their directories."""
  found = []
  for package in sorted({"scaly", *(ep.module.partition(".")[0] for ep in entry_points(group="scaly.solvers"))}):
    spec = importlib.util.find_spec(package)
    found.append((package, tuple(sorted(spec.submodule_search_locations or ())) if spec is not None else ()))
  return tuple(found)


def source_digest() -> str | None:
  """A digest of the installed source of scaly and its solver plugins: every file's path and
  contents, and the interpreter's and NumPy's versions. Each file is read once per process.

  None when the files on disk may not be the code this process runs: a file, a link or a directory
  changed since scaly was loaded (plugins import scaly before anything of their own runs), or
  bytecode older than its source's last change that Python would still load in its place."""
  digest = hashlib.sha256(f"{sys.version}\0{np.__version__}\0".encode())
  mark = LOADED_AT_NS - _CLOCK_SLACK_NS
  for package, roots in _packages():
    digest.update(f"{package}\0".encode())
    if not roots or not all(_scan(root, digest, mark) for root in roots):
      return None
  return digest.hexdigest()


def _scan(root: str, digest: Any, mark: int) -> bool:
  """Add every file under ``root`` to ``digest``; False when something changed at ``mark`` or later."""

  def moved(stat: os.stat_result) -> bool:
    # The change time too: a copy that keeps the modification time of the file it replaces moves it.
    return max(stat.st_mtime_ns, stat.st_ctime_ns) >= mark

  def moved_directory(directory: str) -> bool:
    # A directory changes when an entry is removed from it, which leaves no other trace, and its
    # change time alone moves when it is renamed into place. Python creating `__pycache__` for a
    # module imported for the first time moves both times to one instant, no later than the cache's
    # own: that change is not a swap. A file deleted before bytecode is next written there passes.
    if os.path.islink(directory) and moved(_source_stat(directory, os.lstat(directory), follow_symlinks=False)):
      return True
    stat = _source_stat(directory, os.stat(directory))
    if not moved(stat):
      return False
    try:
      cache = os.stat(os.path.join(directory, "__pycache__"))
    except FileNotFoundError:
      return True
    renamed = stat.st_ctime_ns != stat.st_mtime_ns
    return renamed or cache.st_mtime_ns < stat.st_mtime_ns

  try:
    files = []
    stack = [(root, "")]
    while stack:
      directory, prefix = stack.pop()
      if moved_directory(directory):
        return False
      with os.scandir(directory) as entries:
        for entry in entries:
          if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
          if entry.is_symlink() and moved(_source_stat(entry.path, entry.stat(follow_symlinks=False), follow_symlinks=False)):
            return False  # a link pointed elsewhere
          if entry.is_dir():
            stack.append((entry.path, f"{prefix}{entry.name}/"))
          else:
            files.append((prefix + entry.name, entry.path, _source_stat(entry.path, entry.stat())))
    for name, path, stat in sorted(files):
      if moved(stat):
        return False
      digest.update(f"{name}\0".encode())
      if stat.st_size > _READ_WHOLE:
        digest.update(f"{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
        continue
      key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
      if key not in _CONTENTS:
        with open(path, "rb") as file:
          _CONTENTS[key] = hashlib.sha256(file.read()).digest()
      digest.update(_CONTENTS[key])
      if name.endswith(".py") and _stale_bytecode(path, stat):
        return False
  except OSError:
    return False
  return True


_BYTECODE: dict[tuple[str, int, int, int, int, int], bool] = {}


def _stale_bytecode(source: str, stat: os.stat_result) -> bool:
  """Whether Python would load bytecode that is not ``source``'s: bytecode written before a source
  replaced with its size and modification time kept, or bytecode keyed by a source hash that Python
  does not check and that is not this source's."""
  try:
    cached = importlib.util.cache_from_source(source)
    kept = os.stat(cached)
  except (OSError, NotImplementedError, ValueError):
    return False
  key = (source, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, kept.st_size, kept.st_mtime_ns)
  if key not in _BYTECODE:
    with open(cached, "rb") as file:
      header = file.read(16)
    flags = int.from_bytes(header[4:8], "little") if len(header) == 16 else 0
    if flags & 1:  # keyed by the source's hash; bit 2 makes Python check it
      with open(source, "rb") as file:
        _BYTECODE[key] = not flags & 2 and header[8:] != importlib.util.source_hash(file.read())
    else:
      recorded = (int.from_bytes(header[8:12], "little"), int.from_bytes(header[12:16], "little"))
      written_before = kept.st_mtime_ns < stat.st_ctime_ns and len(header) == 16
      _BYTECODE[key] = written_before and recorded == (int(stat.st_mtime) & 0xFFFFFFFF, stat.st_size & 0xFFFFFFFF)
  return _BYTECODE[key]
