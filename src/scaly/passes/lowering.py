"""Lower expression IR (``Expr`` / ``Function``) into Program IR (``ProgramNode``).

This module + ``codegen/c.py`` are the **sole** CPU path to C (see
``docs/how_it_works/lowering.md``); the legacy tape-based scalar renderer is gone.
The one sanctioned non-Program-IR escape is a Function with an extern body
(``function/extern.py``), a solver for instance: its C comes from the callee, and even
there the Functions that C calls lower through here.

Dispatch is a **registry** keyed by expression ``ExprOp``: each op's lowering is a
self-contained rule registered with ``@lowers(...)``. Adding/deepening an op (or,
later, a GPU schedule) is a local change — a new rule, not an edit to a monolith.

Covered: elementwise unary/binary (with numpy broadcasting), ``RESHAPE`` (alias),
``CONST`` (any size, via ``const_buffer``), general ``SLICE`` (integer / multi-dim /
strided), ``SUM``, ``MATMUL`` (rank <= 2), ``TRANSPOSE`` (rank <= 4), ``GATHER`` /
``SEGMENT_REDUCE`` (any size, affine indices as arithmetic on the trip index and
whatever is left as a ``static const`` table), ``STACK`` / ``CONCAT`` (any axis),
``CALL`` (multi-PROC, deduped) and ``VMAP``; a ``CALL`` to a Function with an extern body is
opaque (see ``lower_function``). The tracking and unbumpercars workloads (forward + ``jac`` +
``spjac``) render and match generated-code / external numeric references. Deferred (re-land from the reference branch):
GPU placement and the new ops tracked in the migration roadmap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..ir import program as p
from ..ir.expr import (
  CALLEE_OPS,
  Expr,
  ExprOp,
  callees_of,
  define_rules,
  define_traits,
  expr_has_trait,
  has_trait,
  op_def,
  put_lanes,
  topo,
)
from ..function import ConcreteFunction, Function
from .arith import constant
from .program import ProgramObserver, optimize_program
from ..ir.program import ProgramNode, ProgramOp, RangeKind
from ..ir.program_spec import verify_program
from ..ir.target import Target, resolve_target
from ..ir.types import DeviceSpec, DType, dtypes
from .affine import affine_index_map
from .expr import cse_many, simplify
from ..utils.names import c_ident


class LoweringError(NotImplementedError):
  """The lowerer (or Program-IR renderer) does not yet cover this op / case."""


LowerRule = Callable[["LowerCtx", Expr], None]


def lowers(*ops: str) -> Callable[[LowerRule], LowerRule]:
  """Define ``fn`` as the lowering rule (``OpDef.lower``) of each expression op in ``ops``."""

  def deco(fn: LowerRule) -> LowerRule:
    for op in ops:
      define_rules(op, lower=fn)
    return fn

  return deco


ExprObserver = Callable[[str, ConcreteFunction], None]


def lower_function(
  fun: Function, observe: ProgramObserver | None = None, observe_expr: ExprObserver | None = None, *, target: Target | str | None = None
) -> ProgramNode:
  """Lower ``fun`` into a Program IR ``PROGRAM`` node (verified before return).

  ``target`` (a ``Target``, a preset name, or None for the target in force) is the processor the
  code is tuned for. Lowering rules read it as ``LowerCtx.target``, and it is recorded on the
  ``PROGRAM`` as its ``tuned_for`` attribute for the program passes.

  Host placement only for now: the returned PROGRAM holds every lowered callee
  PROC in topological order followed by ``fun``'s main PROC last. Non-host
  placement raises ``LoweringError`` — GPU backends re-land from the reference
  branch after CPU parity (see ``internal/notes/program_ir_migration.md``).

  A Function with an extern body (``Function.extern``, a solver for instance) is **opaque**: its
  ``ExprOp.EXTERN_CALL`` body is not lowered, since the callee renders its own C (rule 6), but the
  Functions that C calls, its ``dependencies()``, *are* lowered to PROCs and called as
  ``<name>_raw``. The extern-to-dependency map is recorded on the PROGRAM (``extern_deps``), with
  the workspace its hand-written sources declare (``extern_workspace``), so ``pack_workspace`` can
  size the caller's ``w[]`` to fit them and the CALL into the extern gets ``callee_needs_w`` right.
  """
  fun = fun.concrete
  target = resolve_target(target)
  if fun.device.kind != "host":
    raise LoweringError(f"non-host placement {fun.device} is not lowered yet (GPU backends are deferred to a later migration step)")
  _check_function_names(fun)
  callees: dict[str, ProgramNode] = {}
  extern_fns: dict[str, ConcreteFunction] = {}

  if fun.extern is not None:
    extern_fns[fun.name] = fun
    for dependency in fun.extern.dependencies():
      if dependency.name not in callees:
        callees[dependency.name] = _lower_to_proc(dependency, callees, extern_fns, target, observe_expr=observe_expr)
    prog = p.program([*callees.values()])
    prog = ProgramNode(ProgramOp.PROGRAM, prog.args, {**prog.attrs, "extern_root": fun.name}, prog.dtype)
  else:
    root = _lower_to_proc(fun, callees, extern_fns, target, observe_expr=observe_expr, entry=True)
    prog = p.program([*callees.values(), root])
  prog = ProgramNode(ProgramOp.PROGRAM, prog.args, {**prog.attrs, "tuned_for": target}, prog.dtype)
  if extern_fns:
    externs = {name: ef.extern for name, ef in extern_fns.items() if ef.extern is not None}
    extern_deps = {name: tuple(d.name for d in extern.dependencies()) for name, extern in externs.items()}
    extern_workspace = {name: max((s.workspace_size for s in extern.extern_sources()), default=0) for name, extern in externs.items()}
    prog = ProgramNode(ProgramOp.PROGRAM, prog.args, {**prog.attrs, "extern_deps": extern_deps, "extern_workspace": extern_workspace}, prog.dtype)
  if observe is not None:
    observe("lowered", prog)
  prog = optimize_program(prog, observe=observe)
  verify_program(prog)
  return prog


def _same_function(a: ConcreteFunction, b: ConcreteFunction) -> bool:
  """Two Function objects that would lower to the same procedure: interned Exprs make equal graphs
  the same objects, so identity of inputs and outputs is structural equality."""
  return a is b or (
    a.input_names == b.input_names
    and a.output_names == b.output_names
    and all(x is y for x, y in zip(a.inputs, b.inputs, strict=True))
    and len(a.outputs) == len(b.outputs)
    and all(x is y for x, y in zip(a.outputs, b.outputs, strict=True))
  )


def _check_function_names(fun: ConcreteFunction) -> None:
  """Refuse two different Functions with one name anywhere in ``fun``'s call tree.

  Procedures are emitted once per name, so the second would silently run the first one's body."""
  owners: dict[str, ConcreteFunction] = {}
  todo = [fun]
  while todo:
    f = todo.pop()
    # Keyed by the C spelling: ``f:_3`` and ``f__3`` are two names but one C symbol.
    seen = owners.get(c_ident(f.name))
    if seen is not None:
      if not _same_function(seen, f):
        named = f"named {f.name!r}" if seen.name == f.name else f"named {seen.name!r} and {f.name!r}, both {c_ident(f.name)!r} in C,"
        raise LoweringError(
          f"two different Functions are {named} in the graph of {fun.name!r}; generated code has one procedure per name, so give them distinct names"
        )
      continue
    owners[c_ident(f.name)] = f
    if f.extern is not None:
      todo.extend(f.extern.dependencies())
    for node in topo(f.outputs):
      if node.op in CALLEE_OPS:
        todo.extend(callees_of(node))


def main_proc(program_node: ProgramNode) -> ProgramNode:
  """Return the main (last) PROC inside a lowered PROGRAM."""
  if program_node.op != ProgramOp.PROGRAM:
    raise TypeError(f"main_proc expects a PROGRAM, got {program_node.op}")
  pc = int(program_node.attrs.get("proc_count", 0))
  if pc <= 0:
    raise ValueError("lowered program contains no procs")
  return program_node.args[pc - 1]


def _shape_or_scalar(shape: tuple[int, ...]) -> tuple[int, ...]:
  return shape or (1,)


def _size_of(shape: tuple[int, ...]) -> int:
  n = 1
  for d in shape:
    n *= int(d)
  return n


def _normalize_function(fun: ConcreteFunction) -> ConcreteFunction:
  outputs = fun.outputs
  for _ in range(4):
    normalized = cse_many(simplify(output) for output in outputs)
    if all(new is old for new, old in zip(normalized, outputs, strict=True)):
      break
    outputs = normalized
  return fun._with_outputs(outputs)


def _lower_to_proc(
  fun: ConcreteFunction,
  callees: dict[str, ProgramNode],
  extern_fns: dict[str, ConcreteFunction],
  target: Target,
  *,
  observe_expr: ExprObserver | None = None,
  entry: bool = False,
  in_place: bool = False,
) -> ProgramNode:
  lowering = fun._effective_lowering()
  fun = _normalize_function(fun)
  if observe_expr is not None:
    observe_expr("normalized", fun)
  # The chain is found on exactly the graph being lowered, so its node ids are this graph's. The
  # caller has proven the updates safe, for the body alone or for the loop's own index tables.
  chain = tuple(e.id for e in update_chain(fun) or ()) if in_place else None
  if in_place and not chain:
    raise LoweringError(f"{fun.name!r} was lowered in place but its carry is not an update chain")
  ctx = LowerCtx(fun, callees, extern_fns, target, observe_expr, entry=entry, in_place=chain)
  ctx.emit_inputs()
  ctx.register_outputs()
  ctx.emit_body()
  ctx.emit_outputs()
  proc = p.proc(fun.name, ctx.params, ctx.statements)
  # ``input_count`` lets the renderer ``const``-qualify the first N (input) params of a ``_raw``
  # callee; emit_inputs runs before register_outputs, so inputs are the leading params.
  nodes = topo(fun.outputs)
  # Narrower stores round or truncate; scalar substitution must not erase those conversions.
  return ProgramNode(
    ProgramOp.PROC,
    proc.args,
    {
      **proc.attrs,
      "input_count": len(fun.inputs),
      "lowering": lowering,
      # A procedure may expand on its own and inside an expanding caller. Hoisted prologues use
      # "inline" because they only expand with their caller.
      # An in-place procedure reads its carry output before writing it, which only the caller's
      # aliasing makes defined; scalar expansion models the two as separate buffers, so it is off.
      "scalarize_mode": "procedure"
      if not in_place
      and all(n.type.dtype in _SCALARIZABLE for n in (*fun.inputs, *nodes))
      # Scalar expansion follows every address at generation time; a run-time index has none.
      and not any(expr_has_trait(n, "runtime_index") for n in nodes)
      and (lowering == "scalar" or (lowering == "auto" and not any(_product_in_loops(n, target) for n in nodes)))
      else "disabled",
      **({"in_place": True} if in_place else {}),
      # The entry point: automatic scalar expansion keeps its call boundaries (``scalarize``).
      **({"entry": True} if entry else {}),
    },
    proc.dtype,
  )


def _product_in_loops(node: Expr, target: Target) -> bool:
  """Whether ``node`` is a matrix product that keeps its procedure out of automatic scalar
  expansion: a matrix times a matrix over a reduction of four terms or more, neither of them a
  constant, whose rows fill at least the target's middle column block (``Target.row_blocks``, 8
  columns on the reference machine), or which fills a register tile's rows (``Target.product_tile``)
  and more than half its columns, and more than four. Its loops run vectorized; expanded, every output is a scalar chain
  the C compiler does not vectorize. Narrower products lose less, and a procedure kept in loops for
  them lost more on the rest of its work than the product gained (a Riccati step of four states,
  1.023x; one of six gained 0.86x in tiles, where it had lost 1.7% in single rows); a constant
  operand's zeros and ones, the seeds of a forward-mode Jacobian above all, fold away only when
  expanded."""
  if node.op != ExprOp.MATMUL:
    return False
  a, b = node.args
  if ExprOp.CONST in (a.op, b.op):
    return False  # expanded, a constant operand's zeros and ones fold away, which loops cannot do
  if len(a.shape) != 2 or len(b.shape) != 2 or int(b.shape[0]) < 4:
    return False
  rows, columns = target.product_tile
  n = int(b.shape[1])
  return n >= target.row_blocks[1] or (1 < rows <= int(a.shape[0]) and n > max(columns // 2, 4))


# ``bool`` values only ever come from comparisons, logic and ``isfinite``, never from a narrowing
# store, so scalar substitution cannot erase a conversion for them. ``int64`` values come from
# explicit casts, integer constants and integer arithmetic; scalar expansion keeps a store's
# conversion as a cast wherever the stored value's type differs from the buffer's.
_SCALARIZABLE = (dtypes.float64, dtypes.bool_, dtypes.int64)


class LowerCtx:
  """Per-Function lowering state: buffers, statements, and the Expr-id -> buffer map.

  An op's lowering rule (``OpDef.lower``) receives this context and the node, and programs against
  its public part, which with ``scaly.ir.program``'s builders is all a rule needs:

  - ``buf_of(expr)`` the buffer holding an argument; ``alloc_tmp(node)`` the buffer to write the
    node into (its output buffer when it is one); ``bind(node, name)`` makes an existing buffer the
    node's value instead, such as ``output_buffer(i)`` for a node updated in place (``in_place``);
  - ``new_private``, ``new_alias`` and ``new_const_index`` for scratch, views and index tables;
  - ``emit(*statements)`` appends to the procedure; ``fresh_id()`` and ``fresh_name(prefix)`` give
    names no other statement uses;
  - ``copy_loop``, ``blocked_sum`` and ``lane_loops`` build the loops rules share;
  - ``fun`` the Function being lowered and ``in_place`` whether its carry is updated in place;
  - ``target`` the processor the code is tuned for (``scaly.ir.target.Target``).

  ``entry`` marks the Function whose procedure becomes the pointer-ABI entry. Its parameters are the
  caller's ``double`` arrays whatever the declared dtype, so a ``bool`` input is read into a typed
  temporary (nonzero is true) and a ``bool`` output is written as 0.0 or 1.0 from one.
  """

  def __init__(
    self,
    fun: ConcreteFunction,
    callees: dict[str, ProgramNode],
    extern_fns: dict[str, ConcreteFunction],
    target: Target,
    observe_expr: ExprObserver | None = None,
    *,
    entry: bool = False,
    in_place: tuple[int, ...] | None = None,
  ) -> None:
    self.fun = fun
    self.target = target
    self.entry = entry
    self.in_place = in_place
    self.callees = callees
    self.extern_fns = extern_fns  # name -> Function with an extern body (opaque; the callee renders its C)
    self.observe_expr = observe_expr
    self.params: list[ProgramNode] = []
    self.statements: list[ProgramNode] = []
    self.buffers: dict[str, ProgramNode] = {}
    # Expr.id -> name of the buffer holding that value at runtime.
    self.value_buffers: dict[int, str] = {}
    # Output Expr.id -> output buffer name, so the body writes outputs in place.
    self._output_alias: dict[int, str] = {}
    # Output position -> its buffer's name: the output's own name unless an input already took it.
    self.output_buffer_names: list[str] = []
    # (callee_name, arg_buffer_names) -> output buffer names, to dedup repeated CALL invocations.
    self.call_invocations: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}
    # scan identity -> buffer name per output index (0 final carry, -1 carries, 1.. stacked outputs).
    self.scan_invocations: dict[tuple[object, ...], dict[int, str]] = {}
    self._const_tables: dict[tuple[int, ...], ProgramNode] = {}
    self._tmp = 0

  # --- declarations ---------------------------------------------------------

  def _abi_dtype(self, dtype: DType) -> DType:
    """The entry point takes and returns ``double`` arrays whatever the declared dtype, so its
    non-``float64`` inputs are converted once into buffers of their own dtype (a callee reads them
    through a pointer of that type) and its outputs are converted on the way out."""
    return dtypes.float64 if self.entry else dtype

  def emit_inputs(self) -> None:
    converted: list[tuple[ProgramNode, Expr]] = []
    for name, expr in zip(self.fun.input_names, self.fun.inputs, strict=True):
      buf = p.buffer(name, self._abi_dtype(expr.type.dtype), _shape_or_scalar(expr.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      self.value_buffers[expr.id] = name
      if buf.dtype != expr.type.dtype:
        converted.append((buf, expr))
    for buf, expr in converted:
      tmp = self.new_private(expr.type.dtype, expr.shape)
      vname = f"c_{tmp.attrs['name']}"
      i = p.var(vname)
      value = p.load(p.view(buf, [i]))
      converted_value = p.compare(ProgramOp.NE, value, p.const_float(0.0)) if expr.type.dtype.is_bool else p.cast(value, expr.type.dtype)
      self.statements.append(p.for_(p.range_(vname, 0, _size_of(expr.shape), kind=RangeKind.GLOBAL), [p.store(p.view(tmp, [i]), converted_value)]))
      self.value_buffers[expr.id] = tmp.attrs["name"]

  def register_outputs(self) -> None:
    """Register output BUFFER params and alias each unique computed output Expr to
    its output buffer, so its rule writes directly into the output (no copy)."""
    seen: set[int] = set()
    for out_name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      # An output may share its name with an input (``(x, y) -> (y, z)``); buffers are keyed by
      # name, so such an output gets one of its own, or it would stand for the input everywhere.
      name = out_name if out_name not in self.buffers else self.fresh_name(f"{out_name}_out")
      self.output_buffer_names.append(name)
      buf = p.buffer(name, self._abi_dtype(expr.type.dtype), _shape_or_scalar(expr.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      if expr.op in (ExprOp.INPUT, ExprOp.CONST) or expr.id in seen or buf.dtype != expr.type.dtype:
        continue  # INPUT/CONST, shared, or converted output: emit_outputs inserts the copy
      seen.add(expr.id)
      self._output_alias[expr.id] = name

  def output_buffer(self, i: int) -> ProgramNode:
    """The buffer of output ``i``."""
    return self.buffers[self.output_buffer_names[i]]

  def emit_outputs(self) -> None:
    for out_name, name, expr in zip(self.fun.output_names, self.output_buffer_names, self.fun.outputs, strict=True):
      src = self.value_buffers.get(expr.id)
      if src is None:
        raise LoweringError(f"output {out_name!r} expression was not lowered")
      if src == name:
        continue  # already written in place via the output alias
      self.statements.append(_copy_loop(self.buffers[src], self.buffers[name], expr.shape))

  # --- body -----------------------------------------------------------------

  def emit_body(self) -> None:
    for node in topo(self.fun.outputs):
      if node.id in self.value_buffers:
        continue  # input (or already lowered)
      definition = op_def(node.op)
      rule = definition.lower or (_lower_elementwise if "elementwise" in definition.traits else None)
      if rule is None:
        raise LoweringError(f"Expression op {node.op!r} is not yet lowered to Program IR")
      rule(self, node)

  # --- helpers --------------------------------------------------------------

  def buf_of(self, expr: Expr) -> ProgramNode:
    return self.buffers[self.value_buffers[expr.id]]

  def fresh_id(self) -> int:
    """A number no other generated name in this procedure carries."""
    self._tmp += 1
    return self._tmp - 1

  def fresh_name(self, prefix: str) -> str:
    """A buffer name not yet taken in this procedure: generated names share one namespace with
    the parameters, which are the Function's own input and output names."""
    while True:
      name = f"{prefix}{self.fresh_id()}"
      if name not in self.buffers:
        return name

  def emit(self, *statements: ProgramNode) -> None:
    """Append ``statements`` to the procedure."""
    self.statements.extend(statements)

  def bind(self, expr: Expr, name: str) -> None:
    """Make the buffer ``name`` hold ``expr``'s value: an alias, or a buffer a rule wrote."""
    self.value_buffers[expr.id] = name

  def copy_loop(self, src: ProgramNode, dst: ProgramNode, shape: tuple[int, ...]) -> ProgramNode:
    """A loop copying ``src`` into ``dst`` entry by entry, converting to ``dst``'s dtype."""
    return _copy_loop(src, dst, shape)

  def blocked_sum(
    self, tag: str, start: ProgramNode, stop: ProgramNode, term: Any, dtype: DType, *, lanes: int = 4
  ) -> tuple[list[ProgramNode], ProgramNode]:
    """The sum of ``term(k)`` over ``[start, stop)`` in ``lanes`` partial sums: see ``_blocked_sum``."""
    return _blocked_sum(self, tag, start, stop, term, dtype, lanes=lanes)

  def tile(
    self,
    tag: str,
    rows: list[ProgramNode],
    width: int,
    steps: int | ProgramNode,
    term: Callable[[ProgramNode, ProgramNode, ProgramNode], ProgramNode],
    out_at: Callable[[ProgramNode, ProgramNode], ProgramNode],
    dtype: DType,
    finish: Callable[[ProgramNode, ProgramNode, ProgramNode], ProgramNode] | None = None,
  ) -> list[ProgramNode]:
    """A register tile of ``rows`` by ``width`` sums over ``steps`` steps, from zero: see ``_tile``."""
    return _tile(self, tag, rows, width, steps, term, out_at, False, dtype, finish)

  def tile_segments(self, n: int) -> list[tuple[int, int, int]]:
    """``n`` columns in the target's register tiles, as ``(first, width, count)``: see ``_tile_segments``."""
    return _tile_segments(n, self.target.product_tile[1], self.target.choices.vector_doubles)[0]

  def lane_loops(self, tag: str, rows: int, lanes: int, kind: RangeKind, body: Any) -> None:
    """Loops over ``rows`` and ``lanes`` emitting ``body``: see ``_lane_loops``."""
    _lane_loops(self, tag, rows, lanes, kind, body)

  def new_private(self, dtype: DType, shape: tuple[int, ...], *, lanes: int | None = None) -> ProgramNode:
    """Allocate a fresh private scratch BUFFER (declared as a local array by the renderer); ``lanes``
    marks one that holds a vector's lanes, which keeps its own storage (``pack_workspace``)."""
    name = self.fresh_name("t")
    buf = p.buffer(name, dtype, _shape_or_scalar(shape), address_space="private")
    if lanes is not None:
      buf = ProgramNode(ProgramOp.BUFFER, (), {**buf.attrs, "lanes": lanes}, dtype)
    self.buffers[name] = buf
    self.statements.append(buf)  # marks the local-array declaration for the renderer
    return buf

  def alloc_tmp(self, expr: Expr) -> ProgramNode:
    """Buffer to hold ``expr``'s value: the output buffer if aliased, else a fresh private temp."""
    alias = self._output_alias.get(expr.id)
    if alias is not None:
      self.value_buffers[expr.id] = alias
      return self.buffers[alias]
    buf = self.new_private(expr.type.dtype, expr.shape)
    self.value_buffers[expr.id] = buf.attrs["name"]
    return buf

  def new_alias(self, dtype: DType, shape: tuple[int, ...], src_name: str, offset: int) -> ProgramNode:
    """A zero-copy private BUFFER that aliases ``src_name`` at a flat ``offset`` (rendered
    ``const T* tN = <src> + offset;``). Carries ``alias_of`` / ``alias_offset`` so the workspace
    pass leaves it unpacked and keeps its source live. Port of ``codegen/c.py``'s contiguous
    SLICE / RESHAPE pointer aliasing."""
    name = self.fresh_name("t")
    buf = ProgramNode(
      ProgramOp.BUFFER,
      (),
      attrs={
        "name": name,
        "shape": _shape_or_scalar(shape),
        "address_space": "private",
        "device": DeviceSpec.parse(None),
        "alias_of": src_name,
        "alias_offset": int(offset),
      },
      dtype=dtype,
    )
    self.buffers[name] = buf
    self.statements.append(buf)
    return buf

  def new_const_index(self, idx: Iterable[int]) -> ProgramNode:
    """A read-only int64 index table (for the ops at constant indices), declared ``static const``; one
    table per distinct content in a procedure."""
    values = [int(v) for v in idx]
    key = tuple(values)
    if key in self._const_tables:
      return self._const_tables[key]
    buf = p.const_buffer(self.fresh_name("k"), dtypes.int64, (len(values),), values)
    self.buffers[buf.attrs["name"]] = buf
    self.statements.append(buf)
    self._const_tables[key] = buf
    return buf

  def index_at(self, idx: np.ndarray, k: ProgramNode) -> ProgramNode:
    """The GATHER source (or SEGMENT_REDUCE destination) for element ``k``, as arithmetic where it can be.

    Every range whose contribution is affine becomes a term over ``k``; only the non-affine
    residual is materialized, and a residual of one element is a constant, so a fully affine index
    emits no table at all. An index with no affine structure keeps the whole table, indexed by
    ``k``, which is what every gather used to emit. An empty index is any constant: its loop runs
    zero times."""
    if idx.size == 0:
      return p.const_int(0)
    amap = affine_index_map(idx)
    period = len(amap.residual)
    out: ProgramNode | None = None
    if period > 1:
      table = self.new_const_index(amap.residual)
      out = p.load(p.view(table, [k if period == idx.size else p.mod(k, p.const_int(period))]))
    elif int(amap.residual[0]) or not amap.dims:
      out = p.const_int(int(amap.residual[0]))
    strides: list[int] = []
    stride = period
    for dim in reversed(amap.dims):
      strides.append(stride)
      stride *= dim
    strides.reverse()
    # Level ``i``'s coordinate is ``(k // strides[i]) % dims[i]``, and the modulo is what makes it a
    # second division. Because ``k // strides[i] // dims[i] == k // strides[i - 1]``, the coordinates
    # telescope: the whole sum is a combination of the plain quotients ``q_i = k // strides[i]`` with
    # coefficients ``c_i - c_(i+1) * dims[i+1]``. So no level needs a modulo, and the innermost
    # quotient is ``k`` itself whenever the residual has one element.
    inner = (*(c * d for c, d in zip(amap.coeffs[1:], amap.dims[1:])), 0)
    for coeff, quotient_stride in zip((c - nxt for c, nxt in zip(amap.coeffs, inner)), strides):
      if coeff == 0:
        continue
      quotient = k if quotient_stride == 1 else p.div(k, p.const_int(quotient_stride))
      term = quotient if coeff == 1 else p.mul(quotient, p.const_int(coeff))
      out = term if out is None else p.add(out, term)
    return out if out is not None else p.const_int(0)

  def emit_elementwise(self, node: Expr, pop: ProgramOp, *, arity: int) -> None:
    out = self.alloc_tmp(node)
    vname = f"i_{out.attrs['name']}"
    rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
    i = p.var(vname)
    loads = tuple(p.load(p.view(self.buf_of(a), [_broadcast_index_p(i, a.shape, node.shape)])) for a in node.args[:arity])
    computed = ProgramNode(pop, loads, dtype=node.type.dtype)
    self.statements.append(p.for_(rng, [p.store(p.view(out, [i]), computed)]))


def _stride(shape: tuple[int, ...], dim: int) -> int:
  s = 1
  for d in shape[dim + 1 :]:
    s *= int(d)
  return s


def _coord_p(flat: ProgramNode, shape: tuple[int, ...], dim: int) -> ProgramNode:
  """Decompose flat output index ``flat`` into the ``dim``-th coordinate of ``shape``."""
  stride = _stride(shape, dim)
  v = flat if stride == 1 else p.div(flat, p.const_int(stride))
  # The leading dim needs no modulo: flat < size guarantees (flat // stride) < shape[0].
  return v if dim == 0 else p.mod(v, p.const_int(int(shape[dim])))


def _flat_index_p(coords: list[ProgramNode], shape: tuple[int, ...]) -> ProgramNode:
  terms = [c if (st := _stride(shape, i)) == 1 else p.mul(c, p.const_int(st)) for i, c in enumerate(coords)]
  if not terms:
    return p.const_int(0)
  acc = terms[0]
  for t in terms[1:]:
    acc = p.add(acc, t)
  return acc


def _affine_sum(vars_: list[ProgramNode], coeffs: list[int]) -> ProgramNode:
  """Build ``Σ coeffs[i] * vars_[i]`` as a ProgramNode, dropping zero coeffs and unit multiplies."""
  acc: ProgramNode | None = None
  for v, c in zip(vars_, coeffs, strict=True):
    if c == 0:
      continue
    term = v if c == 1 else p.mul(v, p.const_int(c))
    acc = term if acc is None else p.add(acc, term)
  return acc if acc is not None else p.const_int(0)


def _row_major_strides(shape: tuple[int, ...]) -> list[int]:
  strides = [1] * len(shape)
  for i in range(len(shape) - 2, -1, -1):
    strides[i] = strides[i + 1] * int(shape[i + 1])
  return strides


def _broadcast_index_p(flat: ProgramNode, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> ProgramNode:
  """Map an output flat index to the source flat index under numpy broadcasting
  (right-aligned; size-1 dims and missing leading dims read index 0)."""
  if in_shape == out_shape or not out_shape:
    return flat
  if not in_shape:
    return p.const_int(0)
  offset = len(out_shape) - len(in_shape)
  coords = [p.const_int(0) if d == 1 else _coord_p(flat, out_shape, offset + i) for i, d in enumerate(in_shape)]
  return _flat_index_p(coords, in_shape)


def _copy_loop(src: ProgramNode, dst: ProgramNode, shape: tuple[int, ...]) -> ProgramNode:
  vname = f"c_{dst.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(shape), kind=RangeKind.GLOBAL)
  i = p.var(vname)
  return p.for_(rng, [p.store(p.view(dst, [i]), p.cast(p.load(p.view(src, [i])), dst.dtype))])


# ---------------------------------------------------------------------------
# Lowering rules.
# ---------------------------------------------------------------------------


def _lower_elementwise(ctx: LowerCtx, node: Expr) -> None:
  """The lowering of every op with the ``elementwise`` trait, whose value is its program op."""
  ctx.emit_elementwise(node, op_def(node.op).traits["elementwise"], arity=len(node.args))


@lowers(ExprOp.SELECT)
def _lower_select(ctx: LowerCtx, node: Expr) -> None:
  ctx.emit_elementwise(node, ProgramOp.SELECT, arity=3)


@lowers(ExprOp.RESHAPE)
def _lower_reshape(ctx: LowerCtx, node: Expr) -> None:
  # Metadata-only: the result aliases its source buffer (no copy).
  ctx.value_buffers[node.id] = ctx.value_buffers[node.args[0].id]


@lowers(ExprOp.CONST)
def _lower_const(ctx: LowerCtx, node: Expr) -> None:
  value = node.value
  assert value is not None
  # A constant of any size materializes as a read-only ``constant``-space buffer
  # (rendered ``static const``). Output-aliasing never applies to CONST, so
  # emit_outputs inserts a copy when a CONST is itself an output. ``tolist`` keeps an int64 exact,
  # where ``float`` would round one above 2**53.
  name = ctx.fresh_name("k")
  buf = p.const_buffer(name, node.type.dtype, _shape_or_scalar(node.shape), value.reshape(-1).tolist())
  ctx.buffers[name] = buf
  ctx.value_buffers[node.id] = name
  ctx.emit(buf)


def _contiguous_slice_offset(index: tuple[object, ...], in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> int | None:
  """If the slice selects a contiguous sub-block of ``src`` at a constant flat offset (no stride,
  at most one partial leading slice with full trailing dims), return that offset; else None. Port
  of ``codegen/c.py::_contiguous_slice_offset`` — the precondition for pointer aliasing."""
  if not out_shape:
    offset = 0
    for dim, item in enumerate(index):
      if not isinstance(item, int):
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * _stride(in_shape, dim)
    return offset
  first_slice = None
  offset = 0
  out_dim = 0
  for dim, item in enumerate(index):
    stride = _stride(in_shape, dim)
    if isinstance(item, int):
      if first_slice is not None and in_shape[dim] != 1:
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * stride
      continue
    assert isinstance(item, slice)
    start, stop, step = item.indices(int(in_shape[dim]))
    if step != 1:
      return None
    if first_slice is None:
      first_slice = dim
      offset += start * stride
      out_dim += 1
      continue
    if start != 0 or stop != int(in_shape[dim]) or out_shape[out_dim] != int(in_shape[dim]):
      return None
    out_dim += 1
  return offset


@lowers(ExprOp.SLICE)
def _lower_slice(ctx: LowerCtx, node: Expr) -> None:
  """General SLICE: integer indices drop a dim, slices keep one. A contiguous slice (constant
  flat offset, no stride) becomes a zero-copy pointer alias of its source; otherwise each output
  element reads the source via flat-index arithmetic. Covers rank-1, multi-dim, integer, strided."""
  src = node.args[0]
  src_shape = src.shape
  index = node.attrs["index"]
  # Contiguous + not an output: alias the source pointer instead of copying (legacy parity).
  if node.id not in ctx._output_alias and (offset := _contiguous_slice_offset(index, src_shape, node.shape)) is not None:
    alias = ctx.new_alias(node.type.dtype, node.shape, ctx.value_buffers[src.id], offset)
    ctx.value_buffers[node.id] = alias.attrs["name"]
    return
  out = ctx.alloc_tmp(node)
  vname = f"i_{out.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  k = p.var(vname)
  coords: list[ProgramNode] = []
  out_dim = 0
  for dim, item in enumerate(index):
    if isinstance(item, int):
      coords.append(p.const_int(item if item >= 0 else int(src_shape[dim]) + item))
      continue
    start, _stop, step = item.indices(int(src_shape[dim]))
    c = _coord_p(k, node.shape, out_dim)
    if step != 1:
      c = p.mul(c, p.const_int(step))
    coords.append(c if (start == 0 and step == 1) else p.add(p.const_int(start), c))
    out_dim += 1
  src_idx = _flat_index_p(coords, src_shape)
  ctx.emit(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


# A reduction this long or longer sums in ``REDUCTION_LANES`` partial sums interleaved by index,
# combined pairwise: one chain of dependent adds cannot overlap, four can (2.6-4x faster from 64
# elements on, ``notes/codegen_speed_o2_report.html``). The rounding is that of a blocked sum, as in
# NumPy's own pairwise ``sum``, not that of a sequential one; the order is fixed by the generated
# code. Eight lanes are faster again on long sums but moved the generated IPM off PIQP's path on
# QBEACONF, a problem whose path rounding decides; four keep it, as they do in ``_blocked_sum``.
REDUCTION_LANES = 4
BLOCKED_REDUCTION_MIN = 2 * REDUCTION_LANES


def _reduce_into(ctx: LowerCtx, acc: ProgramNode, n: int, term: Any, dtype: DType) -> None:
  """``acc[0] = sum(term(k) for k < n)``: in one chain below ``BLOCKED_REDUCTION_MIN``, else in
  ``REDUCTION_LANES`` partial sums. Every read of the terms sits in one statement, a loop run once,
  so ``fuse_elementwise`` can inline an elementwise producer; ``unroll_unit_loops`` removes it."""
  z = p.const_int(0)
  zero = p.const_float(0.0, dtype=dtype)
  tag = acc.attrs["name"]
  if n < BLOCKED_REDUCTION_MIN:
    name = f"i_{tag}"
    rng = p.range_(name, 0, n, kind=RangeKind.REDUCE)
    slot = p.view(acc, [z])
    ctx.emit(p.store(slot, zero), p.for_(rng, [p.store(slot, p.add(p.load(slot), term(p.var(name))))]))
    return
  # A private accumulator holds the first partial sum; an output does not, since the ABI's buffers
  # are memory the C compiler must assume the terms may alias, so a sum kept there stays in memory.
  private = all(acc is not param for param in ctx.params)
  stmts, total = _blocked_sum(ctx, tag, z, p.const_int(n), term, dtype, lanes=REDUCTION_LANES, first=p.view(acc, [z]) if private else None)
  ctx.emit(p.for_(p.range_(f"ko_{tag}", 0, 1), [*stmts, p.store(p.view(acc, [z]), total)]))


@lowers(ExprOp.SUM)
def _lower_sum(ctx: LowerCtx, node: Expr) -> None:
  """Full reduction to a scalar (``_reduce_into``)."""
  src_buf = ctx.buf_of(node.args[0])
  _reduce_into(ctx, ctx.alloc_tmp(node), _size_of(node.args[0].shape), lambda k: p.load(p.view(src_buf, [k])), node.type.dtype)


@lowers(ExprOp.MAX, ExprOp.MIN)
def _lower_extremum(ctx: LowerCtx, node: Expr) -> None:
  """A REDUCE loop keeping the larger (smaller) value. A NaN element replaces the accumulator and
  nothing replaces a NaN accumulator, so NaN propagates as ``np.max`` does; C's ``fmax`` would drop
  it. Past eight elements it keeps four accumulators, one per lane of four, combined in lane order:
  one chain of compares and selects cannot overlap, four can. The result is the same value (only
  the sign of a zero extremum could come from another element)."""
  src = node.args[0]
  acc = ctx.alloc_tmp(node)
  c = p.const_int
  src_buf = ctx.buf_of(src)
  n = _size_of(src.shape)

  def pick(cur: ProgramNode, value: ProgramNode) -> ProgramNode:
    better = p.compare(ProgramOp.LT, cur, value) if node.op == ExprOp.MAX else p.compare(ProgramOp.LT, value, cur)
    take = ProgramNode(ProgramOp.OR, (better, p.compare(ProgramOp.NE, value, value)), dtype=dtypes.bool_)
    return p.select(take, value, cur)

  def at(k: ProgramNode) -> ProgramNode:
    return p.load(p.view(src_buf, [k]))

  def reads(stmts: list[ProgramNode]) -> None:
    # Every read of the source in one statement, a loop run once, so that ``fuse_elementwise`` can
    # inline an elementwise producer into the reduction; ``unroll_unit_loops`` then removes the loop.
    ctx.emit(p.for_(p.range_(f"ko_{acc.attrs['name']}", 0, 1), stmts))

  if n < 8:
    slot = p.view(acc, [c(0)])
    first = [p.store(slot, at(c(0)))]
    if n > 1:
      name = f"i_{acc.attrs['name']}"
      rng = p.range_(name, 1, n, kind=RangeKind.REDUCE)
      first.append(p.for_(rng, [p.store(slot, pick(p.load(slot), at(p.var(name))))]))
    reads(first)
    return
  slots = [p.view(ctx.new_private(node.type.dtype, ()), [c(0)]) for _ in range(4)]
  stmts = [p.store(s, at(c(q))) for q, s in enumerate(slots)]
  tail = n - (n - 4) % 4
  kb, kt = f"kb_{acc.attrs['name']}", f"kt_{acc.attrs['name']}"
  block = [p.store(s, pick(p.load(s), at(p.add(p.var(kb), c(q))))) for q, s in enumerate(slots)]
  stmts.append(p.for_(p.range_(kb, 4, tail, step=4, kind=RangeKind.REDUCE), block))
  if tail < n:
    stmts.append(p.for_(p.range_(kt, tail, n, kind=RangeKind.REDUCE), [p.store(slots[0], pick(p.load(slots[0]), at(p.var(kt))))]))
  reads(stmts)
  total = p.load(slots[0])
  for s in slots[1:]:
    total = pick(total, p.load(s))
  ctx.emit(p.store(p.view(acc, [c(0)]), total))


@lowers(ExprOp.TRANSPOSE)
def _lower_transpose(ctx: LowerCtx, node: Expr) -> None:
  """Permuted copy: ``out[t] = src[Σ o_i(t)·src_stride_{axes[i]}]``, one flat loop over the output
  whose coordinates ``o_i(t)`` divide ``t``. A flat loop is what ``fuse_elementwise`` can inline
  into a single consumer, a gather above all (the recovery of a sparse derivative gathers from a
  transposed product and reads only its nonzeros), and ``delinearize_loops`` splits one that stays
  into a loop per output axis without the divisions."""
  src = node.args[0]
  axes = tuple(int(a) for a in node.attrs["axes"])
  src_shape, out_shape = src.shape, node.shape
  if len(src_shape) > 4:
    raise LoweringError(f"TRANSPOSE lowering handles rank <= 4; got {src_shape}")
  out = ctx.alloc_tmp(node)
  src_strides = _row_major_strides(src_shape)
  name = f"d_{out.attrs['name']}"
  t = p.var(name)
  coords = [_coord_p(t, out_shape, i) for i in range(len(out_shape))]
  src_idx = _affine_sum(coords, [src_strides[axes[i]] for i in range(len(out_shape))])
  rng = p.range_(name, 0, _size_of(out_shape), kind=RangeKind.GLOBAL)
  ctx.emit(p.for_(rng, [p.store(p.view(out, [t]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


def _nest(ranges: list[ProgramNode], body: list[ProgramNode]) -> list[ProgramNode]:
  for rng in reversed(ranges):
    body = [p.for_(rng, body)]
  return body


def _mm_init(out: ProgramNode, idx: ProgramNode, dtype: DType) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.const_float(0.0, dtype=dtype))


def _mm_accum(out: ProgramNode, idx: ProgramNode, a_load: ProgramNode, b_load: ProgramNode) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.add(p.load(p.view(out, [idx])), p.mul(a_load, b_load)))


@lowers(ExprOp.MATMUL)
def _lower_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  a_buf, b_buf, out = ctx.buf_of(a), ctx.buf_of(b), ctx.alloc_tmp(node)
  dt = node.type.dtype
  sa, sb = a.shape, b.shape
  nm = out.attrs["name"]
  if len(sa) == 1 and len(sb) == 1:  # dot
    _reduce_into(ctx, out, sa[0], lambda k: p.mul(p.load(p.view(a_buf, [k])), p.load(p.view(b_buf, [k]))), dt)
  elif len(sa) == 2 and len(sb) == 1 and sa[1] >= MATVEC_BLOCKED_MIN:
    _lower_matvec_blocked(ctx, a_buf, b_buf, out, sa, dt)
  elif len(sa) == 2 and len(sb) == 1:  # mat @ vec: the reduction axis is contiguous in ``a``
    m, kk = sa
    i = p.var(f"i_{nm}")
    irng = p.range_(f"i_{nm}", 0, m, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    row_dot = lambda row: _mm_accum(out, row, p.load(p.view(a_buf, [p.add(p.mul(row, p.const_int(kk)), k)])), p.load(p.view(b_buf, [k])))
    ctx.emit(*_nest([irng], [_mm_init(out, i, dt)]))
    # Four output rows per pass as four unrolled statements, so the compiler keeps four independent
    # accumulators; a four-trip inner loop over the rows becomes gathers under -march=native.
    blocks, tail = divmod(m, 4)
    if blocks:
      ib = p.var(f"ib_{nm}")
      ibrng = p.range_(f"ib_{nm}", 0, blocks, kind=RangeKind.GLOBAL)
      rows = [p.add(p.mul(ib, p.const_int(4)), p.const_int(r)) for r in range(4)]
      ctx.emit(*_nest([ibrng, krng], [row_dot(row) for row in rows]))
    if tail:
      it = p.var(f"it_{nm}")
      itrng = p.range_(f"it_{nm}", 4 * blocks, m, kind=RangeKind.GLOBAL)
      ctx.emit(*_nest([itrng, krng], [row_dot(it)]))
  elif len(sa) == 1 and len(sb) == 2:  # vec @ mat
    _lower_columns_blocked(ctx, a_buf, b_buf, out, None, *sb, dt)
  elif len(sa) == 2 and len(sb) == 2:  # mat @ mat
    _lower_columns_blocked(ctx, a_buf, b_buf, out, sa[0], *sb, dt)
  else:
    raise LoweringError(f"matmul shapes {sa}@{sb} not lowered (batched / higher-rank deferred)")


@dataclass(frozen=True)
class _LaneSums:
  """A block's ``width`` running sums, in private buffers of ``lanes`` each. ``each`` makes one
  statement per sum from ``make(sum, column)``, the column counted from the block's first: on a
  target of one lane each sum is its own scalar; on a wider one, each buffer's lanes are a loop of
  kind ``VECTOR`` over the columns they hold, which the renderer writes as one vector statement.
  The lanes are independent columns, so each sum is the scalar one's, in the same order."""

  tag: str
  lanes: int
  buffers: tuple[ProgramNode, ...]

  def each(self, name: str, make: Any) -> list[ProgramNode]:
    c = p.const_int
    if self.lanes == 1:
      return [make(p.view(buf, [c(0)]), c(col)) for col, buf in enumerate(self.buffers)]
    loops = []
    for v, buf in enumerate(self.buffers):
      q = f"q{name}{v}_{self.tag}"
      lane = p.var(q)
      loops.append(p.for_(p.range_(q, 0, self.lanes, kind=RangeKind.VECTOR), [make(p.view(buf, [lane]), p.add(c(v * self.lanes), lane))]))
    return loops


def _lane_sums(ctx: LowerCtx, tag: str, width: int, dtype: DType) -> _LaneSums:
  """``width`` private running sums in buffers of the target's lanes (``_LaneSums``)."""
  lanes = ctx.target.choices.vector_doubles
  lanes = lanes if width % lanes == 0 else 1
  return _LaneSums(
    tag, lanes, tuple(ctx.new_private(dtype, ()) if lanes == 1 else ctx.new_private(dtype, (lanes,), lanes=lanes) for _ in range(width // lanes))
  )


# ``x @ b`` and ``a @ b`` keep a block of ``Target.row_blocks[0]`` outputs of a row in registers
# across the reduction, and then one of each narrower width while one fits: the loads and stores of
# the output row that a reduction loop outermost pays at every step are gone, and each output still
# sums its terms in order of ``k`` as before (``notes/codegen_speed_o6_report.html``). The columns no
# block covers keep the reduction outermost: one to three chains of a long reduction are slower
# than the memory round trips they save. So does a vector's row wider than ``Target.row_blocked_max``:
# there the reduction outermost streams the row contiguously, and the blocks, which read the matrix a
# column block at a time, measured slower (the unbumpercars oracle's 128 x 256 products, 1.13x).
# A matrix of several rows reads ``b`` once per row instead, so when ``b`` is wider than that, or
# larger than the whole L1 data cache, the column blocks run outermost: each block's panel of ``b``
# is copied contiguous, in chunks of ``k`` that fit in ``Target.panel_bytes``, and every row passes
# over the chunk; each chunk resumes its sums from the outputs, so each is still one chain in order
# of ``k``. The copy has to be paid for by the rows that share it: on the reference machine a wide
# row's streaming, slow per row, lost to it from six rows (1.19x at 6 x 256 x 256, even at four),
# and the row blocks re-reading a ``b`` past the L1 cache from L2 only from sixteen (1.09x at 16 x
# 700 x 40, 0.91 at eight; a ``b`` of twice the panel was even at 128 rows).
PANEL_ROWS_WIDE = 6
PANEL_ROWS_LARGE = 16
# A product in register tiles (``Target.product_tile``) reads ``b`` once per tile of rows, not per
# row, and at any width: it copies panels only for a ``b`` larger than the L1 data cache and from 64
# rows, or, from sixteen rows, larger than half the L2 cache, which every tile of rows would read
# from memory again. Below that, reading ``b`` in place measured faster on the reference machine
# (1.1-1.6x at 8 to 32 rows of 256 x 256, 1.08x at 96 x 96 x 96, 1.1-1.2x at 16 rows up to a ``b`` of
# 8 MiB); above it the copy won, by 2-6% from 64 rows (64 x 64 x 512, 192 and 256 square) and by
# 1.1x at 16 rows of 16 MiB to 3.1x at 63 rows of 32 MiB.
PANEL_ROWS_TILED = 64


def _tile(
  ctx: LowerCtx,
  tag: str,
  rows: list[ProgramNode],
  width: int,
  steps: int | ProgramNode,
  term: Callable[[ProgramNode, ProgramNode, ProgramNode], ProgramNode],
  out_at: Callable[[ProgramNode, ProgramNode], ProgramNode],
  resume: bool,
  dtype: DType,
  finish: Callable[[ProgramNode, ProgramNode, ProgramNode], ProgramNode] | None = None,
) -> list[ProgramNode]:
  """A register tile of ``len(rows)`` rows and ``width`` columns of a product: each row's sums in
  buffers of the target's lanes (``_LaneSums``), started at zero or, when ``resume``, from the
  outputs, then ``steps`` steps of ``k`` each adding ``term(row, k, column)`` to every sum, then
  each output stored once: the sum, or ``finish(row, column, sum)``. Each sum is still one chain of
  multiply-adds in order of ``k``."""
  zero = p.const_float(0.0, dtype=dtype)
  k = p.var(f"k_{tag}")
  sums = [_lane_sums(ctx, f"{tag}_{r}", width, dtype) for r in range(len(rows))]
  return [
    *(
      st
      for r, (row, s_r) in enumerate(zip(rows, sums, strict=True))
      for st in s_r.each(f"z{r}", lambda s, col, row=row: p.store(s, p.load(out_at(row, col)) if resume else zero))
    ),
    p.for_(
      p.range_(k.attrs["name"], 0, steps, kind=RangeKind.REDUCE),
      [
        st
        for r, (row, s_r) in enumerate(zip(rows, sums, strict=True))
        for st in s_r.each(f"k{r}", lambda s, col, row=row: p.store(s, p.add(p.load(s), term(row, k, col))))
      ],
    ),
    *(
      st
      for r, (row, s_r) in enumerate(zip(rows, sums, strict=True))
      for st in s_r.each(f"s{r}", lambda s, col, row=row: p.store(out_at(row, col), p.load(s) if finish is None else finish(row, col, p.load(s))))
    ),
  ]


def _tile_segments(n: int, columns: int, lanes: int) -> tuple[list[tuple[int, int, int]], int]:
  """A row's columns in tiles: ``columns`` wide as many times as they fit, then at most one of each
  half as wide down to one vector's lanes, then the columns left, fewer than a vector's, in one of
  scalar sums (streamed with ``k`` outermost they cost more than their share); with the first
  column none covers, ``n``."""
  segments: list[tuple[int, int, int]] = []
  first = 0
  if (count := n // columns) > 0:
    segments.append((0, columns, count))
    first = columns * count
  width = columns // 2
  while width >= lanes:
    if n - first >= width:
      segments.append((first, width, 1))
      first += width
    width //= 2
  if first < n:
    segments.append((first, n - first, 1))
  return segments, n


def _lower_columns_blocked(
  ctx: LowerCtx, a_buf: ProgramNode, b_buf: ProgramNode, out: ProgramNode, m: int | None, kk: int, n: int, dtype: DType
) -> None:
  """``out = a @ b`` for ``b`` of shape ``(kk, n)`` and ``a`` a vector (``m`` None) or ``(m, kk)``:
  row by row, the columns in blocks of the target's ``row_blocks`` widths, the widest as many times
  as it fits, then at most one of each narrower one, each block's sums in private scalars over a
  ``k`` loop inside it and each output stored once; the columns left over, every column of a
  product narrower than the narrowest block and every column of a row wider than
  ``row_blocked_max`` accumulate with ``k`` outermost, as every product did before, unless there are
  rows enough to share the copy of a panel: then a ``b`` wider than that, or larger than the L1 data
  cache, takes the same blocks with the column blocks outermost (``_panel_passes``)."""
  c = p.const_int
  nm = out.attrs["name"]
  zero = p.const_float(0.0, dtype=dtype)
  target = ctx.target
  tile_rows, tile_columns = target.product_tile
  tiled = m is not None and 1 < tile_rows <= m
  large = kk * n * dtype.itemsize > target.choices.l1d_bytes
  outermost = (
    m is not None
    and kk > 0
    and (
      (large and (m >= PANEL_ROWS_TILED or (m >= PANEL_ROWS_LARGE and kk * n * dtype.itemsize > target.choices.l2_bytes // 2)))
      if tiled
      else ((n > target.row_blocked_max and m >= PANEL_ROWS_WIDE) or (large and m >= PANEL_ROWS_LARGE))
    )
  )
  segments: list[tuple[int, int, int]] = []  # (first column, width, blocks)
  j0 = 0  # the first column no block covers; a vector's row wider than row_blocked_max has no blocks
  if tiled:
    segments, j0 = _tile_segments(n, tile_columns, target.choices.vector_doubles)
  elif n <= target.row_blocked_max or outermost:
    widest, *narrower = target.row_blocks
    if (count := n // widest) > 0:
      segments.append((0, widest, count))
      j0 = widest * count
    for width in narrower:
      if n - j0 >= width:
        segments.append((j0, width, 1))
        j0 += width
  i = p.var(f"i_{nm}") if m is not None else None

  def a_at(row: ProgramNode | None, k: ProgramNode) -> ProgramNode:
    return p.load(p.view(a_buf, [k if row is None else p.add(p.mul(row, c(kk)), k)]))

  def row_block(tag: str, first: ProgramNode, width: int) -> list[ProgramNode]:
    sums = _lane_sums(ctx, tag, width, dtype)
    k = p.var(f"k_{tag}")
    a_k = a_at(i, k)
    row = c(0) if i is None else p.mul(i, c(n))
    return [
      *sums.each("z", lambda s, _col: p.store(s, zero)),
      p.for_(
        p.range_(k.attrs["name"], 0, kk, kind=RangeKind.REDUCE),
        sums.each("k", lambda s, col: p.store(s, p.add(p.load(s), p.mul(a_k, p.load(p.view(b_buf, [p.add(p.mul(k, c(n)), p.add(first, col))])))))),
      ),
      *sums.each("s", lambda s, col: p.store(p.view(out, [p.add(row, p.add(first, col))]), p.load(s))),
    ]

  if tiled and not outermost:
    assert m is not None
    _lower_row_tiles(
      ctx,
      nm,
      segments,
      m,
      tile_rows,
      kk,
      lambda row, k, col, first: p.mul(a_at(row, k), p.load(p.view(b_buf, [p.add(p.mul(k, c(n)), p.add(first, col))]))),
      lambda row, col, first: p.view(out, [p.add(p.mul(row, c(n)), p.add(first, col))]),
      dtype,
    )
    segments = []
  stmts: list[ProgramNode] = []
  for number, (first, width, count) in enumerate(segments):
    tag = f"{nm}_{number}"
    if outermost:
      assert m is not None
      block = lambda start: _panel_passes(ctx, tag, a_buf, b_buf, out, m, kk, n, start, width, dtype, tile_rows if tiled else 1)  # noqa: E731
    else:
      block = lambda start: row_block(tag, start, width)  # noqa: E731
    if count > 1:
      jb = p.var(f"jb_{tag}")
      stmts.append(p.for_(p.range_(jb.attrs["name"], 0, count, kind=RangeKind.GLOBAL), block(p.add(c(first), p.mul(jb, c(width))))))
    else:
      stmts += block(c(first))
  if stmts:
    rowwise = i is not None and m is not None and not outermost
    ctx.emit(*([p.for_(p.range_(i.attrs["name"], 0, m, kind=RangeKind.GLOBAL), stmts)] if rowwise else stmts))
  if j0 == n:
    return
  # The columns left over: zero them, then add each ``k`` in turn over every row and column.
  j, k, r = p.var(f"j_{nm}"), p.var(f"k_{nm}"), p.var(f"r_{nm}")
  jrng = p.range_(j.attrs["name"], j0, n, kind=RangeKind.GLOBAL)
  krng = p.range_(k.attrs["name"], 0, kk, kind=RangeKind.REDUCE)
  rows = [] if m is None else [p.range_(r.attrs["name"], 0, m, kind=RangeKind.GLOBAL)]
  row = None if m is None else r
  idx = j if m is None else p.add(p.mul(r, c(n)), j)
  acc = p.store(p.view(out, [idx]), p.add(p.load(p.view(out, [idx])), p.mul(a_at(row, k), p.load(p.view(b_buf, [p.add(p.mul(k, c(n)), j)])))))
  ctx.emit(*_nest([*rows, jrng], [p.store(p.view(out, [idx]), zero)]), *_nest([krng, *rows, jrng], [acc]))


def _lower_row_tiles(
  ctx: LowerCtx,
  nm: str,
  segments: list[tuple[int, int, int]],
  m: int,
  tile_rows: int,
  kk: int,
  term: Callable[[ProgramNode, ProgramNode, ProgramNode, ProgramNode], ProgramNode],
  out_at: Callable[[ProgramNode, ProgramNode, ProgramNode], ProgramNode],
  dtype: DType,
) -> None:
  """The tiled columns of every row of a product: ``tile_rows`` rows at a time over each segment's
  tiles, the rows left over one at a time over the same tiles (``_tile``)."""
  c = p.const_int
  blocks, left = divmod(m, tile_rows)

  def over(label: str, rows: list[ProgramNode]) -> list[ProgramNode]:
    stmts: list[ProgramNode] = []
    for number, (first, width, count) in enumerate(segments):
      tag = f"{nm}_{label}{number}"

      def tile(start: ProgramNode, tag: str = tag, width: int = width) -> list[ProgramNode]:
        return _tile(ctx, tag, rows, width, kk, lambda row, k, col: term(row, k, col, start), lambda row, col: out_at(row, col, start), False, dtype)

      if count > 1:
        jb = p.var(f"jb_{tag}")
        stmts.append(p.for_(p.range_(jb.attrs["name"], 0, count, kind=RangeKind.GLOBAL), tile(p.add(c(first), p.mul(jb, c(width))))))
      else:
        stmts += tile(c(first))
    return stmts

  if blocks:
    ib = p.var(f"ib_{nm}")
    rows = [p.add(p.mul(ib, c(tile_rows)), c(r)) for r in range(tile_rows)]
    ctx.emit(p.for_(p.range_(ib.attrs["name"], 0, blocks, kind=RangeKind.GLOBAL), over("t", rows)))
  if left:
    it = p.var(f"it_{nm}")
    ctx.emit(p.for_(p.range_(it.attrs["name"], blocks * tile_rows, m, kind=RangeKind.GLOBAL), over("l", [it])))


def _panel_passes(
  ctx: LowerCtx,
  tag: str,
  a_buf: ProgramNode,
  b_buf: ProgramNode,
  out: ProgramNode,
  m: int,
  kk: int,
  n: int,
  first: ProgramNode,
  width: int,
  dtype: DType,
  tile_rows: int = 1,
) -> list[ProgramNode]:
  """The ``width`` columns of ``out = a @ b`` from ``first`` on, with the panel of ``b`` they read
  outermost: for each chunk of ``k`` that fits in ``Target.panel_bytes``, the chunk's rows of the
  panel are copied into one contiguous buffer (read down a column of a power-of-two ``n``, the panel
  would fall into a few sets of the cache and evict itself), then every row of ``a`` passes over it
  with its sums in private scalars, which the first chunk starts at zero and every later one
  resumes from the outputs the chunk before stored. Each output is one chain of multiply-adds in
  order of ``k``, as in a single pass. With ``tile_rows`` above one, the rows pass ``tile_rows`` at a
  time as register tiles (``_tile``), the rows left over one at a time."""
  c = p.const_int
  chunk = min(kk, max(1, ctx.target.panel_bytes // (width * dtype.itemsize)))
  if tile_rows > 1:
    return _panel_tile_passes(ctx, tag, a_buf, b_buf, out, m, kk, n, first, width, dtype, tile_rows, chunk)
  sums = _lane_sums(ctx, tag, width, dtype)
  packed = ctx.new_private(dtype, (chunk * width,))
  zero = p.const_float(0.0, dtype=dtype)

  def pass_over(name: str, start: ProgramNode, rows: int, resume: bool) -> list[ProgramNode]:
    """Copy rows ``start`` to ``start + rows`` of the panel, then pass every row of ``a`` over them."""
    r, q, i, k = (p.var(f"{v}{name}_{tag}") for v in ("r", "q", "i", "k"))
    copy = p.for_(
      p.range_(r.attrs["name"], 0, rows, kind=RangeKind.GLOBAL),
      [
        p.for_(
          p.range_(q.attrs["name"], 0, width, kind=RangeKind.GLOBAL),
          [p.store(p.view(packed, [p.add(p.mul(r, c(width)), q)]), p.load(p.view(b_buf, [p.add(p.mul(p.add(start, r), c(n)), p.add(first, q))])))],
        )
      ],
    )

    def at(col: ProgramNode) -> ProgramNode:
      return p.view(out, [p.add(p.mul(i, c(n)), p.add(first, col))])

    a_k = p.load(p.view(a_buf, [p.add(p.mul(i, c(kk)), p.add(start, k))]))
    rows_pass = p.for_(
      p.range_(i.attrs["name"], 0, m, kind=RangeKind.GLOBAL),
      [
        *sums.each(f"z{name}", lambda s, col: p.store(s, p.load(at(col)) if resume else zero)),
        p.for_(
          p.range_(k.attrs["name"], 0, rows, kind=RangeKind.REDUCE),
          sums.each(f"k{name}", lambda s, col: p.store(s, p.add(p.load(s), p.mul(a_k, p.load(p.view(packed, [p.add(p.mul(k, c(width)), col)])))))),
        ),
        *sums.each(f"s{name}", lambda s, col: p.store(at(col), p.load(s))),
      ],
    )
    return [copy, rows_pass]

  full, tail = divmod(kk, chunk)
  passes = pass_over("p0", c(0), chunk, resume=False)
  if full > 1:
    kb = p.var(f"kb_{tag}")
    passes.append(p.for_(p.range_(kb.attrs["name"], 1, full, kind=RangeKind.SERIAL), pass_over("p1", p.mul(kb, c(chunk)), chunk, resume=True)))
  if tail:
    passes += pass_over("p2", c(full * chunk), tail, resume=True)
  return passes


def _panel_tile_passes(
  ctx: LowerCtx,
  tag: str,
  a_buf: ProgramNode,
  b_buf: ProgramNode,
  out: ProgramNode,
  m: int,
  kk: int,
  n: int,
  first: ProgramNode,
  width: int,
  dtype: DType,
  tile_rows: int,
  chunk: int,
) -> list[ProgramNode]:
  """``_panel_passes`` with the rows in register tiles of ``tile_rows``."""
  c = p.const_int
  packed = ctx.new_private(dtype, (chunk * width,))
  blocks, left = divmod(m, tile_rows)

  def pass_over(name: str, start: ProgramNode, steps: int, resume: bool) -> list[ProgramNode]:
    r, q = (p.var(f"{v}{name}_{tag}") for v in ("r", "q"))
    copy = p.for_(
      p.range_(r.attrs["name"], 0, steps, kind=RangeKind.GLOBAL),
      [
        p.for_(
          p.range_(q.attrs["name"], 0, width, kind=RangeKind.GLOBAL),
          [p.store(p.view(packed, [p.add(p.mul(r, c(width)), q)]), p.load(p.view(b_buf, [p.add(p.mul(p.add(start, r), c(n)), p.add(first, q))])))],
        )
      ],
    )

    def term(row: ProgramNode, k: ProgramNode, col: ProgramNode) -> ProgramNode:
      return p.mul(p.load(p.view(a_buf, [p.add(p.mul(row, c(kk)), p.add(start, k))])), p.load(p.view(packed, [p.add(p.mul(k, c(width)), col)])))

    def out_at(row: ProgramNode, col: ProgramNode) -> ProgramNode:
      return p.view(out, [p.add(p.mul(row, c(n)), p.add(first, col))])

    passes = [copy]
    if blocks:
      ib = p.var(f"ib{name}_{tag}")
      rows = [p.add(p.mul(ib, c(tile_rows)), c(t)) for t in range(tile_rows)]
      passes.append(
        p.for_(
          p.range_(ib.attrs["name"], 0, blocks, kind=RangeKind.GLOBAL), _tile(ctx, f"{tag}_{name}t", rows, width, steps, term, out_at, resume, dtype)
        )
      )
    if left:
      it = p.var(f"it{name}_{tag}")
      passes.append(
        p.for_(
          p.range_(it.attrs["name"], blocks * tile_rows, m, kind=RangeKind.GLOBAL),
          _tile(ctx, f"{tag}_{name}l", [it], width, steps, term, out_at, resume, dtype),
        )
      )
    return passes

  full, tail = divmod(kk, chunk)
  passes = pass_over("p0", c(0), chunk, resume=False)
  if full > 1:
    kb = p.var(f"kb_{tag}")
    passes.append(p.for_(p.range_(kb.attrs["name"], 1, full, kind=RangeKind.SERIAL), pass_over("p1", p.mul(kb, c(chunk)), chunk, resume=True)))
  if tail:
    passes += pass_over("p2", c(full * chunk), tail, resume=True)
  return passes


def _pairwise(values: list[ProgramNode]) -> ProgramNode:
  """``values`` summed as a balanced tree, neighbours first: ``(v0 + v1) + (v2 + v3)``."""
  while len(values) > 1:
    values = [p.add(values[i], values[i + 1]) if i + 1 < len(values) else values[i] for i in range(0, len(values), 2)]
  return values[0]


def _blocked_sum(
  ctx: LowerCtx, tag: str, start: ProgramNode, stop: ProgramNode, term: Any, dtype: DType, *, lanes: int = 4, first: ProgramNode | None = None
) -> tuple[list[ProgramNode], ProgramNode]:
  """Statements summing ``term(k)`` over ``start <= k < stop`` into ``lanes`` partial sums, and the total.

  A dot product summed in one accumulator is a chain of dependent adds the C compiler may not
  reorder; interleaved accumulators overlap as many chains. Partial sum ``q`` takes the ``k`` with
  ``(k - start) % lanes == q`` up to the last whole block, the first takes the tail too, and the
  total combines them pairwise (``_pairwise``). The rounding differs from a single sequential sum
  the way blocked library kernels do. ``first``, a view, holds the first partial sum instead of a
  new private scalar: the result's own slot, so that it is live from the start, as a sequential
  accumulator is, and does not take over a slot its terms free."""
  c = p.const_int
  slots = [first if q == 0 and first is not None else p.view(ctx.new_private(dtype, ()), [c(0)]) for q in range(lanes)]
  zero = p.const_float(0.0, dtype=dtype)
  if start.op == stop.op == ProgramOp.CONST_INT:  # a static trip count, which fusion needs to see
    tail = c(int(stop.attrs["value"]) - (int(stop.attrs["value"]) - int(start.attrs["value"])) % lanes)
  else:
    tail = p.sub(stop, p.mod(p.sub(stop, start), c(lanes)))
  kb, kt = f"kb_{tag}", f"kt_{tag}"
  block = [p.store(slots[q], p.add(p.load(slots[q]), term(p.add(p.var(kb), c(q))))) for q in range(lanes)]
  stmts = [
    *(p.store(s, zero) for s in slots),
    p.for_(p.range_(kb, start, tail, step=lanes, kind=RangeKind.REDUCE), block),
    p.for_(p.range_(kt, tail, stop, kind=RangeKind.REDUCE), [p.store(slots[0], p.add(p.load(slots[0]), term(p.var(kt))))]),
  ]
  return stmts, _pairwise([p.load(s) for s in slots])


# ``a @ x`` with a row this long or longer runs four rows at a time, each row's dot product in four
# partial sums: sixteen independent chains, which the C compiler pairs into vector lanes. 2.3-3.6x
# faster than four sequential rows from 12x12 to 256x256; a shorter row keeps the rows' own order.
MATVEC_ROWS = 4
MATVEC_LANES = 4
MATVEC_BLOCKED_MIN = 2 * MATVEC_LANES


def _lower_matvec_blocked(ctx: LowerCtx, a_buf: ProgramNode, x_buf: ProgramNode, out: ProgramNode, shape: tuple[int, ...], dtype: DType) -> None:
  """``out = a @ x`` for a row-major ``a`` of ``shape = (m, kk)``: ``MATVEC_ROWS`` rows per pass
  (the rows left over one at a time), each summed in ``MATVEC_LANES`` partial sums interleaved by
  ``k`` and combined pairwise, the tail of ``k`` into the first. The sums live in private scalars
  and each output is stored once."""
  m, kk = shape
  c = p.const_int
  nm = out.attrs["name"]
  lanes = MATVEC_LANES
  tail = kk - kk % lanes
  zero = p.const_float(0.0, dtype=dtype)

  def rows_pass(tag: str, rows: list[ProgramNode]) -> list[ProgramNode]:
    slots = [[p.view(ctx.new_private(dtype, ()), [c(0)]) for _ in range(lanes)] for _ in rows]
    kb, kt = p.var(f"kb_{tag}"), p.var(f"kt_{tag}")

    def term(row: ProgramNode, k: ProgramNode) -> ProgramNode:
      return p.mul(p.load(p.view(a_buf, [p.add(p.mul(row, c(kk)), k)])), p.load(p.view(x_buf, [k])))

    def add(slot: ProgramNode, row: ProgramNode, k: ProgramNode) -> ProgramNode:
      return p.store(slot, p.add(p.load(slot), term(row, k)))

    body = [add(slots[r][q], row, p.add(kb, c(q))) for r, row in enumerate(rows) for q in range(lanes)]
    stmts = [p.store(s, zero) for row_slots in slots for s in row_slots]
    stmts.append(p.for_(p.range_(kb.attrs["name"], 0, tail, step=lanes, kind=RangeKind.REDUCE), body))
    if tail < kk:
      stmts.append(p.for_(p.range_(kt.attrs["name"], tail, kk, kind=RangeKind.REDUCE), [add(slots[r][0], row, kt) for r, row in enumerate(rows)]))
    stmts += [p.store(p.view(out, [row]), _pairwise([p.load(s) for s in slots[r]])) for r, row in enumerate(rows)]
    return stmts

  blocks, left = divmod(m, MATVEC_ROWS)
  if blocks:
    ib = p.var(f"ib_{nm}")
    rows = [p.add(p.mul(ib, c(MATVEC_ROWS)), c(r)) for r in range(MATVEC_ROWS)]
    ctx.emit(p.for_(p.range_(ib.attrs["name"], 0, blocks, kind=RangeKind.GLOBAL), rows_pass(nm, rows)))
  if left:
    it = p.var(f"it_{nm}")
    ctx.emit(p.for_(p.range_(it.attrs["name"], MATVEC_ROWS * blocks, m, kind=RangeKind.GLOBAL), rows_pass(f"{nm}_t", [it])))


def _ensure_in_place_callee(ctx: LowerCtx, callee: ConcreteFunction, steps: dict[int, np.ndarray] | None = None) -> tuple[str, int] | None:
  """Lower an in-place variant of a loop body, named apart from the ordinary procedure (which other
  call sites may use with separate buffers), and return its name with the scratch slots its carry
  needs past its entries; None when the body does not qualify, or in-place updates are switched off.

  ``steps`` holds, per body input position, the value that input takes at every step when the loop
  slices it from a constant table (the step number, index tables): with it, updates at run-time
  indices are proven safe for exactly the indices this loop uses (``in_place_steps``). The procedure
  is the same whichever loop proved it, so it is shared by name."""
  if not DONATE_CARRIES or callee.device.kind != ctx.fun.device.kind:
    return None
  normalized = _normalize_function(callee)
  if in_place_chain(normalized) is None and not in_place_steps(normalized, steps or {}):
    return None
  name = f"{callee.name}_inplace"
  if name not in ctx.callees:
    renamed = ConcreteFunction.from_exprs(name, normalized.inputs, normalized.outputs, normalized.input_names, normalized.output_names)
    ctx.callees[name] = _lower_to_proc(renamed, ctx.callees, ctx.extern_fns, ctx.target, observe_expr=ctx.observe_expr, in_place=True)
  return name, _put_scratch(update_chain(normalized) or [])


def _put_scratch(chain: Iterable[Expr]) -> int:
  """The scratch slots after the carry's entries that its run-time-index updates write padded lanes to."""
  puts = (e for e in chain if e.op in (ExprOp.PUT_ADD, ExprOp.PUT) and put_lanes(e) is None and e.shape[-1] and not e.attrs.get("in_range"))
  return max((e.size // e.shape[-1] * e.args[1].size for e in puts), default=0)


def _loop_steps(
  callee: ConcreteFunction, outers: Iterable[Expr], starts: Iterable[int], strides: Iterable[int], length: int
) -> dict[int, np.ndarray]:
  """The value of each body input a loop slices from an integer constant, at every step."""
  steps: dict[int, np.ndarray] = {}
  for pos, (outer, start, stride) in enumerate(zip(outers, starts, strides, strict=True), start=1):
    formal = callee.inputs[pos]
    if outer.op == ExprOp.CONST and outer.value is not None and outer.type.dtype == dtypes.int64:
      flat = np.asarray(outer.value, dtype=np.int64).reshape(-1)
      offsets = start + stride * np.arange(length)[:, None] + np.arange(formal.size)[None, :]
      steps[pos] = flat[offsets].reshape((length, *formal.shape))
  return steps


def _ensure_callee(ctx: LowerCtx, callee: ConcreteFunction) -> None:
  if callee.device.kind != ctx.fun.device.kind:
    raise LoweringError(f"mixed-device CALL ({ctx.fun.device} -> {callee.device}) is deferred to a later migration step")
  if callee.extern is not None:
    # Opaque: the callee renders its own C (rule 6); its body is EXTERN_CALL, which has no lowering
    # rule. The Functions that C calls are lowered as usual.
    ctx.extern_fns[callee.name] = callee
    for dependency in callee.extern.dependencies():
      _ensure_callee(ctx, dependency)
    return
  if callee.name not in ctx.callees:
    ctx.callees[callee.name] = _lower_to_proc(callee, ctx.callees, ctx.extern_fns, ctx.target, observe_expr=ctx.observe_expr)


@lowers(ExprOp.CALL)
def _lower_call(ctx: LowerCtx, node: Expr) -> None:
  """An expression CALL output: emit one Program-IR CALL writing all callee outputs into scratch
  buffers (deduped per unique invocation), then map this node to the selected output buffer."""
  callee: ConcreteFunction = node.attrs["callee"]
  out_idx = int(node.attrs["output"])
  arg_names = tuple(ctx.value_buffers[a.id] for a in node.args)
  key = (callee.name, arg_names)
  if key not in ctx.call_invocations:
    _ensure_callee(ctx, callee)
    out_bufs = [ctx.new_private(o.type.dtype, o.shape) for o in callee.outputs]
    in_bufs = [ctx.buffers[n] for n in arg_names]
    ctx.emit(
      ProgramNode(
        ProgramOp.CALL, tuple(in_bufs + out_bufs), attrs={"callee": callee.name, "n_in": len(in_bufs), "n_out": len(out_bufs), "returns": ()}
      )
    )
    ctx.call_invocations[key] = tuple(b.attrs["name"] for b in out_bufs)
  ctx.value_buffers[node.id] = ctx.call_invocations[key][out_idx]


@lowers(ExprOp.VMAP)
def _lower_vmap(ctx: LowerCtx, node: Expr) -> None:
  """A ``length``-iteration loop calling the callee with pointer-offset VIEW args. Iteration ``it``
  reads ``outer_k[start_k + it·stride_k ...]`` and writes the selected output into ``out[it·slice_size ...]``."""
  callee: ConcreteFunction = node.attrs["callee"]
  out_idx = int(node.attrs["output"])
  length = int(node.attrs["length"])
  starts = tuple(int(s) for s in node.attrs["starts"])
  strides = tuple(int(s) for s in node.attrs["strides"])
  slice_size = int(node.attrs["slice_size"])
  _ensure_callee(ctx, callee)
  out = ctx.alloc_tmp(node)
  # Other callee outputs are written every iteration but discarded: one reused scratch each.
  scratch = [out if i == out_idx else ctx.new_private(o.type.dtype, o.shape) for i, o in enumerate(callee.outputs)]
  if length == 0:
    return
  loop = f"it_{out.attrs['name']}"
  rng = p.range_(loop, 0, length, kind=RangeKind.GLOBAL)
  it = p.var(loop)
  in_args = []
  for k, outer in enumerate(node.args):
    off = p.add(p.const_int(starts[k]), p.mul(p.const_int(strides[k]), it)) if strides[k] else p.const_int(starts[k])
    in_args.append(p.view(ctx.buf_of(outer), [off]))
  out_args = []
  for i, sbuf in enumerate(scratch):
    if i == out_idx:
      off = p.mul(it, p.const_int(slice_size)) if slice_size != 1 else it
      out_args.append(p.view(out, [off]))
    else:
      out_args.append(sbuf)
  call = ProgramNode(
    ProgramOp.CALL, tuple(in_args + out_args), attrs={"callee": callee.name, "n_in": len(in_args), "n_out": len(out_args), "returns": ()}
  )
  ctx.emit(p.for_(rng, [call]))


def _scan_key(node: Expr) -> tuple[object, ...]:
  return (id(node.attrs["callee"]), tuple(a.id for a in node.args), node.attrs["length"], node.attrs["starts"], node.attrs["strides"])


@lowers(ExprOp.SCAN)
def _lower_scan(ctx: LowerCtx, node: Expr) -> None:
  """One ``SERIAL`` loop per scan, shared by every output node of it, calling the body procedure.

  The carry lives in one buffer. Without a consumer of the per-step carries it holds two slots and
  step ``k`` reads slot ``k % 2`` and writes the other, so nothing is copied between steps and no
  call reads and writes the same memory. When reverse mode asked for the carries (output ``-1``) it
  holds ``length + 1`` slots and step ``k`` reads slot ``k`` and writes slot ``k + 1``: the
  trajectory is the carry storage itself. Stacked outputs are written in place at ``k * size``."""
  key = _scan_key(node)
  if key not in ctx.scan_invocations:
    ctx.scan_invocations[key] = _emit_scan(ctx, node)
  ctx.value_buffers[node.id] = ctx.scan_invocations[key][int(node.attrs["output"])]


def _emit_scan(ctx: LowerCtx, node: Expr) -> dict[int, str]:
  callee: ConcreteFunction = node.attrs["callee"]
  length, starts, strides = int(node.attrs["length"]), node.attrs["starts"], node.attrs["strides"]
  init, outers = node.args[0], node.args[1:]
  carry = callee.inputs[0]
  cs, dtype = carry.size, carry.type.dtype
  key = _scan_key(node)
  siblings = {int(n.attrs["output"]): n for n in topo(ctx.fun.outputs) if n.op == ExprOp.SCAN and _scan_key(n) == key}
  trajectory = -1 in siblings
  # A stacked output someone reads gets its own buffer (the Function's output buffer when it is one);
  # one nobody reads is written to a single reused slot.
  ys = {j: ctx.alloc_tmp(siblings[j]) if j in siblings else ctx.new_private(out.type.dtype, (out.size,)) for j, out in enumerate(callee.outputs) if j}
  bufs = {j: b.attrs["name"] for j, b in ys.items()}
  if length == 0:
    bufs[0] = ctx.value_buffers[init.id]
    bufs[-1] = ctx.new_private(dtype, (0,)).attrs["name"]
    return bufs
  proof = None if trajectory else _ensure_in_place_callee(ctx, callee, _loop_steps(callee, outers, starts, strides, length))
  in_place, scratch = proof if proof is not None else (None, 0)
  if in_place is None:
    _ensure_callee(ctx, callee)
  store = ctx.new_private(dtype, ((length + 1 if trajectory else 1 if in_place else 2) * cs + scratch,))
  ctx.emit(_copy_loop(ctx.buf_of(init), store, (cs,)))
  name = f"k_{store.attrs['name']}"
  k = p.var(name)
  if in_place is not None:
    # The body overwrites its carry where it changes it; the one slot is read and written.
    read = write = p.const_int(0)
    final = 0
  elif trajectory:
    read = p.mul(k, p.const_int(cs))
    write = p.mul(p.add(k, p.const_int(1)), p.const_int(cs))
    final = length * cs
  else:
    parity = p.mod(k, p.const_int(2))
    read = p.mul(parity, p.const_int(cs))
    write = p.mul(p.sub(p.const_int(1), parity), p.const_int(cs))
    final = (length % 2) * cs
  in_args = [p.view(store, [read])]
  counters: list[ProgramNode] = []
  for formal, outer, start, stride in zip(callee.inputs[1:], outers, starts, strides, strict=True):
    per_step = _constant_per_step(ctx, outer, formal, start, stride, length, k)
    if per_step is not None:
      in_args.append(_scalar_arg(ctx, formal.type.dtype, per_step, counters))
      continue
    offset = p.add(p.const_int(start), p.mul(p.const_int(stride), k)) if stride else p.const_int(start)
    in_args.append(p.view(ctx.buf_of(outer), [offset]))
  out_args = [p.view(store, [write])]
  out_args += [p.view(ys[j], [p.mul(k, p.const_int(out.size)) if j in siblings else p.const_int(0)]) for j, out in enumerate(callee.outputs) if j]
  call = ProgramNode(
    ProgramOp.CALL, tuple(in_args + out_args), attrs={"callee": in_place or callee.name, "n_in": len(in_args), "n_out": len(out_args), "returns": ()}
  )
  ctx.emit(p.for_(p.range_(name, 0, length, kind=RangeKind.SERIAL), [*counters, call]))
  bufs[0] = ctx.new_alias(dtype, carry.shape, store.attrs["name"], final).attrs["name"]
  if trajectory:
    bufs[-1] = ctx.new_alias(dtype, (length * cs,), store.attrs["name"], 0).attrs["name"]
  return bufs


def _constant_per_step(ctx: LowerCtx, outer: Expr, formal: Expr, start: int, stride: int, length: int, k: ProgramNode) -> ProgramNode | None:
  """The entry step ``k`` of a scan reads from a constant table one entry per step, computed instead
  of read where the table allows: arithmetic on the loop counter for an integer table (the step
  number, typically), which keeps only a non-affine residual as a table, and the value itself for
  a table whose entries along the walk are all equal (a cotangent of ones, typically). ``None``
  leaves the slice a read of the table."""
  if outer.op != ExprOp.CONST or formal.size != 1 or outer.value is None:
    return None
  values = np.asarray(outer.value).reshape(-1)[start + stride * np.arange(length)]
  if outer.type.dtype == dtypes.int64:
    return ctx.index_at(values.astype(np.int64), k)
  # Equal bit patterns, so that -0.0 and 0.0 stay apart (and a repeated NaN is still constant).
  bits = np.ascontiguousarray(values).view(np.uint8).reshape(values.size, -1)
  if outer.type.dtype.is_floating and values.size and np.all(bits == bits[0]):
    return p.const_float(float(values[0]), dtype=outer.type.dtype)
  return None


def _scalar_arg(ctx: LowerCtx, dtype: DType, value: ProgramNode, stores: list[ProgramNode]) -> ProgramNode:
  """A one-element buffer set to ``value`` each step (the store goes in ``stores``) and passed by
  pointer like any argument; once the call is inlined the C compiler keeps it in a register."""
  buf = ctx.new_private(dtype, ())
  slot = p.view(buf, [p.const_int(0)])
  stores.append(p.store(slot, value))
  return p.view(buf, [p.const_int(0)])


def _while_key(node: Expr) -> tuple[object, ...]:
  return (
    "while",
    id(node.attrs["callee"]),
    id(node.attrs["cond"]),
    tuple(a.id for a in node.args),
    node.attrs["max_iter"],
    node.attrs.get("index", False),
  )


@lowers(ExprOp.WHILE)
def _lower_while(ctx: LowerCtx, node: Expr) -> None:
  """A ``SERIAL`` loop of at most ``max_iter`` trips that calls the condition, leaves when it is
  false, and otherwise calls the body. The carry alternates between two slots as in ``scan``, or
  keeps every step when reverse mode reads them; the loop variable outlives the loop and is the
  step count. Because the last slot written is only known at run time, the final carry is copied
  out once after the loop. The loop's params are passed to both calls where they are, every step."""
  key = _while_key(node)
  if key not in ctx.scan_invocations:
    ctx.scan_invocations[key] = _emit_while(ctx, node, key)
  ctx.value_buffers[node.id] = ctx.scan_invocations[key][int(node.attrs["output"])]


def _emit_while(ctx: LowerCtx, node: Expr, key: tuple[object, ...]) -> dict[int, str]:
  body, cond = node.attrs["callee"], node.attrs["cond"]
  max_iter, init, params = int(node.attrs["max_iter"]), node.args[0], node.args[1:]
  index = bool(node.attrs.get("index", False))
  carry = body.inputs[0]
  cs, dtype = carry.size, carry.type.dtype
  trajectory = any(n.op == ExprOp.WHILE and n.attrs["output"] == -1 and _while_key(n) == key for n in topo(ctx.fun.outputs))
  steps = {1: np.arange(max_iter, dtype=np.int64)} if index else {}
  first = 1 + int(index)
  for i, param in enumerate(params):
    # An integer constant param is a table the in-place proof can read: the same value every step.
    # Without the step number nothing else varies either, so one step stands for all of them (a
    # table broadcast to every step made the proof's arithmetic cost max_iter times its size).
    if param.op == ExprOp.CONST and param.value is not None and param.type.dtype == dtypes.int64:
      value = np.asarray(param.value, dtype=np.int64).reshape(1, *param.shape)
      steps[first + i] = np.broadcast_to(value, (max_iter, *param.shape)) if index else value
  proof = None if trajectory else _ensure_in_place_callee(ctx, body, steps)
  in_place, scratch = proof if proof is not None else (None, 0)
  if in_place is None:
    _ensure_callee(ctx, body)
  _ensure_callee(ctx, cond)
  store = ctx.new_private(dtype, ((max_iter + 1 if trajectory else 1 if in_place else 2) * cs + scratch,))
  flag = ctx.new_private(dtypes.bool_, (1,))
  ctx.emit(_copy_loop(ctx.buf_of(init), store, (cs,)))
  name = f"k_{store.attrs['name']}"
  k = p.var(name)

  def slot(step: ProgramNode) -> ProgramNode:
    if in_place is not None:
      return p.const_int(0)
    return p.mul(step if trajectory else p.mod(step, p.const_int(2)), p.const_int(cs))

  read = p.view(store, [slot(k)])
  views = [p.view(ctx.buf_of(param), [p.const_int(0)]) for param in params]
  check = ProgramNode(ProgramOp.CALL, (read, *views, flag), attrs={"callee": cond.name, "n_in": 1 + len(views), "n_out": 1, "returns": ()})
  leave = p.break_if(ProgramNode(ProgramOp.NOT, (p.load(p.view(flag, [p.const_int(0)])),), dtype=dtypes.bool_))
  counters: list[ProgramNode] = []
  # A body that takes the step number gets the loop counter itself.
  extra = [_scalar_arg(ctx, dtypes.int64, k, counters)] if index else []
  step = ProgramNode(
    ProgramOp.CALL,
    (read, *extra, *views, p.view(store, [slot(p.add(k, p.const_int(1)))])),
    attrs={"callee": in_place or body.name, "n_in": 1 + len(extra) + len(views), "n_out": 1, "returns": ()},
  )
  ctx.emit(p.for_(p.range_(name, 0, max_iter, kind=RangeKind.SERIAL), [check, leave, *counters, step], exit_var=True))
  count = ctx.new_private(dtypes.float64, ())
  ctx.emit(p.store(p.view(count, [p.const_int(0)]), p.cast(k, dtypes.float64)))
  if not trajectory:
    # The final carry is the first slot, read in place: an in-place loop only has that one, and a
    # loop that stopped on an odd step moves its second slot there first (a loop of zero trips, a
    # gate that did not open, copies nothing).
    if in_place is None:
      c = p.var(f"c_{store.attrs['name']}")
      moved = p.store(p.view(store, [c]), p.load(p.view(store, [p.add(p.const_int(cs), c)])))
      ctx.emit(p.for_(p.range_(c.attrs["name"], 0, p.mul(p.mod(k, p.const_int(2)), p.const_int(cs)), kind=RangeKind.GLOBAL), [moved]))
    final = ctx.new_alias(dtype, carry.shape, store.attrs["name"], 0)
    return {0: final.attrs["name"], 1: count.attrs["name"]}
  final = ctx.new_private(dtype, carry.shape)
  c = p.var(f"c_{final.attrs['name']}")
  source = p.load(p.view(store, [p.add(slot(k), c)]))
  ctx.emit(p.for_(p.range_(c.attrs["name"], 0, cs, kind=RangeKind.GLOBAL), [p.store(p.view(final, [c]), source)]))
  bufs = {0: final.attrs["name"], 1: count.attrs["name"]}
  if trajectory:
    # Slots past the last step taken hold the final carry, so a backward pass that reads them sees
    # finite values; it discards what it computes from them.
    j = p.var(f"j_{store.attrs['name']}")
    fill = p.store(p.view(store, [p.add(p.mul(j, p.const_int(cs)), c)]), p.load(p.view(final, [c])))
    rng = p.range_(j.attrs["name"], p.add(k, p.const_int(1)), max_iter, kind=RangeKind.GLOBAL)
    ctx.emit(p.for_(rng, [p.for_(p.range_(c.attrs["name"], 0, cs, kind=RangeKind.GLOBAL), [fill])]))
    bufs[-1] = ctx.new_alias(dtype, (max_iter * cs,), store.attrs["name"], 0).attrs["name"]
  return bufs


@lowers(ExprOp.GATHER)
def _lower_gather(ctx: LowerCtx, node: Expr) -> None:
  """``out[k] = src[indices[k]]`` via a ``static const`` index table + one GLOBAL loop (any size)."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  vname = f"i_{out.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  k = p.var(vname)
  src_idx = ctx.index_at(idx, k)
  ctx.emit(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


@lowers(ExprOp.SEGMENT_REDUCE)
def _lower_segment_reduce(ctx: LowerCtx, node: Expr) -> None:
  """Fill the output, then fold each value into its bin in order: a sum (``scatter``, ``segment_sum``)
  or an extremum (``segment_max``, ``segment_min``)."""
  if node.attrs["reduce"] == "add":
    _lower_scatter(ctx, node)
  else:
    _lower_segment_extremum(ctx, node)


def _lower_scatter(ctx: LowerCtx, node: Expr) -> None:
  """Fill the output (with zeros, as ``scatter`` builds it), then ``out[indices[k]] += src[k]``, with
  the destination as arithmetic on ``k`` where it is affine and a ``static const`` table otherwise.
  The indices are fixed, so their pattern picks the loop: distinct destinations of a zero fill store
  (a parallel ``GLOBAL`` loop), and repeated ones accumulate ``out[d] = out[d] + src[k]`` in a
  ``REDUCE`` loop."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  zname = f"z_{out.attrs['name']}"
  zrng = p.range_(zname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  z = p.var(zname)
  fill = float(node.attrs["fill"])
  ctx.emit(p.for_(zrng, [p.store(p.view(out, [z]), p.const_float(fill, dtype=node.type.dtype))]))
  iname = f"i_{out.attrs['name']}"
  i = p.var(iname)
  dst = ctx.index_at(idx, i)
  value = p.load(p.view(ctx.buf_of(src), [i]))
  unique = scatter_is_unique(idx)
  if unique and fill == 0.0:
    ctx.emit(p.for_(p.range_(iname, 0, len(idx), kind=RangeKind.GLOBAL), [p.store(p.view(out, [dst]), value)]))
    return
  accumulate = p.store(p.view(out, [dst]), p.add(p.load(p.view(out, [dst])), value))
  ctx.emit(p.for_(p.range_(iname, 0, len(idx), kind=RangeKind.GLOBAL if unique else RangeKind.REDUCE), [accumulate]))


def _runtime_index(ctx: LowerCtx, idx: Expr, j: ProgramNode, n: int, *, in_range: bool = False) -> tuple[ProgramNode, ProgramNode | None]:
  """The index at lane ``j`` and whether it addresses an entry, ``0 <= i < n``; no test when the
  op promises every index is in range."""
  i = p.load(p.view(ctx.buf_of(idx), [j]))
  if in_range:
    return i, None
  inside = ProgramNode(ProgramOp.AND, (p.compare(ProgramOp.LE, p.const_int(0), i), p.compare(ProgramOp.LT, i, p.const_int(n))), dtype=dtypes.bool_)
  return i, inside


def _lane_loops(ctx: LowerCtx, tag: str, rows: int, lanes: int, kind: RangeKind, body: Any) -> None:
  """``for b < rows: for j < lanes: body(b, j)``, with the row loop left out for one row."""
  jname = f"j_{tag}"
  j = p.var(jname)
  if rows == 1:
    ctx.emit(p.for_(p.range_(jname, 0, lanes, kind=kind), body(p.const_int(0), j)))
    return
  bname = f"b_{tag}"
  b = p.var(bname)
  inner = p.for_(p.range_(jname, 0, lanes, kind=kind), body(b, j))
  ctx.emit(p.for_(p.range_(bname, 0, rows, kind=RangeKind.GLOBAL), [inner]))


@lowers(ExprOp.TAKE)
def _lower_take(ctx: LowerCtx, node: Expr) -> None:
  """``out[b, j] = x[b, i]`` for the run-time index ``i = indices[j]`` when ``0 <= i < n``, else the
  fill. The read goes through a clamped index, so no lane reads outside ``x`` whatever it holds."""
  x, idx = node.args
  n, lanes = x.shape[-1], idx.size
  rows = _size_of(node.shape) // lanes if lanes else 0
  out = ctx.alloc_tmp(node)
  if not rows or not lanes:
    return
  fill = constant(node.attrs["fill"], node.type.dtype)

  def body(b: ProgramNode, j: ProgramNode) -> list[ProgramNode]:
    if n == 0:
      value = fill
    else:
      i, inside = _runtime_index(ctx, idx, j, n, in_range=bool(node.attrs.get("in_range")))
      src = p.add(p.mul(b, p.const_int(n)), i if inside is None else p.select(inside, i, p.const_int(0)))
      value = p.load(p.view(ctx.buf_of(x), [src]))
      if inside is not None:
        value = p.select(inside, value, fill)
    # One row writes ``out[j]`` itself, the shape producer fusion recognizes.
    target = j if rows == 1 else p.add(p.mul(b, p.const_int(lanes)), j)
    return [p.store(p.view(out, [target]), value)]

  _lane_loops(ctx, out.attrs["name"], rows, lanes, RangeKind.GLOBAL, body)


@lowers(ExprOp.PUT_ADD, ExprOp.PUT)
def _lower_put(ctx: LowerCtx, node: Expr) -> None:
  """Copy the base, then add (or store) lane ``j`` of row ``b`` at ``[b, i]`` for ``i = indices[j]``.

  The result buffer has one scratch slot per lane after the ``rows * n`` entries, and a lane whose
  index is outside ``[0, n)`` writes its own slot. The write is then unconditional, and padded
  lanes never update one address in turn, which would chain every lane's read-modify-write through
  memory. Lanes run in order: repeated indices accumulate (``put_add``) or keep the last value.
  Constant indices lower through ``_lower_put_constant`` instead."""
  landing = put_lanes(node)
  if landing is not None:
    _lower_put_constant(ctx, node, *landing)
    return
  base, idx, values = node.args
  n, lanes = base.shape[-1], idx.size
  size = _size_of(node.shape)
  rows = size // n if n else 0
  if ctx.in_place is not None and node.id in ctx.in_place:
    # A link of a proven in-place chain: the carry itself, whose caller keeps the scratch slots.
    store = ctx.output_buffer(0)
    ctx.value_buffers[node.id] = store.attrs["name"]
  else:
    store = ctx.new_private(node.type.dtype, (size + rows * lanes,))
    ctx.value_buffers[node.id] = store.attrs["name"]
    if size:
      ctx.emit(_copy_loop(ctx.buf_of(base), store, node.shape))
  if not rows or not lanes:
    return

  def body(b: ProgramNode, j: ProgramNode) -> list[ProgramNode]:
    i, inside = _runtime_index(ctx, idx, j, n, in_range=bool(node.attrs.get("in_range")))
    entry = p.add(p.mul(b, p.const_int(n)), i)
    scratch = p.add(p.const_int(size), p.add(p.mul(b, p.const_int(lanes)), j))
    dst = p.view(store, [entry if inside is None else p.select(inside, entry, scratch)])
    value = p.cast(p.load(p.view(ctx.buf_of(values), [p.add(p.mul(b, p.const_int(lanes)), j)])), node.type.dtype)
    if node.op == ExprOp.PUT_ADD:
      value = p.add(p.load(dst), value)
    return [p.store(dst, value)]

  _lane_loops(ctx, store.attrs["name"], rows, lanes, RangeKind.REDUCE, body)


def _lower_put_constant(ctx: LowerCtx, node: Expr, lanes: np.ndarray, entries: np.ndarray) -> None:
  """A put at constant indices (``index_add``, ``index_set``): copy the base, then add (or store) each
  landing lane at its entry of every row, the entry as arithmetic on the lane where it is affine and
  a ``static const`` table otherwise, as a gather's source is. The lanes that drop are left out when
  the code is generated, so no lane is checked or needs a scratch slot, and distinct entries make a
  parallel ``GLOBAL`` loop. In a procedure that updates its carry in place (``LowerCtx.in_place``)
  every link of the update chain is the carry output itself, which the caller passes aliased to the
  carry input, so nothing is copied and only the indexed entries are touched."""
  base, _, values = node.args
  if ctx.in_place is not None and node.id in ctx.in_place:
    out = ctx.output_buffer(0)
    ctx.value_buffers[node.id] = out.attrs["name"]
  else:
    out = _reshaped_output(ctx, node) or ctx.alloc_tmp(node)
    if ctx.value_buffers[base.id] != out.attrs["name"]:
      ctx.emit(_copy_loop(ctx.buf_of(base), out, node.shape))
  n, width = node.shape[-1], values.shape[-1]
  rows = node.size // n if n else 0
  iname = f"u_{out.attrs['name']}_{ctx.fresh_id()}"
  i = p.var(iname)
  dst, src = ctx.index_at(entries, i), i if np.array_equal(lanes, np.arange(width)) else ctx.index_at(lanes, i)
  if rows != 1:
    b = p.var(f"b_{iname}")
    dst, src = p.add(p.mul(b, p.const_int(n)), dst), p.add(p.mul(b, p.const_int(width)), src)
  value = p.cast(p.load(p.view(ctx.buf_of(values), [src])), node.type.dtype)
  if node.op == ExprOp.PUT_ADD:
    value = p.add(p.load(p.view(out, [dst])), value)
  kind = RangeKind.GLOBAL if scatter_is_unique(entries) else RangeKind.REDUCE
  loop = p.for_(p.range_(iname, 0, lanes.size, kind=kind), [p.store(p.view(out, [dst]), value)])
  ctx.emit(loop if rows == 1 else p.for_(p.range_(f"b_{iname}", 0, rows, kind=RangeKind.GLOBAL), [loop]))


def _reshaped_output(ctx: LowerCtx, node: Expr) -> ProgramNode | None:
  """The buffer of an output that is ``node`` reshaped, now holding ``node``'s value, when ``node``
  is no output itself: ``index_add`` on a matrix returns its flat update reshaped, and the update
  written there directly needs no copy into the output, which the reshape aliases."""
  if node.id in ctx._output_alias:
    return None
  for expr in ctx.fun.outputs:
    if expr.op == ExprOp.RESHAPE and expr.args[0] is node and (name := ctx._output_alias.get(expr.id)) is not None:
      ctx.value_buffers[node.id] = name
      return ctx.buffers[name]
  return None


# Tests switch this off to compare every in-place loop with its two-slot version.
DONATE_CARRIES = True


def in_place_chain(fun: ConcreteFunction) -> tuple[int, ...] | None:
  """The update nodes (by id) through which ``fun`` may overwrite its carry in place, or None.

  ``fun`` takes the carry first and returns the next carry first. The next carry must be a chain
  of ``put``/``put_add`` at constant indices (``index_add``, ``index_set``) rooted at the carry
  input, ``u_0 = carry, u_i = update(u_{i-1})``, and each read of the chain must happen before the
  write that would change what it reads:

  - the values of update ``i`` read no chain link but ``u_{i-1}``, and none of the entries update
    ``i`` writes;
  - no other output reads any chain link.

  Which entries the values read is the structural pattern of the values with respect to
  ``u_{i-1}``, taken only over operations whose pattern is exactly what they read (the
  ``exact_reads`` trait).
  A path through anything else (a comparison, a ``select`` condition, ``copysign``'s sign, a call)
  counts as reading every entry. These are sufficient, not necessary; anything else keeps the
  two-slot carry.
  """
  from ..ad.sparsity import jacobian_sparsity
  from ..ir.expr import substitute

  chain = update_chain(fun)
  if chain is None or any(e.op not in (ExprOp.PUT_ADD, ExprOp.PUT) or put_lanes(e) is None for e in chain):
    return None
  carry = fun.inputs[0]
  links = [carry, *chain]
  link_ids = {e.id for e in links}

  def reads(expr: Expr) -> set[int]:
    """The chain links ``expr`` reads directly: a walk that stops at each link, whose own
    ancestors are read through the link, not by ``expr``."""
    found: set[int] = set()
    seen: set[int] = set()
    pending = [expr]
    while pending:
      node = pending.pop()
      if node.id in seen:
        continue
      seen.add(node.id)
      if node.id in link_ids:
        found.add(node.id)
      else:
        pending.extend(node.args)
    return found

  if any(reads(y) for y in fun.outputs[1:]):
    return None
  for before, update in zip(links[:-1], chain, strict=True):
    values = update.args[2]
    touched = reads(values)
    if touched - {before.id}:
      return None
    if touched:
      # Cut the graph at the link: a fresh symbol in its place is what the pattern is taken against.
      stand_in = Expr.sym("in_place_link", before.shape, dtype=before.type.dtype)
      cut = substitute(values, {before: stand_in})
      reaching: set[int] = set()
      for n in topo([cut]):  # children first, so one pass finds every node with the link below it
        if n is stand_in or any(a.id in reaching for a in n.args):
          reaching.add(n.id)
          if not expr_has_trait(n, "exact_reads"):
            return None
      read = set(jacobian_sparsity(cut, stand_in).cols)
      if read & set(_constant_writes(update).tolist()):
        return None
  return tuple(e.id for e in chain)


def _constant_writes(update: Expr) -> np.ndarray:
  """The flat entries a put at constant indices writes, in every row."""
  landing = put_lanes(update)
  assert landing is not None
  n = update.shape[-1]
  rows = update.size // n if n else 0
  return (np.arange(rows, dtype=np.int64)[:, None] * n + landing[1][None, :]).reshape(-1)


def update_chain(fun: ConcreteFunction) -> list[Expr] | None:
  """The next carry as a chain of updates rooted at the carry input, ``u_1 ... u_m`` in order, or
  None when it is not one. A reshape between two links is looked through: it keeps the flat order
  and lowers to an alias, so ``index_add`` on the flat view of a matrix carry is a link. Structure
  only: whether the chain may run in place is proven apart."""
  carry, node = fun.inputs[0], fun.outputs[0]
  chain: list[Expr] = []
  while node is not carry:
    if node.op == ExprOp.RESHAPE:
      node = node.args[0]
      continue
    if not has_trait(node.op, "update"):
      return None
    chain.append(node)
    node = node.args[0]
  return chain[::-1] or None


def in_place_steps(fun: ConcreteFunction, steps: dict[int, np.ndarray]) -> bool:
  """Whether a loop body's update chain may overwrite its carry in place, for the index values the
  loop feeds it: ``steps[p]`` is input ``p`` at every step, for the inputs sliced from constants
  (none: every index is a constant, the same at every step).

  Every index of every update, and every index through which the values read a chain link, must be
  computable from those inputs and constants alone; it is then computed for all steps at once. The
  values of update ``i`` may read link ``u_j`` (``j < i``) only through ``take``, ``gather`` or a
  slice, possibly of a reshape, and at every step the entries they read must be disjoint from the
  entries updates ``j + 1 ... i`` write. No other output may read a link. Sufficient, not
  necessary; the two-slot carry is kept otherwise.
  """
  chain = update_chain(fun)
  if chain is None:
    return False
  # With no input sliced per step, every index comes from constants and is the same at every
  # step: one step stands for all of them.
  length = next(iter(steps.values())).shape[0] if steps else 1
  if length == 0:
    return True  # a loop of no steps never runs its body
  carry = fun.inputs[0]
  links = [carry, *chain]
  position = {e.id: j for j, e in enumerate(links)}
  memo: dict[int, np.ndarray | None] = {}
  inputs = {e.id: pos for pos, e in enumerate(fun.inputs)}

  def value(e: Expr) -> np.ndarray | None:
    return _step_value(e, steps, inputs, length, memo)

  writes: list[Positions] = []  # per update: the positions it writes at each step
  for update in chain:
    w = op_def(update.op).traits["update"](update, value, length)
    if w is None:
      return False
    writes.append(w)

  def link_of(e: Expr) -> int | None:
    while e.op == ExprOp.RESHAPE and e.id not in position:
      e = e.args[0]
    return position.get(e.id)

  reads: list[tuple[int, int, Positions]] = []  # (update i, link j, positions)
  for i, update in enumerate(chain, start=1):
    # Everything the update reads but its base: its values, and for an update from a source (a
    # ragged run, say) that source.
    seen: set[int] = {update.id}
    pending = [update]
    while pending:
      node = pending.pop()
      if node is not update:
        if node.id in seen:
          continue
        seen.add(node.id)
        if link_of(node) is not None:
          return False  # the values use a whole link (or are one) beyond what the rule can bound
      for a_pos, arg in enumerate(node.args):
        if node is update and a_pos == 0:
          continue  # the base the update writes into
        j = link_of(arg)
        if j is None:
          pending.append(arg)
          continue
        read = op_def(node.op).traits.get("reads")
        positions = None if read is None else read(node, a_pos, value, length)
        if positions is None:
          return False
        reads.append((i, j, positions))
  for y in fun.outputs[1:]:
    for node in topo([y]):
      if any(link_of(a) is not None for a in node.args) or link_of(node) is not None:
        return False
  for i, j, positions in reads:
    for w in range(j + 1, i + 1):
      if _may_overlap(positions, writes[w - 1]):
        return False
  return True


class PositionRanges(Protocol):
  """Positions kept in a compact form: per-step ``bounds()`` and, only where those meet another's,
  ``explicit()`` positions as a ``(length, lanes)`` array padded with -1."""

  def bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

  def explicit(self) -> np.ndarray: ...


Positions = np.ndarray | PositionRanges
"""What an op's ``update`` or ``reads`` trait gives (``OpDef``): the positions it touches at each
step, a ``(length, lanes)`` array with -1 for a dropped lane, or ranges that produce one."""


def _bounds(x: Positions) -> tuple[np.ndarray, np.ndarray]:
  if not isinstance(x, np.ndarray):
    return x.bounds()
  valid = x >= 0
  low = np.where(valid, x, np.iinfo(np.int64).max).min(axis=1) if x.size else np.ones(x.shape[0], dtype=np.int64)
  high = np.where(valid, x, -1).max(axis=1) if x.size else np.zeros(x.shape[0], dtype=np.int64)
  return low, high


def _may_overlap(a: Positions, b: Positions) -> bool:
  """Whether any step's positions in ``a`` and ``b`` share an entry: first by per-step intervals,
  and only where those meet, entry by entry."""
  (alo, ahi), (blo, bhi) = _bounds(a), _bounds(b)
  meet = (alo <= ahi) & (blo <= bhi) & (alo <= bhi) & (blo <= ahi)
  if not meet.any():
    return False
  ea = a if isinstance(a, np.ndarray) else a.explicit()
  eb = b if isinstance(b, np.ndarray) else b.explicit()
  return _overlap(ea, eb)


def _flat_positions(idx: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
  """The flat positions a run-time index addresses in an array of ``shape`` at each step: row ``b``
  of the last axis at ``b * n + i``, and -1 for an index outside ``[0, n)``."""
  n = shape[-1]
  rows = int(np.prod(shape[:-1], dtype=np.int64))
  idx = idx.reshape(idx.shape[0], -1)
  inside = (idx >= 0) & (idx < n)
  flat = np.arange(rows)[None, :, None] * n + idx[:, None, :]
  return np.where(inside[:, None, :], flat, -1).reshape(idx.shape[0], -1)


StepValue = Callable[[Expr], "np.ndarray | None"]
"""An index's value at every step of the loop, or None when it is not computable from constants."""


def _constant_positions(indices: np.ndarray, length: int) -> np.ndarray:
  return np.broadcast_to(np.asarray(indices, dtype=np.int64).reshape(1, -1), (length, indices.size))


def _put_writes(node: Expr, value: StepValue, length: int) -> Positions | None:
  idx = value(node.args[1])
  return None if idx is None else _flat_positions(idx, node.shape)


def _take_reads(node: Expr, position: int, value: StepValue, length: int) -> Positions | None:
  idx = value(node.args[1]) if position == 0 else None
  return None if idx is None else _flat_positions(idx, node.args[0].shape)


def _slice_reads(node: Expr, position: int, value: StepValue, length: int) -> Positions:
  arg = node.args[0]
  flat = np.arange(arg.size).reshape(arg.shape)[node.attrs["index"]].reshape(1, -1)
  return np.broadcast_to(flat, (length, flat.size))


# How the builtin updates and partial reads bound what they touch (the ``update`` and ``reads``
# traits): what lets a loop's carry be updated in place.
for _op in (ExprOp.PUT_ADD, ExprOp.PUT):
  define_traits(_op, update=_put_writes)
define_traits(ExprOp.TAKE, reads=_take_reads)
define_traits(ExprOp.GATHER, reads=lambda node, position, value, length: _constant_positions(node.attrs["indices"], length))
define_traits(ExprOp.SLICE, reads=_slice_reads)
del _op


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
  """Whether, at any step (row), a position >= 0 appears in both ``a`` and ``b``."""
  if not a.size or not b.size:
    return False
  length = a.shape[0]
  span = int(max(a.max(), b.max())) + 1
  step_a = np.broadcast_to(np.arange(length)[:, None], a.shape)
  step_b = np.broadcast_to(np.arange(length)[:, None], b.shape)
  keys_a = np.unique((step_a * span + a)[a >= 0])
  keys_b = np.unique((step_b * span + b)[b >= 0])
  return np.intersect1d(keys_a, keys_b, assume_unique=True).size > 0


def _step_value(e: Expr, steps: dict[int, np.ndarray], inputs: dict[int, int], length: int, memo: dict[int, np.ndarray | None]) -> np.ndarray | None:
  """An integer expression's value at every step, shape ``(length, *e.shape)``, when it depends only
  on constants and the inputs in ``steps``; None otherwise."""
  if e.id in memo:
    return memo[e.id]
  memo[e.id] = None
  if e.type.dtype != dtypes.int64:
    return None
  args = [_step_value(a, steps, inputs, length, memo) if a.type.dtype == dtypes.int64 else None for a in e.args]
  out: np.ndarray | None = None
  vals = [a for a in args if a is not None]
  if e.op == ExprOp.CONST and e.value is not None:
    out = np.broadcast_to(np.asarray(e.value, dtype=np.int64).reshape(1, *e.shape), (length, *e.shape))
  elif e.op == ExprOp.INPUT and inputs.get(e.id) in steps:
    out = steps[inputs[e.id]]
  elif len(vals) != len(args) or not vals:
    out = None  # a float operand (a cast from data) or an unknown input
  elif e.op in (ExprOp.ADD, ExprOp.SUB, ExprOp.MUL, ExprOp.MINIMUM, ExprOp.MAXIMUM):
    fn = {ExprOp.ADD: np.add, ExprOp.SUB: np.subtract, ExprOp.MUL: np.multiply, ExprOp.MINIMUM: np.minimum, ExprOp.MAXIMUM: np.maximum}[ExprOp(e.op)]
    # Align each operand's own axes to the right of the step axis, as NumPy broadcasting would.
    x, y = (a.reshape(length, *(1,) * (len(e.shape) - len(a_e.shape)), *a_e.shape) for a, a_e in zip(vals, e.args, strict=True))
    out = fn(x, y)
  elif e.op == ExprOp.NEG:
    out = -vals[0]
  elif e.op == ExprOp.CAST:
    out = vals[0]
  elif e.op == ExprOp.RESHAPE:
    out = vals[0].reshape(length, *e.shape)
  elif e.op == ExprOp.SLICE:
    out = vals[0][(slice(None), *e.attrs["index"])]
  elif e.op == ExprOp.GATHER:
    out = vals[0].reshape(length, -1)[:, np.asarray(e.attrs["indices"]).reshape(-1)].reshape(length, *e.shape)
  elif e.op == ExprOp.TAKE:
    x, idx = vals
    n = e.args[0].shape[-1]
    fill = int(e.attrs["fill"])
    if not n:
      out = np.full((length, *e.shape), fill, dtype=np.int64)
    else:
      xr = x.reshape(length, -1, n)
      inside = (idx >= 0) & (idx < n)
      where = np.broadcast_to(np.where(inside, idx, 0)[:, None, :], (length, xr.shape[1], idx.shape[1]))
      out = np.where(inside[:, None, :], np.take_along_axis(xr, where, axis=2), fill).reshape(length, *e.shape)
  elif e.op in (ExprOp.CONCAT, ExprOp.STACK):
    axis = int(e.attrs.get("axis", 0)) + 1
    parts = [a.reshape(length, *a_e.shape) for a, a_e in zip(vals, e.args, strict=True)]
    out = np.concatenate(parts, axis=axis) if e.op == ExprOp.CONCAT else np.stack(parts, axis=axis)
  if out is not None and out.shape != (length, *e.shape):
    out = None
  memo[e.id] = out
  return out


def scatter_is_unique(idx: np.ndarray) -> bool:
  """Whether every destination of a fixed index table is distinct, so a plain store suffices."""
  return np.unique(idx).size == idx.size


def _lower_segment_extremum(ctx: LowerCtx, node: Expr) -> None:
  """Fill the output, then each value replaces its bin's entry when it is larger (smaller) or NaN."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  fname = f"z_{out.attrs['name']}"
  f = p.var(fname)
  fill = p.const_float(float(node.attrs["fill"]), dtype=node.type.dtype)
  ctx.emit(p.for_(p.range_(fname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL), [p.store(p.view(out, [f]), fill)]))
  iname = f"i_{out.attrs['name']}"
  i = p.var(iname)
  dst = ctx.index_at(idx, i)
  cur, value = p.load(p.view(out, [dst])), p.load(p.view(ctx.buf_of(src), [i]))

  def pick(cur: ProgramNode, value: ProgramNode) -> ProgramNode:
    better = p.compare(ProgramOp.LT, cur, value) if node.attrs["reduce"] == "max" else p.compare(ProgramOp.LT, value, cur)
    take = ProgramNode(ProgramOp.OR, (better, p.compare(ProgramOp.NE, value, value)), dtype=dtypes.bool_)
    return p.select(take, value, cur)

  starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]]) if idx.size else np.zeros(0, dtype=np.int64)
  if idx.size and np.all(np.diff(idx) >= 0) and starts.size * 2 <= idx.size:
    # Bins in runs (a CSC column order): each run reduces in a register and stores once, where
    # one loop over the entries waited on the previous store whenever an entry's bin repeated.
    # The same comparisons in the same order: the same result.
    stops = np.r_[starts[1:], idx.size]
    rname = f"r_{out.attrs['name']}"
    r = p.var(rname)
    run = p.view(ctx.new_private(node.type.dtype, ()), [p.const_int(0)])
    first = p.store(run, p.load(p.view(out, [ctx.index_at(idx[starts], r)])))
    step = p.store(run, pick(p.load(run), p.load(p.view(ctx.buf_of(src), [i]))))
    inner = p.for_(p.range_(iname, ctx.index_at(starts, r), ctx.index_at(stops, r), kind=RangeKind.REDUCE), [step])
    last = p.store(p.view(out, [ctx.index_at(idx[starts], r)]), p.load(run))
    ctx.emit(p.for_(p.range_(rname, 0, starts.size, kind=RangeKind.GLOBAL), [first, inner, last]))
    return
  kind = RangeKind.GLOBAL if scatter_is_unique(idx) else RangeKind.REDUCE
  ctx.emit(p.for_(p.range_(iname, 0, len(idx), kind=kind), [p.store(p.view(out, [dst]), pick(cur, value))]))


def _lower_tile(ctx: LowerCtx, node: Expr) -> bool:
  """A ``stack`` or ``concat`` along axis 0 of one input repeated (a tile: forward mode stacks a
  primal factor once per seed) as one loop, ``out[i] = src[i % size]``, where one copy loop per
  piece would be: a single producer loop is what ``fuse_elementwise`` can inline into the
  consumer, and ``delinearize_loops`` then splits the remainder away. False for any other."""
  src = node.args[0]
  if len(node.args) < 2 or int(node.attrs.get("axis", 0)) != 0 or any(a is not src for a in node.args):
    return False
  out = ctx.alloc_tmp(node)
  size = _size_of(src.shape)
  name = f"j_{out.attrs['name']}"
  j = p.var(name)
  rng = p.range_(name, 0, size * len(node.args), kind=RangeKind.GLOBAL)
  ctx.emit(p.for_(rng, [p.store(p.view(out, [j]), p.load(p.view(ctx.buf_of(src), [p.mod(j, p.const_int(size))])))]))
  return True


@lowers(ExprOp.STACK)
def _lower_stack(ctx: LowerCtx, node: Expr) -> None:
  """Stack ``n`` rank-r inputs along a new ``axis`` into a rank-(r+1) output: each input
  occupies index ``i`` along the new axis. Per element, decompose the input flat index into
  its coords, insert ``i`` at ``axis``, recombine against the output shape. A tile is one loop
  (``_lower_tile``)."""
  if _lower_tile(ctx, node):
    return
  axis = int(node.attrs.get("axis", 0))
  out_shape = node.shape
  out = ctx.alloc_tmp(node)
  for i, src in enumerate(node.args):
    src_shape = src.shape
    name = f"j_{out.attrs['name']}_{i}"
    rng = p.range_(name, 0, _size_of(src_shape), kind=RangeKind.GLOBAL)
    j = p.var(name)
    src_coords = [_coord_p(j, src_shape, d) for d in range(len(src_shape))]
    out_coords = src_coords[:axis] + [p.const_int(i)] + src_coords[axis:]
    dst = _flat_index_p(out_coords, out_shape)
    ctx.emit(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))


@lowers(ExprOp.CONCAT)
def _lower_concat(ctx: LowerCtx, node: Expr) -> None:
  """Concatenate inputs along ``axis``: each input keeps its shape but its ``axis`` coordinate is
  shifted by the running offset. Per element, decompose / shift / recombine against the output.
  A tile is one loop (``_lower_tile``)."""
  if _lower_tile(ctx, node):
    return
  axis = int(node.attrs.get("axis", 0))
  out_shape = node.shape
  out = ctx.alloc_tmp(node)
  offset = 0
  for i, src in enumerate(node.args):
    src_shape = src.shape
    name = f"j_{out.attrs['name']}_{i}"
    rng = p.range_(name, 0, _size_of(src_shape), kind=RangeKind.GLOBAL)
    j = p.var(name)
    coords = [_coord_p(j, src_shape, d) for d in range(len(src_shape))]
    if offset:
      coords[axis] = p.add(p.const_int(offset), coords[axis])
    dst = _flat_index_p(coords, out_shape)
    ctx.emit(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))
    offset += int(src_shape[axis])


__all__ = ["LoweringError", "LowerCtx", "lower_function", "lowers", "main_proc"]
