"""Own lowering state, rule registration, the loop-nest emitter, and the ConcreteFunction-to-program driver."""

from __future__ import annotations

from typing import Literal
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp, OP_INFO, _index_values, callees_of, gather, invocation_key, topo
from ...ir.match import Pattern, rewrite
from ...function.concrete import ConcreteFunction
from ...function.extern import _Extern
from ...function.model import Function, as_concrete
from ...function.tree import flat_tree
from ...ir.layout import SparseLayout, sparse_layout
from ..program import ProgramObserver, optimize_program
from ..program._common import rebuild_program, substitute_var, trip_count
from ...ir.program import ProgramNode, ProgramOp, RangeKind, walk_program
from ...ir.program_spec import verify_program
from ...ir.support import Support, _make_support, support_indices, specialize_support
from ...ir.types import DeviceSpec, DType, dtypes, stored_indices
from ...utils.names import NameScope
from ..affine import affine_index_map
from ..arith import constant
from ...ad.rules import ELEMENTWISE
from .fusion import ALIASES, _contiguous_slice_offset, plan_fusion
from ..arith import fold_program
from ..program.fold_arith import _fold_index as _fold_bounded
from ..expr import cse_many, simplify

_EXACT_STORES = (dtypes.float64, dtypes.float32, dtypes.int64, dtypes.bool_)
LowerRule = Callable[["LowerCtx", Expr], None]
ProcKey = ConcreteFunction | tuple[ConcreteFunction, tuple[Support, ...]]
_RULES: dict[ExprOp, LowerRule] = {}


def _with_support(buf: ProgramNode, support: Support) -> ProgramNode:
  # Support is for scalar expansion only; later buffer rewrites may invalidate it.
  if support is None or len(support) == _size_of(buf.attrs["shape"]):
    return buf
  return ProgramNode(buf.op, buf.args, {**buf.attrs, "support": support}, dtype=buf.dtype)


def lowers(*ops: ExprOp) -> Callable[[LowerRule], LowerRule]:
  """Register ``fn`` as the lowering rule for each expression op in ``ops``."""

  def deco(fn: LowerRule) -> LowerRule:
    for op in ops:
      if op in _RULES:
        raise ValueError(f"lowering rule already registered for {op}")
    for op in ops:
      _RULES[op] = fn
    return fn

  return deco


ExprObserver = Callable[[str, ConcreteFunction], None]


class LoweringError(NotImplementedError):
  """The lowerer (or Program-IR renderer) does not yet cover this op / case."""


def lower_function(
  fun: Function | ConcreteFunction,
  observe: ProgramObserver | None = None,
  observe_expr: ExprObserver | None = None,
  *,
  reciprocal: bool = False,
  lanes: Literal["auto"] | Literal[1, 2, 4, 8] = 1,
  frame_budget: int = 65536,
  reserved_names: Iterable[str] = (),
) -> ProgramNode:
  """Lower ``fun`` into a Program IR ``PROGRAM`` node (verified before return).

  The returned PROGRAM holds every lowered callee PROC in topological order
  followed by ``fun``'s main PROC last.

  """
  fun = as_concrete(fun)
  names, symbols, residuals, broadcasts = _function_names(fun, reserved_names)
  callees: dict[ProcKey, ProgramNode] = {}
  root = _lower_to_proc(fun, callees, names, symbols, residuals, broadcasts, auto_scalarize=False, observe_expr=observe_expr)
  prog = p.program([*callees.values(), root])
  externs = tuple(fn for fn in symbols if isinstance(fn, _Extern))
  prog = ProgramNode(
    prog.op,
    prog.args,
    {**prog.attrs, "function_symbols": tuple(symbols.items()), "workspace_dtype": workspace_dtype(fun), **({"externs": externs} if externs else {})},
    dtype=prog.dtype,
  )
  if observe is not None:
    observe("lowered", prog)
  prog = optimize_program(prog, observe=observe, reciprocal=reciprocal, lanes=lanes, frame_budget=frame_budget)
  verify_program(prog)
  return prog


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


def _declared_values(stored: Expr, layout: SparseLayout) -> Expr | None:
  """Dense values already in ``layout``'s order: the stored values when the layout is canonical, or,
  when they are gathered, their source gathered straight into the declared order."""
  order, identity = np.asarray(layout.order, dtype=np.int64), np.arange(len(layout.order))
  if np.array_equal(order, identity):
    return stored
  if stored.op != ExprOp.GATHER or stored.size != len(order) or stored.args[0].type.pattern is not None or stored.args[1].value is None:
    return None
  source, indices = stored.args[0], _index_values(stored).reshape(-1)[order]
  return (
    source
    if source.size == len(order) and np.array_equal(indices, identity)
    else gather(source, indices, mode=stored.attrs["mode"], fill_value=stored.attrs["fill_value"])
  )


def _normalize_function(fun: ConcreteFunction, *, residuals: bool = False) -> ConcreteFunction:
  """Simplify the body and drop its derivative rules; a called procedure also writes its residuals."""
  results = fun.results if residuals else fun.outputs
  outputs = tuple(rewrite(out, (Pattern(ExprOp.STOP_GRADIENT, lambda e: True, lambda e: e.args[0]),), revisit=True) for out in results)
  for _ in range(4):
    normalized = cse_many(simplify(output) for output in outputs)
    if all(new is old for new, old in zip(normalized, outputs, strict=True)):
      break
    outputs = normalized
  if len(outputs) == len(fun.outputs):
    return fun._replace(outputs=outputs, rules=None)
  extra = len(outputs) - len(fun.outputs)
  layouts = (
    *fun.output_tree.layouts,
    *(sparse_layout(out.type.pattern, "csr") if out.type.pattern is not None else None for out in outputs[len(fun.outputs) :]),
  )
  return fun._replace(
    outputs=outputs,
    output_tree=flat_tree(fun.result_names, tuple(out.type for out in outputs), layouts=layouts),
    output_coloring_widths=(*fun.output_coloring_widths, *(None,) * extra),
    rules=None,
  )


def _lower_to_proc(
  fun: ConcreteFunction,
  callees: dict[ProcKey, ProgramNode],
  names: NameScope,
  symbols: dict[ConcreteFunction, str],
  residuals: frozenset[ConcreteFunction],
  broadcasts: Mapping[ConcreteFunction, frozenset[frozenset[int]]],
  *,
  auto_scalarize: bool = True,
  observe_expr: ExprObserver | None = None,
  argument_support: tuple[Support, ...] | None = None,
  symbol: str | None = None,
  inout_carries: dict[int, tuple[Expr, ...]] | None = None,
) -> ProgramNode:
  lowering = fun._effective_lowering
  symbol = symbols[fun] if symbol is None else symbol
  invariant = broadcasts.get(fun, frozenset())
  # A reuse proof names nodes in the already normalized function.
  if inout_carries is None:
    fun = _normalize_function(fun, residuals=fun in residuals)
  if observe_expr is not None:
    observe_expr("normalized", fun)
  ctx = LowerCtx(fun, callees, names, symbols, residuals, broadcasts, observe_expr, invariant=invariant)
  if argument_support is not None:
    ctx.seed_pairs = True
    ctx.support_of = specialize_support(fun, argument_support, OP_INFO)
  ctx.inout_carries = inout_carries or {}
  nodes = fun.nodes
  # Float32 constants and operations retain their rounding during scalar substitution.
  ctx.scalarize = all(n.type.dtype in _EXACT_STORES for n in (*fun.inputs, *nodes)) and (
    lowering == "scalar" or lowering == "auto" and auto_scalarize
  )
  ctx.emit_inputs()
  ctx.register_outputs()
  ctx.emit_body()
  ctx.emit_outputs()
  proc = p.proc(symbol, ctx.params, ctx.statements)
  # ``input_count`` lets the renderer ``const``-qualify the first N (input) params of a ``_raw``
  # callee; emit_inputs runs before register_outputs, so inputs are the leading params.
  return ProgramNode(
    ProgramOp.PROC,
    proc.args,
    {
      **proc.attrs,
      "input_count": len(fun.inputs),
      **({"inout_carries": tuple(ctx.inout_carries)} if ctx.inout_carries else {}),
      "lowering": lowering,
      # Apple clang 17 (Xcode 16.4, macOS 15 arm64) miscompiles inlined forward-AD helpers for CALL
      # Jacobians, so they stay real call frames. See internal/notes/macos_clang_call_miscompile.md.
      **({"noinline": True} if fun.role == "forward" else {}),
      # A procedure may expand on its own and inside an expanding caller. Hoisted prologues use
      # "inline" because they only expand with their caller.
      "scalarize_mode": "procedure" if ctx.scalarize else "disabled",
    },
    dtype=proc.dtype,
  )


class LowerCtx:
  """Per-ConcreteFunction lowering state: buffers, statements, and the Expr-id -> buffer map."""

  def __init__(
    self,
    fun: ConcreteFunction,
    callees: dict[ProcKey, ProgramNode],
    names: NameScope,
    symbols: dict[ConcreteFunction, str],
    residuals: frozenset[ConcreteFunction],
    broadcasts: Mapping[ConcreteFunction, frozenset[frozenset[int]]],
    observe_expr: ExprObserver | None = None,
    *,
    invariant: frozenset[frozenset[int]] = frozenset(),
  ) -> None:
    self.fun = fun
    self.seed_pairs = fun.role == "forward"
    self.support_of: Callable[[Expr], Support] = lambda node: node.support
    self.scalarize = False
    self.callees = callees
    self.program_names = names
    self.names = names.child()
    self.symbols = symbols
    # Functions whose procedures also write their residuals, because some call reads one.
    self.residuals = residuals
    # Functions mapped or looped over with some arguments broadcast: the input positions each holds fixed.
    self.broadcasts = broadcasts
    self.input_names = tuple(self.names.allocate(n) for n in fun.input_names)
    self.output_names = tuple(self.names.allocate(n) for n in fun.output_names)
    self.observe_expr = observe_expr
    self.inout_carries: dict[int, tuple[Expr, ...]] = {}
    self.fused_scatter: dict[int, set[Expr]] = {}
    self.params: list[ProgramNode] = []
    self.statements: list[ProgramNode] = []
    self.buffers: dict[str, ProgramNode] = {}
    # Expr.id -> name of the buffer holding that value at runtime.
    self.value_buffers: dict[int, str] = {}
    # Output Expr.id -> output buffer name, so the body writes outputs in place.
    self._output_alias: dict[int, str] = {}
    self.used_results: dict[tuple, list[Expr]] = {}
    for node in fun.nodes:
      if callees_of(node):
        self.used_results.setdefault(invocation_key(node), []).append(node)
    self._tmp = 0
    self._orders: dict[int, np.ndarray] = {}
    self._canonical: dict[int, ProgramNode] = {}
    self._output_orders: dict[int, np.ndarray] = {}
    # What each output computes: its expression, or the values its declared layout stores as they are.
    self._computed: list[Expr] = []
    self._declared: dict[str, Expr] = {}
    self._shared_indices: dict[tuple[int, ...], ProgramNode] = {}
    self.invariant = invariant
    self.deferred: frozenset[int] = frozenset()
    self.sums: dict[int, tuple[Expr, ...]] = {}
    self._reads: dict[tuple[int, bool, tuple[ProgramNode, ...]], ProgramNode] = {}
    # Each index ``index_at`` built, with the table and the variable it indexes.
    self._indexed: dict[ProgramNode, tuple[np.ndarray, ProgramNode]] = {}
    # Nodes evaluated in place that read through a deferred transpose.
    self.permutes: set[int] = set()

  # --- declarations ---------------------------------------------------------

  def emit_inputs(self) -> None:
    for k, (name, expr, layout) in enumerate(zip(self.input_names, self.fun.inputs, self.fun.input_tree.layouts, strict=True)):
      buf = p.buffer(
        name, expr.type.dtype, _shape_or_scalar(expr.type.storage_shape), address_space="global", intent="inout" if k in self.inout_carries else None
      )
      self.params.append(buf)
      self.buffers[name] = buf
      self.value_buffers[expr.id] = name
      if layout is not None and expr.type.pattern is None:
        converted = self.reorder(buf, expr, layout.inverse_permutation)
        self.value_buffers[expr.id] = converted.attrs["name"]
      elif layout is not None:
        self._orders[expr.id] = layout.inverse_permutation

  def register_outputs(self) -> None:
    """Register output BUFFER params and alias each unique computed output Expr to
    its output buffer, so its rule writes directly into the output (no copy)."""
    seen: set[int] = set()
    for k, (name, expr, layout) in enumerate(zip(self.output_names, self.fun.outputs, self.fun.output_tree.layouts, strict=True)):
      if k in self.inout_carries:
        self._computed.append(expr)
        continue
      buf = _with_support(
        p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.storage_shape), address_space="global"),
        self.storage_support(expr, None if layout is None else layout.permutation),
      )
      self.params.append(buf)
      self.buffers[name] = buf
      if expr.op == ExprOp.PACK and isinstance(layout, SparseLayout) and (declared := _declared_values(expr.args[0], layout)) is not None:
        # These dense values, as they are, are the packed value in the declared order.
        self._declared[name] = declared
        expr, layout = declared, None
      self._computed.append(expr)
      if (
        expr.op in (ExprOp.INPUT, ExprOp.CONST)
        or expr.id in seen
        or (layout is not None and (expr.type.pattern is None or expr.op in (ExprOp.CALL, ExprOp.VMAP, ExprOp.LOOP)))
      ):
        continue  # INPUT/CONST or shared output: emit_outputs inserts the copy
      seen.add(expr.id)
      self._output_alias[expr.id] = name
      if layout is not None:
        self._output_orders[expr.id] = layout.inverse_permutation

  def emit_outputs(self) -> None:
    for k, (name, expr, layout) in enumerate(zip(self.output_names, self.fun.outputs, self.fun.output_tree.layouts, strict=True)):
      if k in self.inout_carries:
        continue
      if name in self._declared:
        expr, layout = self._declared[name], None
      src = self.value_buffers.get(expr.id)
      if src is None:
        raise LoweringError(f"output {name!r} expression was not lowered")
      if src == name or expr.type.storage_size == 0:
        continue  # already written in place, or nothing to copy
      source = self.raw_buf_of(expr)
      order = self._orders.get(expr.id)
      permutation = order if layout is None else layout.permutation
      if layout is not None and order is not None:
        permutation = order[permutation]
      if permutation is None:
        self.statements.append(_copy_loop(source, self.buffers[name], expr.type.storage_shape, self.names))
      else:
        self.reorder(source, expr, permutation, self.buffers[name])

  # --- body -----------------------------------------------------------------

  def emit_body(self) -> None:
    from .in_place import same_index_sources

    self.fused_scatter = {
      node.id: sources for chain in self.inout_carries.values() for node in chain if (sources := same_index_sources(self.fun, node))
    }
    deferred = set().union(*self.fused_scatter.values())
    self.deferred, self.sums = plan_fusion(self.fun, self.invariant, scalar=self.scalarize and self.fun._effective_lowering == "scalar")
    absorbed = {leaf.id for leaves in self.sums.values() for leaf in leaves}
    self.permutes = set()
    for node in topo(self._computed):
      if ExprOp(node.op) == ExprOp.TRANSPOSE or any(a.id in self.deferred and a.id in self.permutes for a in node.args):
        self.permutes.add(node.id)
    needed = set()
    for node in topo(self._computed):
      operands = node.args
      if node.op in (ExprOp.GATHER, ExprOp.SCATTER) and node.args[1].value is not None:
        operands = node.args[:1] if node.op == ExprOp.GATHER else (node.args[0], node.args[2])
      if node.op == ExprOp.SCATTER and node.args[0].support == frozenset():
        operands = tuple(arg for arg in operands if arg is not node.args[0])
      needed.update(operands)
    needed.update(self._computed)
    for node in topo(self._computed):
      if node in deferred:
        continue
      if node.op == ExprOp.CONST and node not in needed:
        continue
      if node.id in self.value_buffers or node.id in self.deferred or node.id in absorbed:
        continue  # input (or already lowered)
      rule = _RULES.get(ExprOp(node.op))
      if rule is None:
        raise LoweringError(f"Expression op {node.op!r} is not yet lowered to Program IR")
      for k, chain in self.inout_carries.items():
        if node in chain:
          self.value_buffers[node.id] = self.value_buffers[self.fun.inputs[k].id]
      rule(self, node)

  # --- helpers --------------------------------------------------------------

  def buf_of(self, expr: Expr) -> ProgramNode:
    source = self.raw_buf_of(expr)
    if expr.id in self._orders:
      if expr.id not in self._canonical:
        self._canonical[expr.id] = self.reorder(source, expr, self._orders[expr.id])
      return self._canonical[expr.id]
    return source

  def raw_buf_of(self, expr: Expr) -> ProgramNode:
    if expr.id not in self.value_buffers:
      self._store(expr)
    return self.buffers[self.value_buffers[expr.id]]

  def _store(self, expr: Expr) -> None:
    """Give a deferred value a buffer, for a consumer that needs it whole: its source's buffer for an
    alias, the constant's own rule, else one flat loop that evaluates it."""
    op = ExprOp(expr.op)
    if op in ALIASES:
      self.value_buffers[expr.id] = self.raw_buf_of(expr.args[0]).attrs["name"]
      return
    if op == ExprOp.CONST:
      _RULES[op](self, expr)
      return
    out = self.alloc_tmp(expr)
    ranges, (k,) = self.nest((expr.size,), f"i_{out.attrs['name']}")
    self.emit(ranges, [p.store(p.view(out, [k]), self.compute_flat(expr, k))])

  def physical_slots(self, expr: Expr, slots: np.ndarray) -> np.ndarray:
    order = self._orders.get(expr.id)
    if order is None or not len(order):
      return slots
    width = len(order)
    return np.where(slots < 0, -1, slots // width * width + order[slots % width])

  def read_stored(self, expr: Expr, slots: np.ndarray, k: ProgramNode) -> ProgramNode:
    if expr.type.pattern is None and expr.id not in self._orders:
      return self.read_flat(expr, self.index_at(slots, k, shared=True))
    return p.load(p.view(self.raw_buf_of(expr), [self.index_at(self.physical_slots(expr, slots), k, shared=True)]))

  def stored_view(self, expr: Expr, slot: ProgramNode) -> ProgramNode:
    if (order := self._orders.get(expr.id)) is not None and len(order):
      width = len(order)
      if slot.op == ProgramOp.CONST_INT:
        index = slot.attrs["value"]
        slot = p.const_int(index // width * width + int(order[index % width]))
      else:
        slot = p.add(p.mul(p.div(slot, p.const_int(width)), p.const_int(width)), self.index_at(order, p.mod(slot, p.const_int(width)), shared=True))
    return p.view(self.raw_buf_of(expr), [slot])

  def new_private(self, dtype: DType, shape: tuple[int, ...], *, support: Support = None) -> ProgramNode:
    """Allocate a fresh private scratch BUFFER (declared as a local array by the renderer)."""
    name = self.names.allocate(f"t{self._tmp}")
    self._tmp += 1
    buf = _with_support(p.buffer(name, dtype, _shape_or_scalar(shape), address_space="private"), support)
    if self.seed_pairs:
      buf = ProgramNode(buf.op, buf.args, {**buf.attrs, "per_call": True}, dtype=buf.dtype)
    self.buffers[name] = buf
    self.statements.append(buf)  # marks the local-array declaration for the renderer
    return buf

  def alloc_tmp(self, expr: Expr) -> ProgramNode:
    """Buffer to hold ``expr``'s value: the output buffer if aliased, else a fresh private temp."""
    if expr.id in self.value_buffers:
      return self.raw_buf_of(expr)
    alias = self._output_alias.get(expr.id)
    if alias is not None:
      self.value_buffers[expr.id] = alias
      if expr.id in self._output_orders:
        self._orders[expr.id] = self._output_orders[expr.id]
      return self.buffers[alias]
    buf = self.new_private(expr.type.dtype, expr.type.storage_shape, support=self.storage_support(expr))
    self.value_buffers[expr.id] = buf.attrs["name"]
    return buf

  def storage_support(self, expr: Expr, permutation: np.ndarray | None = None) -> Support:
    support = self.support_of(expr)
    if support is None or expr.type.pattern is None and permutation is None:
      return support
    positions = np.arange(expr.type.storage_size) if expr.type.pattern is None else stored_indices(expr.type)
    if permutation is not None:
      width = len(permutation)
      positions = positions.reshape(-1, width)[:, permutation].reshape(-1) if width else positions
    return _make_support(np.flatnonzero(np.isin(positions, support_indices(support))))

  def reorder(self, source: ProgramNode, expr: Expr, permutation: np.ndarray, target: ProgramNode | None = None) -> ProgramNode:
    """Copy stored batches between canonical order and a leaf's declared order."""
    if target is None and np.array_equal(permutation, np.arange(len(permutation))):
      return source
    target = self.new_private(expr.type.dtype, expr.type.storage_shape) if target is None else target
    if expr.type.storage_size:
      ranges, (k,) = self.nest((expr.type.storage_size,), f"layout_{self._tmp}")
      width = len(permutation)
      slot = p.add(p.mul(p.div(k, p.const_int(width)), p.const_int(width)), self.index_at(permutation, p.mod(k, p.const_int(width))))

      def at(buffer: ProgramNode, index: ProgramNode) -> ProgramNode:
        if buffer.op == ProgramOp.VIEW:
          return p.view(self.buffers[buffer.attrs["buffer"]], [p.add(buffer.args[0], index)])
        return p.view(buffer, [index])

      self.emit(ranges, [p.store(at(target, k), p.load(at(source, slot)))])
    return target

  def new_alias(self, dtype: DType, shape: tuple[int, ...], src_name: str, offset: int) -> ProgramNode:
    """A zero-copy private BUFFER that aliases ``src_name`` at a flat ``offset`` (rendered
    ``const T* tN = <src> + offset;``). Carries ``alias_of`` / ``alias_offset`` so the workspace
    pass leaves it unpacked and keeps its source live. Port of ``codegen/c.py``'s contiguous
    SLICE / RESHAPE pointer aliasing."""
    name = self.names.allocate(f"t{self._tmp}")
    self._tmp += 1
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

  def new_const_index(self, idx: Iterable[int], *, shared: bool = False) -> ProgramNode:
    """A read-only int64 index table (for GATHER/SCATTER), declared ``static const``."""
    values = [int(v) for v in idx]
    key = tuple(values) if shared else ()
    if shared and key in self._shared_indices:
      return self._shared_indices[key]
    buf = p.const_buffer(self.names.allocate(f"k{self._tmp}", generated=True), dtypes.int64, (len(values),), values)
    buf = ProgramNode(buf.op, buf.args, {**buf.attrs, "index_table": True}, dtype=buf.dtype)
    self._tmp += 1
    self.buffers[buf.attrs["name"]] = buf
    self.statements.append(buf)
    if shared:
      self._shared_indices[key] = buf
    return buf

  def index_at(self, idx: np.ndarray, k: ProgramNode, *, shared: bool = False) -> ProgramNode:
    """The GATHER source (or SCATTER destination) for element ``k``, as arithmetic where it can be.

    Every range whose contribution is affine becomes a term over ``k``; only the non-affine
    residual is materialized, and a residual of one element is a constant, so a fully affine index
    emits no table at all. An index with no affine structure keeps the whole table, indexed by
    ``k``, which is what every gather used to emit. An empty index is any constant: its loop runs
    zero times. A result that loads a table remembers ``idx`` and ``k``, so a movement op read at it
    composes its own map with ``idx`` into one table."""
    out = self._index_at(idx, k, shared=shared)
    if any(n.op == ProgramOp.LOAD for n in walk_program(out)):
      self._indexed[out] = (idx, k)
    return out

  def _index_at(self, idx: np.ndarray, k: ProgramNode, *, shared: bool = False) -> ProgramNode:
    if idx.size == 0:
      return p.const_int(0)
    if k.op == ProgramOp.CONST_INT:
      return p.const_int(int(idx[k.attrs["value"]]))
    amap = affine_index_map(idx)
    period = len(amap.residual)
    out: ProgramNode | None = None
    if period > 1:
      table = self.new_const_index(amap.residual, shared=shared)
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

  # --- loop nests -------------------------------------------------------------
  #
  # A kernel is a nest of ranges, a scalar body and its writes. ``nest`` and ``reduce`` give the
  # ranges and one coordinate per axis; an access map is the tuple of coordinates a kernel passes
  # to ``read`` or ``write``, each affine in the loop variables or a table lookup from
  # ``index_at``; ``emit`` places the nest. Bounds may be program expressions, such as the
  # ``rowptr[i]`` to ``rowptr[i + 1]`` of a sparse row, and a coordinate need not start at zero.

  def nest(
    self, extents: Sequence[int | tuple[ProgramNode, ProgramNode]], name: str, *, kind: RangeKind = RangeKind.GLOBAL
  ) -> tuple[list[ProgramNode], list[ProgramNode]]:
    """One range per axis of ``extents`` and the coordinate of each axis, outermost first.

    An extent is a static length or a ``(start, stop)`` pair of index expressions. A static axis of
    length one gets no range and coordinate zero. A domain with no range left runs once, as one
    unit loop, so that every kernel stays a loop for the program passes before ``unroll_unit_loops``.
    """
    loops = [k for k, extent in enumerate(extents) if extent != 1]
    ranges: list[ProgramNode] = []
    coords: list[ProgramNode] = []
    for k, extent in enumerate(extents):
      if extent == 1:
        coords.append(p.const_int(0))
        continue
      var = self.names.allocate(name if k == loops[-1] else f"{name}_{k}")
      start, stop = (0, extent) if isinstance(extent, int) else extent
      ranges.append(p.range_(var, start, stop, kind=kind))
      coords.append(p.var(var))
    if not ranges:
      ranges.append(p.range_(self.names.allocate(name), 0, 1, kind=kind))
    return ranges, coords

  def reduce(self, extents: Sequence[int | tuple[ProgramNode, ProgramNode]], name: str) -> tuple[list[ProgramNode], list[ProgramNode]]:
    """``nest`` over reduction axes: every iteration updates the same outputs, in order."""
    return self.nest(extents, name, kind=RangeKind.REDUCE)

  def read(self, expr: Expr, coords: Sequence[ProgramNode]) -> ProgramNode:
    """The scalar value of ``expr`` at dense logical ``coords``: a load where it is stored, else the
    value evaluated in place."""
    if expr.type.pattern is not None:
      raise LoweringError("sparse logical reads require an explicit stored-slot map")
    return self._resolve((expr, False, tuple(coords)))

  def read_flat(self, expr: Expr, flat: ProgramNode) -> ProgramNode:
    """``read`` at the row-major position ``flat``."""
    return self._resolve((expr, True, (flat,)))

  def compute(self, expr: Expr, coords: Sequence[ProgramNode]) -> ProgramNode:
    """``expr``'s own operation at ``coords``, reading its arguments, whether or not it is stored."""
    return self._finish(self._step((expr, False, tuple(coords))))

  def compute_flat(self, expr: Expr, flat: ProgramNode) -> ProgramNode:
    """``compute`` at the row-major position ``flat``."""
    return self._finish(self._step((expr, True, (flat,))))

  def _finish(self, step: _Step) -> ProgramNode:
    if isinstance(step, ProgramNode):
      return step
    reads, build = step
    return build([self._resolve(r) for r in reads])

  def _resolve(self, request: _Read) -> ProgramNode:
    """The value a read asks for, memoized on ``(expr, coordinates)``. Evaluated with an explicit
    stack, so a long chain of inlined values does not become depth on the Python stack."""
    key = lambda r: (r[0].id, r[1], r[2])
    stack, plans = [request], {}
    while stack:
      top = stack[-1]
      k = key(top)
      if k in self._reads:
        stack.pop()
        continue
      if k not in plans:
        expr, flat, at = top
        if expr.id in self.value_buffers or expr.id not in self.deferred:
          step: _Step = p.load(p.view(self.buf_of(expr), [at[0] if flat else _flat(expr.shape, at)]))
        else:
          step = self._step(top)
        if isinstance(step, ProgramNode):
          self._reads[k] = step
          stack.pop()
          continue
        plans[k] = step
      reads, build = plans[k]
      missing = [r for r in reads if key(r) not in self._reads]
      if missing:
        stack.extend(missing)
        continue
      self._reads[k] = build([self._reads[key(r)] for r in reads])
      stack.pop()
    return self._reads[key(request)]

  def _step(self, request: _Read) -> _Step:
    """One step of evaluating an expression in place: its value, or the reads of its arguments and
    how to combine them. An elementwise operation reads each operand at the broadcast coordinates, a
    constant is its value or a table load, and a movement op is an index remap. A reshape maps each
    group of axes it merges or splits on its own, so quotients and remainders appear only on the axes
    a merge reads through a transpose or a slice. A flat position reaches an operand of the same size,
    an alias and a gather as it is."""
    expr, flat, at = request
    op = ExprOp(expr.op)
    if op in ELEMENTWISE:
      coords = _flat_coordinates(at[0], expr.shape) if flat and any(a.size != expr.size for a in expr.args) else at
      reads = [(a, True, at) if flat and a.size == expr.size else (a, False, _broadcast(coords, a.shape)) for a in expr.args]
      return reads, lambda values: ProgramNode(ELEMENTWISE[op].program, tuple(values), dtype=expr.type.dtype)
    position = at[0] if flat else _flat(expr.shape, at)
    if op == ExprOp.CONST:
      if (value := _uniform(expr)) is not None:
        return constant(value, expr.type.dtype)
      return p.load(p.view(self.raw_buf_of(expr), [position]))
    src = expr.args[0]
    if op in ALIASES:
      if not flat and src.id in self.deferred and src.id not in self.value_buffers and ExprOp(src.op) in (ExprOp.TRANSPOSE, ExprOp.SLICE):
        return [(src, False, tuple(_reshape_coords(at, expr.shape, src.shape)))], _only
      return [(src, True, (position,))], _only
    if op == ExprOp.SLICE and flat and (offset := _contiguous_slice_offset(expr.attrs["index"], src.shape, expr.shape)) is not None:
      return [(src, True, (_shift(position, offset),))], _only
    if op == ExprOp.GATHER or op in (ExprOp.TRANSPOSE, ExprOp.SLICE) and flat and position in self._indexed:
      table = _source_positions(expr)
      if (known := self._indexed.get(position)) is not None and known[0].max() < len(table):
        return [(src, True, (self.index_at(table[known[0]], known[1]),))], _only
      return [(src, True, (self.index_at(table, position),))], _only
    coords = _flat_coordinates(at[0], expr.shape) if flat else at
    if op == ExprOp.TRANSPOSE:
      axes = tuple(int(a) for a in expr.attrs["axes"])
      return [(src, False, tuple(coords[axes.index(dim)] for dim in range(len(axes))))], _only
    if op == ExprOp.SLICE:
      return [(src, False, tuple(_slice_coords(expr.attrs["index"], src.shape, coords)))], _only
    raise LoweringError(f"{op} is not evaluated in place")

  def write(self, expr: Expr, coords: Sequence[ProgramNode], value: ProgramNode) -> ProgramNode:
    """The statement storing ``value`` at dense logical ``coords``."""
    if expr.type.pattern is not None:
      raise LoweringError("sparse logical writes require an explicit stored-slot map")
    return p.store(p.view(self.buf_of(expr), [_flat(expr.shape, coords)]), value)

  def emit(self, ranges: Sequence[ProgramNode], body: Sequence[ProgramNode], *, delinearize: bool = False) -> None:
    """Append ``body`` inside ``ranges``, outermost first.

    See ``nest_body`` for empty ranges, merging and unit-loop indices, and ``delinearize`` for the
    loops of a kernel that reads through a transpose.
    """
    if delinearize:
      ranges, body = self.delinearize(ranges, body)
    self.statements.extend(self.nest_body(ranges, body))

  def delinearize(self, ranges: Sequence[ProgramNode], body: Sequence[ProgramNode]) -> tuple[list[ProgramNode], list[ProgramNode]]:
    """Split each loop whose variable the body divides into one loop per digit the divisors mark, so
    a read through a transpose indexes by affine sums; ``nest_body`` merges back the digits every
    access walks contiguously.

    The block sizes are the divisors of ``k / b`` and ``k % b`` and the products ``s * e`` of
    ``(k / s) % e``. When each divides the next and the trip count, ``k`` is the sum of its digits
    times their block sizes, and the quotients and remainders fold away under the digits' bounds."""
    out: list[ProgramNode] = []
    body = list(body)
    bounds = {r.attrs["name"]: (int(r.args[0].attrs["value"]), int(r.args[1].attrs["value"]) - 1) for r in ranges if trip_count(r)}
    for rng in ranges:
      count, name = trip_count(rng), rng.attrs["name"]
      var = p.var(name)
      blocks = {count}
      for n in (n for stmt in body for n in walk_program(stmt)):
        if n.op in (ProgramOp.DIV, ProgramOp.MOD) and n.args[1].op == ProgramOp.CONST_INT:
          x, c = n.args[0], int(n.args[1].attrs["value"])
          if x is var:
            blocks.add(c)
          elif x.op == ProgramOp.DIV and x.args[0] is var and x.args[1].op == ProgramOp.CONST_INT:
            blocks.add(int(x.args[1].attrs["value"]) * c)
      sizes = sorted(b for b in blocks if b is not None and 1 < b)
      edges = [1, *sizes]
      if (
        count is None
        or len(sizes) < 2
        or rng.args[0] != p.const_int(0)
        or rng.args[2] != p.const_int(1)
        or rng.attrs.get("carried")
        or sizes[-1] != count
        or any(b % a for a, b in zip(edges, sizes))
      ):
        out.append(rng)
        continue
      digits = [self.names.allocate(f"{name}_{k}") for k in range(len(sizes))]
      index: ProgramNode | None = None
      for digit, edge, top in zip(digits, edges, sizes):
        bounds[digit] = (0, top // edge - 1)
        term = p.var(digit) if edge == 1 else p.mul(p.var(digit), p.const_int(edge))
        index = term if index is None else p.add(term, index)
      assert index is not None
      fold = Pattern[ProgramNode](None, lambda n: n.op in (ProgramOp.DIV, ProgramOp.MOD), lambda n: _fold_bounded(n, bounds) or n)
      body = [_fold_index(rewrite(substitute_var(stmt, name, index), [fold], rebuild=rebuild_program)) for stmt in body]
      out += [p.range_(digit, 0, top // edge, kind=rng.attrs["kind"]) for digit, edge, top in reversed(list(zip(digits, edges, sizes)))]
    return out, body

  def nest_body(self, ranges: Sequence[ProgramNode], body: Sequence[ProgramNode]) -> list[ProgramNode]:
    """Return ``body`` inside ``ranges``, outermost first.

    A static empty range drops the whole nest, so an empty tensor emits no store. Adjacent ranges
    merge where every access is contiguous across them. A unit loop from zero offsets each constant
    view index by its variable, which is zero, the flat form of a one-element domain the program
    passes match on."""
    if any(trip_count(r) == 0 for r in ranges):
      return []
    merged, stmts = _merge(list(ranges), list(body))
    if len(merged) == 1 and trip_count(merged[0]) == 1 and merged[0].args[0] == p.const_int(0):
      unit = p.var(merged[0].attrs["name"])

      def offset(view: ProgramNode) -> ProgramNode:
        index = view.args[0]
        return ProgramNode(ProgramOp.VIEW, (unit if index.attrs["value"] == 0 else p.add(index, unit),), view.attrs, dtype=view.dtype)

      pattern = Pattern(ProgramOp.VIEW, lambda v: v.args[0].op == ProgramOp.CONST_INT, offset)
      stmts = [rewrite(stmt, [pattern], rebuild=rebuild_program, fixpoint=False) for stmt in stmts]
    for rng in reversed(merged):
      stmts = [p.for_(rng, stmts)]
    return stmts


def _stride(shape: tuple[int, ...], dim: int) -> int:
  s = 1
  for d in shape[dim + 1 :]:
    s *= int(d)
  return s


def _flat(shape: tuple[int, ...], coords: Sequence[ProgramNode]) -> ProgramNode:
  """The row-major element index of ``coords`` in a tensor of ``shape``: strides times coordinates."""
  if len(coords) != len(shape):
    raise LoweringError(f"{len(coords)} coordinates for a tensor of shape {shape}")
  offset = 0
  out: ProgramNode | None = None
  for dim, coord in enumerate(coords):
    stride = _stride(shape, dim)
    if coord.op == ProgramOp.CONST_INT:
      offset += int(coord.attrs["value"]) * stride
      continue
    term = coord if stride == 1 else p.mul(coord, p.const_int(stride))
    out = term if out is None else p.add(out, term)
  if out is None:
    return p.const_int(offset)
  return p.add(p.const_int(offset), out) if offset else out


def _source_positions(expr: Expr) -> np.ndarray:
  """The flat position in its source of each element of a transpose, a slice or a gather."""
  src = expr.args[0]
  if ExprOp(expr.op) == ExprOp.GATHER:
    return expr.attrs["indices"].reshape(-1)
  positions = np.arange(src.size, dtype=np.int64).reshape(src.shape)
  if ExprOp(expr.op) == ExprOp.TRANSPOSE:
    return positions.transpose(expr.attrs["axes"]).reshape(-1)
  return positions[expr.attrs["index"]].reshape(-1)


_Read = tuple[Expr, bool, tuple[ProgramNode, ...]]
_Step = ProgramNode | tuple[list[_Read], Callable[[list[ProgramNode]], ProgramNode]]


def _only(values: list[ProgramNode]) -> ProgramNode:
  return values[0]


def _broadcast(coords: Sequence[ProgramNode], shape: tuple[int, ...]) -> tuple[ProgramNode, ...]:
  """The coordinates an operand of ``shape`` is read at under broadcasting: the trailing ones, zero on a unit axis."""
  return tuple(c if d != 1 else p.const_int(0) for c, d in zip(coords[len(coords) - len(shape) :], shape, strict=True))


def _uniform(expr: Expr) -> int | float | None:
  """The one value of a constant whose entries are all equal, else None."""
  value = expr.value
  assert value is not None
  flat = value.reshape(-1)
  return flat[0].item() if flat.size and flat.tobytes() == flat[:1].tobytes() * flat.size else None


def _slice_coords(index: tuple[object, ...], src_shape: tuple[int, ...], coords: Sequence[ProgramNode]) -> list[ProgramNode]:
  """The source coordinates a slice reads at its output ``coords``: an integer index is a constant, a
  slice ``start + step * coord``."""
  source: list[ProgramNode] = []
  out_dim = 0
  for dim, item in enumerate(index):
    if isinstance(item, int):
      source.append(p.const_int(item if item >= 0 else int(src_shape[dim]) + item))
      continue
    assert isinstance(item, slice)
    start, _stop, step = item.indices(int(src_shape[dim]))
    source.append(_shift(coords[out_dim], start, step))
    out_dim += 1
  return source


def _reshape_coords(coords: Sequence[ProgramNode], shape: tuple[int, ...], src_shape: tuple[int, ...]) -> list[ProgramNode]:
  """The source coordinates of a reshape from ``src_shape`` to ``shape`` at ``coords``.

  Axes pair into the smallest groups of equal size on both sides. A group's position is affine in its
  output coordinates, and splitting it back costs a quotient and a remainder only on a source group
  of more than one axis."""
  if 0 in shape:
    return [p.const_int(0)] * len(src_shape)
  out: list[ProgramNode] = []
  i = j = 0
  while j < len(src_shape):
    gi, gj, a, b = [], [j], 1, int(src_shape[j])
    j += 1
    while a != b:
      if a < b:
        gi.append(i)
        a *= int(shape[i])
        i += 1
      else:
        gj.append(j)
        b *= int(src_shape[j])
        j += 1
    position = _flat(tuple(int(shape[k]) for k in gi), [coords[k] for k in gi])
    out += _flat_coordinates(position, tuple(int(src_shape[k]) for k in gj))
  return out


def _flat_coordinates(k: ProgramNode, shape: tuple[int, ...]) -> list[ProgramNode]:
  """The coordinates of row-major position ``k`` in ``shape``, as quotients and remainders."""
  coords = []
  for dim, extent in enumerate(shape):
    stride = _stride(shape, dim)
    if k.op == ProgramOp.CONST_INT:
      coords.append(p.const_int(int(k.attrs["value"]) // stride % int(extent) if extent else 0))
      continue
    quotient = k if stride == 1 else p.div(k, p.const_int(stride))
    coords.append(p.const_int(0) if extent == 1 else quotient if dim == 0 else p.mod(quotient, p.const_int(int(extent))))
  return coords


def _shift(coord: ProgramNode, start: int, step: int = 1) -> ProgramNode:
  """The coordinate ``start + step * coord``, folded when ``coord`` is a constant."""
  if coord.op == ProgramOp.CONST_INT:
    return p.const_int(start + step * int(coord.attrs["value"]))
  scaled = coord if step == 1 else p.mul(coord, p.const_int(step))
  return scaled if start == 0 else p.add(p.const_int(start), scaled)


def _coefficients(index: ProgramNode, names: frozenset[str]) -> dict[str, int] | None:
  """The coefficient of each variable in ``names`` if ``index`` is affine in them, else None.

  A literal constant term is reported under the key ``""``; a subterm free of ``names`` counts as
  zero."""
  if index.op == ProgramOp.CONST_INT:
    return {"": int(index.attrs["value"])}
  if index.op == ProgramOp.VAR:
    return {index.attrs["name"]: 1} if index.attrs["name"] in names else {}
  if index.op in (ProgramOp.ADD, ProgramOp.SUB):
    left, right = (_coefficients(a, names) for a in index.args)
    if left is None or right is None:
      return None
    sign = -1 if index.op == ProgramOp.SUB else 1
    return {name: left.get(name, 0) + sign * right.get(name, 0) for name in left.keys() | right.keys()}
  if index.op == ProgramOp.MUL:
    for scale, term in (index.args, index.args[::-1]):
      if scale.op == ProgramOp.CONST_INT and (inner := _coefficients(term, names)) is not None:
        return {name: c * scale.attrs["value"] for name, c in inner.items()}
  mentioned = any(n.op == ProgramOp.VAR and n.attrs["name"] in names for n in walk_program(index))
  return None if mentioned else {}


def _fold_index(node: ProgramNode) -> ProgramNode:
  """Fold the integer additions and products below ``node``, such as those a substituted zero leaves."""
  pattern = Pattern(None, lambda n: n.dtype == dtypes.int64 and n.op in (ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL), fold_program)
  return rewrite(node, [pattern], rebuild=rebuild_program)


def _accesses(body: Sequence[ProgramNode], names: frozenset[str]) -> list[dict[str, int]] | None:
  """Each view's coefficients of ``names``, or None if one is not affine or a loop variable is read as a value."""
  out: list[dict[str, int]] = []
  stack, seen = list(body), set()
  while stack:
    n = stack.pop()
    if n in seen:
      continue
    seen.add(n)
    if n.op == ProgramOp.VIEW:
      coefficients = _coefficients(n.args[0], names)
      if coefficients is None:
        return None
      out.append(coefficients)
    elif n.op == ProgramOp.VAR and n.attrs["name"] in names:
      return None
    else:
      stack.extend(n.args)
  return out


def _merge(ranges: list[ProgramNode], body: list[ProgramNode]) -> tuple[list[ProgramNode], list[ProgramNode]]:
  """Merge adjacent loops wherever every view in ``body``, and in the bounds of the loops inside, is
  contiguous across them.

  Outer ``a`` over ``[0, da)`` and inner ``b`` over ``[0, db)`` merge when each view index reads
  ``ca * a + cb * b + rest`` with ``ca == cb * db``: then ``b`` over ``[0, da * db)`` with ``a = 0``
  gives the same indices. So an elementwise kernel without broadcasting stays one flat loop."""
  k = len(ranges) - 2
  while k >= 0:
    outer, inner = ranges[k], ranges[k + 1]
    da, db = trip_count(outer), trip_count(inner)
    if (
      da is not None
      and db is not None
      and outer.attrs["kind"] == inner.attrs["kind"]
      and not outer.attrs.get("carried")
      and not inner.attrs.get("carried")
      and all(r.args[0].op == ProgramOp.CONST_INT and r.args[0].attrs["value"] == 0 and r.args[2].attrs["value"] == 1 for r in (outer, inner))
      and (accesses := _accesses([*ranges[k + 2 :], *body], frozenset((outer.attrs["name"], inner.attrs["name"])))) is not None
      and all(c.get(outer.attrs["name"], 0) == c.get(inner.attrs["name"], 0) * db for c in accesses)
    ):
      nested, body = ([_fold_index(substitute_var(n, outer.attrs["name"], p.const_int(0))) for n in nodes] for nodes in (ranges[k + 2 :], body))
      ranges[k:] = [p.range_(inner.attrs["name"], 0, da * db, kind=inner.attrs["kind"]), *nested]
    k -= 1
  return ranges, body


def _copy_loop(src: ProgramNode, dst: ProgramNode, shape: tuple[int, ...], names: NameScope) -> ProgramNode:
  vname = names.allocate(f"c_{dst.attrs['name']}")
  rng = p.range_(vname, 0, _size_of(shape), kind=RangeKind.GLOBAL)
  i = p.var(vname)
  return p.for_(rng, [p.store(p.view(dst, [i]), p.load(p.view(src, [i])))])


def _function_names(
  root: ConcreteFunction, reserved_names: Iterable[str] = ()
) -> tuple[NameScope, dict[ConcreteFunction, str], frozenset[ConcreteFunction], dict[ConcreteFunction, frozenset[frozenset[int]]]]:
  """Name every reachable function, find those whose residuals some call reads, and the argument
  positions each map or loop of a function holds the same at every trip."""

  names = NameScope(reserved_names)
  symbols: dict[ConcreteFunction, str] = {}
  functions: list[ConcreteFunction] = []
  seen: set[ConcreteFunction] = set()
  residuals: set[ConcreteFunction] = set()
  broadcasts: dict[ConcreteFunction, frozenset[frozenset[int]]] = {}

  external_symbols: dict[str, _Extern] = {}

  def visit(function: ConcreteFunction) -> None:
    if isinstance(function, _Extern):
      previous = external_symbols.setdefault(function.name, function)
      if (previous.library, previous.signature) != (function.library, function.signature):
        raise ValueError(f"conflicting external declarations for C symbol {function.name!r}")
      symbols[function] = function.name
      names.occupied.add(function.name)
      _check_float64_leaves(function, "an external C function")
      return
    if function in seen:
      return
    seen.add(function)
    functions.append(function)
    for node in function.nodes:
      for callee in callees_of(node):
        if node.op != ExprOp.LOOP and node.attrs["output"] >= len(callee.outputs):
          residuals.add(callee)
        if node.op in (ExprOp.VMAP, ExprOp.LOOP) and 0 in node.attrs["strides"]:
          # A loop's body takes its carries and its step number before the sliced arguments.
          first = node.attrs["n_carry"] + node.attrs["index"] if node.op == ExprOp.LOOP else 0
          fixed = frozenset(first + k for k, stride in enumerate(node.attrs["strides"]) if stride == 0)
          broadcasts[callee] = broadcasts.get(callee, frozenset()) | {fixed}
        visit(callee)

  visit(root)
  if workspace_dtype(root) == dtypes.float32 and (needs := next((fn for fn in external_symbols.values() if fn.workspace), None)):
    raise LoweringError(f"external C function {needs.name!r} needs a double workspace, and {root.name!r} with only float32 leaves has a float one")
  symbols[root] = names.claim(root.name)
  names.occupied.add(f"{symbols[root]}_FRAME_BYTES")
  for function in functions:
    if function not in symbols:
      symbols[function] = names.procedure(function.name)
  return names, symbols, frozenset(residuals), broadcasts


def workspace_dtype(fun: ConcreteFunction) -> DType:
  """The unit of ``fun``'s entry pointers and workspace: float64 or float32 when every leaf has
  that dtype, otherwise uint8 for untyped pointers and a byte-addressed workspace."""
  found = {leaf.type.dtype for leaf in (*fun.inputs, *fun.outputs)}
  if found <= {dtypes.float64}:
    return dtypes.float64
  return dtypes.float32 if found == {dtypes.float32} else dtypes.uint8


def _check_float64_leaves(fun: ConcreteFunction, needs: str) -> None:
  for side, names, leaves in (("input", fun.input_names, fun.inputs), ("output", fun.output_names, fun.outputs)):
    for name, leaf in zip(names, leaves, strict=True):
      if leaf.type.dtype != dtypes.float64:
        raise LoweringError(f"function {fun.name!r} {side} leaf {name!r} has dtype {leaf.type.dtype}; {needs} needs float64 leaves")
