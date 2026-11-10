"""Own lowering state, rule registration, the loop-nest emitter, and the ConcreteFunction-to-program driver."""

from __future__ import annotations

from typing import Literal
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp, callees_of, invocation_key
from ...ir.match import Pattern, rewrite
from ...function.concrete import ConcreteFunction
from ...function.model import Function, as_concrete
from ...function.tree import flat_tree
from ...ir.layout import sparse_layout
from ..program import ProgramObserver, optimize_program
from ..program._common import rebuild_program, substitute_var, trip_count
from ...ir.program import ProgramNode, ProgramOp, RangeKind, walk_program
from ...ir.program_spec import verify_program
from ...ir.support import Support, _make_support, support_indices
from ...ir.types import DeviceSpec, DType, dtypes, stored_indices
from ...utils.names import NameScope
from ..affine import affine_index_map
from ..arith import fold_program
from ..expr import cse_many, simplify

LowerRule = Callable[["LowerCtx", Expr], None]
_RULES: dict[ExprOp, LowerRule] = {}


def _with_support(buf: ProgramNode, support: Support) -> ProgramNode:
  # Support is for scalar expansion only; later buffer rewrites may invalidate it.
  if support is None or len(support) == _size_of(buf.attrs["shape"]):
    return buf
  return ProgramNode(buf.op, buf.args, {**buf.attrs, "support": support}, buf.dtype)


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
) -> ProgramNode:
  """Lower ``fun`` into a Program IR ``PROGRAM`` node (verified before return).

  The returned PROGRAM holds every lowered callee PROC in topological order
  followed by ``fun``'s main PROC last.

  A ``solver ConcreteFunction`` callee is **opaque**: its ``ExprOp.SOLVER_CALL`` body is not
  lowered — the solver wrapper is rendered by the sanctioned ``codegen/solver``
  path (rule 6) — but its oracle Functions *are* lowered to PROCs (the wrapper
  calls them as ``<oracle>_raw``). The solver→oracle-name map is recorded on the
  PROGRAM (``solver_oracles`` attr) so ``pack_workspace`` can size the caller's
  ``w[]`` to fit the oracle and the CALL-to-solver gets ``callee_needs_w`` right.
  """
  fun = as_concrete(fun)
  names, symbols, residuals = _function_names(fun)
  callees: dict[ConcreteFunction, ProgramNode] = {}
  solver_fns: dict[ConcreteFunction, ConcreteFunction] = {}
  from ...solvers.graph import is_solver_function, solver_callees

  if is_solver_function(fun):
    solver_fns[fun] = fun
    for oracle in solver_callees(fun):
      if oracle not in callees:
        callees[oracle] = _lower_to_proc(oracle, callees, solver_fns, names, symbols, residuals, observe_expr=observe_expr)
    prog = p.program([*callees.values()])
    prog = ProgramNode(ProgramOp.PROGRAM, prog.args, {**prog.attrs, "solver_root": symbols[fun]}, prog.dtype)
  else:
    root = _lower_to_proc(fun, callees, solver_fns, names, symbols, residuals, auto_scalarize=False, observe_expr=observe_expr)
    prog = p.program([*callees.values(), root])
  if solver_fns:
    solver_oracles = {symbols[sf]: tuple(symbols[o] for o in solver_callees(sf)) for sf in solver_fns}
    from ...solvers.model import ExternalOracle

    solver_external_workspace = {}
    for sf in solver_fns:
      desc = sf.descriptor
      solver_external_workspace[symbols[sf]] = max(
        (oracle.workspace_size for oracle in (desc.base, desc.grad, desc.jac, desc.hess, desc.bounds) if isinstance(oracle, ExternalOracle)),
        default=0,
      )
    prog = ProgramNode(
      ProgramOp.PROGRAM,
      prog.args,
      {**prog.attrs, "solver_oracles": solver_oracles, "solver_external_workspace": solver_external_workspace},
      prog.dtype,
    )
  prog = ProgramNode(prog.op, prog.args, {**prog.attrs, "function_symbols": tuple(symbols.items())}, prog.dtype)
  if observe is not None:
    observe("lowered", prog)
  prog = optimize_program(prog, observe=observe, reciprocal=reciprocal, lanes=lanes)
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
    output_sparsities=(*fun.output_sparsities, *(None,) * extra),
    output_coloring_widths=(*fun.output_coloring_widths, *(None,) * extra),
    rules=None,
  )


def _lower_to_proc(
  fun: ConcreteFunction,
  callees: dict[ConcreteFunction, ProgramNode],
  solver_fns: dict[ConcreteFunction, ConcreteFunction],
  names: NameScope,
  symbols: dict[ConcreteFunction, str],
  residuals: frozenset[ConcreteFunction],
  *,
  auto_scalarize: bool = True,
  observe_expr: ExprObserver | None = None,
) -> ProgramNode:
  lowering = fun._effective_lowering
  symbol = symbols[fun]
  fun = _normalize_function(fun, residuals=fun in residuals)
  if observe_expr is not None:
    observe_expr("normalized", fun)
  ctx = LowerCtx(fun, callees, solver_fns, names, symbols, residuals, observe_expr)
  ctx.emit_inputs()
  ctx.register_outputs()
  ctx.emit_body()
  ctx.emit_outputs()
  proc = p.proc(symbol, ctx.params, ctx.statements)
  # ``input_count`` lets the renderer ``const``-qualify the first N (input) params of a ``_raw``
  # callee; emit_inputs runs before register_outputs, so inputs are the leading params.
  nodes = fun.nodes
  # Narrower stores round or truncate; scalar substitution must not erase those conversions.
  return ProgramNode(
    ProgramOp.PROC,
    proc.args,
    {
      **proc.attrs,
      "input_count": len(fun.inputs),
      "lowering": lowering,
      # Apple clang 17 (Xcode 16.4, macOS 15 arm64) miscompiles inlined forward-AD helpers for CALL
      # Jacobians, so they stay real call frames. See internal/notes/macos_clang_call_miscompile.md.
      **({"noinline": True} if fun.role == "forward" else {}),
      # A procedure may expand on its own and inside an expanding caller. Hoisted prologues use
      # "inline" because they only expand with their caller.
      "scalarize_mode": "procedure"
      if all(n.type.dtype == dtypes.float64 for n in (*fun.inputs, *nodes)) and (lowering == "scalar" or (lowering == "auto" and auto_scalarize))
      else "disabled",
    },
    proc.dtype,
  )


class LowerCtx:
  """Per-ConcreteFunction lowering state: buffers, statements, and the Expr-id -> buffer map."""

  def __init__(
    self,
    fun: ConcreteFunction,
    callees: dict[ConcreteFunction, ProgramNode],
    solver_fns: dict[ConcreteFunction, ConcreteFunction],
    names: NameScope,
    symbols: dict[ConcreteFunction, str],
    residuals: frozenset[ConcreteFunction],
    observe_expr: ExprObserver | None = None,
  ) -> None:
    self.fun = fun
    self.callees = callees
    self.solver_fns = solver_fns  # identity -> solver ConcreteFunction (opaque callees; rendered by codegen/solver)
    self.program_names = names
    self.names = names.child()
    self.symbols = symbols
    # Functions whose procedures also write their residuals, because some call reads one.
    self.residuals = residuals
    self.input_names = tuple(self.names.allocate(n) for n in fun.input_names)
    self.output_names = tuple(self.names.allocate(n) for n in fun.output_names)
    self.observe_expr = observe_expr
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
    self._zero_padded: dict[int, ProgramNode] = {}

  # --- declarations ---------------------------------------------------------

  def emit_inputs(self) -> None:
    for name, expr, layout in zip(self.input_names, self.fun.inputs, self.fun.input_tree.layouts, strict=True):
      buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.storage_shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      self.value_buffers[expr.id] = name
      if layout is not None:
        converted = self.reorder(buf, expr, layout.inverse_permutation)
        self.value_buffers[expr.id] = converted.attrs["name"]

  def register_outputs(self) -> None:
    """Register output BUFFER params and alias each unique computed output Expr to
    its output buffer, so its rule writes directly into the output (no copy)."""
    seen: set[int] = set()
    for name, expr, layout in zip(self.output_names, self.fun.outputs, self.fun.output_tree.layouts, strict=True):
      buf = _with_support(
        p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.storage_shape), address_space="global"),
        self.storage_support(expr, None if layout is None else layout.permutation),
      )
      self.params.append(buf)
      self.buffers[name] = buf
      if expr.op in (ExprOp.INPUT, ExprOp.CONST) or expr.id in seen or layout is not None:
        continue  # INPUT/CONST or shared output: emit_outputs inserts the copy
      seen.add(expr.id)
      self._output_alias[expr.id] = name

  def emit_outputs(self) -> None:
    for name, expr, layout in zip(self.output_names, self.fun.outputs, self.fun.output_tree.layouts, strict=True):
      src = self.value_buffers.get(expr.id)
      if src is None:
        raise LoweringError(f"output {name!r} expression was not lowered")
      if src == name or expr.type.storage_size == 0:
        continue  # already written in place, or nothing to copy
      if layout is None:
        self.statements.append(_copy_loop(self.buffers[src], self.buffers[name], expr.type.storage_shape, self.names))
      else:
        self.reorder(self.buffers[src], expr, layout.permutation, self.buffers[name])

  # --- body -----------------------------------------------------------------

  def emit_body(self) -> None:
    for node in self.fun.nodes:
      if node.id in self.value_buffers:
        continue  # input (or already lowered)
      rule = _RULES.get(ExprOp(node.op))
      if rule is None:
        raise LoweringError(f"Expression op {node.op!r} is not yet lowered to Program IR")
      rule(self, node)

  # --- helpers --------------------------------------------------------------

  def buf_of(self, expr: Expr) -> ProgramNode:
    return self.buffers[self.value_buffers[expr.id]]

  def new_private(self, dtype: DType, shape: tuple[int, ...], *, support: Support = None) -> ProgramNode:
    """Allocate a fresh private scratch BUFFER (declared as a local array by the renderer)."""
    name = self.names.allocate(f"t{self._tmp}")
    self._tmp += 1
    buf = _with_support(p.buffer(name, dtype, _shape_or_scalar(shape), address_space="private"), support)
    self.buffers[name] = buf
    self.statements.append(buf)  # marks the local-array declaration for the renderer
    return buf

  def alloc_tmp(self, expr: Expr) -> ProgramNode:
    """Buffer to hold ``expr``'s value: the output buffer if aliased, else a fresh private temp."""
    alias = self._output_alias.get(expr.id)
    if alias is not None:
      self.value_buffers[expr.id] = alias
      return self.buffers[alias]
    buf = self.new_private(expr.type.dtype, expr.type.storage_shape, support=self.storage_support(expr))
    self.value_buffers[expr.id] = buf.attrs["name"]
    return buf

  def storage_support(self, expr: Expr, permutation: np.ndarray | None = None) -> Support:
    support = expr.support
    if expr.type.pattern is None or support is None:
      return support
    positions = stored_indices(expr.type)
    if permutation is not None:
      width = expr.type.pattern.nnz
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

  def read_indexed(self, expr: Expr, logical: np.ndarray, k: ProgramNode) -> ProgramNode:
    """Read known logical coordinates through a stored-slot map, including structural zeros."""
    if expr.type.pattern is None:
      return p.load(p.view(self.buf_of(expr), [self.index_at(logical, k)]))
    entries = stored_indices(expr.type)
    slots = np.searchsorted(entries, logical)
    valid = slots < len(entries)
    valid[valid] &= entries[slots[valid]] == logical[valid]
    source = self.buf_of(expr)
    if not np.all(valid):
      slots = np.where(valid, slots, len(entries))
      if expr.id not in self._zero_padded:
        source = self.new_private(expr.type.dtype, (len(entries) + 1,))
        self.statements.append(_copy_loop(self.buf_of(expr), source, expr.type.storage_shape, self.names))
        self.statements.append(p.store(p.view(source, [p.const_int(len(entries))]), p.const_float(0.0, dtype=expr.type.dtype)))
        self._zero_padded[expr.id] = source
      source = self._zero_padded[expr.id]
    return p.load(p.view(source, [self.index_at(slots, k)]))

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

  def new_const_index(self, idx: Iterable[int]) -> ProgramNode:
    """A read-only int64 index table (for GATHER/SCATTER), declared ``static const``."""
    values = [int(v) for v in idx]
    buf = p.const_buffer(self.names.allocate(f"k{self._tmp}", generated=True), dtypes.int64, (len(values),), values)
    self._tmp += 1
    self.buffers[buf.attrs["name"]] = buf
    self.statements.append(buf)
    return buf

  def index_at(self, idx: np.ndarray, k: ProgramNode) -> ProgramNode:
    """The GATHER source (or SCATTER destination) for element ``k``, as arithmetic where it can be.

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
    """The scalar value of ``expr`` at dense logical ``coords``."""
    if expr.type.pattern is not None:
      raise LoweringError("sparse logical reads require an explicit stored-slot map")
    return p.load(p.view(self.buf_of(expr), [_flat(expr.shape, coords)]))

  def write(self, expr: Expr, coords: Sequence[ProgramNode], value: ProgramNode) -> ProgramNode:
    """The statement storing ``value`` at dense logical ``coords``."""
    if expr.type.pattern is not None:
      raise LoweringError("sparse logical writes require an explicit stored-slot map")
    return p.store(p.view(self.buf_of(expr), [_flat(expr.shape, coords)]), value)

  def emit(self, ranges: Sequence[ProgramNode], body: Sequence[ProgramNode]) -> None:
    """Append ``body`` inside ``ranges``, outermost first.

    A static empty range drops the whole nest, so an empty tensor emits no store. Adjacent ranges
    merge where every access is contiguous across them. A unit loop from zero offsets each constant
    view index by its variable, which is zero, the flat form of a one-element domain the program
    passes match on."""
    if any(trip_count(r) == 0 for r in ranges):
      return
    merged, stmts = _merge(list(ranges), list(body))
    if len(merged) == 1 and trip_count(merged[0]) == 1 and merged[0].args[0] == p.const_int(0):
      unit = p.var(merged[0].attrs["name"])

      def offset(view: ProgramNode) -> ProgramNode:
        index = view.args[0]
        return ProgramNode(ProgramOp.VIEW, (unit if index.attrs["value"] == 0 else p.add(index, unit),), view.attrs, view.dtype)

      pattern = Pattern(ProgramOp.VIEW, lambda v: v.args[0].op == ProgramOp.CONST_INT, offset)
      stmts = [rewrite(stmt, [pattern], rebuild=rebuild_program, fixpoint=False) for stmt in stmts]
    for rng in reversed(merged):
      stmts = [p.for_(rng, stmts)]
    self.statements.extend(stmts)


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


def _flat_coordinates(k: ProgramNode, shape: tuple[int, ...]) -> list[ProgramNode]:
  """The coordinates of row-major position ``k`` in ``shape``, as quotients and remainders."""
  coords = []
  for dim, extent in enumerate(shape):
    stride = _stride(shape, dim)
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


def _function_names(root: ConcreteFunction) -> tuple[NameScope, dict[ConcreteFunction, str], frozenset[ConcreteFunction]]:
  """Name every reachable function, and find those whose residuals some call reads."""
  from ...solvers.graph import is_solver_function, solver_callees, external_oracles

  names = NameScope()
  symbols: dict[ConcreteFunction, str] = {}
  functions: list[ConcreteFunction] = []
  seen: set[ConcreteFunction] = set()
  residuals: set[ConcreteFunction] = set()
  oracles: set[ConcreteFunction] = set()

  def visit(function: ConcreteFunction) -> None:
    if function in seen:
      return
    seen.add(function)
    _check_float64_leaves(function)
    functions.append(function)
    for oracle in external_oracles(function):
      names.occupied.add(oracle.raw_symbol)
    for callee in solver_callees(function):
      oracles.add(callee)
      visit(callee)
    for node in function.nodes:
      for callee in callees_of(node):
        if node.attrs["output"] >= len(callee.outputs):
          residuals.add(callee)
        visit(callee)

  visit(root)
  symbols[root] = names.claim(root.name)
  for function in functions:
    if is_solver_function(function):
      if function is not root:
        symbols[function] = names.claim(function.name)
      for suffix in ("stats", "default_options", "stats_data", "raw"):
        names.claim(f"{symbols[function]}_{suffix}")
  for function in functions:
    if function not in symbols:
      symbols[function] = names.procedure(function.name)
  if residuals & oracles:
    raise LoweringError(
      f"a solver passes its oracles their outputs only, but a call reads residuals of {sorted(f.name for f in residuals & oracles)}"
    )
  return names, symbols, frozenset(residuals)


def _check_float64_leaves(fun: ConcreteFunction) -> None:
  for side, names, leaves in (("input", fun.input_names, fun.inputs), ("output", fun.output_names, fun.outputs)):
    for name, leaf in zip(names, leaves, strict=True):
      if leaf.type.dtype != dtypes.float64:
        raise LoweringError(f"function {fun.name!r} {side} leaf {name!r} has dtype {leaf.type.dtype}; only float64 leaves are supported")
