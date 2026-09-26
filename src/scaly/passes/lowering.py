"""Lower expression IR (``Expr`` / ``Function``) into Program IR (``ProgramNode``).

This module + ``codegen/c.py`` are the **sole** CPU path to C (see
``docs/how_it_works/lowering.md``); the legacy tape-based scalar renderer is gone.
The one sanctioned non-Program-IR escape is the ``codegen/solver`` wrapper for a
``solver Function`` — and even there the oracle Functions it drives lower through here.

Dispatch is a **registry** keyed by expression ``ExprOp``: each op's lowering is a
self-contained rule registered with ``@lowers(...)``. Adding/deepening an op (or,
later, a GPU schedule) is a local change — a new rule, not an edit to a monolith.

Covered: elementwise unary/binary (with numpy broadcasting), ``RESHAPE`` (alias),
``CONST`` (any size, via ``const_buffer``), general ``SLICE`` (integer / multi-dim /
strided), ``SUM``, ``MATMUL`` (rank <= 2), ``TRANSPOSE`` (rank <= 4), ``GATHER`` /
``SCATTER`` (any size, affine indices as arithmetic on the trip index and
whatever is left as a ``static const`` table), ``STACK`` / ``CONCAT`` (any axis),
``CALL`` (multi-PROC, deduped) and ``VMAP``; a ``solver Function`` ``CALL`` is opaque
(see ``lower_function``). The tracking and unbumpercars workloads (forward + ``jac`` +
``spjac``) render and match generated-code / external numeric references. Deferred (re-land from the reference branch):
GPU placement and the new ops tracked in the migration roadmap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..ir import program as p
from ..ir.expr import CALLEE_OPS, COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, RUNTIME_INDEX_OPS, Expr, ExprOp, callees_of, topo
from ..function import Function
from .arith import constant
from .program import ProgramObserver, optimize_program
from ..ir.program import ProgramNode, ProgramOp, RangeKind
from ..ir.program_spec import verify_program
from ..ir.types import DeviceSpec, DType, dtypes
from .affine import affine_index_map
from .expr import cse_many, simplify


class LoweringError(NotImplementedError):
  """The lowerer (or Program-IR renderer) does not yet cover this op / case."""


# Expression ExprOp -> Program IR scalar ProgramOp. The op vocabulary grows here as ops migrate.
_UNARY: dict[ExprOp, ProgramOp] = {
  ExprOp.NEG: ProgramOp.NEG,
  ExprOp.SIN: ProgramOp.SIN,
  ExprOp.COS: ProgramOp.COS,
  ExprOp.TAN: ProgramOp.TAN,
  ExprOp.ASIN: ProgramOp.ASIN,
  ExprOp.ACOS: ProgramOp.ACOS,
  ExprOp.ATAN: ProgramOp.ATAN,
  ExprOp.SINH: ProgramOp.SINH,
  ExprOp.COSH: ProgramOp.COSH,
  ExprOp.TANH: ProgramOp.TANH,
  ExprOp.ERF: ProgramOp.ERF,
  ExprOp.EXP: ProgramOp.EXP,
  ExprOp.LOG: ProgramOp.LOG,
  ExprOp.SQRT: ProgramOp.SQRT,
  ExprOp.ABS: ProgramOp.ABS,
  ExprOp.FLOOR: ProgramOp.FLOOR,
  ExprOp.CEIL: ProgramOp.CEIL,
  ExprOp.NOT: ProgramOp.NOT,
  ExprOp.ISFINITE: ProgramOp.ISFINITE,
  ExprOp.CAST: ProgramOp.CAST,
}

_BINARY: dict[ExprOp, ProgramOp] = {
  ExprOp.ADD: ProgramOp.ADD,
  ExprOp.SUB: ProgramOp.SUB,
  ExprOp.MUL: ProgramOp.MUL,
  ExprOp.DIV: ProgramOp.DIV,
  ExprOp.POW: ProgramOp.POW,
  ExprOp.ATAN2: ProgramOp.ATAN2,
  ExprOp.MINIMUM: ProgramOp.MINIMUM,
  ExprOp.MAXIMUM: ProgramOp.MAXIMUM,
  ExprOp.COPYSIGN: ProgramOp.COPYSIGN,
  ExprOp.LT: ProgramOp.LT,
  ExprOp.LE: ProgramOp.LE,
  ExprOp.EQ: ProgramOp.EQ,
  ExprOp.NE: ProgramOp.NE,
  ExprOp.AND: ProgramOp.AND,
  ExprOp.OR: ProgramOp.OR,
}

LowerRule = Callable[["LowerCtx", Expr], None]
_RULES: dict[ExprOp, LowerRule] = {}


def lowers(*ops: ExprOp) -> Callable[[LowerRule], LowerRule]:
  """Register ``fn`` as the lowering rule for each expression op in ``ops``."""

  def deco(fn: LowerRule) -> LowerRule:
    for op in ops:
      _RULES[op] = fn
    return fn

  return deco


ExprObserver = Callable[[str, Function], None]


def lower_function(fun: Function, observe: ProgramObserver | None = None, observe_expr: ExprObserver | None = None) -> ProgramNode:
  """Lower ``fun`` into a Program IR ``PROGRAM`` node (verified before return).

  Host placement only for now: the returned PROGRAM holds every lowered callee
  PROC in topological order followed by ``fun``'s main PROC last. Non-host
  placement raises ``LoweringError`` — GPU backends re-land from the reference
  branch after CPU parity (see ``internal/notes/program_ir_migration.md``).

  A ``solver Function`` callee is **opaque**: its ``ExprOp.SOLVER_CALL`` body is not
  lowered — the solver wrapper is rendered by the sanctioned ``codegen/solver``
  path (rule 6) — but its oracle Functions *are* lowered to PROCs (the wrapper
  calls them as ``<oracle>_raw``). The solver→oracle-name map is recorded on the
  PROGRAM (``solver_oracles`` attr) so ``pack_workspace`` can size the caller's
  ``w[]`` to fit the oracle and the CALL-to-solver gets ``callee_needs_w`` right.
  """
  if fun.device.kind != "host":
    raise LoweringError(f"non-host placement {fun.device} is not lowered yet (GPU backends are deferred to a later migration step)")
  from ..solvers.graph import is_solver_function, solver_callees

  _check_function_names(fun, solver_callees)
  callees: dict[str, ProgramNode] = {}
  solver_fns: dict[str, Function] = {}

  if is_solver_function(fun):
    solver_fns[fun.name] = fun
    for oracle in solver_callees(fun):
      if oracle.name not in callees:
        callees[oracle.name] = _lower_to_proc(oracle, callees, solver_fns, observe_expr=observe_expr)
    prog = p.program([*callees.values()])
    prog = ProgramNode(ProgramOp.PROGRAM, prog.args, {**prog.attrs, "solver_root": fun.name}, prog.dtype)
  else:
    root = _lower_to_proc(fun, callees, solver_fns, auto_scalarize=False, observe_expr=observe_expr, entry=True)
    prog = p.program([*callees.values(), root])
  if solver_fns:
    solver_oracles = {name: tuple(o.name for o in solver_callees(sf)) for name, sf in solver_fns.items()}
    from ..solvers.model import ExternalOracle

    solver_external_workspace = {}
    for name, sf in solver_fns.items():
      desc = getattr(sf, "descriptor")
      solver_external_workspace[name] = max(
        (oracle.workspace_size for oracle in (desc.base, desc.grad, desc.jac, desc.hess, desc.bounds) if isinstance(oracle, ExternalOracle)),
        default=0,
      )
    prog = ProgramNode(
      ProgramOp.PROGRAM,
      prog.args,
      {**prog.attrs, "solver_oracles": solver_oracles, "solver_external_workspace": solver_external_workspace},
      prog.dtype,
    )
  if observe is not None:
    observe("lowered", prog)
  prog = optimize_program(prog, observe=observe)
  verify_program(prog)
  return prog


def _same_function(a: Function, b: Function) -> bool:
  """Two Function objects that would lower to the same procedure: interned Exprs make equal graphs
  the same objects, so identity of inputs and outputs is structural equality."""
  return a is b or (
    a.input_names == b.input_names
    and a.output_names == b.output_names
    and all(x is y for x, y in zip(a.inputs, b.inputs, strict=True))
    and len(a.outputs) == len(b.outputs)
    and all(x is y for x, y in zip(a.outputs, b.outputs, strict=True))
  )


def _check_function_names(fun: Function, solver_callees: Callable[[Function], Iterable[Function]]) -> None:
  """Refuse two different Functions with one name anywhere in ``fun``'s call tree.

  Procedures are emitted once per name, so the second would silently run the first one's body."""
  owners: dict[str, Function] = {}
  todo = [fun]
  while todo:
    f = todo.pop()
    seen = owners.get(f.name)
    if seen is not None:
      if not _same_function(seen, f):
        raise LoweringError(
          f"two different Functions are named {f.name!r} in the graph of {fun.name!r}; generated code has one procedure "
          "per name, so give them distinct names"
        )
      continue
    owners[f.name] = f
    todo.extend(solver_callees(f))
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


def _normalize_function(fun: Function) -> Function:
  outputs = fun.outputs
  for _ in range(4):
    normalized = cse_many(simplify(output) for output in outputs)
    if all(new is old for new, old in zip(normalized, outputs, strict=True)):
      break
    outputs = normalized
  return fun._with_outputs(outputs)


def _lower_to_proc(
  fun: Function,
  callees: dict[str, ProgramNode],
  solver_fns: dict[str, Function],
  *,
  auto_scalarize: bool = True,
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
  ctx = LowerCtx(fun, callees, solver_fns, observe_expr, entry=entry, in_place=chain)
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
      and not any(n.op in RUNTIME_INDEX_OPS for n in nodes)
      and (lowering == "scalar" or (lowering == "auto" and auto_scalarize))
      else "disabled",
      **({"in_place": True} if in_place else {}),
    },
    proc.dtype,
  )


# ``bool`` values only ever come from comparisons, logic and ``isfinite``, never from a narrowing
# store, so scalar substitution cannot erase a conversion for them. ``int64`` values come from
# explicit casts, integer constants and integer arithmetic; scalar expansion keeps a store's
# conversion as a cast wherever the stored value's type differs from the buffer's.
_SCALARIZABLE = (dtypes.float64, dtypes.bool_, dtypes.int64)


class LowerCtx:
  """Per-Function lowering state: buffers, statements, and the Expr-id -> buffer map.

  ``entry`` marks the Function whose procedure becomes the pointer-ABI entry. Its parameters are the
  caller's ``double`` arrays whatever the declared dtype, so a ``bool`` input is read into a typed
  temporary (nonzero is true) and a ``bool`` output is written as 0.0 or 1.0 from one.
  """

  def __init__(
    self,
    fun: Function,
    callees: dict[str, ProgramNode],
    solver_fns: dict[str, Function],
    observe_expr: ExprObserver | None = None,
    *,
    entry: bool = False,
    in_place: tuple[int, ...] | None = None,
  ) -> None:
    self.fun = fun
    self.entry = entry
    self.in_place = in_place
    self.callees = callees
    self.solver_fns = solver_fns  # name -> solver Function (opaque callees; rendered by codegen/solver)
    self.observe_expr = observe_expr
    self.params: list[ProgramNode] = []
    self.statements: list[ProgramNode] = []
    self.buffers: dict[str, ProgramNode] = {}
    # Expr.id -> name of the buffer holding that value at runtime.
    self.value_buffers: dict[int, str] = {}
    # Output Expr.id -> output buffer name, so the body writes outputs in place.
    self._output_alias: dict[int, str] = {}
    # (callee_name, arg_buffer_names) -> output buffer names, to dedup repeated CALL invocations.
    self.call_invocations: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}
    # scan identity -> buffer name per output index (0 final carry, -1 carries, 1.. stacked outputs).
    self.scan_invocations: dict[tuple[object, ...], dict[int, str]] = {}
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
    for name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      buf = p.buffer(name, self._abi_dtype(expr.type.dtype), _shape_or_scalar(expr.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      if expr.op in (ExprOp.INPUT, ExprOp.CONST) or expr.id in seen or buf.dtype != expr.type.dtype:
        continue  # INPUT/CONST, shared, or converted output: emit_outputs inserts the copy
      seen.add(expr.id)
      self._output_alias[expr.id] = name

  def emit_outputs(self) -> None:
    for name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      src = self.value_buffers.get(expr.id)
      if src is None:
        raise LoweringError(f"output {name!r} expression was not lowered")
      if src == name:
        continue  # already written in place via the output alias
      self.statements.append(_copy_loop(self.buffers[src], self.buffers[name], expr.shape))

  # --- body -----------------------------------------------------------------

  def emit_body(self) -> None:
    for node in topo(self.fun.outputs):
      if node.id in self.value_buffers:
        continue  # input (or already lowered)
      rule = _RULES.get(ExprOp(node.op))
      if rule is None:
        raise LoweringError(f"Expression op {node.op!r} is not yet lowered to Program IR")
      rule(self, node)

  # --- helpers --------------------------------------------------------------

  def buf_of(self, expr: Expr) -> ProgramNode:
    return self.buffers[self.value_buffers[expr.id]]

  def new_private(self, dtype: DType, shape: tuple[int, ...]) -> ProgramNode:
    """Allocate a fresh private scratch BUFFER (declared as a local array by the renderer)."""
    name = f"t{self._tmp}"
    self._tmp += 1
    buf = p.buffer(name, dtype, _shape_or_scalar(shape), address_space="private")
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
    name = f"t{self._tmp}"
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
    buf = p.const_buffer(f"k{self._tmp}", dtypes.int64, (len(values),), values)
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


@lowers(*_UNARY)
def _lower_unary(ctx: LowerCtx, node: Expr) -> None:
  ctx.emit_elementwise(node, _UNARY[ExprOp(node.op)], arity=1)


@lowers(*_BINARY)
def _lower_binary(ctx: LowerCtx, node: Expr) -> None:
  ctx.emit_elementwise(node, _BINARY[ExprOp(node.op)], arity=2)


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
  # emit_outputs inserts a copy when a CONST is itself an output.
  name = f"k{ctx._tmp}"
  ctx._tmp += 1
  buf = p.const_buffer(name, node.type.dtype, _shape_or_scalar(node.shape), [float(v) for v in value.reshape(-1)])
  ctx.buffers[name] = buf
  ctx.value_buffers[node.id] = name
  ctx.statements.append(buf)


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
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


@lowers(ExprOp.SUM)
def _lower_sum(ctx: LowerCtx, node: Expr) -> None:
  """Full reduction to a scalar: zero the accumulator, then a REDUCE loop adds every element."""
  src = node.args[0]
  acc = ctx.alloc_tmp(node)
  z = p.const_int(0)
  ctx.statements.append(p.store(p.view(acc, [z]), p.const_float(0.0, dtype=node.type.dtype)))
  name = f"i_{acc.attrs['name']}"
  rng = p.range_(name, 0, _size_of(src.shape), kind=RangeKind.REDUCE)
  i = p.var(name)
  ctx.statements.append(p.for_(rng, [p.store(p.view(acc, [z]), p.add(p.load(p.view(acc, [z])), p.load(p.view(ctx.buf_of(src), [i]))))]))


@lowers(ExprOp.MAX, ExprOp.MIN)
def _lower_extremum(ctx: LowerCtx, node: Expr) -> None:
  """Start from the first element, then a REDUCE loop keeps the larger (smaller) value. A NaN element
  replaces the accumulator and nothing replaces a NaN accumulator, so NaN propagates as ``np.max``
  does; C's ``fmax`` would drop it."""
  src = node.args[0]
  acc = ctx.alloc_tmp(node)
  z = p.const_int(0)
  src_buf = ctx.buf_of(src)
  ctx.statements.append(p.store(p.view(acc, [z]), p.load(p.view(src_buf, [z]))))
  if src.size == 1:
    return
  name = f"i_{acc.attrs['name']}"
  i = p.var(name)
  cur, value = p.load(p.view(acc, [z])), p.load(p.view(src_buf, [i]))
  better = p.compare(ProgramOp.LT, cur, value) if node.op == ExprOp.MAX else p.compare(ProgramOp.LT, value, cur)
  take = ProgramNode(ProgramOp.OR, (better, p.compare(ProgramOp.NE, value, value)), dtype=dtypes.bool_)
  rng = p.range_(name, 1, _size_of(src.shape), kind=RangeKind.REDUCE)
  ctx.statements.append(p.for_(rng, [p.store(p.view(acc, [z]), p.select(take, value, cur))]))


@lowers(ExprOp.TRANSPOSE)
def _lower_transpose(ctx: LowerCtx, node: Expr) -> None:
  """Permuted copy: ``out[Σ o_i·out_stride_i] = src[Σ o_i·src_stride_{axes[i]}]``, one loop per output axis."""
  src = node.args[0]
  axes = tuple(int(a) for a in node.attrs["axes"])
  src_shape, out_shape = src.shape, node.shape
  if len(src_shape) > 4:
    raise LoweringError(f"TRANSPOSE lowering handles rank <= 4; got {src_shape}")
  out = ctx.alloc_tmp(node)
  src_strides = _row_major_strides(src_shape)
  out_strides = _row_major_strides(out_shape)
  ranges, loop_vars = [], []
  for i, d in enumerate(out_shape):
    name = f"d{i}_{out.attrs['name']}"
    ranges.append(p.range_(name, 0, int(d), kind=RangeKind.GLOBAL))
    loop_vars.append(p.var(name))
  out_idx = _affine_sum(loop_vars, out_strides)
  src_idx = _affine_sum(loop_vars, [src_strides[axes[i]] for i in range(len(out_shape))])
  stmt: ProgramNode = p.store(p.view(out, [out_idx]), p.load(p.view(ctx.buf_of(src), [src_idx])))
  for rng in reversed(ranges):
    stmt = p.for_(rng, [stmt])
  ctx.statements.append(stmt)


def _nest(ranges: list[ProgramNode], body: list[ProgramNode]) -> list[ProgramNode]:
  for rng in reversed(ranges):
    body = [p.for_(rng, body)]
  return body


def _mm_init(out: ProgramNode, idx: ProgramNode, dtype: DType) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.const_float(0.0, dtype=dtype))


def _mm_accum(out: ProgramNode, idx: ProgramNode, a_load: ProgramNode, b_load: ProgramNode) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.add(p.load(p.view(out, [idx])), p.mul(a_load, b_load)))


def _mm_accumulate(
  ctx: LowerCtx,
  out: ProgramNode,
  out_idx: ProgramNode,
  a_load: ProgramNode,
  b_load: ProgramNode,
  dtype: DType,
  outer: list[ProgramNode],
  k_rng: ProgramNode,
) -> None:
  """Zero ``out[out_idx]`` over the ``outer`` loops, then accumulate ``a*b`` with the REDUCE-k loop outermost.

  A dot product per output is a serial add chain the C compiler cannot break without reassociation. With
  k outermost the inner loop runs over independent outputs and vectorizes, and each output still sums its
  terms in the same order, so the result is bit-identical to the dot form. Use it when the reduction axis
  is the matrix's slow axis; a contiguous reduction axis would make the compiler gather under -march=native.
  """
  ctx.statements.extend(_nest(outer, [_mm_init(out, out_idx, dtype)]))
  ctx.statements.extend(_nest([k_rng, *outer], [_mm_accum(out, out_idx, a_load, b_load)]))


@lowers(ExprOp.MATMUL)
def _lower_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  a_buf, b_buf, out = ctx.buf_of(a), ctx.buf_of(b), ctx.alloc_tmp(node)
  dt = node.type.dtype
  sa, sb = a.shape, b.shape
  nm = out.attrs["name"]
  if len(sa) == 1 and len(sb) == 1:  # dot
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, sa[0], kind=RangeKind.REDUCE)
    _mm_accumulate(ctx, out, p.const_int(0), p.load(p.view(a_buf, [k])), p.load(p.view(b_buf, [k])), dt, [], krng)
  elif len(sa) == 2 and len(sb) == 1:  # mat @ vec: the reduction axis is contiguous in ``a``
    m, kk = sa
    i = p.var(f"i_{nm}")
    irng = p.range_(f"i_{nm}", 0, m, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    row_dot = lambda row: _mm_accum(out, row, p.load(p.view(a_buf, [p.add(p.mul(row, p.const_int(kk)), k)])), p.load(p.view(b_buf, [k])))
    ctx.statements.extend(_nest([irng], [_mm_init(out, i, dt)]))
    # Four output rows per pass as four unrolled statements, so the compiler keeps four independent
    # accumulators; a four-trip inner loop over the rows becomes gathers under -march=native.
    blocks, tail = divmod(m, 4)
    if blocks:
      ib = p.var(f"ib_{nm}")
      ibrng = p.range_(f"ib_{nm}", 0, blocks, kind=RangeKind.GLOBAL)
      rows = [p.add(p.mul(ib, p.const_int(4)), p.const_int(r)) for r in range(4)]
      ctx.statements.extend(_nest([ibrng, krng], [row_dot(row) for row in rows]))
    if tail:
      it = p.var(f"it_{nm}")
      itrng = p.range_(f"it_{nm}", 4 * blocks, m, kind=RangeKind.GLOBAL)
      ctx.statements.extend(_nest([itrng, krng], [row_dot(it)]))
  elif len(sa) == 1 and len(sb) == 2:  # vec @ mat
    kk, n = sb
    j = p.var(f"j_{nm}")
    jrng = p.range_(f"j_{nm}", 0, n, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    b_idx = p.add(p.mul(k, p.const_int(n)), j)
    _mm_accumulate(ctx, out, j, p.load(p.view(a_buf, [k])), p.load(p.view(b_buf, [b_idx])), dt, [jrng], krng)
  elif len(sa) == 2 and len(sb) == 2:  # mat @ mat
    m, kk = sa
    n = sb[1]
    i = p.var(f"i_{nm}")
    irng = p.range_(f"i_{nm}", 0, m, kind=RangeKind.GLOBAL)
    j = p.var(f"j_{nm}")
    jrng = p.range_(f"j_{nm}", 0, n, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    out_idx = p.add(p.mul(i, p.const_int(n)), j)
    a_idx = p.add(p.mul(i, p.const_int(kk)), k)
    b_idx = p.add(p.mul(k, p.const_int(n)), j)
    _mm_accumulate(ctx, out, out_idx, p.load(p.view(a_buf, [a_idx])), p.load(p.view(b_buf, [b_idx])), dt, [irng, jrng], krng)
  else:
    raise LoweringError(f"matmul shapes {sa}@{sb} not lowered (batched / higher-rank deferred)")


# Factorizations and triangular solves of order at most this are straight-line code, which scalar
# expansion then turns into registers; larger ones are loops with triangular bounds.
DENSE_UNROLL = 8


def _entry(buf: ProgramNode, n: int, i: ProgramNode, j: ProgramNode) -> ProgramNode:
  """The view of entry ``(i, j)`` of a row-major matrix with ``n`` columns."""
  return p.view(buf, [p.add(p.mul(i, p.const_int(n)), j)])


def _unary_node(op: ProgramOp, x: ProgramNode) -> ProgramNode:
  return ProgramNode(op, (x,), dtype=x.dtype)


def _blocked_sum(ctx: LowerCtx, tag: str, start: ProgramNode, stop: ProgramNode, term: Any, dtype: DType) -> tuple[list[ProgramNode], ProgramNode]:
  """Statements summing ``term(k)`` over ``start <= k < stop`` into four partial sums, and the total.

  A dot product summed in one accumulator is a chain of dependent adds the C compiler may not
  reorder; four interleaved accumulators overlap four chains. The rounding differs from a single
  sequential sum the way blocked library kernels do."""
  c = p.const_int
  slots = [p.view(ctx.new_private(dtype, ()), [c(0)]) for _ in range(4)]
  zero = p.const_float(0.0, dtype=dtype)
  tail = p.sub(stop, p.mod(p.sub(stop, start), c(4)))
  kb, kt = f"kb_{tag}", f"kt_{tag}"
  block = [p.store(slots[q], p.add(p.load(slots[q]), term(p.add(p.var(kb), c(q))))) for q in range(4)]
  stmts = [
    *(p.store(s, zero) for s in slots),
    p.for_(p.range_(kb, start, tail, step=4, kind=RangeKind.REDUCE), block),
    p.for_(p.range_(kt, tail, stop, kind=RangeKind.REDUCE), [p.store(slots[0], p.add(p.load(slots[0]), term(p.var(kt))))]),
  ]
  loads = [p.load(s) for s in slots]
  return stmts, p.add(p.add(loads[0], loads[1]), p.add(loads[2], loads[3]))


@lowers(ExprOp.CHOLESKY, ExprOp.LDL)
def _lower_factor(ctx: LowerCtx, node: Expr) -> None:
  """Row-by-row (Crout) ``L L^T`` or ``L D L^T``: entry ``(i, j)``, ``j <= i``, is the matrix entry
  minus a dot product of two rows already computed, both contiguous in row-major storage. For
  ``L D L^T`` the row being computed is kept scaled by ``D`` in a scratch vector, so every update
  is one multiply-add. Small orders are unrolled into straight-line code."""
  a = node.args[0]
  n = a.shape[0]
  src, out, dt = ctx.buf_of(a), ctx.alloc_tmp(node), node.type.dtype
  chol = node.op == ExprOp.CHOLESKY
  scaled = None if chol else ctx.new_private(dt, (n,))
  zero = p.const_float(0.0, dtype=dt)
  c = p.const_int

  def term(i: ProgramNode, j: ProgramNode, k: ProgramNode) -> ProgramNode:
    """The ``k`` term subtracted for entry ``(i, j)``."""
    if chol:
      return p.mul(p.load(_entry(out, n, i, k)), p.load(_entry(out, n, j, k)))
    assert scaled is not None
    return p.mul(p.load(p.view(scaled, [k])), p.load(_entry(out, n, j, k)))

  def finish(i: ProgramNode, j: ProgramNode, value: ProgramNode, diagonal: bool) -> list[ProgramNode]:
    if diagonal:
      return [p.store(_entry(out, n, i, i), _unary_node(ProgramOp.SQRT, value) if chol else value)]
    stores = [p.store(p.view(scaled, [j]), value)] if scaled is not None else []
    return [*stores, p.store(_entry(out, n, i, j), p.div(value, p.load(_entry(out, n, j, j))))]

  if n <= DENSE_UNROLL:
    for i in range(n):
      for j in range(i + 1):
        value = p.load(_entry(src, n, c(i), c(j)))
        for k in range(j):
          value = p.sub(value, term(c(i), c(j), c(k)))
        ctx.statements.extend(finish(c(i), c(j), value, i == j))
      ctx.statements.extend(p.store(_entry(out, n, c(i), c(z)), zero) for z in range(i + 1, n))
    return
  nm = out.attrs["name"]
  i, j, z = (p.var(f"{v}_{nm}") for v in ("fi", "fj", "fz"))
  off_sum, off_total = _blocked_sum(ctx, f"o_{nm}", c(0), j, lambda kk: term(i, j, kk), dt)
  diag_sum, diag_total = _blocked_sum(ctx, f"d_{nm}", c(0), i, lambda kk: term(i, i, kk), dt)
  off = [*off_sum, *finish(i, j, p.sub(p.load(_entry(src, n, i, j)), off_total), False)]
  diag = [*diag_sum, *finish(i, i, p.sub(p.load(_entry(src, n, i, i)), diag_total), True)]
  zeros = p.for_(p.range_(z.attrs["name"], p.add(i, c(1)), n, kind=RangeKind.GLOBAL), [p.store(_entry(out, n, i, z), zero)])
  row = [p.for_(p.range_(j.attrs["name"], 0, i, kind=RangeKind.SERIAL), off), *diag, zeros]
  ctx.statements.append(p.for_(p.range_(i.attrs["name"], 0, n, kind=RangeKind.SERIAL), row))


@lowers(ExprOp.TRISOLVE)
def _lower_trisolve(ctx: LowerCtx, node: Expr) -> None:
  """Substitution reading the triangle by rows. ``T X = B`` takes each unknown as its right-hand side
  minus a dot product with the unknowns already found (row ``i`` of ``T``, contiguous); ``T^T X = B``
  sweeps columns of ``T^T``, which are rows of ``T``: each unknown, once found, is subtracted from
  the right-hand sides still open. A matrix of right-hand sides does each step for a whole row of
  ``X``, contiguous. Small orders are unrolled."""
  t, b = node.args
  n = t.shape[0]
  m = 1 if len(b.shape) == 1 else b.shape[1]
  tb, bb, out, dt = ctx.buf_of(t), ctx.buf_of(b), ctx.alloc_tmp(node), node.type.dtype
  lower, trans, unit = (bool(node.attrs[key]) for key in ("lower", "trans", "unit"))
  c = p.const_int
  nm = out.attrs["name"]

  def x(row: ProgramNode, cc: ProgramNode) -> ProgramNode:
    return p.view(out, [p.add(p.mul(row, c(m)), cc) if m > 1 else row])

  def rhs(row: ProgramNode, cc: ProgramNode) -> ProgramNode:
    return p.view(bb, [p.add(p.mul(row, c(m)), cc) if m > 1 else row])

  def per_col(body: Any) -> list[ProgramNode]:
    """``body(column)`` for every right-hand side: one statement for a vector, a loop otherwise."""
    if m == 1:
      return body(c(0))
    name = f"sc_{nm}_{ctx._tmp}"
    ctx._tmp += 1
    return [p.for_(p.range_(name, 0, m, kind=RangeKind.GLOBAL), body(p.var(name)))]

  def scale(i: ProgramNode) -> list[ProgramNode]:
    if unit:
      return []
    return per_col(lambda cc: [p.store(x(i, cc), p.div(p.load(x(i, cc)), p.load(_entry(tb, n, i, i))))])

  # ``order(s)`` is the row handled at step ``s``; ``others(i)`` the range of the other index.
  forward = lower != trans
  row_at = (lambda s: s) if forward else (lambda s: p.sub(c(n - 1), s))  # noqa: E731
  if n <= DENSE_UNROLL:
    steps = range(n) if forward else range(n - 1, -1, -1)
    if trans:
      ctx.statements.append(_copy_loop(bb, out, b.shape))
      for r in steps:
        ctx.statements.extend(scale(c(r)))
        for k in range(r) if lower else range(r + 1, n):
          ctx.statements.extend(
            per_col(
              lambda cc, r=r, k=k: [p.store(x(c(k), cc), p.sub(p.load(x(c(k), cc)), p.mul(p.load(_entry(tb, n, c(r), c(k))), p.load(x(c(r), cc)))))]
            )
          )
      return
    for r in steps:

      def solve_row(cc: ProgramNode, r: int = r) -> list[ProgramNode]:
        value = p.load(rhs(c(r), cc))
        for k in range(r) if lower else range(r + 1, n):
          value = p.sub(value, p.mul(p.load(_entry(tb, n, c(r), c(k))), p.load(x(c(k), cc))))
        if not unit:
          value = p.div(value, p.load(_entry(tb, n, c(r), c(r))))
        return [p.store(x(c(r), cc), value)]

      ctx.statements.extend(per_col(solve_row))
    return
  s, k = p.var(f"ss_{nm}"), p.var(f"sk_{nm}")
  i = row_at(s)
  k_rng = (
    (lambda kind: p.range_(k.attrs["name"], 0, i, kind=kind)) if lower else (lambda kind: p.range_(k.attrs["name"], p.add(i, c(1)), n, kind=kind))
  )  # noqa: E731
  if trans:
    ctx.statements.append(_copy_loop(bb, out, b.shape))
    sweep = per_col(lambda cc: [p.store(x(k, cc), p.sub(p.load(x(k, cc)), p.mul(p.load(_entry(tb, n, i, k)), p.load(x(i, cc)))))])
    body = [*scale(i), p.for_(k_rng(RangeKind.GLOBAL), sweep)]
  elif m == 1:
    lo, hi = (c(0), i) if lower else (p.add(i, c(1)), c(n))
    sums, total = _blocked_sum(ctx, f"t_{nm}", lo, hi, lambda kk: p.mul(p.load(_entry(tb, n, i, kk)), p.load(x(kk, c(0)))), dt)
    value = p.sub(p.load(rhs(i, c(0))), total)
    body = [*sums, p.store(x(i, c(0)), value if unit else p.div(value, p.load(_entry(tb, n, i, i))))]
  else:
    # Four rows of ``X`` per pass over the row being solved: a quarter of its loads and stores.
    lo, hi = (c(0), i) if lower else (p.add(i, c(1)), c(n))
    tail = p.sub(hi, p.mod(p.sub(hi, lo), c(4)))
    kb = p.var(f"kb_{nm}")

    def four(cc: ProgramNode) -> list[ProgramNode]:
      terms = [p.mul(p.load(_entry(tb, n, i, p.add(kb, c(q)))), p.load(x(p.add(kb, c(q)), cc))) for q in range(4)]
      return [p.store(x(i, cc), p.sub(p.load(x(i, cc)), p.add(p.add(terms[0], terms[1]), p.add(terms[2], terms[3]))))]

    def one(cc: ProgramNode) -> list[ProgramNode]:
      return [p.store(x(i, cc), p.sub(p.load(x(i, cc)), p.mul(p.load(_entry(tb, n, i, k)), p.load(x(k, cc)))))]

    body = [
      *per_col(lambda cc: [p.store(x(i, cc), p.load(rhs(i, cc)))]),
      p.for_(p.range_(kb.attrs["name"], lo, tail, step=4, kind=RangeKind.SERIAL), per_col(four)),
      p.for_(p.range_(k.attrs["name"], tail, hi, kind=RangeKind.SERIAL), per_col(one)),
      *scale(i),
    ]
  ctx.statements.append(p.for_(p.range_(s.attrs["name"], 0, n, kind=RangeKind.SERIAL), body))


def _ensure_in_place_callee(ctx: LowerCtx, callee: Function, steps: dict[int, np.ndarray] | None = None) -> tuple[str, int] | None:
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
  if in_place_chain(normalized) is None and (not steps or not in_place_steps(normalized, steps)):
    return None
  name = f"{callee.name}_inplace"
  if name not in ctx.callees:
    renamed = Function._from_exprs(name, normalized.inputs, normalized.outputs, normalized.input_names, normalized.output_names)
    ctx.callees[name] = _lower_to_proc(renamed, ctx.callees, ctx.solver_fns, observe_expr=ctx.observe_expr, in_place=True)
  return name, _put_scratch(update_chain(normalized) or [])


def _put_scratch(chain: Iterable[Expr]) -> int:
  """The scratch slots after the carry's entries that its run-time-index updates write padded lanes to."""
  puts = (e for e in chain if e.op in (ExprOp.PUT_ADD, ExprOp.PUT) and e.shape[-1] and not e.attrs.get("in_range"))
  return max((e.size // e.shape[-1] * e.args[1].size for e in puts), default=0)


def _loop_steps(callee: Function, outers: Iterable[Expr], starts: Iterable[int], strides: Iterable[int], length: int) -> dict[int, np.ndarray]:
  """The value of each body input a loop slices from an integer constant, at every step."""
  steps: dict[int, np.ndarray] = {}
  for pos, (outer, start, stride) in enumerate(zip(outers, starts, strides, strict=True), start=1):
    formal = callee.inputs[pos]
    if outer.op == ExprOp.CONST and outer.value is not None and outer.type.dtype == dtypes.int64:
      flat = np.asarray(outer.value, dtype=np.int64).reshape(-1)
      offsets = start + stride * np.arange(length)[:, None] + np.arange(formal.size)[None, :]
      steps[pos] = flat[offsets].reshape((length, *formal.shape))
  return steps


def _ensure_callee(ctx: LowerCtx, callee: Function) -> None:
  if callee.device.kind != ctx.fun.device.kind:
    raise LoweringError(f"mixed-device CALL ({ctx.fun.device} -> {callee.device}) is deferred to a later migration step")
  from ..solvers.graph import is_solver_function, solver_callees

  if is_solver_function(callee):
    # Opaque: the solver wrapper is rendered by codegen/solver (rule 6), not lowered. Its body is
    # SOLVER_CALL (no lowering rule). We still lower the oracle Functions the wrapper drives.
    ctx.solver_fns[callee.name] = callee
    for oracle in solver_callees(callee):
      _ensure_callee(ctx, oracle)
    return
  if callee.name not in ctx.callees:
    ctx.callees[callee.name] = _lower_to_proc(callee, ctx.callees, ctx.solver_fns, observe_expr=ctx.observe_expr)


@lowers(ExprOp.CALL)
def _lower_call(ctx: LowerCtx, node: Expr) -> None:
  """An expression CALL output: emit one Program-IR CALL writing all callee outputs into scratch
  buffers (deduped per unique invocation), then map this node to the selected output buffer."""
  callee: Function = node.attrs["callee"]
  out_idx = int(node.attrs["output"])
  arg_names = tuple(ctx.value_buffers[a.id] for a in node.args)
  key = (callee.name, arg_names)
  if key not in ctx.call_invocations:
    _ensure_callee(ctx, callee)
    out_bufs = [ctx.new_private(o.type.dtype, o.shape) for o in callee.outputs]
    in_bufs = [ctx.buffers[n] for n in arg_names]
    ctx.statements.append(
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
  callee: Function = node.attrs["callee"]
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
  ctx.statements.append(p.for_(rng, [call]))


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
  callee: Function = node.attrs["callee"]
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
  ctx.statements.append(_copy_loop(ctx.buf_of(init), store, (cs,)))
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
  ctx.statements.append(p.for_(p.range_(name, 0, length, kind=RangeKind.SERIAL), [*counters, call]))
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
  if outer.type.dtype.is_floating and values.size and np.all(values == values[0]):
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
  return ("while", id(node.attrs["callee"]), id(node.attrs["cond"]), node.args[0].id, node.attrs["max_iter"])


@lowers(ExprOp.WHILE)
def _lower_while(ctx: LowerCtx, node: Expr) -> None:
  """A ``SERIAL`` loop of at most ``max_iter`` trips that calls the condition, leaves when it is
  false, and otherwise calls the body. The carry alternates between two slots as in ``scan``, or
  keeps every step when reverse mode reads them; the loop variable outlives the loop and is the
  step count. Because the last slot written is only known at run time, the final carry is copied
  out once after the loop."""
  key = _while_key(node)
  if key not in ctx.scan_invocations:
    ctx.scan_invocations[key] = _emit_while(ctx, node, key)
  ctx.value_buffers[node.id] = ctx.scan_invocations[key][int(node.attrs["output"])]


def _emit_while(ctx: LowerCtx, node: Expr, key: tuple[object, ...]) -> dict[int, str]:
  body, cond = node.attrs["callee"], node.attrs["cond"]
  max_iter, init = int(node.attrs["max_iter"]), node.args[0]
  carry = body.inputs[0]
  cs, dtype = carry.size, carry.type.dtype
  trajectory = any(n.op == ExprOp.WHILE and n.attrs["output"] == -1 and _while_key(n) == key for n in topo(ctx.fun.outputs))
  steps = {1: np.arange(max_iter, dtype=np.int64)} if len(body.inputs) == 2 else {}
  proof = None if trajectory else _ensure_in_place_callee(ctx, body, steps)
  in_place, scratch = proof if proof is not None else (None, 0)
  if in_place is None:
    _ensure_callee(ctx, body)
  _ensure_callee(ctx, cond)
  store = ctx.new_private(dtype, ((max_iter + 1 if trajectory else 1 if in_place else 2) * cs + scratch,))
  flag = ctx.new_private(dtypes.bool_, (1,))
  ctx.statements.append(_copy_loop(ctx.buf_of(init), store, (cs,)))
  name = f"k_{store.attrs['name']}"
  k = p.var(name)

  def slot(step: ProgramNode) -> ProgramNode:
    if in_place is not None:
      return p.const_int(0)
    return p.mul(step if trajectory else p.mod(step, p.const_int(2)), p.const_int(cs))

  read = p.view(store, [slot(k)])
  check = ProgramNode(ProgramOp.CALL, (read, flag), attrs={"callee": cond.name, "n_in": 1, "n_out": 1, "returns": ()})
  leave = p.break_if(ProgramNode(ProgramOp.NOT, (p.load(p.view(flag, [p.const_int(0)])),), dtype=dtypes.bool_))
  counters: list[ProgramNode] = []
  # A body that takes the step number gets the loop counter itself.
  extra = [_scalar_arg(ctx, dtypes.int64, k, counters)] if len(body.inputs) == 2 else []
  step = ProgramNode(
    ProgramOp.CALL,
    (read, *extra, p.view(store, [slot(p.add(k, p.const_int(1)))])),
    attrs={"callee": in_place or body.name, "n_in": 1 + len(extra), "n_out": 1, "returns": ()},
  )
  ctx.statements.append(p.for_(p.range_(name, 0, max_iter, kind=RangeKind.SERIAL), [check, leave, *counters, step], exit_var=True))
  count = ctx.new_private(dtypes.float64, ())
  ctx.statements.append(p.store(p.view(count, [p.const_int(0)]), p.cast(k, dtypes.float64)))
  final = ctx.new_private(dtype, carry.shape)
  c = p.var(f"c_{final.attrs['name']}")
  source = p.load(p.view(store, [p.add(slot(k), c)]))
  ctx.statements.append(p.for_(p.range_(c.attrs["name"], 0, cs, kind=RangeKind.GLOBAL), [p.store(p.view(final, [c]), source)]))
  bufs = {0: final.attrs["name"], 1: count.attrs["name"]}
  if trajectory:
    # Slots past the last step taken hold the final carry, so a backward pass that reads them sees
    # finite values; it discards what it computes from them.
    j = p.var(f"j_{store.attrs['name']}")
    fill = p.store(p.view(store, [p.add(p.mul(j, p.const_int(cs)), c)]), p.load(p.view(final, [c])))
    rng = p.range_(j.attrs["name"], p.add(k, p.const_int(1)), max_iter, kind=RangeKind.GLOBAL)
    ctx.statements.append(p.for_(rng, [p.for_(p.range_(c.attrs["name"], 0, cs, kind=RangeKind.GLOBAL), [fill])]))
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
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


@lowers(ExprOp.SCATTER)
def _lower_scatter(ctx: LowerCtx, node: Expr) -> None:
  """Zero the output, then ``out[indices[k]] = src[k]``, with the destination as arithmetic on ``k``
  where it is affine and a ``static const`` table otherwise. The indices are fixed, so their pattern
  picks the loop: distinct destinations store (a parallel ``GLOBAL`` loop), and repeated ones
  accumulate ``out[d] = out[d] + src[k]`` in a ``REDUCE`` loop."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  zname = f"z_{out.attrs['name']}"
  zrng = p.range_(zname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  z = p.var(zname)
  ctx.statements.append(p.for_(zrng, [p.store(p.view(out, [z]), p.const_float(0.0, dtype=node.type.dtype))]))
  iname = f"i_{out.attrs['name']}"
  i = p.var(iname)
  dst = ctx.index_at(idx, i)
  value = p.load(p.view(ctx.buf_of(src), [i]))
  if scatter_is_unique(idx):
    ctx.statements.append(p.for_(p.range_(iname, 0, len(idx), kind=RangeKind.GLOBAL), [p.store(p.view(out, [dst]), value)]))
    return
  accumulate = p.store(p.view(out, [dst]), p.add(p.load(p.view(out, [dst])), value))
  ctx.statements.append(p.for_(p.range_(iname, 0, len(idx), kind=RangeKind.REDUCE), [accumulate]))


@lowers(ExprOp.INDEX_ADD, ExprOp.INDEX_SET)
def _lower_index_update(ctx: LowerCtx, node: Expr) -> None:
  """Copy the base, then add (or store) each value at its index. In a procedure that updates its
  carry in place (``LowerCtx.in_place``) every link of the update chain is the carry output itself,
  which the caller passes aliased to the carry input, so nothing is copied and only the indexed
  entries are touched."""
  base, values = node.args
  idx = node.attrs["indices"]
  if ctx.in_place is not None and node.id in ctx.in_place:
    out = ctx.buffers[ctx.fun.output_names[0]]
    ctx.value_buffers[node.id] = out.attrs["name"]
  else:
    out = ctx.alloc_tmp(node)
    if ctx.value_buffers[base.id] != out.attrs["name"]:
      ctx.statements.append(_copy_loop(ctx.buf_of(base), out, node.shape))
  iname = f"u_{out.attrs['name']}_{ctx._tmp}"
  ctx._tmp += 1
  i = p.var(iname)
  dst = ctx.index_at(idx, i)
  value = p.load(p.view(ctx.buf_of(values), [i]))
  if node.op == ExprOp.INDEX_ADD:
    value = p.add(p.load(p.view(out, [dst])), value)
  kind = RangeKind.GLOBAL if scatter_is_unique(idx) else RangeKind.REDUCE
  ctx.statements.append(p.for_(p.range_(iname, 0, idx.size, kind=kind), [p.store(p.view(out, [dst]), value)]))


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
    ctx.statements.append(p.for_(p.range_(jname, 0, lanes, kind=kind), body(p.const_int(0), j)))
    return
  bname = f"b_{tag}"
  b = p.var(bname)
  inner = p.for_(p.range_(jname, 0, lanes, kind=kind), body(b, j))
  ctx.statements.append(p.for_(p.range_(bname, 0, rows, kind=RangeKind.GLOBAL), [inner]))


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
  memory. Lanes run in order: repeated indices accumulate (``put_add``) or keep the last value."""
  base, idx, values = node.args
  n, lanes = base.shape[-1], idx.size
  size = _size_of(node.shape)
  rows = size // n if n else 0
  if ctx.in_place is not None and node.id in ctx.in_place:
    # A link of a proven in-place chain: the carry itself, whose caller keeps the scratch slots.
    store = ctx.buffers[ctx.fun.output_names[0]]
    ctx.value_buffers[node.id] = store.attrs["name"]
  else:
    store = ctx.new_private(node.type.dtype, (size + rows * lanes,))
    ctx.value_buffers[node.id] = store.attrs["name"]
    if size:
      ctx.statements.append(_copy_loop(ctx.buf_of(base), store, node.shape))
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


# Tests switch this off to compare every in-place loop with its two-slot version.
DONATE_CARRIES = True


def in_place_chain(fun: Function) -> tuple[int, ...] | None:
  """The update nodes (by id) through which ``fun`` may overwrite its carry in place, or None.

  ``fun`` takes the carry first and returns the next carry first. The next carry must be a chain
  of ``index_add``/``index_set`` rooted at the carry input, ``u_0 = carry, u_i = update(u_{i-1})``,
  and each read of the chain must happen before the write that would change what it reads:

  - the values of update ``i`` read no chain link but ``u_{i-1}``, and none of the entries update
    ``i`` writes;
  - no other output reads any chain link.

  Which entries the values read is the structural pattern of the values with respect to
  ``u_{i-1}``, taken only over operations whose pattern is exactly what they read (``_EXACT_READS``).
  A path through anything else (a comparison, a ``select`` condition, ``copysign``'s sign, a call)
  counts as reading every entry. These are sufficient, not necessary; anything else keeps the
  two-slot carry.
  """
  from ..ad.sparsity import jacobian_sparsity
  from ..ir.expr import substitute

  chain = update_chain(fun)
  if chain is None or any(e.op not in (ExprOp.INDEX_ADD, ExprOp.INDEX_SET) for e in chain):
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
    values = update.args[1]
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
          if n.op not in _EXACT_READS:
            return None
      read = set(jacobian_sparsity(cut, stand_in).cols)
      if read & set(update.attrs["indices"].tolist()):
        return None
  return tuple(e.id for e in chain)


_UPDATE_OPS = (ExprOp.INDEX_ADD, ExprOp.INDEX_SET, ExprOp.PUT_ADD, ExprOp.PUT)


def update_chain(fun: Function) -> list[Expr] | None:
  """The next carry as a chain of updates rooted at the carry input, ``u_1 ... u_m`` in order, or
  None when it is not one. Structure only: whether the chain may run in place is proven apart."""
  carry, node = fun.inputs[0], fun.outputs[0]
  chain: list[Expr] = []
  while node is not carry:
    if node.op not in _UPDATE_OPS:
      return None
    chain.append(node)
    node = node.args[0]
  return chain[::-1] or None


def in_place_steps(fun: Function, steps: dict[int, np.ndarray]) -> bool:
  """Whether a loop body's update chain may overwrite its carry in place, for the index values the
  loop feeds it: ``steps[p]`` is input ``p`` at every step, for the inputs sliced from constants.

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
  length = next(iter(steps.values())).shape[0] if steps else 0
  carry = fun.inputs[0]
  links = [carry, *chain]
  position = {e.id: j for j, e in enumerate(links)}
  memo: dict[int, np.ndarray | None] = {}
  inputs = {e.id: pos for pos, e in enumerate(fun.inputs)}

  def value(e: Expr) -> np.ndarray | None:
    return _step_value(e, steps, inputs, length, memo)

  writes: list[np.ndarray] = []  # per update: (length, lanes) flat positions, -1 for a dropped lane
  for update in chain:
    if update.op in (ExprOp.INDEX_ADD, ExprOp.INDEX_SET):
      w = np.broadcast_to(np.asarray(update.attrs["indices"], dtype=np.int64).reshape(1, -1), (length, update.attrs["indices"].size))
    else:
      idx = value(update.args[1])
      if idx is None:
        return False
      w = _flat_positions(idx, update.shape)
    writes.append(w)

  def link_of(e: Expr) -> int | None:
    while e.op == ExprOp.RESHAPE and e.id not in position:
      e = e.args[0]
    return position.get(e.id)

  reads: list[tuple[int, int, np.ndarray]] = []  # (update i, link j, positions)
  for i, update in enumerate(chain, start=1):
    values = update.args[-1]
    seen: set[int] = set()
    pending = [values]
    while pending:
      node = pending.pop()
      if node.id in seen:
        continue
      seen.add(node.id)
      if link_of(node) is not None:
        return False  # the values use a whole link (or are one) beyond what the rule can bound
      for a_pos, arg in enumerate(node.args):
        j = link_of(arg)
        if j is None:
          pending.append(arg)
          continue
        if node.op == ExprOp.TAKE and a_pos == 0:
          idx = value(node.args[1])
          if idx is None:
            return False
          positions = _flat_positions(idx, arg.shape)
        elif node.op == ExprOp.GATHER:
          positions = np.broadcast_to(np.asarray(node.attrs["indices"], dtype=np.int64).reshape(1, -1), (length, node.attrs["indices"].size))
        elif node.op == ExprOp.SLICE:
          flat = np.arange(arg.size).reshape(arg.shape)[node.attrs["index"]].reshape(1, -1)
          positions = np.broadcast_to(flat, (length, flat.size))
        else:
          return False
        reads.append((i, j, positions))
        if node.op == ExprOp.TAKE:
          pending.append(node.args[1])
  for y in fun.outputs[1:]:
    for node in topo([y]):
      if any(link_of(a) is not None for a in node.args) or link_of(node) is not None:
        return False
  for i, j, positions in reads:
    written = np.concatenate([writes[w - 1] for w in range(j + 1, i + 1)], axis=1)
    if _overlap(positions, written):
      return False
  return True


def _flat_positions(idx: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
  """The flat positions a run-time index addresses in an array of ``shape`` at each step: row ``b``
  of the last axis at ``b * n + i``, and -1 for an index outside ``[0, n)``."""
  n = shape[-1]
  rows = int(np.prod(shape[:-1], dtype=np.int64))
  idx = idx.reshape(idx.shape[0], -1)
  inside = (idx >= 0) & (idx < n)
  flat = np.arange(rows)[None, :, None] * n + idx[:, None, :]
  return np.where(inside[:, None, :], flat, -1).reshape(idx.shape[0], -1)


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


# Operations whose structural sparsity pattern is exactly the set of entries they read, so the pattern
# can stand in for a read set. Everything else (predicates, ``select``'s condition, ``copysign``'s
# sign, casts, calls, maps and loops) may read entries its pattern omits.
_EXACT_READS = frozenset(
  {
    ExprOp.INPUT,
    ExprOp.CONST,
    *(op for op in COMMON_ELEMENTWISE_UNARY),
    *(op for op in COMMON_ELEMENTWISE_BINARY if op != ExprOp.COPYSIGN),
    ExprOp.SUM,
    ExprOp.MAX,
    ExprOp.MIN,
    ExprOp.RESHAPE,
    ExprOp.TRANSPOSE,
    ExprOp.SLICE,
    ExprOp.GATHER,
    ExprOp.SCATTER,
    ExprOp.SEGMENT_MAX,
    ExprOp.SEGMENT_MIN,
    ExprOp.STACK,
    ExprOp.CONCAT,
    ExprOp.MATMUL,
    ExprOp.INDEX_ADD,
    ExprOp.INDEX_SET,
  }
)


def scatter_is_unique(idx: np.ndarray) -> bool:
  """Whether every destination of a fixed index table is distinct, so a plain store suffices."""
  return np.unique(idx).size == idx.size


@lowers(ExprOp.SEGMENT_MAX, ExprOp.SEGMENT_MIN)
def _lower_segment_extremum(ctx: LowerCtx, node: Expr) -> None:
  """Fill the output, then each value replaces its bin's entry when it is larger (smaller) or NaN."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  fname = f"z_{out.attrs['name']}"
  f = p.var(fname)
  fill = p.const_float(float(node.attrs["fill"]), dtype=node.type.dtype)
  ctx.statements.append(p.for_(p.range_(fname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL), [p.store(p.view(out, [f]), fill)]))
  iname = f"i_{out.attrs['name']}"
  i = p.var(iname)
  dst = ctx.index_at(idx, i)
  cur, value = p.load(p.view(out, [dst])), p.load(p.view(ctx.buf_of(src), [i]))
  better = p.compare(ProgramOp.LT, cur, value) if node.op == ExprOp.SEGMENT_MAX else p.compare(ProgramOp.LT, value, cur)
  take = ProgramNode(ProgramOp.OR, (better, p.compare(ProgramOp.NE, value, value)), dtype=dtypes.bool_)
  kind = RangeKind.GLOBAL if scatter_is_unique(idx) else RangeKind.REDUCE
  ctx.statements.append(p.for_(p.range_(iname, 0, len(idx), kind=kind), [p.store(p.view(out, [dst]), p.select(take, value, cur))]))


@lowers(ExprOp.STACK)
def _lower_stack(ctx: LowerCtx, node: Expr) -> None:
  """Stack ``n`` rank-r inputs along a new ``axis`` into a rank-(r+1) output: each input
  occupies index ``i`` along the new axis. Per element, decompose the input flat index into
  its coords, insert ``i`` at ``axis``, recombine against the output shape."""
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
    ctx.statements.append(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))


@lowers(ExprOp.CONCAT)
def _lower_concat(ctx: LowerCtx, node: Expr) -> None:
  """Concatenate inputs along ``axis``: each input keeps its shape but its ``axis`` coordinate is
  shifted by the running offset. Per element, decompose / shift / recombine against the output."""
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
    ctx.statements.append(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))
    offset += int(src_shape[axis])


__all__ = ["LoweringError", "LowerCtx", "lower_function", "lowers", "main_proc"]
