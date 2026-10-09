"""Own lowering state, rule registration, flat-index helpers, and the ConcreteFunction-to-program driver."""

from __future__ import annotations

from typing import Literal
from collections.abc import Callable, Iterable

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp, topo
from ...function.concrete import ConcreteFunction
from ...function.model import Function, as_concrete
from ..program import ProgramObserver, optimize_program
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ...ir.program_spec import verify_program
from ...ir.types import DeviceSpec, DType, dtypes
from ..affine import affine_index_map
from ..expr import cse_many, simplify

LowerRule = Callable[["LowerCtx", Expr], None]
_RULES: dict[ExprOp, LowerRule] = {}


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
  _check_callee_names(fun)
  callees: dict[str, ProgramNode] = {}
  solver_fns: dict[str, ConcreteFunction] = {}
  from ...solvers.graph import is_solver_function, solver_callees

  if is_solver_function(fun):
    solver_fns[fun.name] = fun
    for oracle in solver_callees(fun):
      if oracle.name not in callees:
        callees[oracle.name] = _lower_to_proc(oracle, callees, solver_fns, observe_expr=observe_expr)
    prog = p.program([*callees.values()])
    prog = ProgramNode(ProgramOp.PROGRAM, prog.args, {**prog.attrs, "solver_root": fun.name}, prog.dtype)
  else:
    root = _lower_to_proc(fun, callees, solver_fns, auto_scalarize=False, observe_expr=observe_expr)
    prog = p.program([*callees.values(), root])
  if solver_fns:
    solver_oracles = {name: tuple(o.name for o in solver_callees(sf)) for name, sf in solver_fns.items()}
    from ...solvers.model import ExternalOracle

    solver_external_workspace = {}
    for name, sf in solver_fns.items():
      desc = sf.descriptor
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
  solver_fns: dict[str, ConcreteFunction],
  *,
  auto_scalarize: bool = True,
  observe_expr: ExprObserver | None = None,
) -> ProgramNode:
  lowering = fun._effective_lowering()
  fun = _normalize_function(fun)
  if observe_expr is not None:
    observe_expr("normalized", fun)
  ctx = LowerCtx(fun, callees, solver_fns, observe_expr)
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
    callees: dict[str, ProgramNode],
    solver_fns: dict[str, ConcreteFunction],
    observe_expr: ExprObserver | None = None,
  ) -> None:
    self.fun = fun
    self.callees = callees
    self.solver_fns = solver_fns  # name -> solver ConcreteFunction (opaque callees; rendered by codegen/solver)
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
    self._tmp = 0

  # --- declarations ---------------------------------------------------------

  def emit_inputs(self) -> None:
    for name, expr in zip(self.fun.input_names, self.fun.inputs, strict=True):
      buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      self.value_buffers[expr.id] = name

  def register_outputs(self) -> None:
    """Register output BUFFER params and alias each unique computed output Expr to
    its output buffer, so its rule writes directly into the output (no copy)."""
    seen: set[int] = set()
    for name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      if expr.op in (ExprOp.INPUT, ExprOp.CONST) or expr.id in seen:
        continue  # INPUT/CONST or shared output: emit_outputs inserts the copy
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
  return p.for_(rng, [p.store(p.view(dst, [i]), p.load(p.view(src, [i])))])


def _check_callee_names(root: ConcreteFunction) -> None:
  from ...solvers.graph import solver_callees
  from ...utils.names import c_ident

  names: dict[str, ConcreteFunction] = {}
  seen: set[int] = set()

  def visit(function: ConcreteFunction) -> None:
    if id(function) in seen:
      return
    seen.add(id(function))
    symbol = c_ident(function.name)
    if symbol in names and names[symbol] is not function:
      raise LoweringError(f"distinct function instances share generated C identifier {symbol!r}; give them distinct names")
    names[symbol] = function
    for callee in solver_callees(function):
      visit(callee)
    for node in topo(function.outputs):
      if node.op in (ExprOp.CALL, ExprOp.VMAP):
        visit(node.attrs["callee"])

  visit(root)
