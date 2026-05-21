"""Lower semantic IR (``Expr`` / ``Function``) into Program IR (``PNode``).

Phase 5 of the roadmap. This module owns the explicit choice of how each
semantic op maps to a Program IR construct: buffers, loops, loads/stores,
calls. Today the lowerer only covers a small slice of the op set:

- ``INPUT`` / ``CONST`` → ``BUFFER`` params / constant buffers
- ``NEG`` / ``SIN`` / ``COS`` / ``EXP`` / ``LOG`` / ``SQRT`` → per-element ``FOR`` over a ``GLOBAL`` range that ``STORE``s ``unary(LOAD)`` into a workspace buffer
- ``ADD`` / ``SUB`` / ``MUL`` / ``DIV`` → per-element binary ``FOR`` (broadcasting not yet supported here — semantic IR resolves shape before lowering)
- ``RESHAPE`` → a buffer aliasing the source view (no copy)

This is intentionally a tiny slice; expanding to ``SUM``, ``MATMUL``, ``GATHER``,
``CALL``, ``MAP``, and ``SOLVER_CALL`` comes in subsequent phases. The lowerer
verifies its output via ``verify_program`` so future extensions cannot regress
silently.
"""

from __future__ import annotations

from . import program as p
from .expr import Expr, topo
from .function import Function
from .ops import Ops
from .program import PNode, POps, RangeKind, verify_program
from .types import DType, dtypes


SUPPORTED_UNARY = {
  Ops.NEG: POps.NEG,
  Ops.SIN: POps.SIN,
  Ops.COS: POps.COS,
  Ops.TAN: POps.TAN,
  Ops.ASIN: POps.ASIN,
  Ops.ACOS: POps.ACOS,
  Ops.ATAN: POps.ATAN,
  Ops.SINH: POps.SINH,
  Ops.COSH: POps.COSH,
  Ops.TANH: POps.TANH,
  Ops.EXP: POps.EXP,
  Ops.LOG: POps.LOG,
  Ops.SQRT: POps.SQRT,
  Ops.ABS: POps.ABS,
  Ops.FLOOR: POps.FLOOR,
  Ops.CEIL: POps.CEIL,
}

SUPPORTED_BINARY = {
  Ops.ADD: POps.ADD,
  Ops.SUB: POps.SUB,
  Ops.MUL: POps.MUL,
  Ops.DIV: POps.DIV,
  Ops.POW: POps.POW,
  Ops.ATAN2: POps.ATAN2,
  Ops.MINIMUM: POps.MINIMUM,
  Ops.MAXIMUM: POps.MAXIMUM,
}


class LoweringError(NotImplementedError):
  """Lowering cannot handle the requested semantic op yet."""


def lower_function(fun: Function) -> PNode:
  """Lower ``fun`` into a Program IR ``PROGRAM`` node.

  ``host`` placement (the default): returns a PROGRAM with all lowered callee
  PROCs in topological order followed by the main host PROC for ``fun`` as the
  last proc.

  Non-host placement (``cuda:N``, ``opencl:N``, ``metal:N``): runs the
  ``kernelize_for_device`` schedule pass on the lowered body. The returned
  PROGRAM holds the device KERNEL alongside a thin host driver PROC that
  ``LAUNCH``es it. Today no GPU backend renders the KERNEL — the renderer
  raises a clear diagnostic that points at Phase 8.

  ``main_proc(program)`` returns the last PROC for convenience (the host
  driver in mixed CPU/GPU programs, the only PROC in host-only ones).
  """
  callees_registry: dict[str, PNode] = {}
  root = _lower_to_proc(fun, callees_registry)
  if fun.device.kind != "host":
    root, kernel_node = _kernelize_for_device(root, fun)
    procs = list(callees_registry.values()) + [root]
    prog = p.program(procs, [kernel_node])
  else:
    procs = list(callees_registry.values()) + [root]
    prog = p.program(procs)
  verify_program(prog)
  return prog


def _kernelize_for_device(host_proc: PNode, fun: Function) -> tuple[PNode, PNode]:
  """Turn a device-placed function's lowered PROC into (host_driver, kernel).

  Phase 7 first slice: move the entire body of ``host_proc`` into a single
  ``KERNEL``. The host driver PROC keeps the same param signature and replaces
  its body with one ``LAUNCH`` of the kernel. Range kinds inside the body are
  left as ``GLOBAL``; later passes can rebind them to ``THREAD``/``WARP``/etc.

  Launch geometry is inferred from the first ``GLOBAL`` ``FOR`` in the body
  (the natural "grid" dim). Functions with no GLOBAL FOR launch with grid=1.
  """
  param_count = int(host_proc.attrs["param_count"])
  params = host_proc.args[:param_count]
  body = list(host_proc.args[param_count:])
  grid_size = _infer_launch_grid_size(body)
  kernel_name = f"{fun.name}_kernel"
  kernel_node = p.kernel(kernel_name, params, body, grid_dims=1, device=fun.device)
  driver_body = [p.launch(kernel_name, grid=[grid_size], block_dims=[1], args=list(params))]
  driver = p.proc(fun.name, params, driver_body, device="host")
  return driver, kernel_node


def _infer_launch_grid_size(body: list[PNode]) -> PNode:
  for stmt in body:
    if stmt.op == POps.FOR:
      rng = stmt.args[0]
      kind = rng.attrs.get("kind")
      if kind == RangeKind.GLOBAL:
        # use the upper bound (stop) as grid size if it's a CONST_INT
        stop = rng.args[1]
        if stop.op == POps.CONST_INT:
          return p.const_int(int(stop.attrs["value"]))
  return p.const_int(1)


def main_proc(program_node: PNode) -> PNode:
  """Return the main (last) PROC inside a lowered PROGRAM."""
  if program_node.op != POps.PROGRAM:
    raise TypeError(f"main_proc expects a PROGRAM, got {program_node.op}")
  pc = int(program_node.attrs.get("proc_count", 0))
  if pc <= 0:
    raise ValueError("lowered program contains no procs")
  return program_node.args[pc - 1]


def _lower_to_proc(fun: Function, callees_registry: dict[str, PNode]) -> PNode:
  builder = _Builder(fun, callees_registry)
  builder.emit_inputs()
  builder.emit_body()
  builder.emit_outputs()
  return p.proc(fun.name, builder.params, builder.statements)


class _Builder:
  def __init__(self, fun: Function, callees_registry: dict[str, PNode]) -> None:
    self.fun = fun
    self.params: list[PNode] = []
    self.statements: list[PNode] = []
    # Map each Expr id to the *buffer name* that holds its value at runtime.
    self.value_buffers: dict[int, str] = {}
    self.buffers: dict[str, PNode] = {}
    self._tmp_counter = 0
    self.callees_registry = callees_registry
    # Map (callee_name, tuple_of_arg_buffer_names) -> tuple of output buffer names.
    self.call_invocations: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}

  # --- inputs ---------------------------------------------------------------

  def emit_inputs(self) -> None:
    for name, expr in zip(self.fun.input_names, self.fun.inputs, strict=True):
      buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      self.value_buffers[expr.id] = name

  # --- outputs --------------------------------------------------------------

  def emit_outputs(self) -> None:
    for name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      out_buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.shape), address_space="global")
      self.params.append(out_buf)
      self.buffers[name] = out_buf
      src_buf_name = self.value_buffers.get(expr.id)
      if src_buf_name is None:
        raise LoweringError(f"output {name!r} expression was not lowered")
      if src_buf_name == name:
        continue  # already written into the output param
      self.statements.append(_copy_loop(self.buffers[src_buf_name], out_buf, expr.type.shape, expr.type.dtype))

  # --- body -----------------------------------------------------------------

  def emit_body(self) -> None:
    for node in topo(self.fun.outputs):
      if node.id in self.value_buffers:
        continue  # already an input
      if node.op == Ops.CONST:
        self._emit_const(node)
      elif node.op in SUPPORTED_UNARY:
        self._emit_elementwise(node, SUPPORTED_UNARY[Ops(node.op)], arity=1)
      elif node.op in SUPPORTED_BINARY:
        self._emit_elementwise(node, SUPPORTED_BINARY[Ops(node.op)], arity=2)
      elif node.op == Ops.RESHAPE:
        self._emit_reshape(node)
      elif node.op == Ops.SUM:
        self._emit_sum(node)
      elif node.op == Ops.SUM_AXIS:
        self._emit_sum_axis(node)
      elif node.op == Ops.MATMUL:
        self._emit_matmul(node)
      elif node.op == Ops.GATHER:
        self._emit_gather(node)
      elif node.op == Ops.SCATTER:
        self._emit_scatter(node)
      elif node.op == Ops.STACK:
        self._emit_stack(node)
      elif node.op == Ops.CONCAT:
        self._emit_concat(node)
      elif node.op == Ops.SLICE:
        self._emit_slice(node)
      elif node.op == Ops.CALL:
        self._emit_call(node)
      elif node.op == Ops.MAP:
        self._emit_map(node)
      elif node.op == Ops.TRANSPOSE:
        self._emit_transpose(node)
      else:
        raise LoweringError(f"semantic op {node.op!r} is not yet lowered to Program IR (Phase 5 slice)")

  def _alloc_tmp(self, expr: Expr) -> PNode:
    name = f"t{self._tmp_counter}"
    self._tmp_counter += 1
    buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.shape), address_space="private")
    self.buffers[name] = buf
    self.value_buffers[expr.id] = name
    # Emit the buffer declaration as a statement so the renderer can size it.
    self.statements.append(buf)
    return buf

  def _emit_const(self, node: Expr) -> None:
    # Constants are emitted as private-address-space "constant" buffers; lowering
    # doesn't yet have a CONST_ARRAY op so we store each scalar element with a
    # CONST_FLOAT initializer loop. Practical phase 5+ work will introduce a
    # dedicated CONST_BUFFER op; for now this keeps the structure simple.
    buf = self._alloc_tmp(node)
    value = node.value
    assert value is not None
    flat = value.reshape(-1)
    shape = node.type.shape or (1,)
    rng = p.range_(f"k_{buf.attrs['name']}", 0, int(value.size), kind=RangeKind.SERIAL)
    k_var = p.var(f"k_{buf.attrs['name']}", dtype=dtypes.int64)
    # Without a dedicated CONST_ARRAY op we serialize the values into a sequence
    # of guarded stores. That's only viable for tiny constants; the real solution
    # comes when CONST_BUFFER lands.
    if int(value.size) > 16:
      raise LoweringError(f"CONST of size {value.size} too large for the Phase 5 lowerer's inline serialization")
    body: list[PNode] = []
    for i, item in enumerate(flat):
      cond_var = p.var(f"k_{buf.attrs['name']}", dtype=dtypes.int64)
      _ = cond_var  # placeholder until IF lands
      body.append(p.store(p.view(buf, [p.const_int(i)]), p.const_float(float(item), dtype=node.type.dtype)))
    _ = shape, rng, k_var  # reserved for the future CONST_BUFFER lowering path
    self.statements.extend(body)

  def _emit_elementwise(self, node: Expr, pop: POps, *, arity: int) -> None:
    if any(a.shape != node.shape for a in node.args):
      raise LoweringError(f"Phase 5 lowerer requires identical shapes for elementwise op {node.op!r}; got {[a.shape for a in node.args]}")
    out_buf = self._alloc_tmp(node)
    size = node.size or 1
    rng = p.range_(f"i_{out_buf.attrs['name']}", 0, size, kind=RangeKind.GLOBAL)
    i = p.var(f"i_{out_buf.attrs['name']}", dtype=dtypes.int64)
    operand_loads = [p.load(p.view(self.buffers[self.value_buffers[a.id]], [i])) for a in node.args]
    if arity == 1:
      computed = PNode(pop, (operand_loads[0],), dtype=node.type.dtype)
    else:
      computed = PNode(pop, (operand_loads[0], operand_loads[1]), dtype=node.type.dtype)
    body = [p.store(p.view(out_buf, [i]), computed)]
    self.statements.append(p.for_(rng, body))

  def _emit_reshape(self, node: Expr) -> None:
    # RESHAPE is metadata-only at this layer: the result aliases the source buffer.
    src_buf_name = self.value_buffers[node.args[0].id]
    self.value_buffers[node.id] = src_buf_name

  def _emit_matmul(self, node: Expr) -> None:
    """Lower ``a @ b`` for the four supported rank combinations.

    Each case becomes a buffer (output) plus nested loops with an inner
    REDUCE-kind range that accumulates ``a[...] * b[...]`` into the output
    cell. The accumulator lives in the output buffer directly: we STORE 0
    before the reduction loop and STORE acc + product inside it.
    """
    a, b = node.args
    a_buf = self.buffers[self.value_buffers[a.id]]
    b_buf = self.buffers[self.value_buffers[b.id]]
    out = self._alloc_tmp(node)
    dtype = node.type.dtype
    if len(a.shape) == 1 and len(b.shape) == 1:
      self._emit_matmul_dot(a_buf, b_buf, out, a.shape[0], dtype)
    elif len(a.shape) == 2 and len(b.shape) == 1:
      self._emit_matmul_matvec(a_buf, b_buf, out, a.shape[0], a.shape[1], dtype)
    elif len(a.shape) == 1 and len(b.shape) == 2:
      # treat 1xK @ KxN: emit as a row-vector matvec by reusing the matvec helper
      # with K on the contraction axis.
      self._emit_matmul_vecmat(a_buf, b_buf, out, b.shape[0], b.shape[1], dtype)
    elif len(a.shape) == 2 and len(b.shape) == 2:
      self._emit_matmul_matmat(a_buf, b_buf, out, a.shape[0], a.shape[1], b.shape[1], dtype)
    else:
      raise LoweringError(f"matmul shape combination {a.shape}@{b.shape} not lowered")

  def _emit_matmul_dot(self, a_buf: PNode, b_buf: PNode, out: PNode, k: int, dtype) -> None:
    zero_idx = p.const_int(0)
    self.statements.append(p.store(p.view(out, [zero_idx]), p.const_float(0.0, dtype=dtype)))
    name = f"k_{out.attrs['name']}"
    rng = p.range_(name, 0, k, kind=RangeKind.REDUCE)
    i = p.var(name, dtype=dtypes.int64)
    body = [
      p.store(
        p.view(out, [zero_idx]),
        p.add(
          p.load(p.view(out, [zero_idx])),
          p.mul(p.load(p.view(a_buf, [i])), p.load(p.view(b_buf, [i]))),
        ),
      )
    ]
    self.statements.append(p.for_(rng, body))

  def _emit_matmul_matvec(self, a_buf: PNode, b_buf: PNode, out: PNode, m: int, k: int, dtype) -> None:
    """``a[m,k] @ b[k]``: outer loop over rows, inner REDUCE over k."""
    row_name = f"i_{out.attrs['name']}"
    k_name = f"k_{out.attrs['name']}"
    row_rng = p.range_(row_name, 0, m, kind=RangeKind.GLOBAL)
    row_var = p.var(row_name, dtype=dtypes.int64)
    k_rng = p.range_(k_name, 0, k, kind=RangeKind.REDUCE)
    k_var = p.var(k_name, dtype=dtypes.int64)
    # flat index = row*K + k
    a_idx = p.add(p.mul(row_var, p.const_int(k)), k_var)
    inner = [
      p.store(
        p.view(out, [row_var]),
        p.add(
          p.load(p.view(out, [row_var])),
          p.mul(p.load(p.view(a_buf, [a_idx])), p.load(p.view(b_buf, [k_var]))),
        ),
      )
    ]
    init = p.store(p.view(out, [row_var]), p.const_float(0.0, dtype=dtype))
    self.statements.append(p.for_(row_rng, [init, p.for_(k_rng, inner)]))

  def _emit_matmul_vecmat(self, a_buf: PNode, b_buf: PNode, out: PNode, k: int, n: int, dtype) -> None:
    """``a[k] @ b[k,n]``: outer loop over columns, inner REDUCE over k."""
    col_name = f"j_{out.attrs['name']}"
    k_name = f"k_{out.attrs['name']}"
    col_rng = p.range_(col_name, 0, n, kind=RangeKind.GLOBAL)
    col_var = p.var(col_name, dtype=dtypes.int64)
    k_rng = p.range_(k_name, 0, k, kind=RangeKind.REDUCE)
    k_var = p.var(k_name, dtype=dtypes.int64)
    b_idx = p.add(p.mul(k_var, p.const_int(n)), col_var)
    inner = [
      p.store(
        p.view(out, [col_var]),
        p.add(
          p.load(p.view(out, [col_var])),
          p.mul(p.load(p.view(a_buf, [k_var])), p.load(p.view(b_buf, [b_idx]))),
        ),
      )
    ]
    init = p.store(p.view(out, [col_var]), p.const_float(0.0, dtype=dtype))
    self.statements.append(p.for_(col_rng, [init, p.for_(k_rng, inner)]))

  def _emit_matmul_matmat(self, a_buf: PNode, b_buf: PNode, out: PNode, m: int, k: int, n: int, dtype) -> None:
    """``a[m,k] @ b[k,n]``: triple-nested loop with REDUCE on the inner k loop."""
    row_name = f"i_{out.attrs['name']}"
    col_name = f"j_{out.attrs['name']}"
    k_name = f"k_{out.attrs['name']}"
    row_rng = p.range_(row_name, 0, m, kind=RangeKind.GLOBAL)
    row_var = p.var(row_name, dtype=dtypes.int64)
    col_rng = p.range_(col_name, 0, n, kind=RangeKind.GLOBAL)
    col_var = p.var(col_name, dtype=dtypes.int64)
    k_rng = p.range_(k_name, 0, k, kind=RangeKind.REDUCE)
    k_var = p.var(k_name, dtype=dtypes.int64)
    out_idx = p.add(p.mul(row_var, p.const_int(n)), col_var)
    a_idx = p.add(p.mul(row_var, p.const_int(k)), k_var)
    b_idx = p.add(p.mul(k_var, p.const_int(n)), col_var)
    inner = [
      p.store(
        p.view(out, [out_idx]),
        p.add(
          p.load(p.view(out, [out_idx])),
          p.mul(p.load(p.view(a_buf, [a_idx])), p.load(p.view(b_buf, [b_idx]))),
        ),
      )
    ]
    init = p.store(p.view(out, [out_idx]), p.const_float(0.0, dtype=dtype))
    inner_for = p.for_(k_rng, inner)
    col_for = p.for_(col_rng, [init, inner_for])
    self.statements.append(p.for_(row_rng, [col_for]))

  def _emit_gather(self, node: Expr) -> None:
    src = node.args[0]
    src_buf = self.buffers[self.value_buffers[src.id]]
    out = self._alloc_tmp(node)
    idx = node.attrs["indices"].reshape(-1)
    if idx.size > 32:
      raise LoweringError(f"GATHER of {idx.size} indices exceeds the Phase 5 unrolled threshold; constant-buffer lowering not implemented yet")
    for i, src_idx in enumerate(idx):
      self.statements.append(
        p.store(
          p.view(out, [p.const_int(i)]),
          p.load(p.view(src_buf, [p.const_int(int(src_idx))])),
        )
      )

  def _emit_scatter(self, node: Expr) -> None:
    src = node.args[0]
    src_buf = self.buffers[self.value_buffers[src.id]]
    out = self._alloc_tmp(node)
    idx = node.attrs["indices"].reshape(-1)
    if idx.size > 32:
      raise LoweringError(f"SCATTER of {idx.size} indices exceeds the Phase 5 unrolled threshold; constant-buffer lowering not implemented yet")
    # initialize all output cells to zero
    size = node.size or 1
    for j in range(size):
      self.statements.append(p.store(p.view(out, [p.const_int(j)]), p.const_float(0.0, dtype=node.type.dtype)))
    for i, dst_idx in enumerate(idx):
      self.statements.append(
        p.store(
          p.view(out, [p.const_int(int(dst_idx))]),
          p.load(p.view(src_buf, [p.const_int(i)])),
        )
      )

  def _emit_stack(self, node: Expr) -> None:
    """Stack along axis 0 only: out[i*base + j] = inputs[i][j]. Other axes deferred."""
    axis = int(node.attrs.get("axis", 0))
    if axis != 0:
      raise LoweringError(f"Phase 5 STACK lowering only handles axis=0; got axis={axis}")
    out = self._alloc_tmp(node)
    base = node.args[0].size or 1
    for i, src in enumerate(node.args):
      src_buf = self.buffers[self.value_buffers[src.id]]
      name = f"j_{out.attrs['name']}_{i}"
      rng = p.range_(name, 0, base, kind=RangeKind.GLOBAL)
      j = p.var(name, dtype=dtypes.int64)
      dst_idx = p.add(p.mul(p.const_int(i), p.const_int(base)), j) if base > 0 else p.const_int(0)
      body = [p.store(p.view(out, [dst_idx]), p.load(p.view(src_buf, [j])))]
      self.statements.append(p.for_(rng, body))

  def _emit_concat(self, node: Expr) -> None:
    """Concat along axis 0 only: contiguous flat chunks in row-major flatten."""
    axis = int(node.attrs.get("axis", 0))
    if axis != 0:
      raise LoweringError(f"Phase 5 CONCAT lowering only handles axis=0; got axis={axis}")
    out = self._alloc_tmp(node)
    offset = 0
    for i, src in enumerate(node.args):
      src_buf = self.buffers[self.value_buffers[src.id]]
      size = src.size or 1
      name = f"j_{out.attrs['name']}_{i}"
      rng = p.range_(name, 0, size, kind=RangeKind.GLOBAL)
      j = p.var(name, dtype=dtypes.int64)
      dst_idx = p.add(p.const_int(offset), j) if offset else j
      body = [p.store(p.view(out, [dst_idx]), p.load(p.view(src_buf, [j])))]
      self.statements.append(p.for_(rng, body))
      offset += size

  def _emit_transpose(self, node: Expr) -> None:
    """Lower TRANSPOSE: emit nested loops that copy elements at the permuted index.

    Row-major flat indexing: ``out_flat[Σ_i o_i * out_stride_i] = src_flat[Σ_i o_axes[i] * src_stride_i]``.
    """
    src = node.args[0]
    src_buf = self.buffers[self.value_buffers[src.id]]
    axes = tuple(int(a) for a in node.attrs["axes"])
    src_shape = src.shape
    out_shape = node.shape
    if len(src_shape) > 4:
      raise LoweringError(f"TRANSPOSE lowering only handles ranks <= 4 today; got {src_shape}")
    out = self._alloc_tmp(node)
    # row-major strides
    src_strides = [1] * len(src_shape)
    for i in range(len(src_shape) - 2, -1, -1):
      src_strides[i] = src_strides[i + 1] * src_shape[i + 1]
    out_strides = [1] * len(out_shape)
    for i in range(len(out_shape) - 2, -1, -1):
      out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
    # one loop per output axis
    loop_vars: list[PNode] = []
    ranges: list[PNode] = []
    for i, d in enumerate(out_shape):
      name = f"d{i}_{out.attrs['name']}"
      ranges.append(p.range_(name, 0, int(d), kind=RangeKind.GLOBAL))
      loop_vars.append(p.var(name, dtype=dtypes.int64))
    # output flat index = Σ o_i * out_strides[i]
    out_idx = _affine_sum(loop_vars, out_strides)
    # src flat index = Σ (output-axis-i-loop-var-mapped-to-src-axis-axes[i]) * src_strides[axes[i]]
    src_idx = _affine_sum([loop_vars[i] for i in range(len(out_shape))], [src_strides[axes[i]] for i in range(len(out_shape))])
    body = [p.store(p.view(out, [out_idx]), p.load(p.view(src_buf, [src_idx])))]
    # wrap in nested fors from innermost to outermost
    stmt = body[0]
    for rng in reversed(ranges):
      stmt = p.for_(rng, [stmt])
    self.statements.append(stmt)

  def _emit_map(self, node: Expr) -> None:
    """Lower ``Ops.MAP``: a ``length``-iteration loop calling the callee with sliced outer args.

    Each iteration ``it`` reads ``outer_k[start_k + it*stride_k : ... + formal_k.size]``
    as input ``k`` and writes the selected callee output into a contiguous slice of
    a flat output buffer ``[it*slice_size : (it+1)*slice_size]``. The CALL receives
    pointer-offset VIEW args so the callee_raw signature stays plain
    ``double*``s; the renderer emits ``(buf + offset)``.
    """
    callee: Function = node.attrs["callee"]
    out_idx = int(node.attrs["output"])
    length = int(node.attrs["length"])
    starts = tuple(int(s) for s in node.attrs["starts"])
    strides = tuple(int(s) for s in node.attrs["strides"])
    slice_size = int(node.attrs["slice_size"])
    # Lower the callee once if needed.
    if callee.name not in self.callees_registry:
      self.callees_registry[callee.name] = _lower_to_proc(callee, self.callees_registry)
    # The MAP produces a flat output buffer of length*slice_size for the selected output index.
    out = self._alloc_tmp(node)
    # All other callee outputs are unused at this MAP node, but the callee writes
    # all of them every iteration. We allocate a scratch buffer per *other* output
    # of size = formal_out.size (one iteration's worth, reused across iterations).
    scratch_outs: list[PNode] = []
    for i, callee_out in enumerate(callee.outputs):
      if i == out_idx:
        scratch_outs.append(out)  # placeholder; real slice computed below
        continue
      tmp_name = f"t{self._tmp_counter}"
      self._tmp_counter += 1
      sbuf = p.buffer(
        tmp_name,
        callee_out.type.dtype,
        _shape_or_scalar(callee_out.type.shape),
        address_space="private",
      )
      self.buffers[tmp_name] = sbuf
      self.statements.append(sbuf)
      scratch_outs.append(sbuf)
    # Loop body: build pointer-offset VIEW args, then a CALL.
    loop_name = f"it_{out.attrs['name']}"
    rng = p.range_(loop_name, 0, length, kind=RangeKind.GLOBAL)
    it = p.var(loop_name, dtype=dtypes.int64)
    in_args: list[PNode] = []
    for k, outer in enumerate(node.args):
      outer_buf = self.buffers[self.value_buffers[outer.id]]
      offset = p.add(p.const_int(starts[k]), p.mul(p.const_int(strides[k]), it)) if strides[k] else p.const_int(starts[k])
      in_args.append(p.view(outer_buf, [offset]))
    out_args: list[PNode] = []
    for i, sbuf in enumerate(scratch_outs):
      if i == out_idx:
        # Write into out[it*slice_size + ...]; pass pointer to out + it*slice_size.
        offset = p.mul(it, p.const_int(slice_size)) if slice_size != 1 else it
        out_args.append(p.view(out, [offset]))
      else:
        out_args.append(sbuf)  # scratch reused every iteration
    call_stmt = PNode(
      POps.CALL,
      tuple(in_args + out_args),
      attrs={
        "callee": callee.name,
        "n_in": len(in_args),
        "n_out": len(out_args),
        "returns": (),
      },
    )
    self.statements.append(p.for_(rng, [call_stmt]))

  def _emit_call(self, node: Expr) -> None:
    """Lower a single semantic ``CALL`` output node.

    The same ``(callee, args)`` invocation may appear under multiple CALL nodes
    (one per output index). We deduplicate by an invocation key and emit one
    Program-IR CALL statement that writes into ``len(callee.outputs)``
    workspace buffers, then map each semantic node's id to the right output
    buffer name.
    """
    callee: Function = node.attrs["callee"]
    out_idx = int(node.attrs["output"])
    arg_buf_names = tuple(self.value_buffers[a.id] for a in node.args)
    key = (callee.name, arg_buf_names)
    is_external = callee.device.kind != self.fun.device.kind
    if key not in self.call_invocations:
      # Lower the callee body once per unique callee name.
      if callee.name not in self.callees_registry:
        self.callees_registry[callee.name] = _lower_to_proc(callee, self.callees_registry)
      if is_external:
        # Mixed-device CALL — callee is its own translation unit (the host
        # driver handles cudaMalloc/launch/cudaMemcpy when the callee is on a
        # GPU). We emit a CALL statement tagged as external so the renderer
        # invokes it through the universal ABI instead of inlining the body.
        pass
      # Allocate output workspace buffers for each callee output.
      out_buf_names: list[str] = []
      out_bufs: list[PNode] = []
      for callee_out in callee.outputs:
        tmp_name = f"t{self._tmp_counter}"
        self._tmp_counter += 1
        buf = p.buffer(
          tmp_name,
          callee_out.type.dtype,
          _shape_or_scalar(callee_out.type.shape),
          address_space="private",
        )
        self.buffers[tmp_name] = buf
        self.statements.append(buf)
        out_bufs.append(buf)
        out_buf_names.append(tmp_name)
      in_bufs = [self.buffers[n] for n in arg_buf_names]
      attrs: dict = {
        "callee": callee.name,
        "n_in": len(in_bufs),
        "n_out": len(out_bufs),
        "returns": (),
      }
      if is_external:
        attrs["external"] = True
        attrs["callee_device"] = str(callee.device)
      call_stmt = PNode(
        POps.CALL,
        tuple(in_bufs + out_bufs),
        attrs=attrs,
      )
      self.statements.append(call_stmt)
      self.call_invocations[key] = tuple(out_buf_names)
    self.value_buffers[node.id] = self.call_invocations[key][out_idx]

  def _emit_slice(self, node: Expr) -> None:
    """Lower SLICE: copy a contiguous run from the source buffer.

    Today only the rank-1 single-slice case (``x[start:stop:1]``) is wired.
    Multi-dim slices need rank-N views which the renderer does not yet model.
    """
    src = node.args[0]
    src_buf = self.buffers[self.value_buffers[src.id]]
    index = node.attrs["index"]
    if len(src.shape) != 1 or len(index) != 1 or not isinstance(index[0], slice):
      raise LoweringError(f"Phase 5 SLICE lowering only handles rank-1 single-slice indices; got src.shape={src.shape}, index={index}")
    start, stop, step = index[0].indices(src.shape[0])
    if step != 1:
      raise LoweringError(f"Phase 5 SLICE lowering only handles step=1; got step={step}")
    out = self._alloc_tmp(node)
    size = max(0, stop - start)
    if size == 0:
      return
    name = f"i_{out.attrs['name']}"
    rng = p.range_(name, 0, size, kind=RangeKind.GLOBAL)
    i = p.var(name, dtype=dtypes.int64)
    src_idx = p.add(p.const_int(start), i) if start else i
    self.statements.append(p.for_(rng, [p.store(p.view(out, [i]), p.load(p.view(src_buf, [src_idx])))]))

  def _emit_sum_axis(self, node: Expr) -> None:
    """Lower SUM_AXIS by emitting nested loops: outer ``GLOBAL`` per output axis,
    inner ``REDUCE`` per axis being reduced. Accumulator lives in the output buffer.
    """
    src = node.args[0]
    src_buf = self.buffers[self.value_buffers[src.id]]
    axes = tuple(int(a) for a in node.attrs["axes"])
    src_shape = src.shape
    if len(src_shape) > 4:
      raise LoweringError(f"SUM_AXIS lowering only handles ranks <= 4 today; got {src_shape}")
    out = self._alloc_tmp(node)
    dtype = node.type.dtype
    # row-major strides over the source
    src_strides = [1] * len(src_shape)
    for i in range(len(src_shape) - 2, -1, -1):
      src_strides[i] = src_strides[i + 1] * src_shape[i + 1]
    # output dims correspond to non-reduced axes in order
    out_dims = [i for i in range(len(src_shape)) if i not in axes]
    out_strides_list = [1] * len(out_dims)
    for i in range(len(out_dims) - 2, -1, -1):
      out_strides_list[i] = out_strides_list[i + 1] * src_shape[out_dims[i + 1]]
    # build loop vars
    outer_vars: list[PNode] = []
    outer_ranges: list[PNode] = []
    for k, axis in enumerate(out_dims):
      name = f"o{k}_{out.attrs['name']}"
      outer_ranges.append(p.range_(name, 0, int(src_shape[axis]), kind=RangeKind.GLOBAL))
      outer_vars.append(p.var(name, dtype=dtypes.int64))
    inner_vars: list[PNode] = []
    inner_ranges: list[PNode] = []
    for k, axis in enumerate(axes):
      name = f"r{k}_{out.attrs['name']}"
      inner_ranges.append(p.range_(name, 0, int(src_shape[axis]), kind=RangeKind.REDUCE))
      inner_vars.append(p.var(name, dtype=dtypes.int64))
    # output flat idx
    out_idx = _affine_sum(outer_vars, out_strides_list)
    # source flat idx mixes outer + inner contributions in original axis order
    src_axis_vars: dict[int, PNode] = {}
    for axis, v in zip(out_dims, outer_vars, strict=True):
      src_axis_vars[axis] = v
    for axis, v in zip(axes, inner_vars, strict=True):
      src_axis_vars[axis] = v
    src_idx_vars = [src_axis_vars[i] for i in range(len(src_shape))]
    src_idx = _affine_sum(src_idx_vars, src_strides)
    # init the output cell to 0, then accumulate
    init = p.store(p.view(out, [out_idx]), p.const_float(0.0, dtype=dtype))
    accum = p.store(
      p.view(out, [out_idx]),
      p.add(p.load(p.view(out, [out_idx])), p.load(p.view(src_buf, [src_idx]))),
    )
    inner_stmt: PNode = accum
    for rng in reversed(inner_ranges):
      inner_stmt = p.for_(rng, [inner_stmt])
    body: list[PNode] = [init, inner_stmt]
    stmt: PNode = p.block(*body) if len(body) > 1 else body[0]
    for rng in reversed(outer_ranges):
      stmt = p.for_(rng, [init, inner_stmt] if rng is outer_ranges[-1] else [stmt])
    # If there are no outer axes (full reduction), still emit init + inner.
    if not outer_ranges:
      self.statements.append(init)
      self.statements.append(inner_stmt)
    else:
      # The loop nest above re-emits init + inner directly inside the innermost outer.
      # Rebuild cleanly to avoid edge cases.
      stmt = p.for_(outer_ranges[-1], [init, inner_stmt])
      for rng in reversed(outer_ranges[:-1]):
        stmt = p.for_(rng, [stmt])
      self.statements.append(stmt)

  def _emit_sum(self, node: Expr) -> None:
    src = node.args[0]
    src_buf = self.buffers[self.value_buffers[src.id]]
    dtype = node.type.dtype
    acc = self._alloc_tmp(node)
    zero_idx = p.const_int(0)
    # init accumulator to 0
    self.statements.append(p.store(p.view(acc, [zero_idx]), p.const_float(0.0, dtype=dtype)))
    # reduction loop
    size = src.size or 1
    name = f"i_{acc.attrs['name']}"
    rng = p.range_(name, 0, size, kind=RangeKind.REDUCE)
    i = p.var(name, dtype=dtypes.int64)
    body = [
      p.store(
        p.view(acc, [zero_idx]),
        p.add(p.load(p.view(acc, [zero_idx])), p.load(p.view(src_buf, [i]))),
      )
    ]
    self.statements.append(p.for_(rng, body))


def _affine_sum(vars_: list[PNode], coeffs: list[int]) -> PNode:
  """Build a PNode for Σ coeffs[i] * vars_[i] using ADD/MUL/CONST_INT, omitting zero/one ops."""
  acc: PNode | None = None
  for v, c in zip(vars_, coeffs, strict=True):
    if c == 0:
      continue
    term = v if c == 1 else p.mul(v, p.const_int(c))
    acc = term if acc is None else p.add(acc, term)
  return acc if acc is not None else p.const_int(0)


def _shape_or_scalar(shape: tuple[int, ...]) -> tuple[int, ...]:
  return shape if shape else (1,)


def _copy_loop(src_buf: PNode, dst_buf: PNode, shape: tuple[int, ...], dtype: DType) -> PNode:
  size = 1
  for d in shape:
    size *= d
  size = size or 1
  rng = p.range_(f"i_{dst_buf.attrs['name']}", 0, size, kind=RangeKind.GLOBAL)
  i = p.var(f"i_{dst_buf.attrs['name']}", dtype=dtypes.int64)
  return p.for_(rng, [p.store(p.view(dst_buf, [i]), p.load(p.view(src_buf, [i])))])


__all__ = ["LoweringError", "lower_function", "main_proc"]
