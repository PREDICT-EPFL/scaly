"""CUDA C source rendering for device-placed functions.

The renderer emits a complete translation unit for each ``device='cuda:N'``
Function:

- a ``__global__`` kernel for the lowered body,
- ``__device__`` inline callees for any ``CALL`` / ``MAP`` invocations,
- an ``extern "C"`` host driver implementing the universal Alloy ABI
  (``int f(const double** arg, double** res, int* iw, double* w, void* mem)``)
  that does ``cudaMalloc`` / ``cudaMemcpy`` / launch / ``cudaDeviceSynchronize``
  / ``cudaFree`` synchronously.

Thread binding: the top-level ``GLOBAL`` ``FOR`` in the kernel body is bound
to ``blockIdx.x * blockDim.x + threadIdx.x`` (with the obvious bounds check).
When the kernel's ``bind_threads`` attribute is False (set by the schedule
pass in ``lowering._kernelize_for_device`` for shapes where a top-level
``REDUCE`` would race against per-thread workspace), the renderer leaves the
loop serial and the host driver launches with ``<<<1, 1>>>``.

Dtypes: the kernel uses the input/output dtype directly (``float`` for
``dtypes.float32``, ``double`` for ``dtypes.float64``). The universal ABI is
locked to ``const double**`` / ``double**``, so the host driver stages
non-double inputs/outputs through a host-side ``malloc`` + cast loop on the
way in and out.

End-to-end dispatch (``fn(np.array(...))``) goes through
``alloy.cuda_runtime.CudaCompiledFunction``, which auto-detects a nvcc whose
PTX the installed driver accepts.
"""

from __future__ import annotations

from ..function import Function
from ..lowering import lower_function, main_proc
from ..program import PNode, POps


_ABI_DEFINES = (
  "#define ALLOY_SUCCESS 0",
  "#define ALLOY_ERR_NULL_ABI 1",
  "#define ALLOY_ERR_NULL_WORK 2",
  "#define ALLOY_ERR_NULL_RESULT 3",
  "#define ALLOY_ERR_NULL_INPUT 4",
  "#define ALLOY_ERR_CUDA 5",
)


def can_render_cuda(fun: Function) -> bool:
  return fun.device.kind == "cuda"


def render_cuda_source(fun: Function) -> str:
  if fun.device.kind != "cuda":
    raise ValueError(f"render_cuda_source needs device='cuda:N', got {fun.device}")
  prog = lower_function(fun)
  driver = main_proc(prog)
  pc = int(prog.attrs["proc_count"])
  kernel = prog.args[pc]  # one kernel per Phase 7 schedule pass
  # Callees: every PROC before the main one is rendered as a ``__device__``
  # function the kernel can call. This mirrors the Metal renderer's
  # ``static inline void`` callee emission.
  callees = [p for p in prog.args[:pc] if p is not driver]
  symbol = fun.name
  lines: list[str] = [
    "#include <cuda_runtime.h>",
    "#include <math.h>",
    "#include <stdlib.h>",
    "",
    *_ABI_DEFINES,
    "",
  ]
  for callee in callees:
    lines += _render_cuda_callee(callee)
    lines.append("")
  lines += _render_cuda_kernel(kernel)
  lines.append("")
  lines += _render_host_driver(symbol, fun, driver, kernel)
  return "\n".join(lines).rstrip() + "\n"


def _render_cuda_callee(proc: PNode) -> list[str]:
  """Emit a callee PROC as a CUDA ``__device__ void`` function.

  Arguments mirror the PROC's BUFFER params (raw pointers). The body uses the
  same statement emitter as the kernel, minus the thread-binding step (a callee
  runs in the calling thread's context).
  """
  param_count = int(proc.attrs["param_count"])
  params = list(proc.args[:param_count])
  body = list(proc.args[param_count:])
  decls = ", ".join(f"{pp.dtype.c_type}* {pp.attrs['name']}" for pp in params)
  out: list[str] = [f"__device__ void {proc.attrs['name']}({decls}) {{"]
  ptr_expr = {pp.attrs["name"]: pp.attrs["name"] for pp in params}
  for stmt in body:
    if stmt.op == POps.BUFFER and stmt.attrs["name"] not in ptr_expr:
      size = 1
      for d in stmt.attrs["shape"]:
        size *= int(d)
      size = size or 1
      out.append(f"  {stmt.dtype.c_type} {stmt.attrs['name']}[{size}];")
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    _emit_stmt_cuda(stmt, ptr_expr, out, indent=2)
  out.append("}")
  return out


def _render_cuda_kernel(kernel: PNode) -> list[str]:
  param_count = int(kernel.attrs["param_count"])
  params = list(kernel.args[:param_count])
  body = list(kernel.args[param_count:])
  bind_threads = bool(kernel.attrs.get("bind_threads", True))
  param_decls = ", ".join(f"{pp.dtype.c_type}* {pp.attrs['name']}" for pp in params)
  out: list[str] = [f"__global__ void {kernel.attrs['name']}({param_decls}) {{"]
  ptr_expr = {pp.attrs["name"]: pp.attrs["name"] for pp in params}
  ts: list[PNode] = []
  for stmt in body:
    if stmt.op == POps.BUFFER and stmt.attrs["name"] not in ptr_expr and stmt not in ts:
      ts.append(stmt)
  for tb in ts:
    size = 1
    for d in tb.attrs["shape"]:
      size *= int(d)
    size = size or 1
    out.append(f"  {tb.dtype.c_type} {tb.attrs['name']}[{size}];")
  # When ``bind_threads`` is True (the safe case, see ``_can_thread_bind`` in
  # lowering.py) we bind the first top-level GLOBAL FOR to (blockIdx, threadIdx).
  # Otherwise the kernel runs serially on a single thread (the host driver
  # launches with grid=1, block=1) so reductions that read across thread-private
  # workspace stay correct.
  bound = False
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    kind = stmt.args[0].attrs.get("kind") if stmt.op == POps.FOR else None
    if bind_threads and not bound and kind is not None and getattr(kind, "value", None) == "global":
      _emit_thread_bound_for(stmt, ptr_expr, out, indent=2)
      bound = True
    else:
      _emit_stmt_cuda(stmt, ptr_expr, out, indent=2)
  out.append("}")
  return out


def _emit_thread_bound_for(stmt: PNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  """Map a top-level ``GLOBAL`` FOR loop onto ``blockIdx.x * blockDim.x + threadIdx.x``."""
  pad = " " * indent
  rng = stmt.args[0]
  name = rng.attrs["name"]
  start = _emit_scalar(rng.args[0])
  stop = _emit_scalar(rng.args[1])
  # Materialize the thread index at the loop variable's name.
  lines.append(f"{pad}long long {name} = blockIdx.x * blockDim.x + threadIdx.x + {start};")
  lines.append(f"{pad}if ({name} >= {stop}) return;")
  for sub in stmt.args[1:]:
    _emit_stmt_cuda(sub, ptr_expr, lines, indent)


def _render_host_driver(symbol: str, fun: Function, driver: PNode, kernel: PNode) -> list[str]:
  in_sizes = [int(_size(e.type.shape)) for e in fun.inputs]
  out_sizes = [int(_size(e.type.shape)) for e in fun.outputs]
  in_ctypes = [e.type.dtype.c_type for e in fun.inputs]
  out_ctypes = [e.type.dtype.c_type for e in fun.outputs]
  param_count = int(driver.attrs["param_count"])
  body = list(driver.args[param_count:])
  launch_stmt = next(s for s in body if s.op == POps.LAUNCH)
  grid_pnode = launch_stmt.args[: int(launch_stmt.attrs["grid_dims"])][0]
  # The schedule pass writes the trip count of the bound GLOBAL loop as
  # ``grid``. When thread-binding is enabled the true grid is ``ceil(trip /
  # block)`` and block defaults to 256. When the kernel sets ``bind_threads=False``
  # (e.g. for kernels with a top-level REDUCE feeding from a thread-bound
  # elementwise — see _kernelize_for_device), we launch with grid=1, block=1
  # so the kernel runs serially on a single thread.
  bind_threads = bool(kernel.attrs.get("bind_threads", True))
  trip = _emit_scalar(grid_pnode)
  if bind_threads:
    block_size = 256
    grid_str = f"({trip} + {block_size - 1}) / {block_size}"
    block_str = str(block_size)
  else:
    grid_str = "1"
    block_str = "1"
  lines: list[str] = [
    f'extern "C" int {symbol}(const double** arg, double** res, int* iw, double* w, void* mem) {{',
    "  (void)iw; (void)w; (void)mem;",
    "  if (!arg || !res) return ALLOY_ERR_NULL_ABI;",
  ]
  for i in range(len(fun.inputs)):
    lines.append(f"  if (!arg[{i}]) return ALLOY_ERR_NULL_INPUT;")
  for i in range(len(fun.outputs)):
    lines.append(f"  if (!res[{i}]) return ALLOY_ERR_NULL_RESULT;")
  # Device-side buffers: use the per-input dtype. When dtype != double, allocate
  # a host-side staging buffer and cast double→dtype before the H2D copy (and
  # dtype→double after D2H on the way out). The universal ABI is locked to
  # const double**, so the cast happens inside the driver.
  for i, (sz, ct) in enumerate(zip(in_sizes, in_ctypes, strict=True)):
    lines.append(f"  {ct}* d_in{i} = NULL;")
    lines.append(f"  if (cudaMalloc((void**)&d_in{i}, sizeof({ct}) * {sz}) != cudaSuccess) return ALLOY_ERR_CUDA;")
    if ct == "double":
      lines.append(f"  if (cudaMemcpy(d_in{i}, arg[{i}], sizeof({ct}) * {sz}, cudaMemcpyHostToDevice) != cudaSuccess) return ALLOY_ERR_CUDA;")
    else:
      lines.append(f"  {ct}* h_in{i} = ({ct}*)malloc(sizeof({ct}) * {sz});")
      lines.append(f"  if (!h_in{i}) return ALLOY_ERR_CUDA;")
      lines.append(f"  for (long _i = 0; _i < {sz}; _i++) h_in{i}[_i] = ({ct})arg[{i}][_i];")
      lines.append(f"  if (cudaMemcpy(d_in{i}, h_in{i}, sizeof({ct}) * {sz}, cudaMemcpyHostToDevice) != cudaSuccess) return ALLOY_ERR_CUDA;")
      lines.append(f"  free(h_in{i});")
  for i, (sz, ct) in enumerate(zip(out_sizes, out_ctypes, strict=True)):
    lines.append(f"  {ct}* d_out{i} = NULL;")
    lines.append(f"  if (cudaMalloc((void**)&d_out{i}, sizeof({ct}) * {sz}) != cudaSuccess) return ALLOY_ERR_CUDA;")
  kernel_args = ", ".join([*(f"d_in{i}" for i in range(len(in_sizes))), *(f"d_out{i}" for i in range(len(out_sizes)))])
  lines.append(f"  {kernel.attrs['name']}<<<dim3({grid_str}), dim3({block_str})>>>({kernel_args});")
  lines.append("  if (cudaDeviceSynchronize() != cudaSuccess) return ALLOY_ERR_CUDA;")
  for i, (sz, ct) in enumerate(zip(out_sizes, out_ctypes, strict=True)):
    if ct == "double":
      lines.append(f"  if (cudaMemcpy(res[{i}], d_out{i}, sizeof({ct}) * {sz}, cudaMemcpyDeviceToHost) != cudaSuccess) return ALLOY_ERR_CUDA;")
    else:
      lines.append(f"  {ct}* h_out{i} = ({ct}*)malloc(sizeof({ct}) * {sz});")
      lines.append(f"  if (!h_out{i}) return ALLOY_ERR_CUDA;")
      lines.append(f"  if (cudaMemcpy(h_out{i}, d_out{i}, sizeof({ct}) * {sz}, cudaMemcpyDeviceToHost) != cudaSuccess) return ALLOY_ERR_CUDA;")
      lines.append(f"  for (long _i = 0; _i < {sz}; _i++) res[{i}][_i] = (double)h_out{i}[_i];")
      lines.append(f"  free(h_out{i});")
  for i in range(len(in_sizes)):
    lines.append(f"  cudaFree(d_in{i});")
  for i in range(len(out_sizes)):
    lines.append(f"  cudaFree(d_out{i});")
  lines.append("  return ALLOY_SUCCESS;")
  lines.append("}")
  return lines


def _emit_stmt_cuda(stmt: PNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  pad = " " * indent
  if stmt.op == POps.FOR:
    rng = stmt.args[0]
    name = rng.attrs["name"]
    start = _emit_scalar(rng.args[0])
    stop = _emit_scalar(rng.args[1])
    step = _emit_scalar(rng.args[2])
    incr = "++" if step == "1" else f" += {step}"
    lines.append(f"{pad}for (long long {name} = {start}; {name} < {stop}; {name}{incr}) {{")
    for sub in stmt.args[1:]:
      _emit_stmt_cuda(sub, ptr_expr, lines, indent + 2)
    lines.append(f"{pad}}}")
  elif stmt.op == POps.STORE:
    view = stmt.args[0]
    value = stmt.args[1]
    buf = view.attrs["buffer"]
    ptr = ptr_expr.get(buf, buf)
    idx = _emit_scalar(view.args[0]) if view.args else "0"
    rhs = _emit_scalar(value)
    lines.append(f"{pad}{ptr}[{idx}] = {rhs};")
  elif stmt.op == POps.ASSIGN:
    lines.append(f"{pad}{stmt.attrs['target']} = {_emit_scalar(stmt.args[0])};")
  elif stmt.op == POps.CALL:
    n_in = int(stmt.attrs["n_in"])
    n_out = int(stmt.attrs["n_out"])
    in_args = stmt.args[:n_in]
    out_args = stmt.args[n_in : n_in + n_out]
    parts = [_emit_call_arg_cuda(a, ptr_expr) for a in (*in_args, *out_args)]
    lines.append(f"{pad}{stmt.attrs['callee']}({', '.join(parts)});")
  else:
    raise NotImplementedError(f"CUDA renderer: statement op {stmt.op} not handled")


def _emit_call_arg_cuda(node: PNode, ptr_expr: dict[str, str]) -> str:
  """Render a CALL argument: BUFFER (bare pointer) or VIEW (pointer + offset)."""
  if node.op == POps.BUFFER:
    return ptr_expr.get(node.attrs["name"], node.attrs["name"])
  if node.op == POps.VIEW:
    buf = node.attrs["buffer"]
    ptr = ptr_expr.get(buf, buf)
    idx = _emit_scalar(node.args[0]) if node.args else "0"
    return f"({ptr} + {idx})"
  raise NotImplementedError(f"CUDA CALL arg op {node.op}")


def _emit_scalar(n: PNode) -> str:
  # CUDA scalar expressions reuse the same shapes as the host C renderer; we
  # duplicate the small subset rather than importing it to keep CUDA code
  # generation self-contained and independent of host C changes.
  if n.op == POps.CONST_INT:
    return str(n.attrs["value"])
  if n.op == POps.CONST_FLOAT:
    return f"{n.attrs['value']:.17g}"
  if n.op == POps.VAR:
    return str(n.attrs["name"])
  if n.op == POps.LOAD:
    view = n.args[0]
    buf = view.attrs["buffer"]
    idx = _emit_scalar(view.args[0]) if view.args else "0"
    return f"{buf}[{idx}]"
  binop = {POps.ADD: "+", POps.SUB: "-", POps.MUL: "*", POps.DIV: "/"}.get(n.op)
  if binop is not None:
    return f"({_emit_scalar(n.args[0])} {binop} {_emit_scalar(n.args[1])})"
  if n.op == POps.NEG:
    return f"(-{_emit_scalar(n.args[0])})"
  unary = {
    POps.SIN: "sin",
    POps.COS: "cos",
    POps.TAN: "tan",
    POps.EXP: "exp",
    POps.LOG: "log",
    POps.SQRT: "sqrt",
    POps.TANH: "tanh",
    POps.ABS: "fabs",
    POps.FLOOR: "floor",
    POps.CEIL: "ceil",
  }.get(n.op)
  if unary is not None:
    return f"{unary}({_emit_scalar(n.args[0])})"
  raise NotImplementedError(f"CUDA renderer scalar op {n.op} not handled")


def _size(shape: tuple[int, ...]) -> int:
  n = 1
  for d in shape:
    n *= int(d)
  return n


__all__ = ["can_render_cuda", "render_cuda_source"]
