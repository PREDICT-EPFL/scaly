"""Phase 8 first slice: CUDA C source rendering for device-placed functions.

This is source generation only — no nvcc integration yet. The output uses the
universal ABI on the host side (``int f(const double** arg, double** res, ...)``)
and `__global__` for the kernel. Host code allocates device memory via
``cudaMalloc``/``cudaMemcpy``, launches the kernel synchronously, and copies
results back.

Range kinds inside the kernel are still ``GLOBAL`` today; a future pass will
rebind them to ``threadIdx``/``blockIdx``. For now every range becomes a serial
``for`` inside the kernel, which is correct but not yet GPU-shaped — the
kernel still runs on a single thread of a single block. That is enough to
ship a stable text contract that the future thread-binding pass and the
actual CUDA backend (Phase 8 follow-up) can extend.
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
  symbol = fun.name
  lines: list[str] = [
    "#include <cuda_runtime.h>",
    "#include <math.h>",
    "",
    *_ABI_DEFINES,
    "",
  ]
  lines += _render_cuda_kernel(kernel)
  lines.append("")
  lines += _render_host_driver(symbol, fun, driver, kernel)
  return "\n".join(lines).rstrip() + "\n"


def _render_cuda_kernel(kernel: PNode) -> list[str]:
  param_count = int(kernel.attrs["param_count"])
  params = list(kernel.args[:param_count])
  body = list(kernel.args[param_count:])
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
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    _emit_stmt_cuda(stmt, ptr_expr, out, indent=2)
  out.append("}")
  return out


def _render_host_driver(symbol: str, fun: Function, driver: PNode, kernel: PNode) -> list[str]:
  in_sizes = [int(_size(e.type.shape)) for e in fun.inputs]
  out_sizes = [int(_size(e.type.shape)) for e in fun.outputs]
  param_count = int(driver.attrs["param_count"])
  body = list(driver.args[param_count:])
  launch_stmt = next(s for s in body if s.op == POps.LAUNCH)
  grid = launch_stmt.args[: int(launch_stmt.attrs["grid_dims"])]
  block_dims = launch_stmt.args[int(launch_stmt.attrs["grid_dims"]) : int(launch_stmt.attrs["grid_dims"]) + int(launch_stmt.attrs["block_dims"])]
  grid_str = ", ".join(_emit_scalar(g) for g in grid)
  block_str = ", ".join(_emit_scalar(b) for b in block_dims)
  lines: list[str] = [
    f'extern "C" int {symbol}(const double** arg, double** res, int* iw, double* w, void* mem) {{',
    "  (void)iw; (void)w; (void)mem;",
    "  if (!arg || !res) return ALLOY_ERR_NULL_ABI;",
  ]
  for i in range(len(fun.inputs)):
    lines.append(f"  if (!arg[{i}]) return ALLOY_ERR_NULL_INPUT;")
  for i in range(len(fun.outputs)):
    lines.append(f"  if (!res[{i}]) return ALLOY_ERR_NULL_RESULT;")
  # Device-side buffers
  for i, sz in enumerate(in_sizes):
    lines.append(f"  double* d_in{i} = NULL;")
    lines.append(f"  if (cudaMalloc((void**)&d_in{i}, sizeof(double) * {sz}) != cudaSuccess) return ALLOY_ERR_CUDA;")
    lines.append(f"  if (cudaMemcpy(d_in{i}, arg[{i}], sizeof(double) * {sz}, cudaMemcpyHostToDevice) != cudaSuccess) return ALLOY_ERR_CUDA;")
  for i, sz in enumerate(out_sizes):
    lines.append(f"  double* d_out{i} = NULL;")
    lines.append(f"  if (cudaMalloc((void**)&d_out{i}, sizeof(double) * {sz}) != cudaSuccess) return ALLOY_ERR_CUDA;")
  kernel_args = ", ".join([*(f"d_in{i}" for i in range(len(in_sizes))), *(f"d_out{i}" for i in range(len(out_sizes)))])
  lines.append(f"  {kernel.attrs['name']}<<<dim3({grid_str}), dim3({block_str})>>>({kernel_args});")
  lines.append("  if (cudaDeviceSynchronize() != cudaSuccess) return ALLOY_ERR_CUDA;")
  for i, sz in enumerate(out_sizes):
    lines.append(f"  if (cudaMemcpy(res[{i}], d_out{i}, sizeof(double) * {sz}, cudaMemcpyDeviceToHost) != cudaSuccess) return ALLOY_ERR_CUDA;")
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
  else:
    raise NotImplementedError(f"CUDA renderer: statement op {stmt.op} not handled")


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
