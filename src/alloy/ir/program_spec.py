"""Program-dialect verifier: the rule tables and ``verify_program``.

``spec_program_shared`` holds the invariants every node must satisfy, ``spec_host_program`` and
``spec_kernel_program`` the host-only and device-only restrictions, and ``spec_program_full`` both.
The shared ``Rule``/``Spec`` machinery is ``ir/spec.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

from .program import ADDRESS_SPACES, DEVICE_ONLY_OPS, HOST_ONLY_OPS, SCALAR_OPS, ProgramNode, ProgramOp, RangeKind
from .spec import Rule, Spec, VerifyError
from .types import DType, DeviceSpec


def _walk(root: ProgramNode) -> Iterable[ProgramNode]:
  seen: set[int] = set()
  stack: list[ProgramNode] = [root]
  while stack:
    n = stack.pop()
    if id(n) in seen:
      continue
    seen.add(id(n))
    yield n
    stack.extend(n.args)


def verify_program(root: ProgramNode, spec: "Spec | None" = None) -> None:
  if spec is None:
    spec = spec_program_full
  for node in _walk(root):
    result = spec.check(node)
    if result is not None:
      rule, diag = result
      raise VerifyError(f"verify_program: node op={node.op.value} failed rule {rule.description!r}: {diag}")


# ---------- shared rules ----------


def _dtype_is_dtype(n: ProgramNode) -> str | None:
  if not isinstance(n.dtype, DType):
    return f"dtype is {type(n.dtype).__name__}, expected DType"
  return None


def _const_int_has_value(n: ProgramNode) -> str | None:
  if "value" not in n.attrs or not isinstance(n.attrs["value"], int):
    return "CONST_INT missing integer 'value' attr"
  if not n.dtype.is_integer:
    return f"CONST_INT must have integer dtype, got {n.dtype}"
  return None


def _const_float_has_value(n: ProgramNode) -> str | None:
  if "value" not in n.attrs:
    return "CONST_FLOAT missing 'value' attr"
  if not n.dtype.is_floating:
    return f"CONST_FLOAT must have floating dtype, got {n.dtype}"
  return None


def _var_has_name(n: ProgramNode) -> str | None:
  if "name" not in n.attrs:
    return "VAR missing 'name' attr"
  return None


def _buffer_attrs(n: ProgramNode) -> str | None:
  for k in ("name", "shape", "address_space", "device"):
    if k not in n.attrs:
      return f"BUFFER missing {k!r} attr"
  if n.attrs["address_space"] not in ADDRESS_SPACES:
    return f"BUFFER address_space {n.attrs['address_space']!r} not in {sorted(ADDRESS_SPACES)}"
  if not isinstance(n.attrs["device"], DeviceSpec):
    return f"BUFFER 'device' must be DeviceSpec, got {type(n.attrs['device']).__name__}"
  return None


def _view_args_scalar(n: ProgramNode) -> str | None:
  for a in n.args:
    if a.op not in SCALAR_OPS:
      return f"VIEW index component op={a.op} is not scalar"
  if "buffer" not in n.attrs:
    return "VIEW missing 'buffer' name attr"
  return None


def _load_takes_view(n: ProgramNode) -> str | None:
  if len(n.args) != 1 or n.args[0].op != ProgramOp.VIEW:
    return "LOAD must wrap exactly one VIEW arg"
  return None


def _store_attrs(n: ProgramNode) -> str | None:
  if len(n.args) != 2:
    return f"STORE expects 2 args (view, value), got {len(n.args)}"
  if n.args[0].op != ProgramOp.VIEW:
    return "STORE target must be VIEW"
  if n.args[1].op not in SCALAR_OPS:
    return f"STORE value op {n.args[1].op} not scalar"
  return None


def _range_kind(n: ProgramNode) -> str | None:
  if "kind" not in n.attrs or not isinstance(n.attrs["kind"], RangeKind):
    return "RANGE missing valid 'kind' attr"
  if "name" not in n.attrs:
    return "RANGE missing 'name' attr"
  if len(n.args) != 3:
    return f"RANGE expects 3 args (start, stop, step), got {len(n.args)}"
  for arg, label in zip(n.args, ("start", "stop", "step"), strict=True):
    if arg.op not in SCALAR_OPS:
      return f"RANGE {label} op {arg.op} not scalar"
  return None


def _for_body(n: ProgramNode) -> str | None:
  if not n.args or n.args[0].op != ProgramOp.RANGE:
    return "FOR first arg must be RANGE"
  return None


def _call_attrs(n: ProgramNode) -> str | None:
  if "callee" not in n.attrs:
    return "CALL missing 'callee' name attr"
  return None


def _launch_attrs(n: ProgramNode) -> str | None:
  for k in ("kernel", "grid_dims", "block_dims"):
    if k not in n.attrs:
      return f"LAUNCH missing {k!r} attr"
  return None


def _barrier_kind(n: ProgramNode) -> str | None:
  if n.attrs.get("kind") not in {"device", "group", "warp"}:
    return f"BARRIER kind {n.attrs.get('kind')!r} not in device/group/warp"
  return None


def _proc_or_kernel_params(n: ProgramNode) -> str | None:
  pc = n.attrs.get("param_count")
  if pc is None:
    return "PROC/KERNEL missing 'param_count'"
  for i in range(pc):
    if n.args[i].op != ProgramOp.BUFFER:
      return f"PROC/KERNEL param {i} op={n.args[i].op} is not BUFFER"
  return None


spec_program_shared = Spec(
  [
    Rule(None, "dtype-is-DType", _dtype_is_dtype),
    Rule(ProgramOp.CONST_INT, "const-int-value", _const_int_has_value),
    Rule(ProgramOp.CONST_FLOAT, "const-float-value", _const_float_has_value),
    Rule(ProgramOp.VAR, "var-name", _var_has_name),
    Rule(ProgramOp.BUFFER, "buffer-attrs", _buffer_attrs),
    Rule(ProgramOp.VIEW, "view-scalar-args", _view_args_scalar),
    Rule(ProgramOp.LOAD, "load-takes-view", _load_takes_view),
    Rule(ProgramOp.STORE, "store-attrs", _store_attrs),
    Rule(ProgramOp.RANGE, "range-attrs", _range_kind),
    Rule(ProgramOp.FOR, "for-body", _for_body),
    Rule(ProgramOp.CALL, "call-attrs", _call_attrs),
    Rule(ProgramOp.LAUNCH, "launch-attrs", _launch_attrs),
    Rule(ProgramOp.BARRIER, "barrier-kind", _barrier_kind),
    Rule(ProgramOp.PROC, "proc-params", _proc_or_kernel_params),
    Rule(ProgramOp.KERNEL, "kernel-params", _proc_or_kernel_params),
  ]
)


def _host_proc_no_device_only(n: ProgramNode) -> str | None:
  for sub in _walk(n):
    if sub is n:
      continue
    if sub.op in DEVICE_ONLY_OPS:
      return f"host PROC contains device-only op {sub.op.value}"
  return None


def _kernel_no_host_only(n: ProgramNode) -> str | None:
  for sub in _walk(n):
    if sub is n:
      continue
    if sub.op in HOST_ONLY_OPS:
      return f"KERNEL contains host-only op {sub.op.value}"
  return None


spec_host_program = Spec(
  [
    *spec_program_shared.any,
    *(r for rs in spec_program_shared.by_op.values() for r in rs),
    Rule(ProgramOp.PROC, "host-proc-no-device-only", _host_proc_no_device_only),
  ]
)


spec_kernel_program = Spec(
  [
    *spec_program_shared.any,
    *(r for rs in spec_program_shared.by_op.values() for r in rs),
    Rule(ProgramOp.KERNEL, "kernel-no-host-only", _kernel_no_host_only),
  ]
)


# Full spec: shared + host + kernel checks together. Suitable for whole-program verify.
spec_program_full = Spec(
  [
    *spec_program_shared.any,
    *(r for rs in spec_program_shared.by_op.values() for r in rs),
    Rule(ProgramOp.PROC, "host-proc-no-device-only", _host_proc_no_device_only),
    Rule(ProgramOp.KERNEL, "kernel-no-host-only", _kernel_no_host_only),
  ]
)


__all__ = [
  "spec_host_program",
  "spec_kernel_program",
  "spec_program_full",
  "spec_program_shared",
  "verify_program",
]
