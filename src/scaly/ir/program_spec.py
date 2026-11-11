"""Program-dialect verifier: the rule tables and ``verify_program``.

``spec_program_full`` holds the invariants every node must satisfy. The shared ``Rule``/``Spec`` machinery is ``ir/spec.py``.
"""

from __future__ import annotations

from .program import ADDRESS_SPACES, SCALAR_OPS, ProgramNode, ProgramOp, RangeKind, walk_program
from .spec import Rule, Spec, VerifyError
from .types import DType, DeviceSpec
from ..utils.names import c_ident


def verify_program(root: ProgramNode, spec: "Spec | None" = None) -> None:
  if spec is None:
    spec = spec_program_full
  if root.op == ProgramOp.PROGRAM:
    pc = int(root.attrs.get("proc_count", 0))
    names = [node.attrs.get("name") for node in root.args[:pc]]
    duplicate = next((name for i, name in enumerate(names) if name in names[:i]), None)
    if duplicate is not None:
      raise VerifyError(f"verify_program: duplicate procedure name {duplicate!r}")
  for node in walk_program(root):
    result = spec.check(node)
    if result is not None:
      rule, diag = result
      raise VerifyError(f"verify_program: node op={node.op.value} failed rule {rule.description!r}: {diag}")
  pending: list[tuple[ProgramNode, bool, set[str]]] = [(root, False, set())]
  while pending:
    node, can_break, declared = pending.pop()
    if node.op == ProgramOp.BREAK_IF and not can_break:
      raise VerifyError("verify_program: BREAK_IF requires the nearest loop to be carried with exit_var")
    if node.op == ProgramOp.PROC:
      can_break = False
      declared = {"w"}  # raw procedures also take a workspace pointer
      pc = int(node.attrs["param_count"])
      names = [arg.attrs["name"] for arg in node.args[:pc]]
      names.extend(dict.fromkeys(arg.attrs["name"] for arg in node.args[pc:] if arg.op == ProgramOp.BUFFER))
      for name in names:
        _declare_name(name, declared)
    elif node.op == ProgramOp.FOR:
      can_break = bool(node.attrs.get("exit_var"))  # the FOR rule has checked that its range is carried
      if can_break:
        _declare_name(node.args[0].attrs["name"], declared)
      declared = set()
    elif node.op == ProgramOp.ASSIGN and node.attrs.get("declare"):
      _declare_name(node.attrs["target"], declared)
    if node.op in (ProgramOp.PROGRAM, ProgramOp.PROC, ProgramOp.BLOCK, ProgramOp.FOR):
      pending.extend((arg, can_break, declared) for arg in node.args)
  procs = root.args if root.op == ProgramOp.PROGRAM else (root,)
  params = {pr.attrs["name"]: pr.args[: int(pr.attrs.get("param_count", 0))] for pr in procs if pr.op == ProgramOp.PROC}
  for proc in procs:
    buffers = {n.attrs["name"]: n for n in walk_program(proc) if n.op == ProgramOp.BUFFER}
    for node in walk_program(proc):
      if node.op == ProgramOp.VIEW and (buf := buffers.get(node.attrs["buffer"])) is not None and node.dtype != buf.dtype:
        raise VerifyError(f"verify_program: view of {buf.attrs['name']!r} has dtype {node.dtype}, buffer has dtype {buf.dtype}")
      if node.op == ProgramOp.CALL and (formals := params.get(node.attrs.get("callee"))) is not None:
        for arg, formal in zip(node.args, formals, strict=False):
          if arg.dtype != formal.dtype:
            raise VerifyError(
              f"verify_program: call to {node.attrs['callee']!r} passes {arg.dtype} for its {formal.dtype} parameter {formal.attrs['name']!r}"
            )


def _declare_name(name: str, declared: set[str]) -> None:
  ident = c_ident(name)
  if ident in declared:
    raise VerifyError(f"verify_program: duplicate C identifier {ident!r} in one scope")
  declared.add(ident)


# ---------- shared rules ----------


def _dtype_is_dtype(n: ProgramNode) -> str | None:
  if not isinstance(n.dtype, DType):
    return f"dtype is {type(n.dtype).__name__}, expected DType"
  return None


def _const_int_has_value(n: ProgramNode) -> str | None:
  if "value" not in n.attrs or not isinstance(n.attrs["value"], int):
    return "CONST_INT missing integer 'value' attr"
  if not n.dtype.is_integer and not n.dtype.is_bool:
    return f"CONST_INT must have integer or bool dtype, got {n.dtype}"
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
  if len(n.args) != 1:
    return "VIEW requires exactly one flat index"
  if n.args[0].op not in SCALAR_OPS:
    return f"VIEW index op={n.args[0].op} is not scalar"
  if "buffer" not in n.attrs:
    return "VIEW missing 'buffer' name attr"
  if "lane_stride" in n.attrs and not isinstance(n.attrs["lane_stride"], int):
    return "VIEW lane_stride must be an integer"
  if "lane_storage" in n.attrs and n.attrs["lane_storage"] not in (1, 2, 4, 8):
    return "VIEW lane storage must reserve one, two, four or eight elements"
  return None


def _load_takes_view(n: ProgramNode) -> str | None:
  if len(n.args) != 1 or n.args[0].op != ProgramOp.VIEW:
    return "LOAD must wrap exactly one VIEW arg"
  if n.dtype != n.args[0].dtype:
    return f"LOAD dtype {n.dtype} does not match view dtype {n.args[0].dtype}"
  return None


def _store_attrs(n: ProgramNode) -> str | None:
  if len(n.args) != 2:
    return f"STORE expects 2 args (view, value), got {len(n.args)}"
  if n.args[0].op != ProgramOp.VIEW:
    return "STORE target must be VIEW"
  if n.args[1].op not in SCALAR_OPS:
    return f"STORE value op {n.args[1].op} not scalar"
  if n.dtype != n.args[0].dtype or n.dtype != n.args[1].dtype:
    return f"STORE dtype {n.dtype} must match target dtype {n.args[0].dtype} and value dtype {n.args[1].dtype}"
  return None


def _store_pair_attrs(n: ProgramNode) -> str | None:
  if len(n.args) != 3:
    return f"STORE_PAIR expects 3 args (view, first, second), got {len(n.args)}"
  if n.args[0].op != ProgramOp.VIEW:
    return "STORE_PAIR target must be VIEW"
  if n.args[0].dtype.c_type != "double" or n.dtype.c_type != "double":
    return "STORE_PAIR target must have double dtype"
  if any(value.op not in SCALAR_OPS or value.dtype.c_type != "double" for value in n.args[1:]):
    return "STORE_PAIR values must be double scalars"
  return None


def _assign_attrs(n: ProgramNode) -> str | None:
  if not isinstance(n.attrs.get("target"), str):
    return "ASSIGN missing string 'target' attr"
  if len(n.args) != 1 or n.args[0].op not in SCALAR_OPS:
    return "ASSIGN expects one scalar value"
  if not isinstance(n.attrs.get("declare", False), bool):
    return "ASSIGN 'declare' must be boolean"
  return None


def _range_kind(n: ProgramNode) -> str | None:
  if "kind" not in n.attrs or not isinstance(n.attrs["kind"], RangeKind):
    return "RANGE missing valid 'kind' attr"
  if "name" not in n.attrs:
    return "RANGE missing 'name' attr"
  if not isinstance(n.attrs.get("carried", False), bool):
    return "RANGE 'carried' must be boolean"
  if n.attrs.get("carried") and n.attrs["kind"] == RangeKind.VECTOR:
    return "a carried range cannot be VECTOR"
  if "lanes" in n.attrs:
    if n.attrs["kind"] != RangeKind.VECTOR or n.attrs["lanes"] not in ("auto", 1, 2, 4, 8):
      return "lane width requires a VECTOR range and auto or 1, 2, 4, 8 lanes"
    if n.attrs.get("lane_cap") not in (1, 2, 4, 8):
      return "lane cap must be 1, 2, 4 or 8"
  if len(n.args) != 3:
    return f"RANGE expects 3 args (start, stop, step), got {len(n.args)}"
  for arg, label in zip(n.args, ("start", "stop", "step"), strict=True):
    if arg.op not in SCALAR_OPS:
      return f"RANGE {label} op {arg.op} not scalar"
  return None


def _for_body(n: ProgramNode) -> str | None:
  if not n.args or n.args[0].op != ProgramOp.RANGE:
    return "FOR first arg must be RANGE"
  if not isinstance(n.attrs.get("exit_var", False), bool):
    return "FOR 'exit_var' must be boolean"
  if n.attrs.get("exit_var"):
    rng = n.args[0]
    if not rng.attrs.get("carried"):
      return "FOR exit_var requires a carried range"
    if len(rng.args) != 3 or any(
      arg.op != ProgramOp.CONST_INT or arg.attrs.get("value") != value for arg, value in ((rng.args[0], 0), (rng.args[2], 1))
    ):
      return "FOR exit_var requires start 0 and step 1 to report the trip count"
  if "vector_helper" in n.attrs:
    if len(n.args) != 2 or n.args[1].op != ProgramOp.FOR or n.args[1].args[0].attrs.get("kind") != RangeKind.VECTOR:
      return "vector helper requires exactly one nested VECTOR range"
    if not isinstance(n.attrs.get("vector_count"), int) or n.attrs["vector_count"] < 1:
      return "vector helper requires a positive static trip count"
  return None


def _break_if_attrs(n: ProgramNode) -> str | None:
  if len(n.args) != 1 or n.args[0].op not in SCALAR_OPS or not n.args[0].dtype.is_bool:
    return "BREAK_IF takes one bool scalar"
  return None


def _call_attrs(n: ProgramNode) -> str | None:
  if "callee" not in n.attrs:
    return "CALL missing 'callee' name attr"
  return None


def _print_pieces(n: ProgramNode) -> str | None:
  pieces = n.attrs.get("pieces")
  if not isinstance(pieces, tuple) or not all(isinstance(piece, str) for piece in pieces):
    return "PRINT missing tuple of string 'pieces'"
  if len(pieces) != len(n.args) + 1:
    return f"PRINT has {len(pieces)} pieces of text for {len(n.args)} values"
  if any(a.op not in SCALAR_OPS for a in n.args):
    return "PRINT values must be scalar"
  return None


def _bit_select_operands(n: ProgramNode) -> str | None:
  if len(n.args) != 3:
    return f"BIT_SELECT takes word, bit and value, got {len(n.args)} args"
  word, bit, value = n.args
  if not word.dtype.is_integer or not bit.dtype.is_integer:
    return f"BIT_SELECT word and bit must be integer, got {word.dtype} and {bit.dtype}"
  if not value.dtype.is_floating or n.dtype != value.dtype:
    return f"BIT_SELECT value must be floating and match the node dtype, got {value.dtype} for {n.dtype}"
  return None


_BOOL_RESULT = frozenset({ProgramOp.LT, ProgramOp.LE, ProgramOp.EQ, ProgramOp.NE, ProgramOp.AND, ProgramOp.OR, ProgramOp.NOT, ProgramOp.ISFINITE})


def _typed_scalar(n: ProgramNode) -> str | None:
  """Predicates and logic give bool, a select picks between values of its own dtype, and integer
  floor division and remainder stay integer."""
  if n.op in _BOOL_RESULT and not n.dtype.is_bool:
    return f"{n.op.value} gives bool, got {n.dtype}"
  if n.op in {ProgramOp.AND, ProgramOp.OR, ProgramOp.NOT} and not all(a.dtype.is_bool for a in n.args):
    return f"{n.op.value} takes bool operands"
  if n.op == ProgramOp.SELECT and (len(n.args) != 3 or not n.args[0].dtype.is_bool or any(a.dtype != n.dtype for a in n.args[1:])):
    return "SELECT takes a bool condition and two values of its dtype"
  if n.op in {ProgramOp.FLOORDIV, ProgramOp.FLOORMOD} and not (n.dtype.is_integer and all(a.dtype == n.dtype for a in n.args)):
    return f"{n.op.value} takes and gives integers of one dtype"
  return None


def _proc_params(n: ProgramNode) -> str | None:
  pc = n.attrs.get("param_count")
  if pc is None:
    return "PROC missing 'param_count'"
  for i in range(pc):
    if n.args[i].op != ProgramOp.BUFFER:
      return f"PROC param {i} op={n.args[i].op} is not BUFFER"
  mode = n.attrs.get("scalarize_mode")
  if mode is not None and mode not in {"disabled", "inline", "procedure"}:
    return f"PROC scalarize_mode {mode!r} is not disabled/inline/procedure"
  return None


spec_program_full = Spec(
  [
    Rule(None, "dtype-is-DType", _dtype_is_dtype),
    Rule(ProgramOp.CONST_INT, "const-int-value", _const_int_has_value),
    Rule(ProgramOp.CONST_FLOAT, "const-float-value", _const_float_has_value),
    Rule(ProgramOp.VAR, "var-name", _var_has_name),
    Rule(ProgramOp.BUFFER, "buffer-attrs", _buffer_attrs),
    Rule(ProgramOp.VIEW, "view-scalar-args", _view_args_scalar),
    Rule(ProgramOp.LOAD, "load-takes-view", _load_takes_view),
    Rule(ProgramOp.STORE, "store-attrs", _store_attrs),
    Rule(ProgramOp.STORE_PAIR, "store-pair-attrs", _store_pair_attrs),
    Rule(ProgramOp.ASSIGN, "assign-attrs", _assign_attrs),
    Rule(ProgramOp.RANGE, "range-attrs", _range_kind),
    Rule(ProgramOp.FOR, "for-body", _for_body),
    Rule(ProgramOp.BREAK_IF, "break-if-attrs", _break_if_attrs),
    Rule(ProgramOp.CALL, "call-attrs", _call_attrs),
    Rule(ProgramOp.PRINT, "print-pieces", _print_pieces),
    Rule(ProgramOp.PROC, "proc-params", _proc_params),
    Rule(ProgramOp.BIT_SELECT, "bit-select-operands", _bit_select_operands),
    *(Rule(op, "typed-scalar", _typed_scalar) for op in (*_BOOL_RESULT, ProgramOp.SELECT, ProgramOp.FLOORDIV, ProgramOp.FLOORMOD)),
  ]
)
