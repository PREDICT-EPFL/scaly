from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
from typing import Callable

from alloy.abi import c_api_signature
from alloy.codegen.solver_c import (
  is_solver_function,
  render_solver_raw,
  solver_callees,
  solver_includes,
)
from alloy.function import Function
from alloy.ops import Ops
from alloy.tape import Instruction


@dataclass(frozen=True, slots=True)
class CModule:
  header_name: str
  source_name: str
  header: str
  source: str


_WORKSPACE_SPILL_THRESHOLD = 1024  # doubles; slots this large or bigger move from the stack to w[].


def _c_array(values: tuple[int, ...]) -> str:
  return "{" + ", ".join(str(v) for v in values) + "}"


def _c_ident(name: str) -> str:
  ident = re.sub(r"\W", "_", name)
  return f"_{ident}" if ident[:1].isdigit() else ident


def _c_float(value: float) -> str:
  if math.isnan(value):
    return "NAN"
  if math.isinf(value):
    return "INFINITY" if value > 0 else "-INFINITY"
  return repr(float(value))


def _abi_status_defines() -> list[str]:
  return [
    "#ifndef ALLOY_SUCCESS",
    "#define ALLOY_SUCCESS 0",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_ABI",
    "#define ALLOY_ERR_NULL_ABI 1",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_WORK",
    "#define ALLOY_ERR_NULL_WORK 2",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_RESULT",
    "#define ALLOY_ERR_NULL_RESULT 3",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_INPUT",
    "#define ALLOY_ERR_NULL_INPUT 4",
    "#endif",
  ]


def _typed_cpp_wrapper(fun: Function, symbol: str) -> list[str]:
  params: list[str] = []
  arg_values: list[str] = []
  res_values: list[str] = []
  checks: list[str] = []
  for name, expr in zip(fun.input_names, fun.inputs, strict=True):
    assert isinstance(name, str)
    ident = _c_ident(name)
    type_name = f"{symbol}_{ident}_in"
    params.append(f"const {type_name}& in_{ident}")
    arg_values.append(f"in_{ident}.data")
    checks.append(f'static_assert(sizeof({type_name}) == sizeof(double) * {expr.size}, "{type_name} size mismatch");')
  for name, expr in zip(fun.output_names, fun.outputs, strict=True):
    assert isinstance(name, str)
    ident = _c_ident(name)
    type_name = f"{symbol}_{ident}_out"
    params.append(f"{type_name}& out_{ident}")
    res_values.append(f"out_{ident}.data")
    checks.append(f'static_assert(sizeof({type_name}) == sizeof(double) * {expr.size}, "{type_name} size mismatch");')

  arg_init = ", ".join(arg_values) or "nullptr"
  res_init = ", ".join(res_values) or "nullptr"
  return [
    "#ifdef __cplusplus",
    *checks,
    f"static inline int {symbol}_call({', '.join(params)}) {{",
    f"  double w[{symbol}_SZ_W > 0 ? {symbol}_SZ_W : 1];",
    f"  const double* arg[{symbol}_SZ_ARG > 0 ? {symbol}_SZ_ARG : 1] = {{{arg_init}}};",
    f"  double* res[{symbol}_SZ_RES > 0 ? {symbol}_SZ_RES : 1] = {{{res_init}}};",
    f"  return {symbol}(arg, res, nullptr, {symbol}_SZ_W ? w : nullptr, nullptr);",
    "}",
    "#endif",
  ]


def render_c_api_header(fun: Function, *, typed_buffers: bool = True) -> str:
  symbol = _c_ident(fun.name)
  lines = [
    "#pragma once",
    "",
    *_abi_status_defines(),
    "",
    f"#define {symbol}_SZ_ARG {len(fun.inputs)}",
    f"#define {symbol}_SZ_RES {len(fun.outputs)}",
    f"#define {symbol}_SZ_IW 0",
    f"#define {symbol}_SZ_W {_workspace_size(fun)}",
    "",
    f"// Universal CasADi-style ABI for {fun.name}.",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    c_api_signature(symbol) + ";",
  ]
  lines += [
    f"int {symbol}_sz_arg(void);",
    f"int {symbol}_sz_res(void);",
    f"int {symbol}_sz_iw(void);",
    f"int {symbol}_sz_w(void);",
    f"void* {symbol}_alloc_mem(void);",
    f"int {symbol}_init_mem(void* mem);",
    f"void {symbol}_free_mem(void* mem);",
    "#ifdef __cplusplus",
    "}",
    "#endif",
  ]
  if typed_buffers:
    lines += ["", "// Optional typed buffer wrappers for statically known shapes."]
    for name, expr in zip(fun.input_names, fun.inputs, strict=True):
      lines.append(f"typedef struct {{ double data[{expr.size}]; }} {symbol}_{_c_ident(name)}_in;")
    for name, expr in zip(fun.output_names, fun.outputs, strict=True):
      lines.append(f"typedef struct {{ double data[{expr.size}]; }} {symbol}_{_c_ident(name)}_out;")
    lines += _typed_cpp_wrapper(fun, symbol)
  sparse_outputs = [(name, sp) for name, sp in zip(fun.output_names, fun.output_sparsities, strict=True) if sp is not None]
  if sparse_outputs:
    lines += ["", "// Sparse output metadata for compact derivative buffers."]
    for name, sp in sparse_outputs:
      assert sp is not None
      prefix = f"{symbol}_{_c_ident(name)}"
      lines.append(f"#define {prefix}_NNZ {sp.nnz}")
      lines.append(f"#define {prefix}_NROW {sp.shape[0]}")
      lines.append(f"#define {prefix}_NCOL {sp.shape[1]}")
      row_ptr, col_ind = sp.to_csr()
      col_ptr, row_ind = sp.to_csc()
      lines.append(f"static const int {prefix}_rows[{sp.nnz}] = {_c_array(sp.rows)};")
      lines.append(f"static const int {prefix}_cols[{sp.nnz}] = {_c_array(sp.cols)};")
      lines.append(f"static const int {prefix}_csr_row_ptr[{sp.shape[0] + 1}] = {_c_array(row_ptr)};")
      lines.append(f"static const int {prefix}_csr_col_ind[{sp.nnz}] = {_c_array(col_ind)};")
      lines.append(f"static const int {prefix}_csc_col_ptr[{sp.shape[1] + 1}] = {_c_array(col_ptr)};")
      lines.append(f"static const int {prefix}_csc_row_ind[{sp.nnz}] = {_c_array(row_ind)};")
  return "\n".join(lines) + "\n"


def render_c_source(fun: Function) -> str:
  """Render a standalone scalar C implementation of ``fun`` and its callees."""

  extra_includes = solver_includes(fun)
  lines = [
    "#include <math.h>",
    "#include <stddef.h>",
    *extra_includes,
    "",
    *_abi_status_defines(),
    "",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  ordered = _function_order(fun)
  for fn in ordered[:-1]:
    lines += _render_c_raw_function(fn)
    lines.append("")
  lines += _render_c_function(ordered[-1])
  lines.append("")
  lines += ["#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def render_c_module(fun: Function, *, header_name: str | None = None, source_name: str | None = None, typed_buffers: bool = True) -> CModule:
  symbol = _c_ident(fun.name)
  header_name = f"{symbol}.h" if header_name is None else header_name
  source_name = f"{symbol}.c" if source_name is None else source_name
  include_name = header_name.replace("\\", "\\\\").replace('"', '\\"')
  source = f'#include "{include_name}"\n\n' + render_c_source(fun)
  return CModule(header_name, source_name, render_c_api_header(fun, typed_buffers=typed_buffers), source)


def _render_c_function(fun: Function) -> list[str]:
  symbol = _c_ident(fun.name)
  workspace_size = _workspace_size(fun)
  lines = [*_render_c_raw_function(fun), ""]
  lines += [
    f"int {symbol}_sz_arg(void) {{ return {len(fun.inputs)}; }}",
    f"int {symbol}_sz_res(void) {{ return {len(fun.outputs)}; }}",
    f"int {symbol}_sz_iw(void) {{ return 0; }}",
    f"int {symbol}_sz_w(void) {{ return {workspace_size}; }}",
    f"void* {symbol}_alloc_mem(void) {{ return NULL; }}",
    f"int {symbol}_init_mem(void* mem) {{ (void)mem; return ALLOY_SUCCESS; }}",
    f"void {symbol}_free_mem(void* mem) {{ (void)mem; }}",
    "",
    c_api_signature(symbol) + " {",
    "  (void)iw;",
    "  (void)mem;",
    "  if (!arg || !res) return ALLOY_ERR_NULL_ABI;",
  ]
  if workspace_size:
    lines.append("  if (!w) return ALLOY_ERR_NULL_WORK;")
  for i in range(len(fun.inputs)):
    lines.append(f"  if (!arg[{i}]) return ALLOY_ERR_NULL_INPUT;")
  for i in range(len(fun.outputs)):
    lines.append(f"  if (!res[{i}]) return ALLOY_ERR_NULL_RESULT;")
  args = [*(f"arg[{i}]" for i in range(len(fun.inputs))), *(f"res[{i}]" for i in range(len(fun.outputs))), "w"]
  lines += [f"  {_raw_symbol(fun)}({', '.join(args)});", "  return ALLOY_SUCCESS;", "}"]
  return lines


_NAME_MAP: dict[int, str] = {}


def _is_alias(inst: Instruction) -> bool:
  return inst.op == Ops.RESHAPE or (inst.op == Ops.SLICE and _slice_offset(inst) is not None)


def _alias_target(inst: Instruction, instructions: tuple[Instruction, ...]) -> int:
  """Follow alias chain through RESHAPE / contiguous SLICE and return the storage-owner index."""
  while _is_alias(inst):
    inst = instructions[inst.inputs[0]]
  return inst.index


def _compute_lifetimes(tape, uses: dict[int, list[Instruction]], skip: set[int], inline: dict) -> dict[int, int]:
  """last_use[idx] = max index that transitively reads inst idx, propagating through inlined chains, peephole skips, and aliases."""
  last_use = {inst.index: inst.index for inst in tape.instructions}
  instructions = tape.instructions

  def bump(target: int, reader: int) -> None:
    if reader > last_use[target]:
      last_use[target] = reader

  def propagate(input_idx: int, reader_idx: int, seen: set[int]) -> None:
    if input_idx in seen:
      return
    seen.add(input_idx)
    bump(input_idx, reader_idx)
    # An inlined or peephole-skipped instruction does not materialize: its consumers
    # read its inputs directly at the consumer's index, so we have to recurse here.
    if input_idx in inline or input_idx in skip:
      for sub in instructions[input_idx].inputs:
        propagate(sub, reader_idx, seen)

  for inst in instructions:
    if inst.index in inline or inst.index in skip:
      continue
    for src in inst.inputs:
      propagate(src, inst.index, set())

  for out_idx in tape.outputs:
    propagate(out_idx, len(instructions), set())

  for inst in reversed(instructions):
    if _is_alias(inst) and inst.inputs:
      src = inst.inputs[0]
      if last_use[inst.index] > last_use[src]:
        last_use[src] = last_use[inst.index]
  return last_use


def _pack_slots(tape, skip: set[int], last_use: dict[int, int]) -> tuple[dict[int, int], list[int]]:
  """Lifetime-pack owner instructions into shared slots."""
  free_at_per_slot: list[int] = []
  slot_size: list[int] = []
  slot_of: dict[int, int] = {}
  for inst in tape.instructions:
    if inst.index in skip:
      continue
    if inst.op in {Ops.INPUT, Ops.CONST}:
      continue
    if _is_alias(inst):
      continue
    chosen = -1
    for i, free_at in enumerate(free_at_per_slot):
      if free_at <= inst.index:
        chosen = i
        break
    if chosen == -1:
      chosen = len(free_at_per_slot)
      free_at_per_slot.append(0)
      slot_size.append(0)
    free_at_per_slot[chosen] = last_use[inst.index] + 1
    slot_size[chosen] = max(slot_size[chosen], inst.size)
    slot_of[inst.index] = chosen
  return slot_of, slot_size


def _spill_plan(slot_size: list[int]) -> tuple[dict[int, int], int]:
  """Sequentially assign workspace offsets to slots above ``_WORKSPACE_SPILL_THRESHOLD``."""
  offsets: dict[int, int] = {}
  total = 0
  for slot, size in enumerate(slot_size):
    if size >= _WORKSPACE_SPILL_THRESHOLD:
      offsets[slot] = total
      total += size
  return offsets, total


def _render_c_raw_function(fun: Function) -> list[str]:
  if is_solver_function(fun):
    return render_solver_raw(fun)
  tape = fun.tape()
  input_ref = {name: f"in{i}" for i, name in enumerate(fun.input_names)}
  uses = _use_map(tape)
  skip = _skipped_instructions(tape, uses)
  inline = _inline_scalar_table(tape, uses, skip)
  for idx in inline:
    skip.add(idx)
  last_use = _compute_lifetimes(tape, uses, skip, inline)
  slot_of, slot_size = _pack_slots(tape, skip, last_use)
  spill_offset, ws_size_self = _spill_plan(slot_size)
  call_w_offset = ws_size_self
  name_map: dict[int, str] = {idx: f"s{slot}" for idx, slot in slot_of.items()}

  global _NAME_MAP
  prev_name_map = _NAME_MAP
  _NAME_MAP = name_map
  try:
    params = [*(f"const double* in{i}" for i in range(len(fun.inputs))), *(f"double* out{i}" for i in range(len(fun.outputs))), "double* w"]
    lines = [f"static void {_raw_symbol(fun)}({', '.join(params)}) {{", "  (void)w;"]
    for slot, size in enumerate(slot_size):
      if slot in spill_offset:
        lines.append(f"  double* s{slot} = w + {spill_offset[slot]};")
      else:
        lines.append(f"  double s{slot}[{size}];")
    declared_slots: set[int] = set()
    for inst in tape:
      if inst.index in skip:
        continue
      if inst.index in slot_of:
        slot = slot_of[inst.index]
        if slot in declared_slots:
          continue
        declared_slots.add(slot)
        continue
      lines += _declare_raw_instruction(inst, input_ref, skip)
    for inst in tape:
      if inst.index in skip:
        continue
      lines += _render_instruction(inst, tape.instructions, call_w_offset, inline)
    for out_slot, inst_index in enumerate(tape.outputs):
      out = tape.instructions[inst_index]
      lines += _copy_loop(f"out{out_slot}", _value_ref(out), out.size)
    lines.append("}")
    return lines
  finally:
    _NAME_MAP = prev_name_map


def _function_order(fun: Function) -> list[Function]:
  seen: set[int] = set()
  ordered: list[Function] = []

  def visit(fn: Function) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    for callee in _callees(fn):
      visit(callee)
    ordered.append(fn)

  visit(fun)
  return ordered


def _callees(fun: Function) -> list[Function]:
  ret: list[Function] = []
  seen: set[int] = set()
  if is_solver_function(fun):
    # SolverFunctions render via a custom template that calls the oracle (and
    # for NLP, the derivative Functions) — these aren't reachable through the
    # solver's own tape (it only contains SOLVER_CALL nodes), so surface them
    # explicitly here.
    for callee in solver_callees(fun):
      if id(callee) not in seen:
        seen.add(id(callee))
        ret.append(callee)
    return ret
  for inst in fun.tape():
    if inst.op not in {Ops.CALL, Ops.MAP}:
      continue
    callee = inst.attrs["callee"]
    if id(callee) not in seen:
      seen.add(id(callee))
      ret.append(callee)
  return ret


def _workspace_size(fun: Function, memo: dict[int, int] | None = None) -> int:
  memo = memo if memo is not None else {}
  if id(fun) in memo:
    return memo[id(fun)]
  memo[id(fun)] = 0  # break cycles defensively
  if is_solver_function(fun):
    # Solver wrappers carry no scalar workspace themselves (QP data is on the
    # stack); they just need enough to call the oracle.
    callee_max = 0
    for callee in solver_callees(fun):
      callee_max = max(callee_max, _workspace_size(callee, memo))
    memo[id(fun)] = callee_max
    return callee_max
  tape = fun.tape()
  uses = _use_map(tape)
  skip = _skipped_instructions(tape, uses)
  inline = _inline_scalar_table(tape, uses, skip)
  for idx in inline:
    skip.add(idx)
  last_use = _compute_lifetimes(tape, uses, skip, inline)
  _, slot_size = _pack_slots(tape, skip, last_use)
  _, own = _spill_plan(slot_size)
  callee_max = 0
  for inst in tape.instructions:
    if inst.op in {Ops.CALL, Ops.MAP}:
      callee_max = max(callee_max, _workspace_size(inst.attrs["callee"], memo))
  total = own + callee_max
  memo[id(fun)] = total
  return total


def _use_map(tape) -> dict[int, list[Instruction]]:
  uses: dict[int, list[Instruction]] = {}
  for inst in tape.instructions:
    for src in inst.inputs:
      uses.setdefault(src, []).append(inst)
  return uses


def _skipped_instructions(tape, uses: dict[int, list[Instruction]]) -> set[int]:
  output_set = set(tape.outputs)
  ret: set[int] = set()
  for inst in tape.instructions:
    if inst.index in output_set:
      # Outputs must materialize so the final copy loop has a buffer to read from.
      continue
    if inst.op == Ops.TRANSPOSE and inst.attrs.get("axes") == (1, 0):
      consumers = uses.get(inst.index, [])
      if consumers and all(user.op == Ops.MATMUL and len(user.inputs) == 2 and user.inputs[1] == inst.index for user in consumers):
        ret.add(inst.index)
      if len(inst.inputs) == 1 and consumers and all(user.op == Ops.GATHER for user in consumers) and len(uses.get(inst.inputs[0], [])) == 1:
        inner = tape.instructions[inst.inputs[0]]
        if (inner.op == Ops.CONCAT and inner.attrs.get("axis", 0) == 1) or (inner.op == Ops.STACK and inner.attrs.get("axis", 0) == 1):
          # The peephole reads CONCAT/STACK args directly per output entry (or per per-block group),
          # so we can skip materializing the transpose-of-concat. The tile path needs x materialized,
          # so don't skip when any consumer gather has a useful tile pattern.
          if not any(_detect_tile(user.attrs["indices"]) is not None for user in uses.get(inst.index, [])):
            ret.add(inst.index)
            ret.add(inst.inputs[0])
  changed = True
  while changed:
    changed = False
    for inst in tape.instructions:
      if inst.index in ret or inst.op not in {Ops.ADD, Ops.SUB} or inst.index in output_set:
        continue
      if uses.get(inst.index) and all(user.index in ret for user in uses[inst.index]):
        ret.add(inst.index)
        changed = True
  return ret


_INLINE_UNARY = {
  Ops.NEG: "-",
  Ops.SIN: "sin",
  Ops.COS: "cos",
  Ops.TAN: "tan",
  Ops.ASIN: "asin",
  Ops.ACOS: "acos",
  Ops.ATAN: "atan",
  Ops.SINH: "sinh",
  Ops.COSH: "cosh",
  Ops.TANH: "tanh",
  Ops.EXP: "exp",
  Ops.LOG: "log",
  Ops.SQRT: "sqrt",
  Ops.ABS: "fabs",
}
# Expensive unaries lower to a libm call. Inlining them into a vector consumer (where the
# consumer's elementwise rendering replays the inline once per scalar position) puts the
# same ``sin(x)`` / ``exp(x)`` text in the C source N times. The C compiler can CSE
# pure-libm calls under ``-O2``, but it doesn't always — and it costs source size either
# way. Keep these in scratch slots when the broadcast factor exceeds 1.
_EXPENSIVE_UNARY = {Ops.SIN, Ops.COS, Ops.TAN, Ops.ASIN, Ops.ACOS, Ops.ATAN, Ops.SINH, Ops.COSH, Ops.TANH, Ops.EXP, Ops.LOG, Ops.SQRT}
_INLINE_BINARY = {Ops.ADD: "+", Ops.SUB: "-", Ops.MUL: "*", Ops.DIV: "/"}


_INLINE_CONSUMER_OPS = (
  set(_INLINE_UNARY) | set(_INLINE_BINARY) | {Ops.POW, Ops.ATAN2, Ops.MINIMUM, Ops.MAXIMUM, Ops.FLOOR, Ops.CEIL, Ops.STACK, Ops.CONCAT}
)

InlineReader = Callable[[str], str]


def _inline_read(inst: Instruction, idx: str, inline: dict[int, InlineReader]) -> str:
  if inst.index in inline:
    return inline[inst.index]("0" if inst.size == 1 else idx)
  return f"{_value_ref(inst)}[{'0' if inst.size == 1 else idx}]"


def _effective_inline_size(
  consumer: Instruction, instructions: tuple[Instruction, ...], uses: dict[int, list[Instruction]], consumer_ops: set, max_depth: int = 8
) -> int:
  """Estimate the largest rendered size the immediate consumer would replicate into. Walks
  up the consumer chain past inline-eligible nodes (single-use elementwise ops) until it
  hits a non-inlineable consumer or runs out of depth. Used to gate expensive-unary inlining
  so a libm call doesn't get replayed at every scalar position of a size-N broadcast."""
  size = consumer.size
  cur = consumer
  for _ in range(max_depth):
    # Would ``cur`` itself be inlineable? Same predicate as ``_inline_scalar_table`` head.
    if cur.op not in consumer_ops:
      break
    cur_uses = uses.get(cur.index, [])
    if len(cur_uses) != 1:
      break
    cur_args = tuple(instructions[i] for i in cur.inputs)
    if not all(a.shape == cur.shape or a.size == 1 for a in cur_args):
      break
    cur = cur_uses[0]
    size = max(size, cur.size)
  return size


def _inline_scalar_table(tape, uses: dict[int, list[Instruction]], skip: set[int]) -> dict[int, InlineReader]:
  inline: dict[int, InlineReader] = {}
  for inst in tape.instructions:
    if inst.index in skip:
      continue
    consumers = uses.get(inst.index, [])
    if len(consumers) != 1 or consumers[0].op not in _INLINE_CONSUMER_OPS:
      continue
    args = tuple(tape.instructions[i] for i in inst.inputs)
    if not all(arg.shape == inst.shape or arg.size == 1 for arg in args):
      continue
    # Expensive unaries (libm calls) get replayed once per scalar position whenever the
    # surrounding rendered context has a higher broadcast size than the inst itself. The
    # immediate consumer may be size-equal to inst (e.g. NEG of SIN where both are size 1)
    # while its own consumer up the chain is the size-N vector that replicates everything
    # down to the leaves. Walk through inlined consumers to find the effective rendered
    # size, and bail out if the libm call would get duplicated.
    if inst.op in _EXPENSIVE_UNARY:
      rendered_size = _effective_inline_size(consumers[0], tape.instructions, uses, _INLINE_CONSUMER_OPS)
      if rendered_size > inst.size:
        continue
    if inst.op in _INLINE_UNARY:
      op = _INLINE_UNARY[inst.op]
      arg = args[0]
      inline[inst.index] = (
        (lambda idx, arg=arg, op=op: f"-{_inline_read(arg, idx, inline)}")
        if op == "-"
        else (lambda idx, arg=arg, op=op: f"{op}({_inline_read(arg, idx, inline)})")
      )
    elif inst.op in _INLINE_BINARY:
      op_str = _INLINE_BINARY[inst.op]
      a, b = args
      inline[inst.index] = lambda idx, a=a, b=b, op_str=op_str: f"({_inline_read(a, idx, inline)} {op_str} {_inline_read(b, idx, inline)})"
    elif inst.op == Ops.POW and args[1].op == Ops.CONST and args[1].value is not None and args[1].value.size == 1:
      exp = float(args[1].value.reshape(-1)[0])
      a = args[0]
      if exp == 2.0:
        inline[inst.index] = lambda idx, a=a: (lambda r: f"({r} * {r})")(_inline_read(a, idx, inline))
      elif exp == 0.5:
        inline[inst.index] = lambda idx, a=a: f"sqrt({_inline_read(a, idx, inline)})"
      elif exp == -1.0:
        inline[inst.index] = lambda idx, a=a: f"(1.0 / {_inline_read(a, idx, inline)})"
  return inline


def _declare_raw_instruction(inst: Instruction, input_ref: dict[str, str], skip: set[int]) -> list[str]:
  if inst.index in skip:
    return []
  if inst.op == Ops.INPUT:
    assert inst.name is not None
    return [f"  const double* {_value_ref(inst)} = {input_ref[inst.name]};"]
  if inst.op == Ops.CONST:
    assert inst.value is not None
    values = "0.0" if inst.size and bool((inst.value.reshape(-1) == 0).all()) else ", ".join(_c_float(x) for x in inst.value.reshape(-1))
    return [f"  const double {_value_ref(inst)}[{inst.size}] = {{{values}}};"]
  if inst.op == Ops.RESHAPE or (inst.op == Ops.SLICE and _slice_offset(inst) is not None):
    return [f"  const double* {_value_ref(inst)};"]
  return [f"  double {_value_ref(inst)}[{inst.size}];"]


def _render_instruction(inst: Instruction, instructions: tuple[Instruction, ...], call_w_offset: int, inline: dict[int, InlineReader]) -> list[str]:
  if inst.op in {Ops.INPUT, Ops.CONST}:
    return []
  args = tuple(instructions[i] for i in inst.inputs)
  if inst.op in {
    Ops.NEG,
    Ops.SIN,
    Ops.COS,
    Ops.TAN,
    Ops.ASIN,
    Ops.ACOS,
    Ops.ATAN,
    Ops.SINH,
    Ops.COSH,
    Ops.TANH,
    Ops.EXP,
    Ops.LOG,
    Ops.SQRT,
    Ops.ABS,
    Ops.FLOOR,
    Ops.CEIL,
  }:
    return _elementwise_unary(inst, args[0], inline)
  if inst.op in {Ops.ADD, Ops.SUB, Ops.MUL, Ops.DIV, Ops.POW, Ops.ATAN2, Ops.MINIMUM, Ops.MAXIMUM}:
    return _elementwise_binary(inst, args[0], args[1], inline)
  if inst.op == Ops.SUM:
    src = _value_ref(args[0])
    lines = [f"  {_value_ref(inst)}[0] = 0.0;"]
    lines += _loop(args[0].size, lambda k: f"{_value_ref(inst)}[0] += {src}[{k}];")
    return lines
  if inst.op == Ops.RESHAPE:
    return [f"  {_value_ref(inst)} = {_value_ref(args[0])};"]
  if inst.op == Ops.TRANSPOSE:
    return _transpose(inst, args[0])
  if inst.op == Ops.SLICE:
    return _slice(inst, args[0])
  if inst.op == Ops.GATHER:
    return _gather(inst, args[0], instructions, inline)
  if inst.op == Ops.SCATTER:
    return _scatter(inst, args[0])
  if inst.op == Ops.STACK:
    return _stack(inst, args, inline)
  if inst.op == Ops.CONCAT:
    return _concat(inst, args, inline)
  if inst.op == Ops.MATMUL:
    return _matmul(inst, args[0], args[1], instructions)
  if inst.op == Ops.CALL:
    return _call(inst, args, call_w_offset)
  if inst.op == Ops.MAP:
    return _map(inst, args, call_w_offset)
  raise NotImplementedError(f"C renderer does not support op {inst.op.value!r}")


def _raw_symbol(fun: Function) -> str:
  return f"{_c_ident(fun.name)}_raw"


def _value_ref(inst: Instruction) -> str:
  return _NAME_MAP.get(inst.index, f"v{inst.index}")


def _read(inst: Instruction, idx: str = "0") -> str:
  return f"{_value_ref(inst)}[{idx}]"


def _loop(size: int, body: Callable[[str], str], *, var: str = "k", limit: int = 32) -> list[str]:
  if size <= limit:
    return ["  " + body(str(i)) for i in range(size)]
  return [f"  for (int {var} = 0; {var} < {size}; ++{var}) " + body(var)]


def _copy_loop(dst: str, src: str, size: int) -> list[str]:
  return _loop(size, lambda k: f"{dst}[{k}] = {src}[{k}];")


def _try_int(value: str) -> int | None:
  try:
    return int(value)
  except ValueError:
    return None


def _coord(flat: str, shape: tuple[int, ...], dim: int) -> str:
  stride = math.prod(shape[dim + 1 :])
  if (n := _try_int(flat)) is not None:
    return str((n // stride) % shape[dim])
  if stride == 1:
    return f"({flat} % {shape[dim]})"
  return f"(({flat} / {stride}) % {shape[dim]})"


def _flat_index(coords: list[str], shape: tuple[int, ...]) -> str:
  if not shape:
    return "0"
  total = 0
  symbolic: list[str] = []
  for i, coord in enumerate(coords):
    stride = math.prod(shape[i + 1 :])
    if (n := _try_int(coord)) is not None:
      total += n * stride
    else:
      symbolic.append(coord if stride == 1 else f"({coord}) * {stride}")
  if not symbolic:
    return str(total)
  if total:
    symbolic.append(str(total))
  return " + ".join(symbolic)


def _broadcast_index(flat: str, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> str:
  if in_shape == out_shape or not out_shape:
    return flat
  if not in_shape:
    return "0"
  offset = len(out_shape) - len(in_shape)
  coords = []
  for i, dim in enumerate(in_shape):
    coords.append("0" if dim == 1 else _coord(flat, out_shape, offset + i))
  return _flat_index(coords, in_shape)


def _read_arg(x: Instruction, idx_expr: str, inline: dict[int, InlineReader]) -> str:
  return _inline_read(x, idx_expr, inline)


def _elementwise_unary(inst: Instruction, x: Instruction, inline: dict[int, InlineReader]) -> list[str]:
  op = {
    Ops.NEG: "-",
    Ops.SIN: "sin",
    Ops.COS: "cos",
    Ops.TAN: "tan",
    Ops.ASIN: "asin",
    Ops.ACOS: "acos",
    Ops.ATAN: "atan",
    Ops.SINH: "sinh",
    Ops.COSH: "cosh",
    Ops.TANH: "tanh",
    Ops.EXP: "exp",
    Ops.LOG: "log",
    Ops.SQRT: "sqrt",
    Ops.ABS: "fabs",
    Ops.FLOOR: "floor",
    Ops.CEIL: "ceil",
  }[inst.op]

  def expr(k: str) -> str:
    idx = _broadcast_index(k, x.shape, inst.shape)
    val = _read_arg(x, idx, inline)
    return f"-{val}" if op == "-" else f"{op}({val})"

  return _loop(inst.size, lambda k: f"{_value_ref(inst)}[{k}] = {expr(k)};")


def _elementwise_binary(inst: Instruction, x: Instruction, y: Instruction, inline: dict[int, InlineReader]) -> list[str]:
  op = {
    Ops.ADD: "+",
    Ops.SUB: "-",
    Ops.MUL: "*",
    Ops.DIV: "/",
    Ops.POW: "pow",
    Ops.ATAN2: "atan2",
    Ops.MINIMUM: "fmin",
    Ops.MAXIMUM: "fmax",
  }[inst.op]

  def expr(k: str) -> str:
    xval = _read_arg(x, _broadcast_index(k, x.shape, inst.shape), inline)
    yval = _read_arg(y, _broadcast_index(k, y.shape, inst.shape), inline)
    if inst.op == Ops.POW and y.op == Ops.CONST and y.value is not None and y.value.size == 1:
      exp = float(y.value.reshape(-1)[0])
      if exp == 2.0:
        return f"({xval} * {xval})"
      if exp == 0.5:
        return f"sqrt({xval})"
      if exp == -1.0:
        return f"(1.0 / {xval})"
    return f"{xval} {op} {yval}" if op in {"+", "-", "*", "/"} else f"{op}({xval}, {yval})"

  return _loop(inst.size, lambda k: f"{_value_ref(inst)}[{k}] = {expr(k)};")


def _transpose(inst: Instruction, x: Instruction) -> list[str]:
  axes = inst.attrs["axes"]

  def src_index(k: str) -> str:
    in_coords = ["0"] * len(x.shape)
    for out_dim, in_dim in enumerate(axes):
      in_coords[in_dim] = _coord(k, inst.shape, out_dim)
    return _flat_index(in_coords, x.shape)

  return _loop(inst.size, lambda k: f"{_value_ref(inst)}[{k}] = {_value_ref(x)}[{src_index(k)}];")


def _slice_offset(inst: Instruction) -> int | None:
  source_shape = inst.attrs.get("source_shape")
  return None if source_shape is None else _contiguous_slice_offset(inst.attrs["index"], source_shape, inst.shape)


def _contiguous_slice_offset(index: tuple[object, ...], in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> int | None:
  if not out_shape:
    offset = 0
    for dim, item in enumerate(index):
      if not isinstance(item, int):
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * math.prod(in_shape[dim + 1 :])
    return offset
  first_slice = None
  offset = 0
  out_dim = 0
  for dim, item in enumerate(index):
    stride = math.prod(in_shape[dim + 1 :])
    if isinstance(item, int):
      if first_slice is not None and in_shape[dim] != 1:
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * stride
      continue
    assert isinstance(item, slice)
    start, stop, step = item.indices(in_shape[dim])
    if step != 1:
      return None
    if first_slice is None:
      first_slice = dim
      offset += start * stride
      out_dim += 1
      continue
    if start != 0 or stop != in_shape[dim] or out_shape[out_dim] != in_shape[dim]:
      return None
    out_dim += 1
  return offset


def _slice(inst: Instruction, x: Instruction) -> list[str]:
  index = inst.attrs["index"]
  if (offset := _contiguous_slice_offset(index, x.shape, inst.shape)) is not None:
    return [f"  {_value_ref(inst)} = {_value_ref(x)} + {offset};"]

  def src_index(k: str) -> str:
    out_dim = 0
    coords = []
    for dim, item in enumerate(index):
      if isinstance(item, int):
        coords.append(str(item if item >= 0 else x.shape[dim] + item))
        continue
      start, _, step = item.indices(x.shape[dim])
      coord = _coord(k, inst.shape, out_dim)
      coords.append(coord if start == 0 and step == 1 else f"({start} + ({coord}) * {step})")
      out_dim += 1
    return _flat_index(coords, x.shape)

  return _loop(inst.size, lambda k: f"{_value_ref(inst)}[{k}] = {_value_ref(x)}[{src_index(k)}];")


def _gather(inst: Instruction, x: Instruction, instructions: tuple[Instruction, ...], inline: dict[int, InlineReader]) -> list[str]:
  if (lines := _gather_tile_pattern(inst, x)) is not None:
    return lines
  if (lines := _gather_transposed_concat(inst, x, instructions, inline)) is not None:
    return lines
  indices = ", ".join(str(int(i)) for i in inst.attrs["indices"].reshape(-1))
  lines = [f"  static const int idx{inst.index}[{inst.size}] = {{{indices}}};"]
  lines += _loop(inst.size, lambda k: f"{_value_ref(inst)}[{k}] = {_value_ref(x)}[idx{inst.index}[{k}]];")
  return lines


_MAX_TILE_SIZE = 256
_MAX_PREFIX = 64
_TILE_CACHE: dict[bytes, tuple[int, int, int, int, tuple[int, ...]] | None] = {}


def _detect_tile(indices: np.ndarray) -> tuple[int, int, int, int, tuple[int, ...]] | None:
  """Detect a trailing tile in flat indices.

  Returns ``(prefix_size, tile_size, length, stride, base)`` such that
  ``indices[prefix_size + i*tile_size + j] == base[j] + i*stride`` for ``i in [0, length)``
  and ``length >= 2``. Returns ``None`` when no useful pattern is found.

  Heuristics: small prefix (capped at ``_MAX_PREFIX``); small tile_size (capped at ``_MAX_TILE_SIZE``);
  iterate ``tile_size`` from small to large to favour tight inner loops.
  """
  flat = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
  cache_key = flat.tobytes()
  if cache_key in _TILE_CACHE:
    return _TILE_CACHE[cache_key]
  n = int(flat.size)
  if n < 4:
    _TILE_CACHE[cache_key] = None
    return None
  prefix_cap = min(n // 2, _MAX_PREFIX)
  for prefix in range(prefix_cap + 1):
    rem = n - prefix
    if rem < 4:
      continue
    max_tile = min(rem // 2, _MAX_TILE_SIZE)
    for tile_size in range(1, max_tile + 1):
      if rem % tile_size != 0:
        continue
      length = rem // tile_size
      tile = flat[prefix : prefix + tile_size]
      stride = int(flat[prefix + tile_size]) - int(flat[prefix])
      arr = flat[prefix:].reshape(length, tile_size)
      expected = tile[None, :] + stride * np.arange(length, dtype=np.int64)[:, None]
      if np.array_equal(arr, expected):
        ret = (prefix, tile_size, length, stride, tuple(int(x) for x in tile))
        _TILE_CACHE[cache_key] = ret
        return ret
  _TILE_CACHE[cache_key] = None
  return None


def _gather_tile_pattern(inst: Instruction, x: Instruction) -> list[str] | None:
  found = _detect_tile(inst.attrs["indices"])
  if found is None:
    return None
  prefix, tile_size, length, stride, base = found
  dst = _value_ref(inst)
  src = _value_ref(x)
  flat = np.asarray(inst.attrs["indices"], dtype=np.int64).reshape(-1)
  lines: list[str] = []
  for k in range(prefix):
    lines.append(f"  {dst}[{k}] = {src}[{int(flat[k])}];")
  if tile_size == 1:
    base0 = base[0]
    if stride == 1 and prefix == 0:
      lines.append(f"  for (int it = 0; it < {length}; ++it) {dst}[it] = {src}[{base0} + it];")
    else:
      lines.append(f"  for (int it = 0; it < {length}; ++it) {dst}[{prefix} + it] = {src}[{base0}{f' + it * {stride}' if stride != 1 else ' + it'}];")
    return lines
  base_array = "{" + ", ".join(str(b) for b in base) + "}"
  lines.append(f"  static const int tile{inst.index}[{tile_size}] = {base_array};")
  lines.append(f"  for (int it = 0; it < {length}; ++it) {{")
  lines.append(f"    for (int j = 0; j < {tile_size}; ++j) {{")
  stride_term = f" + it * {stride}" if stride else ""
  lines.append(f"      {dst}[{prefix} + it * {tile_size} + j] = {src}[tile{inst.index}[j]{stride_term}];")
  lines.append("    }")
  lines.append("  }")
  return lines


def _read_element(inst: Instruction, idx: str, instructions: tuple[Instruction, ...], inline: dict[int, InlineReader]) -> str:
  if inst.index in inline:
    return inline[inst.index]("0" if inst.size == 1 else idx)
  if inst.op == Ops.ADD:
    return f"({_read_element(instructions[inst.inputs[0]], idx, instructions, inline)} + {_read_element(instructions[inst.inputs[1]], idx, instructions, inline)})"
  if inst.op == Ops.SUB:
    return f"({_read_element(instructions[inst.inputs[0]], idx, instructions, inline)} - {_read_element(instructions[inst.inputs[1]], idx, instructions, inline)})"
  return _read(inst, idx)


_PEEPHOLE_INLINE_PER_BLOCK = 32  # per-block group size above which we emit a loop + idx[] table


def _gather_transposed_concat(
  inst: Instruction, x: Instruction, instructions: tuple[Instruction, ...], inline: dict[int, InlineReader]
) -> list[str] | None:
  if x.op != Ops.TRANSPOSE or x.attrs.get("axes") != (1, 0) or len(x.inputs) != 1:
    return None
  inner = instructions[x.inputs[0]]
  # Both CONCAT axis=1 with rank-2 args (shape (m, k_i)) and STACK axis=1 with rank-1 args
  # (shape (m,)) produce inner of shape (m, ncolor). The transpose makes (ncolor, m) and
  # gather flat indices decompose as outer=f//m, m_idx=f%m. Each block is either a column
  # range of width k_i or a single column.
  if inner.op == Ops.CONCAT and inner.attrs.get("axis", 0) == 1:
    blocks = [instructions[i] for i in inner.inputs]
    if not blocks or any(len(b.shape) != 2 or b.shape[0] != inner.shape[0] for b in blocks):
      return None
    widths = [b.shape[1] for b in blocks]
  elif inner.op == Ops.STACK and inner.attrs.get("axis", 0) == 1:
    blocks = [instructions[i] for i in inner.inputs]
    if not blocks or any(len(b.shape) != 1 or b.shape[0] != inner.shape[0] for b in blocks):
      return None
    widths = [1] * len(blocks)
  else:
    return None
  starts = np.cumsum([0, *widths])
  m_dim = inner.shape[0]

  # Group entries by block_idx and compute per-entry block-relative flat index.
  groups: list[tuple[int, list[tuple[int, int]]]] = []  # (block_idx, [(out_idx, block_flat_idx), ...])
  cur_block = -1
  cur_entries: list[tuple[int, int]] = []
  for out_idx, flat in enumerate(inst.attrs["indices"].reshape(-1)):
    outer, m_idx = int(flat) // m_dim, int(flat) % m_dim
    block_idx = int(np.searchsorted(starts, outer, side="right") - 1)
    local_col = outer - int(starts[block_idx])
    block = blocks[block_idx]
    if len(block.shape) == 2:
      block_flat = m_idx * block.shape[1] + local_col
    else:
      assert local_col == 0
      block_flat = m_idx
    if block_idx != cur_block:
      if cur_entries:
        groups.append((cur_block, cur_entries))
      cur_block = block_idx
      cur_entries = []
    cur_entries.append((out_idx, block_flat))
  if cur_entries:
    groups.append((cur_block, cur_entries))

  lines: list[str] = []
  dst = _value_ref(inst)
  for block_idx, entries in groups:
    block = blocks[block_idx]
    if len(entries) <= _PEEPHOLE_INLINE_PER_BLOCK:
      # Small group: emit per-entry inline-aware reads.
      for out_idx, block_flat in entries:
        lines.append(f"  {dst}[{out_idx}] = {_read_element(block, str(block_flat), instructions, inline)};")
      continue
    # Large group: one static idx[] table + a single loop. _read_element with a runtime index
    # expression keeps the inline-through-ADD/SUB chains working, so the body stays compact even
    # when the block is itself a skipped ADD of materialized tensors.
    out_base = entries[0][0]
    idx_table = ", ".join(str(b) for _, b in entries)
    idx_name = f"gidx{inst.index}_b{block_idx}"
    lines.append(f"  static const int {idx_name}[{len(entries)}] = {{{idx_table}}};")
    read_expr = _read_element(block, f"{idx_name}[k]", instructions, inline)
    lines.append(f"  for (int k = 0; k < {len(entries)}; ++k) {dst}[{out_base} + k] = {read_expr};")
  return lines


def _scatter(inst: Instruction, x: Instruction) -> list[str]:
  indices = ", ".join(str(int(i)) for i in inst.attrs["indices"].reshape(-1))
  lines = [f"  static const int idx{inst.index}[{x.size}] = {{{indices}}};"]
  lines += _loop(inst.size, lambda k: f"{_value_ref(inst)}[{k}] = 0.0;")
  lines += _loop(x.size, lambda k: f"{_value_ref(inst)}[idx{inst.index}[{k}]] = {_value_ref(x)}[{k}];")
  return lines


def _stack(inst: Instruction, args: tuple[Instruction, ...], inline: dict[int, InlineReader]) -> list[str]:
  axis = inst.attrs.get("axis", 0)
  lines: list[str] = []
  for i, arg in enumerate(args):

    def out_index(k: str) -> str:
      coords = []
      for dim in range(len(inst.shape)):
        if dim == axis:
          continue
        coords.append(_coord(k, arg.shape, dim if dim < axis else dim - 1))
      out_coords = coords[:axis] + [str(i)] + coords[axis:]
      return _flat_index(out_coords, inst.shape)

    lines += _loop(arg.size, lambda k: f"{_value_ref(inst)}[{out_index(k)}] = {_inline_read(arg, k, inline)};")
  return lines


def _concat(inst: Instruction, args: tuple[Instruction, ...], inline: dict[int, InlineReader]) -> list[str]:
  axis = inst.attrs.get("axis", 0)
  start = 0
  lines: list[str] = []
  for arg in args:

    def out_index(k: str) -> str:
      coords = [_coord(k, arg.shape, dim) for dim in range(len(arg.shape))]
      out_coords = list(coords)
      out_coords[axis] = coords[axis] if start == 0 else f"({coords[axis]} + {start})"
      return _flat_index(out_coords, inst.shape)

    lines += _loop(arg.size, lambda k: f"{_value_ref(inst)}[{out_index(k)}] = {_inline_read(arg, k, inline)};")
    start += arg.shape[axis]
  return lines


def _call(inst: Instruction, args: tuple[Instruction, ...], call_w_offset: int) -> list[str]:
  callee = inst.attrs["callee"]
  output = inst.attrs["output"]
  prefix = f"call{inst.index}"
  res_refs: list[str] = []
  lines: list[str] = []
  for i, out in enumerate(callee.outputs):
    if i == output:
      res_refs.append(_value_ref(inst))
    else:
      lines.append(f"  double {prefix}_unused_res{i}[{out.size}];")
      res_refs.append(f"{prefix}_unused_res{i}")
  workspace = "NULL" if _workspace_size(callee) == 0 else f"w + {call_w_offset}"
  call_args = [*(_value_ref(arg) for arg in args), *res_refs, workspace]
  lines.append(f"  {_raw_symbol(callee)}({', '.join(call_args)});")
  return lines


def _ptr_with_offset(base: str, start: int, stride: int, var: str = "it") -> str:
  parts: list[str] = []
  if start:
    parts.append(str(start))
  if stride:
    parts.append(var if stride == 1 else f"{var} * {stride}")
  return base if not parts else f"{base} + {' + '.join(parts)}"


def _map(inst: Instruction, args: tuple[Instruction, ...], call_w_offset: int) -> list[str]:
  callee = inst.attrs["callee"]
  output = inst.attrs["output"]
  length = inst.attrs["length"]
  starts = inst.attrs["starts"]
  strides = inst.attrs["strides"]
  slice_size = inst.attrs["slice_size"]
  prefix = f"map{inst.index}"

  lines: list[str] = []
  unused_refs: dict[int, str] = {}
  for i, out in enumerate(callee.outputs):
    if i == output:
      continue
    name = f"{prefix}_unused_res{i}"
    lines.append(f"  double {name}[{out.size}];")
    unused_refs[i] = name

  if length == 0:
    return lines

  workspace = "NULL" if _workspace_size(callee) == 0 else f"w + {call_w_offset}"
  out_ref = _value_ref(inst)

  in_refs = [_ptr_with_offset(_value_ref(arg), start, stride) for arg, start, stride in zip(args, starts, strides, strict=True)]
  result_refs = [_ptr_with_offset(out_ref, 0, slice_size) if i == output else unused_refs[i] for i in range(len(callee.outputs))]
  call_args = [*in_refs, *result_refs, workspace]
  raw = _raw_symbol(callee)
  lines.append(f"  for (int it = 0; it < {length}; ++it) {{")
  lines.append(f"    {raw}({', '.join(call_args)});")
  lines.append("  }")
  return lines


def _sparse_const(value, limit: int) -> list[tuple[int, float]] | None:
  flat = value.reshape(-1)
  nz = np.nonzero(flat)[0]
  return [(int(i), float(flat[i])) for i in nz] if nz.size <= limit else None


def _sparse_term(value: float, ref: str, *, leading: bool) -> str:
  if value == 1.0:
    return ref if leading else f"+ {ref}"
  if value == -1.0:
    return f"-{ref}" if leading else f"- {ref}"
  return f"{_c_float(value)} * {ref}" if leading or value < 0 else f"+ {_c_float(value)} * {ref}"


def _sparse_terms(nz: list[tuple[int, float]], ref_for: Callable[[int], str]) -> str:
  if not nz:
    return "0.0"
  parts = [_sparse_term(v, ref_for(col), leading=i == 0) for i, (col, v) in enumerate(nz)]
  return " ".join(parts)


def _matmul(inst: Instruction, x: Instruction, y: Instruction, instructions: tuple[Instruction, ...]) -> list[str]:
  dst, xv, yv = _value_ref(inst), _value_ref(x), _value_ref(y)
  if len(x.shape) == 1 and len(y.shape) == 1:
    return [f"  {dst}[0] = 0.0;", f"  for (int k = 0; k < {x.shape[0]}; ++k) {dst}[0] += {xv}[k] * {yv}[k];"]
  if len(x.shape) == 2 and len(y.shape) == 1:
    rows, inner = x.shape
    if y.op == Ops.CONST and y.value is not None and (nz := _sparse_const(y.value, limit=4)) is not None:
      terms = _sparse_terms(nz, lambda col: f"{xv}[i * {inner} + {col}]")
      return [f"  for (int i = 0; i < {rows}; ++i) {dst}[i] = {terms};"]
    return [
      f"  for (int i = 0; i < {rows}; ++i) {{",
      f"    {dst}[i] = 0.0;",
      f"    for (int k = 0; k < {inner}; ++k) {dst}[i] += {xv}[i * {inner} + k] * {yv}[k];",
      "  }",
    ]
  if len(x.shape) == 1 and len(y.shape) == 2:
    inner, cols = y.shape
    if x.op == Ops.CONST and x.value is not None and (nz := _sparse_const(x.value, limit=4)) is not None:
      terms = _sparse_terms(nz, lambda row: f"{yv}[{row} * {cols} + j]")
      return [f"  for (int j = 0; j < {cols}; ++j) {dst}[j] = {terms};"]
    return [
      f"  for (int j = 0; j < {cols}; ++j) {{",
      f"    {dst}[j] = 0.0;",
      f"    for (int k = 0; k < {inner}; ++k) {dst}[j] += {xv}[k] * {yv}[k * {cols} + j];",
      "  }",
    ]
  if len(x.shape) == 2 and len(y.shape) == 2:
    rows, inner = x.shape
    cols = y.shape[1]
    if y.op == Ops.TRANSPOSE and y.attrs.get("axes") == (1, 0):
      yt = instructions[y.inputs[0]]
      ytv = _value_ref(yt)
      return [
        f"  for (int i = 0; i < {rows}; ++i) {{",
        f"    for (int j = 0; j < {cols}; ++j) {{",
        f"      {dst}[i * {cols} + j] = 0.0;",
        f"      for (int k = 0; k < {inner}; ++k) {dst}[i * {cols} + j] += {xv}[i * {inner} + k] * {ytv}[j * {inner} + k];",
        "    }",
        "  }",
      ]
    return [
      f"  for (int i = 0; i < {rows}; ++i) {{",
      f"    for (int j = 0; j < {cols}; ++j) {{",
      f"      {dst}[i * {cols} + j] = 0.0;",
      f"      for (int k = 0; k < {inner}; ++k) {dst}[i * {cols} + j] += {xv}[i * {inner} + k] * {yv}[k * {cols} + j];",
      "    }",
      "  }",
    ]
  raise NotImplementedError(f"C renderer does not support matmul shapes {x.shape} @ {y.shape}")
