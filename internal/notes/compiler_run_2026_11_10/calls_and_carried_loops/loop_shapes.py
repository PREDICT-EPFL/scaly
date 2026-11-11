"""Carried ranges in the shapes that the lowering of LOOP may emit, through every pass, against NumPy.

Three hand-built programs: a scan (call the step, copy the carry back), a while loop (break, call, copy, exit
variable) and a loop whose body is one call. Each runs through ``optimize_program`` under 108 settings, and the
compiled C is compared with a NumPy loop. Run from a checkout with
``uv run internal/notes/compiler_run_2026_11_10/calls_and_carried_loops/loop_shapes.py``.
"""

import ctypes
import itertools
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scaly.codegen.c import _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.program_spec import verify_program
from scaly.ir.types import dtypes
from scaly.passes.program import optimize_program

f64 = dtypes.float64


def attrs(node, **kw):
  return ProgramNode(node.op, node.args, {**node.attrs, **kw}, node.dtype)


def at(buf, i=0):
  return p.view(buf, [p.const_int(i) if isinstance(i, int) else i])


def call(name, ins, outs):
  return ProgramNode(ProgramOp.CALL, (*ins, *outs), {"callee": name, "n_in": len(ins), "n_out": len(outs), "returns": ()})


def build(shape, mode, hint, length):
  n = 2
  x, state, weights, nxt = p.buffer("x", f64, (1,)), p.buffer("state", f64, (n,)), p.buffer("weights", f64, (n,)), p.buffer("next", f64, (n,))
  tmp = p.buffer("tmp", f64, (n,), address_space="private")
  j = p.var("j")
  step = p.proc(
    "step",
    [x, state, weights, nxt],
    [
      tmp,
      p.for_(p.range_("j", 0, n), [p.store(p.view(tmp, [j]), p.mul(p.load(p.view(weights, [j])), p.load(p.view(weights, [j]))))]),
      p.for_(
        p.range_("j", 0, n),
        [p.store(p.view(nxt, [j]), p.add(p.div(p.load(p.view(state, [j])), p.const_float(2)), p.mul(p.load(p.view(tmp, [j])), p.load(at(x)))))],
      ),
    ],
  )
  step = attrs(step, input_count=3, scalarize_mode=mode, lowering=hint)
  xs, flags, out, count = (
    p.buffer("xs", f64, (max(1, length),)),
    p.buffer("flags", dtypes.bool_, (max(1, length),)),
    p.buffer("out", f64, (n,)),
    p.buffer("count", dtypes.int64, (1,)),
  )
  scratch = p.buffer("scratch", f64, (n,), address_space="private")
  k = p.var("k")
  init = p.for_(p.range_("j", 0, n), [p.store(p.view(out, [j]), p.load(p.view(weights, [j])))])
  copy = p.for_(p.range_("j2", 0, n), [p.store(p.view(out, [p.var("j2")]), p.load(p.view(scratch, [p.var("j2")])))])
  brk = p.break_if(p.load(p.view(flags, [k])))
  if shape == "scan":  # carried, no exit: call then copy the carry back
    loop = p.for_(p.range_("k", 0, length, carried=True), [call("step", [p.view(xs, [k]), out, weights], [scratch]), copy])
    tail = [p.store(at(count), p.const_int(length))]
  elif shape == "while":  # exit variable and break, call then copy
    loop = p.for_(p.range_("k", 0, length, carried=True), [brk, call("step", [p.view(xs, [k]), out, weights], [scratch]), copy], exit_var=True)
    tail = [p.store(at(count), k)]
  elif shape == "single_call":  # the body is one call that reads slot k of a history buffer and writes slot k + 1
    hist = p.buffer("hist", f64, (n * (length + 1),), address_space="private")
    init = p.for_(p.range_("j", 0, n), [p.store(p.view(hist, [j]), p.load(p.view(weights, [j])))])
    loop = p.for_(
      p.range_("k", 0, length, carried=True),
      [
        call(
          "step",
          [p.view(xs, [k]), p.view(hist, [p.mul(k, p.const_int(n))]), weights],
          [p.view(hist, [p.add(p.mul(k, p.const_int(n)), p.const_int(n))])],
        )
      ],
      exit_var=True,
    )
    copy_out = p.for_(
      p.range_("j2", 0, n), [p.store(p.view(out, [p.var("j2")]), p.load(p.view(hist, [p.add(p.mul(k, p.const_int(n)), p.var("j2"))])))]
    )
    root = p.proc("root", [xs, weights, flags, out, count], [hist, init, loop, copy_out, p.store(at(count), k)])
    return p.program([step, attrs(root, input_count=3, scalarize_mode=mode, lowering=hint)])
  root = p.proc("root", [xs, weights, flags, out, count], [scratch, init, loop, *tail])
  return p.program([step, attrs(root, input_count=3, scalarize_mode=mode, lowering=hint)])


def reference(shape, xs, weights, flags, length):
  state, n = weights.copy(), 0
  for k in range(length):
    if shape == "while" and flags[k]:
      break
    state = state / 2 + weights * weights * xs[k]
    n += 1
  return state, n


def run(prog, tmp, tag, xs, weights, flags):
  lines = [*_includes(dialect="gnu", prog=prog)]
  for pr in prog.args:
    lines += _render_raw_callee(pr, dialect="gnu")
  src = (
    "\n".join(lines)
    + "\nvoid run(double *xs, double *weights, uint8_t *flags, double *out, int64_t *count) { double w[4096]; root_raw(xs, weights, flags, out, count, w); }\n"
  )
  c, so = Path(tmp, f"{tag}.c"), Path(tmp, f"{tag}.so")
  c.write_text(src)
  r = subprocess.run(
    ["cc", "-shared", "-fPIC", "-O2", "-Werror=incompatible-pointer-types", str(c), "-lm", "-o", str(so)], capture_output=True, text=True
  )
  if r.returncode:
    return None, r.stderr[:600], src
  fn = ctypes.CDLL(str(so)).run
  out, count = np.full(2, -1.0), np.array([-1], dtype=np.int64)
  dp = ctypes.POINTER(ctypes.c_double)
  fn.argtypes = [dp, dp, ctypes.POINTER(ctypes.c_uint8), dp, ctypes.POINTER(ctypes.c_int64)]
  fn(
    xs.ctypes.data_as(dp),
    weights.ctypes.data_as(dp),
    flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    out.ctypes.data_as(dp),
    count.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
  )
  return (out, int(count[0])), None, src


bad = total = 0
shown = set()
with tempfile.TemporaryDirectory() as tmp:
  for i, (shape, mode, hint, length, lanes, recip) in enumerate(
    itertools.product(
      ["scan", "while", "single_call"], ["procedure", "inline", "disabled"], ["auto", "scalar"], [0, 1, 5], [1, 2, "auto"], [False, True]
    )
  ):
    total += 1
    xs, weights = np.linspace(0.3, 1.7, max(1, length)), np.array([0.7, -1.3])
    flags = (np.arange(max(1, length)) >= 3).astype(np.uint8)
    prog = build(shape, mode, hint, length)
    stages = []
    try:
      verify_program(prog)
      opt = optimize_program(prog, observe=lambda name, st: (verify_program(st), stages.append(name)), lanes=lanes, reciprocal=recip)
    except Exception as e:
      bad += 1
      print("PASS FAILURE", shape, mode, hint, length, lanes, recip, "after", stages[-1:], type(e).__name__, str(e)[:300])
      continue
    got, err, src = run(opt, tmp, f"p{i}", xs, weights, flags)
    exp = reference(shape, xs, weights, flags, length)
    if shape == "scan":
      exp = (exp[0], length)
    ok = err is None and np.allclose(got[0], exp[0], rtol=1e-13, atol=0) and got[1] == exp[1]
    carried = [n for n in walk_program(opt) if n.op == ProgramOp.RANGE and n.attrs.get("carried")]
    key = (shape, len(carried), any("_hoist_" in pr.attrs["name"] for pr in opt.args), any(pr.attrs.get("scalarized") for pr in opt.args))
    if not ok:
      bad += 1
      print("MISMATCH", shape, mode, hint, length, lanes, recip, got, exp, err)
      print(src[-1500:])
    elif key not in shown and length == 5:
      shown.add(key)
      print("ok", shape, mode, hint, "lanes", lanes, "carried ranges kept:", key[1], "hoisted prologue:", key[2], "scalarized procs:", key[3])
print(f"{total} configurations, {bad} bad")
