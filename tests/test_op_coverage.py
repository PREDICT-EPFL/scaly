"""Pin expression operation coverage and keep generic passes independent of elementwise members."""

from __future__ import annotations

import ast
from collections import Counter
import inspect
from pathlib import Path

from scaly.ad import forward, reverse, sparsity
from scaly.ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, ExprOp
from scaly.ir.expr_spec import spec_expr
from scaly.passes.lowering.ctx import _RULES

ROOT = Path(__file__).resolve().parents[1] / "src" / "scaly"
ELEMENTWISE = COMMON_ELEMENTWISE_UNARY | COMMON_ELEMENTWISE_BINARY

# #20 replaces the existing AD formulas with shared partials. Until then, pin their references
# rather than exempting the entire modules. Arithmetic identities and op-mapping tables remain.
EXISTING_ELEMENTWISE_REFERENCES = {
  (
    "ad/forward.py",
    "_jvp",
  ): "NEG ADD SUB MUL DIV POW SIN COS TAN ASIN ACOS ATAN ATAN2 SINH COSH TANH ERF EXP LOG SQRT ABS FLOOR CEIL MINIMUM MAXIMUM",
  ("ad/forward.py", "_jvp_many_structural"): "ADD SUB NEG MUL DIV POW SIN COS TAN EXP LOG SQRT TANH ERF COSH SINH",
  (
    "ad/reverse.py",
    "_local_vjp",
  ): "NEG ADD SUB MUL DIV POW SIN COS TAN ASIN ACOS ATAN ATAN2 SINH COSH TANH ERF EXP LOG SQRT ABS FLOOR CEIL MINIMUM MAXIMUM",
  ("passes/expr.py", "_structural_key"): "ADD MUL",
  ("passes/expr.py", "SIMPLIFY_PATTERNS"): "ADD SUB MUL DIV NEG POW",
  ("passes/lowering/elementwise.py", "_UNARY"): "NEG SIN COS TAN ASIN ACOS ATAN SINH COSH TANH ERF EXP LOG SQRT ABS FLOOR CEIL",
  ("passes/lowering/elementwise.py", "_BINARY"): "ADD SUB MUL DIV POW ATAN2 MINIMUM MAXIMUM",
}


def _dispatch_ops(*functions) -> set[ExprOp]:
  ops = set()
  families = {"COMMON_ELEMENTWISE_UNARY": COMMON_ELEMENTWISE_UNARY, "COMMON_ELEMENTWISE_BINARY": COMMON_ELEMENTWISE_BINARY}
  for fn in functions:
    for node in ast.walk(ast.parse(inspect.getsource(fn))):
      if not isinstance(node, ast.If):
        continue
      for condition in ast.walk(node.test):
        if isinstance(condition, ast.Attribute) and isinstance(condition.value, ast.Name) and condition.value.id == "ExprOp":
          ops.add(ExprOp[condition.attr])
        elif isinstance(condition, ast.Name) and condition.id in families:
          ops.update(families[condition.id])
  return ops


def _elementwise_references() -> dict[tuple[str, str], Counter[str]]:
  references = {}
  paths = [ROOT / "ir" / "expr_spec.py", *(ROOT / "ad").rglob("*.py"), *(ROOT / "passes").rglob("*.py")]
  for path in paths:
    for statement in ast.parse(path.read_text()).body:
      if isinstance(statement, ast.FunctionDef | ast.ClassDef):
        scope = statement.name
      elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        scope = statement.target.id
      elif isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
        scope = statement.targets[0].id
      else:
        scope = "<module>"
      members = Counter(
        node.attr
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ExprOp" and ExprOp[node.attr] in ELEMENTWISE
      )
      if members:
        key = (path.relative_to(ROOT).as_posix(), scope)
        references.setdefault(key, Counter()).update(members)
  return references


def test_every_expr_op_is_classified_in_every_pass() -> None:
  classifications = {
    "verifier": set(spec_expr.by_op),
    "forward AD": _dispatch_ops(forward._jvp),
    "reverse AD": _dispatch_ops(reverse.vjp, reverse._local_vjp),
    "sparsity": _dispatch_ops(sparsity._jac_mask_uncached),
    # INPUT is bound by emit_inputs; SOLVER_CALL is opaque and handled by the solver wrapper.
    "lowering": set(_RULES) | {ExprOp.INPUT, ExprOp.SOLVER_CALL},
    # Add support rules here in #40, and graph_digest here in #64. Each must cover set(ExprOp).
  }
  for name, classified in classifications.items():
    assert classified == set(ExprOp), f"{name}: missing {set(ExprOp) - classified}, extra {classified - set(ExprOp)}"
  expected = {key: Counter(names.split()) for key, names in EXISTING_ELEMENTWISE_REFERENCES.items()}
  assert _elementwise_references() == expected, "individual elementwise references changed outside the pinned formulas, identities and tables"
