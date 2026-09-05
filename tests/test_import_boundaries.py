from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Sequence, get_type_hints

import alloy as al
import alloy.codegen as codegen
from alloy.codegen import aot
from alloy.function.factory import Adj, DerivSpec, Fwd, Grad, Hess, Jac, SpHess, SpJac
from alloy.function.model import Function
from alloy.function.tree import G, L, Tree
from alloy.ir.expr import Expr, ExprOp
from alloy.ir.expr_spec import spec_expr
from alloy.ir.program import ProgramNode, ProgramOp
from alloy.ir.program_spec import spec_program_full
from alloy.ir.spec import Rule, Spec, VerifyError
from alloy.solvers.problem import NO_LB, NO_UB, Bounded, Problem, ProblemSpec
from alloy.solvers.qp import NotQuadratic, QPData, qp_problem


def test_public_exports_are_canonical() -> None:
  program = __import__("alloy.ir.program", fromlist=["ProgramNode"])
  assert al.Expr is Expr
  assert al.ExprOp is ExprOp
  assert al.Function is Function
  assert al.L is L
  assert al.G is G
  assert al.Bounded is Bounded
  assert al.Problem is Problem
  assert al.ProblemSpec is ProblemSpec
  assert al.NotQuadratic is NotQuadratic
  assert al.NO_LB is NO_LB
  assert al.NO_UB is NO_UB
  assert al.QPData is QPData
  assert al.qp_problem is qp_problem
  assert callable(al.bounded) and callable(al.problem) and callable(al.solver)
  assert {"Bounded", "NO_LB", "NO_UB", "NotQuadratic", "Problem", "ProblemSpec", "QPData", "bounded", "problem", "qp_problem", "solver"} <= set(
    al.__all__
  )
  assert not hasattr(al, "nlp")
  assert not hasattr(al, "qp")
  assert not hasattr(al, "SolverFunction")
  assert not hasattr(al, "Tree")
  assert not hasattr(al, "Buffer")
  assert {"L", "G"} <= set(al.__all__)
  function_module = __import__("alloy.function", fromlist=["Tree"])
  assert function_module.Tree is Tree
  # The call surface is the two named tree methods plus the dispatching __call__; the flat leaf
  # seams under them are private and must not reappear as public names.
  assert not hasattr(Function, "call")
  assert not hasattr(Function, "eval_list")
  assert all(callable(getattr(Function, name)) for name in ("__call__", "symbolic_call", "numerical_call"))
  assert all(callable(getattr(Function, name)) for name in ("_flat_symbolic_call", "_flat_numerical_call"))
  factory_hints = get_type_hints(Function.factory)
  assert factory_hints["outputs"] == Sequence[str | DerivSpec]
  assert factory_hints["return"] is Function
  assert not hasattr(al, "Port")
  assert al.factory.DerivSpec is DerivSpec
  assert al.factory.Jac is Jac
  assert al.factory.Grad is Grad
  assert al.factory.Hess is Hess
  assert al.factory.SpJac is SpJac
  assert al.factory.SpHess is SpHess
  assert al.factory.Fwd is Fwd
  assert al.factory.Adj is Adj
  assert al.factory.__all__ == ["Jac", "Grad", "Hess", "SpJac", "SpHess", "Fwd", "Adj", "DerivSpec"]
  assert al.Rule is Rule
  assert al.Spec is Spec
  assert al.VerifyError is VerifyError
  assert program.ProgramNode is ProgramNode
  assert program.ProgramOp is ProgramOp
  assert codegen.render_c_source is aot.render_c_source
  assert callable(al.vmap)
  assert not hasattr(al, "map_")
  assert not hasattr(al, "scan")
  assert all(
    not hasattr(al, name)
    for name in ("DerivSpec", "expr_jacobian", "expr_gradient", "expr_hessian", "jac", "grad", "hess", "spjac", "sphess", "spjacobian", "sphessian")
  )
  assert {"ExprOp", "Rule", "Spec", "VerifyError"} <= set(al.__all__)
  assert "vmap" in al.__all__
  assert {"map_", "scan"}.isdisjoint(al.__all__)
  assert {"Ops", "VerifyRule", "spec_semantic", "spec_semantic_shared"}.isdisjoint(al.__all__)


def test_both_dialects_use_the_shared_spec_types() -> None:
  assert isinstance(spec_expr, Spec)
  assert isinstance(spec_program_full, Spec)
  assert all(
    type(rule) is Rule for spec in (spec_expr, spec_program_full) for rule in (*spec.any, *(r for rules in spec.by_op.values() for r in rules))
  )
  assert __import__("alloy.ir.expr_spec", fromlist=["VerifyError"]).VerifyError is VerifyError
  assert __import__("alloy.ir.program_spec", fromlist=["VerifyError"]).VerifyError is VerifyError


def test_obsolete_module_paths_and_vocabulary_are_absent() -> None:
  obsolete_modules = (
    "alloy.abi",
    "alloy.api",
    "alloy.assembly",
    "alloy.codegen.program_c",
    "alloy.codegen.solver_c",
    "alloy.expr",
    "alloy.jit",
    "alloy.lowering",
    "alloy.ops",
    "alloy.program",
    "alloy.rewrite",
    "alloy.sparse",
    "alloy.sparsity",
    "alloy.spec",
    "alloy.toolchain",
    "alloy.types",
    "alloy.viz._recording",
  )
  assert all(importlib.util.find_spec(module) is None for module in obsolete_modules)
  assert not hasattr(al, "Ops")
  assert not hasattr(al, "VerifyRule")
  program = __import__("alloy.ir.program", fromlist=["ProgramNode"])
  assert not hasattr(program, "PNode")
  assert not hasattr(program, "POps")


def test_tests_do_not_import_benchmark_problems() -> None:
  violations = []
  root = Path(__file__).resolve().parent
  for path in root.rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
      if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
      elif isinstance(node, ast.ImportFrom):
        names = [node.module or "", *(f"{node.module}.{alias.name}" for alias in node.names)]
      else:
        continue
      if any(name == "benchmarks.problems" or name.startswith("benchmarks.problems.") for name in names):
        violations.append(f"{path.relative_to(root)}:{node.lineno}")
  assert not violations, "Benchmark problem imports belong in problem checks: " + ", ".join(violations)
