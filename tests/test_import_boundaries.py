from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
from typing import Sequence, get_type_hints

import scaly as sc
import scaly.codegen as codegen
from scaly.codegen import aot
from scaly.function.factory import Adj, DerivSpec, Fwd, Grad, Hess, Jac, SpHess, SpJac
from scaly.function.model import ConcreteFunction, Function, NotConcrete
from scaly.function.tree import G, L, Tree
from scaly.ir.expr import Expr, ExprOp
from scaly.ir.expr_spec import spec_expr
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.program_spec import spec_program_full
from scaly.ir.spec import Rule, Spec, VerifyError
from scaly.solvers.problem import NO_LB, NO_UB, Bounded, Problem, ProblemSpec
from scaly.solvers.qp import NotQuadratic, QPData, qp_problem


def test_public_exports_are_canonical() -> None:
  program = __import__("scaly.ir.program", fromlist=["ProgramNode"])
  assert sc.Expr is Expr
  assert sc.ExprOp is ExprOp
  assert sc.Function is Function
  assert sc.ConcreteFunction is ConcreteFunction and issubclass(ConcreteFunction, Function)
  assert sc.NotConcrete is NotConcrete and issubclass(NotConcrete, TypeError)
  assert sc.L is L
  assert sc.G is G
  assert sc.Bounded is Bounded
  assert sc.Problem is Problem
  assert sc.ProblemSpec is ProblemSpec
  assert sc.NotQuadratic is NotQuadratic
  assert sc.NO_LB is NO_LB
  assert sc.NO_UB is NO_UB
  assert sc.QPData is QPData
  assert sc.qp_problem is qp_problem
  assert callable(sc.bounded) and callable(sc.problem) and callable(sc.solver)
  assert {"Bounded", "NO_LB", "NO_UB", "NotQuadratic", "Problem", "ProblemSpec", "QPData", "bounded", "problem", "qp_problem", "solver"} <= set(
    sc.__all__
  )
  assert not hasattr(sc, "nlp")
  assert not hasattr(sc, "qp")
  assert not hasattr(sc, "SolverFunction")
  assert not hasattr(sc, "Tree")
  assert not hasattr(sc, "Buffer")
  assert {"L", "G"} <= set(sc.__all__)
  function_module = __import__("scaly.function", fromlist=["Tree"])
  assert function_module.Tree is Tree
  # The call surface is the two named tree methods plus the dispatching __call__; the flat leaf
  # seams under them are private and must not reappear as public names.
  assert not hasattr(Function, "call")
  assert not hasattr(Function, "eval_list")
  assert all(callable(getattr(Function, name)) for name in ("__call__", "symbolic_call", "numerical_call"))
  assert all(callable(getattr(ConcreteFunction, name)) for name in ("_flat_symbolic_call", "_flat_numerical_call"))
  factory_hints = get_type_hints(ConcreteFunction.factory)
  assert factory_hints["outputs"] == Sequence[str | DerivSpec]
  assert factory_hints["return"] is ConcreteFunction
  assert not hasattr(sc, "Port")
  assert sc.factory.DerivSpec is DerivSpec
  assert sc.factory.Jac is Jac
  assert sc.factory.Grad is Grad
  assert sc.factory.Hess is Hess
  assert sc.factory.SpJac is SpJac
  assert sc.factory.SpHess is SpHess
  assert sc.factory.Fwd is Fwd
  assert sc.factory.Adj is Adj
  assert sc.factory.__all__ == ["Jac", "Grad", "Hess", "SpJac", "SpHess", "Fwd", "Adj", "DerivSpec"]
  assert sc.Rule is Rule
  assert sc.Spec is Spec
  assert sc.VerifyError is VerifyError
  assert program.ProgramNode is ProgramNode
  assert program.ProgramOp is ProgramOp
  assert codegen.render_c_source is aot.render_c_source
  assert callable(sc.vmap)
  assert callable(sc.scan)
  assert not hasattr(sc, "map_")
  assert all(
    not hasattr(sc, name)
    for name in ("DerivSpec", "expr_jacobian", "expr_gradient", "expr_hessian", "jac", "grad", "hess", "spjac", "sphess", "spjacobian", "sphessian")
  )
  assert {"ExprOp", "Rule", "Spec", "VerifyError"} <= set(sc.__all__)
  assert {"vmap", "scan"} <= set(sc.__all__)
  assert "map_" not in sc.__all__
  assert {"Ops", "VerifyRule", "spec_semantic", "spec_semantic_shared"}.isdisjoint(sc.__all__)


def test_the_integrator_surface() -> None:
  integrators = importlib.import_module("scaly.integrators")
  assert sc.integrators is integrators
  assert integrators.__all__ == [
    "Collocation",
    "FAMILIES",
    "Interval",
    "MultipleShooting",
    "Pseudospectral",
    "TABLEAUS",
    "Tableau",
    "Transcription",
    "UNROLL_STEPS",
    "adaptive",
    "explicit",
    "foh",
    "gauss_legendre",
    "implicit",
    "linearize",
    "lobatto_iiia",
    "lobatto_iiic",
    "order_conditions",
    "radau_iia",
    "rk4",
    "symplectic",
    "tableau",
    "zoh",
  ]
  assert "integrators" not in sc.__all__ and not hasattr(sc, "rk4")


def test_the_mpc_surface() -> None:
  mpc = importlib.import_module("scaly.mpc")
  assert sc.mpc is mpc
  assert mpc.__all__ == ["MPC", "OCP", "ClosedLoop", "Path", "Quadratic", "Solution", "TerminalEquality", "simulate"]
  assert "mpc" not in sc.__all__ and not hasattr(sc, "OCP")


def test_both_dialects_use_the_shared_spec_types() -> None:
  assert isinstance(spec_expr, Spec)
  assert isinstance(spec_program_full, Spec)
  assert all(
    type(rule) is Rule for spec in (spec_expr, spec_program_full) for rule in (*spec.any, *(r for rules in spec.by_op.values() for r in rules))
  )
  assert __import__("scaly.ir.expr_spec", fromlist=["VerifyError"]).VerifyError is VerifyError
  assert __import__("scaly.ir.program_spec", fromlist=["VerifyError"]).VerifyError is VerifyError


def test_obsolete_module_paths_and_vocabulary_are_absent() -> None:
  obsolete_modules = (
    "scaly.abi",
    "scaly.api",
    "scaly.assembly",
    "scaly.codegen.program_c",
    "scaly.codegen.solver_c",
    "scaly.expr",
    "scaly.jit",
    "scaly.lowering",
    "scaly.ops",
    "scaly.program",
    "scaly.rewrite",
    "scaly.sparse",
    "scaly.sparsity",
    "scaly.spec",
    "scaly.toolchain",
    "scaly.types",
    "scaly.viz._recording",
  )
  assert all(importlib.util.find_spec(module) is None for module in obsolete_modules)
  assert not hasattr(sc, "Ops")
  assert not hasattr(sc, "VerifyRule")
  program = __import__("scaly.ir.program", fromlist=["ProgramNode"])
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
