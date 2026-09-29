from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from pathlib import Path
from typing import Sequence, get_type_hints

import pytest

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
from scaly.opt.problem import NLP, NO_LB, NO_UB, Bounded, ProblemSpec
from scaly.opt.qp import QP, NotQuadratic, QPData


def test_public_exports_are_canonical() -> None:
  program = __import__("scaly.ir.program", fromlist=["ProgramNode"])
  assert sc.Expr is Expr
  assert sc.ExprOp is ExprOp
  assert sc.Function is Function
  assert sc.ConcreteFunction is ConcreteFunction and issubclass(ConcreteFunction, Function)
  assert sc.NotConcrete is NotConcrete and issubclass(NotConcrete, TypeError)
  assert sc.L is L
  assert sc.G is G
  assert sc.opt.Bounded is Bounded
  assert sc.opt.NLP is NLP
  assert sc.opt.ProblemSpec is ProblemSpec
  assert sc.opt.NotQuadratic is NotQuadratic
  assert sc.opt.NO_LB is NO_LB
  assert sc.opt.NO_UB is NO_UB
  assert sc.opt.QPData is QPData
  assert sc.opt.QP is QP and issubclass(QP, NLP)
  assert callable(sc.opt.bounded) and callable(sc.opt.problem) and callable(sc.opt.solver)
  assert {"Bounded", "NLP", "NO_LB", "NO_UB", "NotQuadratic", "ProblemSpec", "QP", "QPData", "bounded", "problem", "solver"} <= set(sc.opt.__all__)
  # Optimization lives in ``sc.opt`` alone.
  assert all(not hasattr(sc, name) for name in ("problem", "solver", "qp_problem", "Problem", "ProblemSpec", "bounded", "solver_stats"))
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


def test_both_dialects_use_the_shared_spec_types() -> None:
  assert isinstance(spec_expr, Spec)
  assert isinstance(spec_program_full, Spec)
  assert all(
    type(rule) is Rule for spec in (spec_expr, spec_program_full) for rule in (*spec.any, *(r for rules in spec.by_op.values() for r in rules))
  )
  assert all(type(rule) is Rule for rules in spec_expr.op_rules().values() for rule in rules)  # the per-op rules live on each OpDef
  assert __import__("scaly.ir.expr_spec", fromlist=["VerifyError"]).VerifyError is VerifyError
  assert __import__("scaly.ir.program_spec", fromlist=["VerifyError"]).VerifyError is VerifyError


def test_obsolete_module_paths_and_vocabulary_are_absent() -> None:
  obsolete_modules = (
    "scaly.abi",
    "scaly.api",
    "scaly.assembly",
    "scaly.codegen.program_c",
    "scaly.codegen.solver",
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
  assert not hasattr(sc.ExprOp, "SOLVER_CALL")
  assert not hasattr(sc, "OP_INFO")  # the op registry replaced the table: scaly.ir.expr.op_def
  assert not hasattr(__import__("scaly.ir.expr", fromlist=["OpDef"]), "OpInfo")
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
      if any(name == "bench.problems" or name.startswith("bench.problems.") for name in names):
        violations.append(f"{path.relative_to(root)}:{node.lineno}")
  assert not violations, "Benchmark problem imports belong in problem checks: " + ", ".join(violations)


def test_library_authors_use_the_public_function_api() -> None:
  """``from_exprs``, ``lift``, ``tokens`` and ``load_library`` are public (``scaly.ext``); the private
  spellings they replaced stay gone from every package, test, example and benchmark."""
  root = Path(__file__).resolve().parents[1]
  private = re.compile(r"\b(_from_exprs|_lift|_tokens|_load_library)\b")
  found = []
  for folder in ("src", "tests", "plugins", "examples", "bench"):
    for path in (root / folder).rglob("*"):
      if path.suffix not in {".py", ".ipynb", ".md"} or any(part in {"third_party", ".ipynb_checkpoints", "results"} for part in path.parts):
        continue
      if path == Path(__file__).resolve():
        continue
      for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        if private.search(line):
          found.append(f"{path.relative_to(root)}:{number}")
  assert not found, "private library-author names: " + ", ".join(found[:20])


def test_the_extension_api_is_versioned_and_complete() -> None:
  import scaly.ext as ext

  assert ext.EXT_API_VERSION == 1
  assert all(hasattr(ext, name) for name in ext.__all__)
  assert ext.from_exprs is sc.Function.from_exprs and ext.register_op is __import__("scaly.ir.expr", fromlist=["register_op"]).register_op
  ext.require_ext_api(ext.EXT_API_VERSION, "a package")
  with pytest.raises(ImportError, match="extension API 0"):
    ext.require_ext_api(0, "an old package")


def test_the_linalg_ops_take_only_public_names_from_the_compiler() -> None:
  """``linalg.ops`` registers its ops as a package outside the compiler would (``scaly.ext``): every
  name it imports from the compiler is public, so the extension API is enough to write it."""
  ops = Path(sc.__file__).parent / "linalg" / "ops"
  private = []
  for path in sorted(ops.glob("*.py")):
    for stmt in ast.walk(ast.parse(path.read_text())):
      into_core = isinstance(stmt, ast.ImportFrom) and (stmt.level == 3 or (stmt.level == 0 and (stmt.module or "").startswith("scaly.")))
      if into_core:
        private += [f"{path.name}:{stmt.lineno} {alias.name}" for alias in stmt.names if alias.name.startswith("_")]
  assert not private, "linalg.ops imports private compiler names: " + ", ".join(private)


def test_nothing_outside_scaly_opt_imports_its_private_names() -> None:
  """``extract_qp``, ``nlp_oracles`` and the method registry are the public way into a problem's
  normal forms and solvers; ``scaly.opt``'s underscore names stay inside it."""
  root = Path(__file__).resolve().parents[1]
  pattern = re.compile(r"from (?:scaly\.opt|\.+opt)(?:\.[\w.]+)? import \(?([^)\n]*)")
  found = []
  for folder in ("src", "tests", "plugins", "examples", "bench"):
    for path in (root / folder).rglob("*.py"):
      if ".ipynb_checkpoints" in path.parts or path.is_relative_to(root / "src" / "scaly" / "opt"):
        continue
      for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        match = pattern.search(line)
        if match and any(name.strip().split(" as ")[0].startswith("_") for name in match.group(1).split(",")):
          found.append(f"{path.relative_to(root)}:{number}")
  assert not found, "private scaly.opt names imported outside it: " + ", ".join(found)
