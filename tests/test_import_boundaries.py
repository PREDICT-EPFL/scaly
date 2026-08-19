from __future__ import annotations

import importlib.util

import alloy as al
import alloy.codegen as codegen
from alloy.codegen import aot
from alloy.function.model import Function, Port
from alloy.ir.expr import Expr, ExprOp
from alloy.ir.expr_spec import spec_expr
from alloy.ir.program import ProgramNode, ProgramOp
from alloy.ir.program_spec import spec_program_full
from alloy.ir.spec import Rule, Spec, VerifyError


def test_public_exports_are_canonical() -> None:
  program = __import__("alloy.ir.program", fromlist=["ProgramNode"])
  assert al.Expr is Expr
  assert al.ExprOp is ExprOp
  assert al.Function is Function
  assert al.Port is Port
  assert al.Rule is Rule
  assert al.Spec is Spec
  assert al.VerifyError is VerifyError
  assert program.ProgramNode is ProgramNode
  assert program.ProgramOp is ProgramOp
  assert codegen.render_c_source is aot.render_c_source
  assert {"ExprOp", "Rule", "Spec", "VerifyError"} <= set(al.__all__)
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
