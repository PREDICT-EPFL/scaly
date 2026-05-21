"""Phase 12 cleanup: ``ALLOY_REQUIRE_JIT`` disables silent interpreter fallback."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.jit import JitUnavailable


def test_jit_required_propagates_jit_unavailable(monkeypatch) -> None:
  """When ALLOY_REQUIRE_JIT is set, a function whose codegen raises
  NotImplementedError no longer silently falls back to the interpreter."""
  # Build a function whose codegen will fail: the solver path raises
  # JitUnavailable in the legacy renderer for unsupported solver wiring.
  # Easiest reproducer: monkeypatch CompiledFunction to raise.
  from alloy import jit as jit_module

  x = al.sym("x", 3)
  fn = al.Function("f_req_jit", [x], [x.sin()], ["x"], ["y"])

  class _BoomCompiled:
    def __init__(self, fun):
      raise JitUnavailable("synthetic")

  monkeypatch.setattr(jit_module, "CompiledFunction", _BoomCompiled)
  monkeypatch.setenv("ALLOY_REQUIRE_JIT", "1")
  fn.recompile()
  with pytest.raises(JitUnavailable):
    fn(np.array([1.0, 2.0, 3.0]))


def test_default_silent_fallback_still_works(monkeypatch) -> None:
  """Without ALLOY_REQUIRE_JIT, the interpreter fallback continues to take over."""
  from alloy import jit as jit_module

  x = al.sym("x", 3)
  fn = al.Function("f_silent_fallback", [x], [x.sin()], ["x"], ["y"])

  class _BoomCompiled:
    def __init__(self, fun):
      raise JitUnavailable("synthetic")

  monkeypatch.setattr(jit_module, "CompiledFunction", _BoomCompiled)
  monkeypatch.delenv("ALLOY_REQUIRE_JIT", raising=False)
  fn.recompile()
  out = fn(np.array([1.0, 2.0, 3.0]))
  np.testing.assert_allclose(out, np.sin([1.0, 2.0, 3.0]))


def test_jit_required_helper_reads_env(monkeypatch) -> None:
  from alloy.jit import jit_required

  monkeypatch.delenv("ALLOY_REQUIRE_JIT", raising=False)
  assert jit_required() is False
  monkeypatch.setenv("ALLOY_REQUIRE_JIT", "1")
  assert jit_required() is True
  monkeypatch.setenv("ALLOY_REQUIRE_JIT", "0")
  assert jit_required() is False
