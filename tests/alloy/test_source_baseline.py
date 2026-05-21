"""Phase 0 baseline: lock generated C source for a small deterministic corpus.

These tests are the regression guard for the semantic-IR -> Program-IR migration
(see ``docs/roadmap.md``). Each corpus entry exercises a small piece of the
current scalar C renderer; the rendered source is normalized and hashed so that
later phases can either:

- preserve the exact output bit-for-bit (Phase 1 dtype model with ``float64``
  default), or
- emit a new golden via ``ALLOY_REGEN_SOURCE_BASELINE=1`` when a phase
  intentionally changes the lowering (Phase 5 Program IR C renderer behind a
  feature flag).

The corpus is intentionally tiny: scalar elementwise, vector reduction, matmul,
named call, MAP loop. Workload-shape coverage stays in the existing
``test_tracking_workload`` / ``test_unbumpercars_workload`` files.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import alloy as al
from alloy.codegen import render_c_source

GOLDEN_DIR = Path(__file__).parent / "golden_source"


def _normalize(src: str) -> str:
  src = src.replace("\r\n", "\n")
  src = re.sub(r"[ \t]+\n", "\n", src)
  return src.rstrip() + "\n"


def _digest(src: str) -> str:
  return hashlib.sha256(_normalize(src).encode("utf-8")).hexdigest()


def _corpus_scalar_elementwise() -> al.Function:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  return al.Function("scalar_elem", [x], [y], ["x"], ["y"])


def _corpus_vector_dot() -> al.Function:
  x = al.sym("x", 5)
  y = al.sym("y", 5)
  return al.Function("vec_dot", [x, y], [al.dot(x, y)], ["x", "y"], ["d"])


def _corpus_matmul() -> al.Function:
  a = al.sym("a", (3, 4))
  b = al.sym("b", (4, 2))
  return al.Function("mm", [a, b], [a @ b], ["a", "b"], ["c"])


def _corpus_named_call() -> al.Function:
  x = al.sym("x", 3)
  inner = al.Function("inner", [x], [(x * x).sum()], ["x"], ["y"])
  z = al.sym("z", 3)
  (out,) = inner.call([z])
  return al.Function("outer", [z], [out + 1.0], ["z"], ["w"])


def _corpus_map() -> al.Function:
  stage_in = al.sym("u", 2)
  stage = al.Function("stage", [stage_in], [stage_in.sin().sum()], ["u"], ["y"])
  batch = al.sym("batch", 8)
  mapped = al.map_(stage, length=4, inputs=[(batch, 0, 2)])
  return al.Function("mapped", [batch], [mapped], ["batch"], ["m"])


CORPUS: tuple[tuple[str, Callable[[], al.Function]], ...] = (
  ("scalar_elementwise", _corpus_scalar_elementwise),
  ("vector_dot", _corpus_vector_dot),
  ("matmul", _corpus_matmul),
  ("named_call", _corpus_named_call),
  ("map_loop", _corpus_map),
)


def _golden_path(name: str) -> Path:
  return GOLDEN_DIR / f"{name}.c"


def _maybe_regen() -> bool:
  return os.environ.get("ALLOY_REGEN_SOURCE_BASELINE") == "1"


@pytest.mark.parametrize("name,builder", CORPUS, ids=[name for name, _ in CORPUS])
def test_source_baseline_matches_golden(name: str, builder, monkeypatch) -> None:
  # The goldens are pinned to the legacy scalar C renderer. Disable the
  # Program-IR-backed renderer for this test so the comparison is meaningful.
  monkeypatch.delenv("ALLOY_USE_PROGRAM_IR_C", raising=False)
  fn = builder()
  src = _normalize(render_c_source(fn))
  golden = _golden_path(name)
  if _maybe_regen() or not golden.exists():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden.write_text(src)
    if _maybe_regen():
      pytest.skip(f"regenerated baseline {golden.name}")
  expected = _normalize(golden.read_text())
  assert _digest(src) == _digest(expected), (
    f"generated C for {name} drifted from golden {golden.name}. If the change is intentional, regenerate with ALLOY_REGEN_SOURCE_BASELINE=1."
  )


@pytest.mark.parametrize("name,builder", CORPUS, ids=[name for name, _ in CORPUS])
def test_source_baseline_compiles_and_evaluates(name: str, builder) -> None:
  """Sanity check: each corpus entry compiles and matches the interpreter."""
  fn = builder()
  if name == "scalar_elementwise":
    args = [np.array([0.1, 0.5, -0.3])]
  elif name == "vector_dot":
    args = [np.arange(5.0), np.arange(5.0) * 0.5]
  elif name == "matmul":
    args = [np.arange(12.0).reshape(3, 4), np.arange(8.0).reshape(4, 2)]
  elif name == "named_call":
    args = [np.array([0.2, -0.1, 0.4])]
  elif name == "map_loop":
    args = [np.linspace(-0.5, 0.5, 8)]
  else:  # pragma: no cover
    pytest.fail(f"unhandled corpus entry {name}")
  ref = fn.eval_interpreter(*args)
  out = fn(*args)
  ref_arr = ref[0] if isinstance(ref, list) else np.asarray(ref)
  out_arr = out[0] if isinstance(out, list) else np.asarray(out)
  np.testing.assert_allclose(np.asarray(out_arr), np.asarray(ref_arr), atol=1e-12, rtol=1e-12)
