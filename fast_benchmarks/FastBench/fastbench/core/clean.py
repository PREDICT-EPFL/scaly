"""Remove regenerable build/run artifacts.

Two levels:
* **clean**     - caches and build/run outputs that are always regenerable
                  (``__pycache__``, ``fastsqp_build``, ``results``, codegen,
                  ``.pytest_cache``, ``*.egg-info``, ``build``/``dist``).
* **deep** (all) - additionally the heavy, fully-reinstallable trees
                  (``.venv``, ``third_party`` acados build, ``examples``),
                  i.e. a reset back to source-only for shipping.

Everything is scoped to the project root and the current working directory; a
safety check refuses to delete anything outside them.
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import List

# project root = .../FastBench  (this file is FastBench/fastbench/core/clean.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# directories never descended into when scanning for __pycache__
_PRUNE = {".venv", "third_party", ".git"}

# regenerable artifacts (relative names matched in ROOT and CWD)
_DIRS = ["results", "fastsqp_build", "build", "dist", ".pytest_cache"]
_GLOBS = ["*.egg-info", "c_generated_code*", "acados_ocp_*.json"]
# only removed with deep=True
_DEEP = [".venv", "third_party", "examples"]


def _within(path: str, roots) -> bool:
    ap = os.path.abspath(path)
    for r in roots:
        r = os.path.abspath(r)
        if ap == r:                       # never delete a root itself
            return False
        if ap.startswith(r + os.sep):
            return True
    return False


def collect_targets(deep: bool = False) -> List[str]:
    roots = {ROOT, os.getcwd()}
    targets = set()
    for base in roots:
        for d in _DIRS:
            p = os.path.join(base, d)
            if os.path.exists(p):
                targets.add(p)
        for g in _GLOBS:
            targets.update(glob.glob(os.path.join(base, g)))
    # __pycache__ anywhere under ROOT (skipping heavy/vendored trees)
    for dirpath, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE]
        if os.path.basename(dirpath) == "__pycache__":
            targets.add(dirpath)
    if deep:
        for d in _DEEP:
            p = os.path.join(ROOT, d)
            if os.path.exists(p):
                targets.add(p)
    return sorted(t for t in targets if _within(t, roots))


def clean(deep: bool = False, dry_run: bool = False) -> List[str]:
    targets = collect_targets(deep=deep)
    for t in targets:
        if dry_run:
            continue
        if os.path.isdir(t) and not os.path.islink(t):
            shutil.rmtree(t, ignore_errors=True)
        else:
            try:
                os.remove(t)
            except OSError:
                pass
    return targets
