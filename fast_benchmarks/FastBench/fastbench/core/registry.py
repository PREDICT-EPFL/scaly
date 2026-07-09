"""Simple registries mapping names -> problem/solver factories."""
from __future__ import annotations
from typing import Callable, Dict, List

_PROBLEMS: Dict[str, Callable] = {}
_SOLVERS: Dict[str, Callable] = {}


def register_problem(name: str):
    def deco(factory: Callable):
        _PROBLEMS[name] = factory
        return factory
    return deco


def register_solver(name: str):
    def deco(factory: Callable):
        _SOLVERS[name] = factory
        return factory
    return deco


def _ensure_problems():
    import fastbench.problems  # noqa: F401  (triggers registration)


def _ensure_solvers():
    import fastbench.solvers  # noqa: F401


def get_problem(name: str):
    _ensure_problems()
    if name not in _PROBLEMS:
        raise KeyError(f"unknown problem '{name}'. Known: {sorted(_PROBLEMS)}")
    return _PROBLEMS[name]()


def get_solver(name: str):
    _ensure_solvers()
    if name not in _SOLVERS:
        raise KeyError(f"unknown solver '{name}'. Known: {sorted(_SOLVERS)}")
    return _SOLVERS[name]()


def list_problems() -> List[str]:
    _ensure_problems()
    return sorted(_PROBLEMS)


def list_solvers() -> List[str]:
    _ensure_solvers()
    return sorted(_SOLVERS)
