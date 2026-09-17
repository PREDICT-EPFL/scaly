"""Both dialects: their definitions, verifiers, stable text, and the machinery passes are built on.

Deliberately re-exports nothing — a node class reachable under two module paths would give
hash-consing two intern tables (``docs/dev/codebase.md``).
"""
