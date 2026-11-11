"""Build Einstein contractions from gathers, reductions and matrix products."""

from collections import Counter
from math import prod
import string
from typing import Any

import numpy as np

from .expr import Expr, as_expr, matmul


def einsum(spec: str, *operands: Any) -> Expr:
  """Contract labelled axes, with NumPy's string subscript and broadcasting rules.

  Contractions run left to right through matrix products. Repeated labels select
  diagonals. Ellipses stand for broadcast batch axes. No order search is performed.
  """
  if not isinstance(spec, str):
    raise TypeError("einsum spec must be a string")
  spec = spec.replace(" ", "")
  parts = spec.split("->")
  if len(parts) > 2:
    raise ValueError("einsum spec has multiple output arrows")
  terms = parts[0].split(",")
  xs = [as_expr(x) for x in operands]
  if not xs or len(terms) != len(xs):
    raise ValueError("einsum operand count does not match subscripts")

  def parse(term: str) -> list[str]:
    if term.count("...") > 1 or any(c not in string.ascii_letters for c in term.replace("...", "")):
      raise ValueError(f"invalid einsum subscript {term!r}")
    return list(term.replace("...", "@"))

  parsed = [parse(t) for t in terms]
  ranks = []
  for labels, x in zip(parsed, xs, strict=True):
    rank = len(x.shape) - len(labels) + ("@" in labels)
    if rank < 0 or ("@" not in labels and rank != 0):
      raise ValueError("einsum subscripts do not match operand rank")
    ranks.append(rank)
  ell = [f"@{i}" for i in range(max(ranks))]
  counts = Counter(c for labels in parsed for c in labels if c != "@")
  if len(parts) == 2:
    output = parse(parts[1])
    output = [v for c in output for v in (ell if c == "@" else [c])]
    if len(set(output)) != len(output) or any(c not in counts and c not in ell for c in output):
      raise ValueError("einsum output labels must be unique and present in inputs")
    if ell and not all(c in output for c in ell):
      raise ValueError("einsum output must retain ellipsis axes")
  else:
    output = ell + sorted(c for c, n in counts.items() if n == 1)
  labels_all = []
  sizes: dict[str, int] = {}
  for i, (labels, x, rank) in enumerate(zip(parsed, xs, ranks, strict=True)):
    labels = [v for c in labels for v in (ell[len(ell) - rank :] if c == "@" else [c])]
    local: dict[str, int] = {}
    for c, n in zip(labels, x.shape, strict=True):
      if c in local and local[c] != n:
        raise ValueError(f"einsum repeated label {c!r} has unequal sizes")
      local[c] = n
      old = sizes.get(c, 1)
      if old != n and old != 1 and n != 1:
        raise ValueError(f"einsum label {c!r} has incompatible sizes {old} and {n}")
      sizes[c] = n if old == 1 else old
    unique = list(dict.fromkeys(labels))
    if len(unique) != len(labels):
      shape = tuple(local[c] for c in unique)
      coordinates = np.indices(shape, dtype=np.int64)
      idx = np.zeros(shape, dtype=np.int64)
      stride = 1
      for c, n in reversed(list(zip(labels, x.shape, strict=True))):
        idx += coordinates[unique.index(c)] * stride
        stride *= n
      xs[i] = x.gather(idx)
    labels_all.append(unique)

  def expand(x: Expr, labels: list[str]) -> Expr:
    shape = tuple(sizes[c] for c in labels)
    if x.shape == shape:
      return x
    idx = np.broadcast_to(np.arange(x.size).reshape(x.shape), shape)
    return x.gather(idx)

  xs = [expand(x, labels) for x, labels in zip(xs, labels_all, strict=True)]
  x, labels = xs[0], labels_all[0]
  for i in range(1, len(xs)):
    y, right = xs[i], labels_all[i]
    needed = set(output).union(*(set(v) for v in labels_all[i + 1 :]))
    for side in (0, 1):
      own, other = (labels, right) if side == 0 else (right, labels)
      axes = tuple(j for j, c in enumerate(own) if c not in needed and c not in other)
      if axes:
        if side == 0:
          x = x.sum(axis=axes)
          labels = [c for j, c in enumerate(labels) if j not in axes]
        else:
          y = y.sum(axis=axes)
          right = [c for j, c in enumerate(right) if j not in axes]
    shared = [c for c in labels if c in right]
    batch = [c for c in shared if c in needed]
    contracted = [c for c in shared if c not in needed]
    left_free = [c for c in labels if c not in shared]
    right_free = [c for c in right if c not in shared]

    def matrix(value: Expr, own: list[str], order: list[str], free: list[str], left: bool) -> Expr:
      value = value.transpose(tuple(own.index(c) for c in order))
      b = tuple(sizes[c] for c in batch)
      f, k = prod(sizes[c] for c in free), prod(sizes[c] for c in contracted)
      return value.reshape(b + ((f, k) if left else (k, f)))

    x = matmul(
      matrix(x, labels, batch + left_free + contracted, left_free, True), matrix(y, right, batch + contracted + right_free, right_free, False)
    )
    labels = batch + left_free + right_free
    x = x.reshape(tuple(sizes[c] for c in labels))
  axes = tuple(i for i, c in enumerate(labels) if c not in output)
  if axes:
    x = x.sum(axis=axes)
    labels = [c for c in labels if c in output]
  return x.transpose(tuple(labels.index(c) for c in output))
