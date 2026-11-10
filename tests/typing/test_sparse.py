"""Static sparse numerical declarations, inferred results, calls and maps."""

from typing import TYPE_CHECKING, assert_type

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.function import Tree
from scaly.function.concrete import ConcreteFunction
from scaly.function.tree import NumericalArray


if TYPE_CHECKING:
  pattern = sc.SparsityPattern((3, 4), (0, 0, 1, 2), (0, 3, 1, 0))
  canonical, _ = pattern.canonicalize()
  tensor = sc.TensorType(pattern.shape, pattern=canonical)
  default = sc.arg("x", tensor)
  csc = sc.arg("x", tensor, layout="csc")
  csr = sc.arg("x", tensor, layout="csr")
  coordinates = sc.arg("x", tensor, layout=pattern)
  assert_type(default, Tree[sc.Expr, NumericalArray])
  assert_type(csc, Tree[sc.Expr, NumericalArray])
  assert_type(csr, Tree[sc.Expr, NumericalArray])
  assert_type(coordinates, Tree[sc.Expr, NumericalArray])
  assert_type(sc.arg("holed", pattern.shape, pattern=...), Tree[sc.Expr, NumericalArray])
  assert_type(sc.arg("holed", pattern=...), Tree[sc.Expr, NumericalArray])
  assert_type(sc.arg("dense", 3), Tree[sc.Expr, np.ndarray])
  assert_type(sc.arg("dense", (3, 4)), Tree[sc.Expr, np.ndarray])

  @sc.function(default, outputs=default.relabel("out_"))
  def declared(x: sc.Expr) -> sc.Expr:
    return -x

  @sc.function(csc, outputs=csr.relabel("out_"))
  def compressed(x: sc.Expr) -> sc.Expr:
    return declared(x)

  @sc.function(coordinates, outputs=coordinates.relabel("out_"))
  def coordinate_order(x: sc.Expr) -> sc.Expr:
    return compressed(x)

  @sc.function(default)
  def inferred(x: sc.Expr) -> sc.Expr:
    return coordinate_order(x)

  @sc.function()
  def constant() -> sc.Expr:
    return sc.pack(sc.const([1.0, 2.0, 3.0, 4.0]), canonical)

  data: np.ndarray = np.ones(canonical.nnz)
  source = sparse.csc_array((data, (canonical.rows, canonical.cols)), shape=canonical.shape)
  csr_source: sparse.csr_array = source.tocsr()
  coordinate_source: sparse.coo_array = source.tocoo()
  symbol = default.symbols()
  assert_type(declared, sc.Function[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(declared(source), NumericalArray)
  assert_type(declared(data), NumericalArray)
  assert_type(compressed(csr_source), NumericalArray)
  assert_type(coordinate_order(coordinate_source), NumericalArray)
  assert_type(inferred(source), NumericalArray)
  assert_type(inferred.numerical_call(source), NumericalArray)
  assert_type(inferred(symbol), sc.Expr)
  assert_type(inferred.symbolic_call(symbol), sc.Expr)
  assert_type(constant(), NumericalArray)
  assert_type(constant.symbolic_call(), sc.Expr)
  concrete = inferred.instantiate()
  assert_type(concrete, ConcreteFunction[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(concrete(source), NumericalArray)
  assert_type(concrete.numerical_call(data), NumericalArray)
  assert_type(concrete.symbolic_call(symbol), sc.Expr)

  mapped = sc.vmap(inferred, 2)
  assert_type(mapped, sc.Function[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(mapped(np.ones((2, canonical.nnz))), NumericalArray)
  assert_type(mapped(sc.broadcast(source)), NumericalArray)
  assert_type(mapped(sc.window(np.ones(2 * canonical.nnz), 0, canonical.nnz)), NumericalArray)
  assert_type(mapped(sc.broadcast(symbol)), sc.Expr)
  assert_type(mapped.numerical_call(sc.broadcast(source)), NumericalArray)
  assert_type(sc.vmap(coordinate_order, 2)(sc.broadcast(source)), NumericalArray)
  assert_type(sc.vmap(sc.vmap(compressed, 2), 3)(sc.broadcast(source)), NumericalArray)

  @sc.function(sc.group(default, sc.arg("dense", 3)), outputs=sc.group(csr.relabel("out_"), sc.arg("dense_out", 3)))
  def grouped(values: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    return values

  assert_type(grouped((source, np.ones(3))), tuple[NumericalArray, np.ndarray])
  assert_type(sc.vmap(grouped, 2)((sc.broadcast(source), np.ones((2, 3)))), tuple[NumericalArray, np.ndarray])
  assert_type(grouped((symbol, sc.sym("dense", 3))), tuple[sc.Expr, sc.Expr])
  declared.numerical_call(symbol)  # ty: ignore[invalid-argument-type]
  declared.symbolic_call(source)  # ty: ignore[invalid-argument-type]
  declared(source, source)  # ty: ignore[no-matching-overload]
  concrete.numerical_call(symbol)  # ty: ignore[invalid-argument-type]
  mapped(symbol, source)  # ty: ignore[no-matching-overload]
  grouped((source, sc.sym("dense", 3)))  # ty: ignore[no-matching-overload]
  grouped((symbol, np.ones(3)))  # ty: ignore[no-matching-overload]
  sc.vmap(grouped, 2)((sc.broadcast(source), sc.sym("dense", (2, 3))))  # ty: ignore[no-matching-overload]

  def needs_dense(value: np.ndarray) -> None: ...

  needs_dense(declared(source))  # ty: ignore[invalid-argument-type]
  needs_dense(inferred(source))  # ty: ignore[invalid-argument-type]
  needs_dense(constant())  # ty: ignore[invalid-argument-type]
  needs_dense(mapped(sc.broadcast(source)))  # ty: ignore[invalid-argument-type]
