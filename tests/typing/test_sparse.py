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


if TYPE_CHECKING:
  diagonal, _ = sc.SparsityPattern((2, 2), (0, 1), (0, 1)).canonicalize()
  diagonal_type = sc.TensorType((2, 2), pattern=diagonal)
  diagonal_input = sc.arg("x", diagonal_type)
  matrix = sparse.csc_array(np.eye(2))
  expr = diagonal_input.symbols()
  assert_type(sc.arg("out"), Tree[sc.Expr, NumericalArray])
  assert_type(sc.arg("out", None), Tree[sc.Expr, NumericalArray])

  @sc.function(diagonal_input, outputs=sc.arg("out"))
  def hole(x: sc.Expr) -> sc.Expr:
    return x.sin()

  @sc.function(diagonal_input, outputs=sc.arg("out", None))
  def none_hole(x: sc.Expr) -> sc.Expr:
    return x.sin()

  @sc.function(diagonal_input)
  def inferred_diagonal(x: sc.Expr) -> sc.Expr:
    return x.sin()

  assert_type(hole, sc.Function[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(none_hole, sc.Function[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(hole(matrix), NumericalArray)
  assert_type(none_hole(matrix), NumericalArray)
  assert_type(inferred_diagonal(matrix), NumericalArray)
  assert_type(hole.numerical_call(matrix), NumericalArray)
  assert_type(none_hole.numerical_call(matrix), NumericalArray)
  assert_type(hole(expr), sc.Expr)
  assert_type(none_hole.symbolic_call(expr), sc.Expr)
  hole_concrete = hole.instantiate()
  assert_type(hole_concrete, ConcreteFunction[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(hole_concrete(matrix), NumericalArray)
  assert_type(none_hole.instantiate().numerical_call(matrix), NumericalArray)
  assert_type(hole_concrete.symbolic_call(expr), sc.Expr)
  assert_type(sc.vmap(hole, 2)(sc.broadcast(matrix)), NumericalArray)
  assert_type(sc.vmap(none_hole, 2).numerical_call(sc.window(np.ones(4), 0, 2)), NumericalArray)
  assert_type(sc.vmap(hole_concrete, 2)(sc.broadcast(expr)), sc.Expr)

  @sc.function(diagonal_input, outputs=sc.group(sc.arg("out"), sc.group(sc.arg("other", None), sc.arg("dense", ()))))
  def nested_holes(x: sc.Expr) -> tuple[sc.Expr, tuple[sc.Expr, sc.Expr]]:
    return x, (x.sin(), x.sum())

  assert_type(nested_holes(matrix), tuple[NumericalArray, tuple[NumericalArray, np.ndarray]])
  assert_type(nested_holes.instantiate().numerical_call(matrix), tuple[NumericalArray, tuple[NumericalArray, np.ndarray]])
  assert_type(sc.vmap(nested_holes, 2)(sc.broadcast(matrix)), tuple[NumericalArray, tuple[NumericalArray, np.ndarray]])
  assert_type(nested_holes(expr), tuple[sc.Expr, tuple[sc.Expr, sc.Expr]])

  @sc.function(sc.arg("dense", 2), outputs=sc.arg("out"))
  def sparse_from_dense(x: sc.Expr) -> sc.Expr:
    return sc.pack(x, diagonal)

  @sc.function(outputs=sc.arg("out", None))
  def sparse_constant() -> sc.Expr:
    return sc.pack(sc.const([1.0, 2.0]), diagonal)

  assert_type(sparse_from_dense(np.ones(2)), NumericalArray)
  assert_type(sparse_constant(), NumericalArray)
  hole.numerical_call(expr)  # ty: ignore[invalid-argument-type]
  none_hole.symbolic_call(matrix)  # ty: ignore[invalid-argument-type]
  needs_dense(hole(matrix))  # ty: ignore[invalid-argument-type]
  needs_dense(none_hole(matrix))  # ty: ignore[invalid-argument-type]
  needs_dense(hole_concrete(matrix))  # ty: ignore[invalid-argument-type]
  needs_dense(sc.vmap(hole, 2)(sc.broadcast(matrix)))  # ty: ignore[invalid-argument-type]
  needs_dense(nested_holes(matrix)[1][0])  # ty: ignore[invalid-argument-type]


if TYPE_CHECKING:

  @sc.function(diagonal_input, outputs=sc.arg("cost", ()))
  def sparse_cost(x: sc.Expr) -> sc.Expr:
    return (x * x).sum()

  grad = sc.gradient(sparse_cost)
  fwd = sc.forward(hole)
  adj = sc.adjoint(hole)
  assert_type(grad, sc.Function[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(fwd, sc.Function[tuple[sc.Expr, sc.Expr], tuple[NumericalArray, NumericalArray], sc.Expr, NumericalArray])
  assert_type(adj, sc.Function[tuple[sc.Expr, sc.Expr], tuple[NumericalArray, NumericalArray], sc.Expr, NumericalArray])
  assert_type(grad(matrix), NumericalArray)
  assert_type(fwd(matrix, matrix), NumericalArray)
  assert_type(adj(matrix, matrix), NumericalArray)
  assert_type(grad.numerical_call(matrix), NumericalArray)
  assert_type(fwd.numerical_call(matrix, matrix), NumericalArray)
  assert_type(adj.numerical_call(matrix, matrix), NumericalArray)
  assert_type(grad(expr), sc.Expr)
  assert_type(fwd.symbolic_call(expr, expr), sc.Expr)
  assert_type(adj(expr, expr), sc.Expr)
  assert_type(grad.instantiate(), ConcreteFunction[tuple[sc.Expr], tuple[NumericalArray], sc.Expr, NumericalArray])
  assert_type(grad.instantiate()(matrix), NumericalArray)
  assert_type(fwd.instantiate().numerical_call(matrix, matrix), NumericalArray)
  assert_type(adj.instantiate()(matrix, matrix), NumericalArray)
  assert_type(sc.gradient(sparse_cost.instantiate())(matrix), NumericalArray)
  assert_type(sc.forward(hole_concrete)(matrix, matrix), NumericalArray)
  assert_type(sc.adjoint(hole_concrete)(matrix, matrix), NumericalArray)
  assert_type(sc.forward(none_hole)(matrix, matrix), NumericalArray)
  assert_type(sc.adjoint(inferred_diagonal)(matrix, matrix), NumericalArray)
  assert_type(sc.vmap(grad, 2)(sc.broadcast(matrix)), NumericalArray)
  assert_type(sc.vmap(fwd, 2)(sc.broadcast(matrix), sc.broadcast(matrix)), NumericalArray)
  assert_type(sc.vmap(adj.instantiate(), 2).numerical_call(sc.broadcast(matrix), sc.broadcast(matrix)), NumericalArray)
  assert_type(sc.vmap(sc.vmap(grad, 2), 3)(sc.broadcast(matrix)), NumericalArray)
  needs_dense(grad(matrix))  # ty: ignore[invalid-argument-type]
  needs_dense(fwd(matrix, matrix))  # ty: ignore[invalid-argument-type]
  needs_dense(adj(matrix, matrix))  # ty: ignore[invalid-argument-type]
  needs_dense(sc.vmap(grad, 2)(sc.broadcast(matrix)))  # ty: ignore[invalid-argument-type]
  needs_dense(fwd.instantiate()(matrix, matrix))  # ty: ignore[invalid-argument-type]
  fwd.numerical_call(expr, matrix)  # ty: ignore[invalid-argument-type]
  adj.symbolic_call(expr, matrix)  # ty: ignore[invalid-argument-type]
  grad(matrix, matrix)  # ty: ignore[no-matching-overload]

  packed_fwd = sc.forward(sparse_from_dense)
  packed_adj = sc.adjoint(sparse_from_dense)
  assert_type(packed_fwd, sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, NumericalArray])
  assert_type(packed_adj, sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, NumericalArray], sc.Expr, np.ndarray])
  assert_type(packed_fwd(np.ones(2), np.ones(2)), NumericalArray)
  assert_type(packed_fwd.instantiate()(np.ones(2), np.ones(2)), NumericalArray)
  assert_type(sc.vmap(packed_fwd, 2)(np.ones((2, 2)), np.ones((2, 2))), NumericalArray)
  assert_type(packed_adj(np.ones(2), matrix), np.ndarray)
  needs_dense(packed_fwd(np.ones(2), np.ones(2)))  # ty: ignore[invalid-argument-type]

  @sc.function(sc.group(diagonal_input, sc.group(sc.arg("scale", ()))), outputs=sc.arg("cost", ()))
  def mixed_cost(inputs: tuple[sc.Expr, tuple[sc.Expr]]) -> sc.Expr:
    x, (scale,) = inputs
    return (x * x).sum() * scale

  mixed = (matrix, (np.ones(()),))
  mixed_grad = sc.gradient(mixed_cost, wrt="x")
  assert_type(mixed_grad(mixed), NumericalArray)
  assert_type(mixed_grad.instantiate().numerical_call(mixed), NumericalArray)
  assert_type(sc.vmap(mixed_grad, 2)((sc.broadcast(matrix), (np.ones(2),))), NumericalArray)
  assert_type(sc.forward(mixed_cost, wrt="x")(mixed, matrix), np.ndarray)
  assert_type(sc.adjoint(mixed_cost, wrt="x")(mixed, np.ones(())), NumericalArray)
  assert_type(sc.gradient(mixed_cost, wrt="scale")(mixed), NumericalArray)
  assert_type(sc.forward(nested_holes, of="other")(matrix, matrix), NumericalArray)
  assert_type(sc.adjoint(nested_holes, of="other")(matrix, matrix), NumericalArray)
  needs_dense(mixed_grad(mixed))  # ty: ignore[invalid-argument-type]

  assert_type(sc.jacobian(hole)(matrix), np.ndarray)
  assert_type(sc.hessian(sparse_cost)(matrix), np.ndarray)
  assert_type(sc.sparse_jacobian(hole)(matrix), np.ndarray)
  assert_type(sc.sparse_hessian(sparse_cost)(matrix), np.ndarray)
  assert_type(sc.lagrangian_hessian(hole)(matrix, matrix), np.ndarray)
  assert_type(sc.sparse_lagrangian_hessian(hole)(matrix, matrix), np.ndarray)
