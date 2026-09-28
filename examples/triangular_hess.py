import numpy as np

import scaly as sc


@sc.function(sc.L("x", 4), sc.L("cost", ...))
def cost(x: sc.Expr) -> sc.Expr:
  return sc.sumsqr(x) + x[0] * x[1]


sparse_hess = sc.sparse_hessian(cost, "cost", "x", triangle="lower")
hess_values = sparse_hess(np.ones(4))
hess_pattern = sparse_hess.output_sparsities[0]
assert hess_pattern is not None
lower = np.zeros(hess_pattern.shape)
lower[np.asarray(hess_pattern.rows), np.asarray(hess_pattern.cols)] = hess_values
print(lower)
# H = lower + lower.T - np.diag(np.diag(lower))
# print(H)
# [[2. 1. 0. 0.]
#  [1. 2. 0. 0.]
#  [0. 0. 2. 0.]
#  [0. 0. 0. 2.]]

sparse_hess_full = sc.sparse_hessian(cost, "cost", "x", triangle="full")
hess_values_full = sparse_hess_full(np.ones(4))
hess_pattern_full = sparse_hess_full.output_sparsities[0]
assert hess_pattern_full is not None
full = np.zeros(hess_pattern_full.shape)
full[np.asarray(hess_pattern_full.rows), np.asarray(hess_pattern_full.cols)] = hess_values_full
print(full)
