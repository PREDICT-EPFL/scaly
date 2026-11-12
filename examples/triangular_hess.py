import numpy as np
from scipy import sparse

import scaly as sc


@sc.function(sc.arg("x", 4), outputs=sc.arg("cost"))
def cost(x: sc.Expr) -> sc.Expr:
  return sc.sumsqr(x) + x[0] * x[1]


sparse_hess = sc.sparse_hessian(cost, "cost", "x", triangle="lower")
lower = sparse.coo_array(sparse_hess(np.ones(4))).toarray()
print(lower)
# H = lower + lower.T - np.diag(np.diag(lower))
# print(H)
# [[2. 1. 0. 0.]
#  [1. 2. 0. 0.]
#  [0. 0. 2. 0.]
#  [0. 0. 0. 2.]]

sparse_hess_full = sc.sparse_hessian(cost, "cost", "x", triangle="full")
print(sparse.coo_array(sparse_hess_full(np.ones(4))).toarray())
