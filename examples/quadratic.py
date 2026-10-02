from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module


@sc.function(sc.arg("x", 2), sc.arg("target", 2), outputs=sc.arg("cost", ...))
def tracking_cost(x: sc.Expr, target: sc.Expr) -> sc.Expr:
  return sc.sumsqr(x - target)


grad = sc.gradient(tracking_cost, "cost", "x")
hess = sc.hessian(tracking_cost, "cost", "x")
data = (np.array([3.0, 5.0]), np.array([1.0, 2.0]))
print(grad(*data))  # [4. 6.]
print(hess(*data))  # [[2. 0.]
#  [0. 2.]]

gen_dir = Path(__file__).parent / "generated"

write_module(tracking_cost, gen_dir)
write_module(grad, gen_dir)
write_module(hess, gen_dir)
