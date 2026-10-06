from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

N = 6


def chain(x: sc.Expr) -> sc.Expr:
  return x[:-1].sin() * x[1:]


@sc.function(sc.arg("x", N), sc.arg("v", N), outputs=sc.arg("jv"))
def jvp_via_sparse_jac(x: sc.Expr, v: sc.Expr) -> sc.Expr:
  return sc.sparse_jacobian(chain(x), x).to_dense() @ v


@sc.function(sc.arg("x", N), sc.arg("v", N), outputs=sc.arg("jv"))
def jvp_native(x: sc.Expr, v: sc.Expr) -> sc.Expr:
  return sc.jvp(chain(x), x, v)


data = (np.linspace(0.1, 0.6, N), np.arange(1.0, N + 1))
print(jvp_via_sparse_jac(*data))
print(jvp_native(*data))

gen_dir = Path(__file__).parent / "generated"
write_module(jvp_via_sparse_jac, gen_dir)
write_module(jvp_native, gen_dir)
