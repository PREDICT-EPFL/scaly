from pathlib import Path

import numpy as np

import scaly as sc


@sc.function(sc.arg("x", 2), sc.arg("gain", ()), outputs=sc.arg("y", 2))
def damped_with_print(x: sc.Expr, gain: sc.Expr) -> sc.Expr:
  x = sc.print("x={} gain={}", x, gain)
  return gain * x


print(damped_with_print(np.array([1.0, 0.1]), np.array(0.5)))
# x=[1, 0.10000000000000001] gain=0.5
# [0.5  0.05]

sc.codegen.write_module(damped_with_print, Path(__file__).parent / "generated")  # generated/damped_with_print.h and .c
