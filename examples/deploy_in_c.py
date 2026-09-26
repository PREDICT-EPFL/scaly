"""Shipping a function to C: an attitude propagator and its Jacobians, called from a C program.

A rigid body's attitude is a unit quaternion ``q``; a gyroscope reads the body rate plus a slowly
drifting bias ``b``. One step of length ``dt`` rotates ``q`` by ``theta = (omega - b) dt`` in closed
form,

    q+ = (cos(|theta|/2) I + sin(|theta|/2) / |theta| Omega(theta)) q,

which is what the prediction step of an attitude EKF needs, together with ``dq+/dq`` and ``dq+/db``.
``Function.factory`` builds all three in one function, so the generated C evaluates the shared
subexpressions once.

The example then walks the ahead-of-time path end to end:

1. ``write_module`` renders a C header with one struct per buffer, and separately a C++ header;
2. a small C program written here includes the header, integrates a gyro trace with ``_call`` and
   prints the result, and is compiled with ``cc`` and run;
3. its output is compared with the same function called from Python (the JIT path), which compiles
   the very same translation unit;
4. if CasADi is importable, the CasADi-compatible symbols (``casadi=True``) are compiled into a
   shared library and loaded with ``casadi.external``, the route acados and CasADi users take. CasADi
   stores matrices column-major, so that build returns the Jacobians as compact sparse outputs
   (``sc.factory.SpJac``), which CasADi receives with their patterns.

The generated files land in ``examples/generated/deploy_in_c/``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "deploy_in_c"
STEPS, DT = 200, 0.01


def omega_matrix(w: sc.Expr) -> sc.Expr:
  """``Omega(w)`` with ``q_dot = 0.5 Omega(w) q`` for a scalar-first quaternion."""
  x, y, z = w[0], w[1], w[2]
  zero = sc.const(0.0)
  return sc.stack(
    [
      sc.stack([zero, -x, -y, -z]),
      sc.stack([x, zero, z, -y]),
      sc.stack([y, -z, zero, x]),
      sc.stack([z, y, -x, zero]),
    ]
  )


@sc.function(
  sc.G(sc.L("q", 4), sc.L("bias", 3), sc.L("gyro", 3), sc.L("dt", ())),
  sc.L("q_next", ...),
)
def propagate(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  q, bias, gyro, dt = inputs
  theta = (gyro - bias) * dt
  angle = (sc.sumsqr(theta) + 1e-30).sqrt()  # never exactly zero, so sin(a/2)/a stays finite
  rotation = sc.const(np.eye(4)) * (0.5 * angle).cos() + omega_matrix(theta) * ((0.5 * angle).sin() / angle)
  return rotation @ q


# One C function for the value and both Jacobians.
attitude_step = propagate.factory(
  "attitude_step",
  ["q", "bias", "gyro", "dt"],
  ["q_next", sc.factory.Jac("q_next", "q"), sc.factory.Jac("q_next", "bias")],
)


def reference(q: np.ndarray, bias: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
  """The same step in NumPy."""
  theta = (gyro - bias) * dt
  a = np.linalg.norm(theta)
  x, y, z = theta
  om = np.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])
  return (np.cos(a / 2) * np.eye(4) + np.sin(a / 2) / a * om) @ q


def gyro_trace(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
  t = DT * np.arange(STEPS)
  rates = np.stack([0.8 * np.sin(1.3 * t), 0.5 * np.cos(0.7 * t), 0.3 + 0.1 * t], axis=1)
  bias = np.array([0.02, -0.01, 0.015])
  return rates + bias + 1e-3 * np.random.default_rng(seed).standard_normal(rates.shape), bias


C_MAIN = r"""
#include <stdio.h>
#include "attitude_step.h"

/* The gyro trace is read from stdin as STEPS lines of three numbers. */
int main(void) {
  attitude_step_q_t q = {{1.0, 0.0, 0.0, 0.0}};
  attitude_step_bias_t bias = {{%(bias)s}};
  attitude_step_dt_t dt = {{%(dt)r}};
  attitude_step_gyro_t gyro;
  attitude_step_q_next_t q_next;
  attitude_step_jac_q_next_q_t jq;
  attitude_step_jac_q_next_bias_t jb;
  attitude_step_workspace_t work;
  for (int k = 0; k < %(steps)d; ++k) {
    if (scanf("%%lf %%lf %%lf", &gyro.data[0], &gyro.data[1], &gyro.data[2]) != 3) return 2;
    int rc = attitude_step_call(&q, &bias, &gyro, &dt, &q_next, &jq, &jb, &work);
    if (rc != SCALY_SUCCESS) return rc;
    for (int i = 0; i < 4; ++i) q.data[i] = q_next.data[i];
  }
  for (int i = 0; i < 4; ++i) printf("%%.17g ", q.data[i]);
  for (int i = 0; i < 12; ++i) printf("%%.17g ", jb.data[i]);
  printf("\n");
  return 0;
}
"""


def run_in_c(gyro: np.ndarray, bias: np.ndarray) -> np.ndarray | None:
  """Compile the generated C with a small driver and run it; ``None`` without a C compiler."""
  cc = shutil.which("cc")
  if cc is None:
    return None
  out = GENERATED / "c"
  write_module(attitude_step, out)
  (out / "main.c").write_text(C_MAIN % {"bias": ", ".join(repr(float(b)) for b in bias), "dt": DT, "steps": STEPS})
  exe = out / "attitude_demo"
  subprocess.run([cc, "-O2", "-std=c11", "-o", str(exe), str(out / "main.c"), str(out / "attitude_step.c"), "-lm"], check=True)
  text = "\n".join(" ".join(repr(float(v)) for v in row) for row in gyro)
  result = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True)
  return np.array([float(v) for v in result.stdout.split()])


# CasADi is column-major, so its layer takes vectors and compact sparse outputs: sparse Jacobians here.
attitude_step_casadi = propagate.factory(
  "attitude_step_casadi",
  ["q", "bias", "gyro", "dt"],
  ["q_next", sc.factory.SpJac("q_next", "q"), sc.factory.SpJac("q_next", "bias")],
)


def run_in_casadi(q: np.ndarray, bias: np.ndarray, gyro: np.ndarray) -> list[np.ndarray] | None:
  """Load the CasADi-compatible build with ``casadi.external``; ``None`` without CasADi or ``cc``."""
  try:
    import casadi
  except ImportError:
    return None
  cc = shutil.which("cc")
  if cc is None:
    return None
  out = GENERATED / "casadi"
  write_module(attitude_step_casadi, out, casadi=True)
  lib = out / "attitude_step_casadi.so"
  subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(lib), str(out / "attitude_step_casadi.c"), "-lm"], check=True)
  ext = casadi.external("attitude_step_casadi", str(lib))
  return [np.array(casadi.DM(v).full()).squeeze() for v in ext(q, bias, gyro, DT)]


def main() -> dict[str, np.ndarray]:
  gyro, bias = gyro_trace()
  q, q_ref = np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
  for w in gyro:
    q, _, jac_bias = attitude_step((q, bias, w, np.array(DT)))
    q_ref = reference(q_ref, bias, w, DT)
  # The bias Jacobian of the last step against central differences.
  eps = 1e-6
  q_prev = q_ref.copy()
  fd = np.stack(
    [(reference(q_prev, bias + eps * e, gyro[-1], DT) - reference(q_prev, bias - eps * e, gyro[-1], DT)) / (2 * eps) for e in np.eye(3)], axis=1
  )
  _, _, jac_last = attitude_step((q_prev, bias, gyro[-1], np.array(DT)))
  return {"q": q, "q_ref": q_ref, "jac_bias": jac_bias, "jac_bias_fd": fd, "jac_last": jac_last, "gyro": gyro, "bias": bias}


if __name__ == "__main__":
  out = main()
  print(f"after {STEPS} steps q = {np.array2string(out['q'], precision=6)}, |q| - 1 = {np.linalg.norm(out['q']) - 1:.1e}")
  print(
    f"against NumPy: {np.abs(out['q'] - out['q_ref']).max():.1e}; dq+/db against central differences: {np.abs(out['jac_last'] - out['jac_bias_fd']).max():.1e}"
  )
  write_module(attitude_step, GENERATED / "cpp", lang="cpp")
  c_out = run_in_c(out["gyro"], out["bias"])
  if c_out is not None:
    print(
      f"the C program agrees with the Python call: q to {np.abs(c_out[:4] - out['q']).max():.1e}, dq+/db to {np.abs(c_out[4:] - out['jac_bias'].reshape(-1)).max():.1e}"
    )
  cas = run_in_casadi(out["q_ref"], out["bias"], out["gyro"][0])
  if cas is not None:
    ours = attitude_step((out["q_ref"], out["bias"], out["gyro"][0], np.array(DT)))
    print(f"casadi.external agrees: {max(np.abs(a - b).max() for a, b in zip(cas, ours, strict=True)):.1e}")
  print(f"generated C, C++ and CasADi builds in {GENERATED}")
