"""Scaly as a client of the authors' real-time simulator server, beside their C++ clients.

Their `scripts/run_qube_server.py` runs the plant in its own process at 4 kHz on wall-clock time and
streams the state at 500 Hz; a client answers with torques. This client follows their C++ clients
(`run_furuta_{np,eq}_laopt_client.cpp`): it drains the socket to the newest state, solves from it,
sends the first torque at once, and starts the next solve; each solve is warm-started from the
previous solution, and the first is repeated until it converges before its torque is sent. The run
lasts `sim_steps * dt` of server time from the first state.

    # in one shell, in the upstream clone, under its environment
    python scripts/run_qube_server.py --no-plot --dump scaly_np
    # in another
    uv run examples/case_studies/np_mpc/realtime_scaly.py --method neural --out rt.json
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as pr  # noqa: E402
import scaly_impl as si  # noqa: E402

SHIPPED_Z = [-2.447049617767334, 0.23672091960906982, -9.962054252624512, -1.7461178302764893]  # model/mpc_config.yaml
MAX_INITIAL_SOLVES = 500  # as the laOPT clients


def send(conn: socket.socket, obj: dict) -> None:
  payload = json.dumps(obj).encode()
  conn.sendall(struct.pack(">I", len(payload)) + payload)


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
  buf = b""
  while len(buf) < n:
    chunk = conn.recv(n - len(buf))
    if not chunk:
      return None
    buf += chunk
  return buf


def recv(conn: socket.socket) -> dict | None:
  header = recv_exact(conn, 4)
  if header is None:
    return None
  payload = recv_exact(conn, struct.unpack(">I", header)[0])
  return None if payload is None else json.loads(payload)


def recv_latest(conn: socket.socket) -> tuple[dict | None, int]:
  """Block for one state, then drain every state already queued and keep the newest (QubeClient::receiveLatestState)."""
  msg = recv(conn)
  if msg is None:
    return None, 0
  skipped = 0
  conn.setblocking(False)
  try:
    while True:
      try:
        peek = conn.recv(4, socket.MSG_PEEK)
      except BlockingIOError:
        break
      if len(peek) < 4:
        break
      conn.setblocking(True)
      newer = recv(conn)
      conn.setblocking(False)
      if newer is None:
        break
      msg, skipped = newer, skipped + 1
  finally:
    conn.setblocking(True)
  return msg, skipped


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--method", choices=["neural", "equation"], default="neural")
  ap.add_argument("--form", choices=si.FORMS, default="laopt")
  ap.add_argument("--solver", choices=["ipopt", "sqp"], default="ipopt")
  ap.add_argument("--theta", type=float, nargs=4, help="z (neural) or p (equation); default: the shipped mpc_config YAMLs'")
  ap.add_argument("--sim-steps", type=int, default=50, help="run length in periods (their mpc_config's sim_steps)")
  ap.add_argument("--min-solve-ms", type=float, default=0.0, help="send no earlier than this after the solve starts (their laOPT clients: 4.3)")
  ap.add_argument("--host", default="127.0.0.1")
  ap.add_argument("--port", type=int, default=56123)
  ap.add_argument("--connect-timeout", type=float, default=60.0)
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()

  theta = np.array(args.theta if args.theta else (SHIPPED_Z if args.method == "neural" else pr.SYSTEMS["sys3"]))
  model = si.Model(args.method, si.load_cnp())
  P = model.terminal_weight(theta)
  c = si.Controller(model, args.form, args.solver)
  compile_s = c.compile(theta, P)

  deadline = time.monotonic() + args.connect_timeout
  while True:
    try:
      conn = socket.create_connection((args.host, args.port))
      break
    except OSError:
      if time.monotonic() > deadline:
        raise
      time.sleep(0.2)
  conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

  run_time = args.sim_steps * pr.DT
  x_guess, u_guess = pr.cold_guess(pr.X_HANGING)  # replaced by the cold start from the first state
  s_guess = np.zeros(pr.NX)
  t_start, log = None, []
  while True:
    state, skipped = recv_latest(conn)
    if state is None:
      break
    x = np.asarray(state["x"], np.float64)
    if t_start is None:
      t_start = state["t"]
      x_guess, u_guess = pr.cold_guess(x)
    if state["t"] - t_start >= run_time:
      break
    t0 = time.perf_counter()
    out = c.solve(x, x_guess, u_guess, s_guess, theta, P)
    calls = 1
    while not log and not out["converged"] and calls < MAX_INITIAL_SOLVES:
      out = c.solve(x, out["x"], out["u"], out["slack"], theta, P)
      calls += 1
    solve_ms = 1e3 * (time.perf_counter() - t0)
    if solve_ms < args.min_solve_ms:
      time.sleep((args.min_solve_ms - solve_ms) / 1e3)
    msg = {"u": [float(out["u"][0])], "solve_ms": solve_ms}
    if state.get("pred"):
      msg.update(t_state=state["t"], x_pred=out["x"].tolist(), u_pred=[[float(v)] for v in out["u"]])
    send(conn, msg)
    x_guess, u_guess, s_guess = out["x"], out["u"], out["slack"]
    log.append(
      {
        "t": state["t"] - t_start,
        "skipped": skipped,
        "solve_ms": solve_ms,
        "calls": calls,
        **{k: out[k] for k in ("converged", "iter", "t_total", "status")},
      }
    )
  conn.close()

  ms = np.array([row["solve_ms"] for row in log[1:]]) if len(log) > 1 else np.zeros(1)
  summary = {
    "implementation": f"scaly-{args.form}-{args.solver}",
    "method": args.method,
    "theta": theta.tolist(),
    "compile_s": compile_s,
    "solves": len(log),
    "converged": int(sum(row["converged"] for row in log)),
    "initial_solve_ms": log[0]["solve_ms"] if log else None,
    "solve_ms_mean": float(ms.mean()),
    "solve_ms_max": float(ms.max()),
    "steps": log,
  }
  print(
    f"{summary['implementation']} {args.method}: {len(log)} solves ({summary['converged']} converged), mean {summary['solve_ms_mean']:.2f} ms, max {summary['solve_ms_max']:.2f} ms"
  )
  if args.out:
    args.out.write_text(json.dumps(pr.jsonable(summary)))


if __name__ == "__main__":
  main()
