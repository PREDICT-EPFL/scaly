"""Interpolant evaluation, one point per call, every search and strategy, against CasADi, timed in C.

    uv run internal/notes/perf_2026_09_28_interp/bench_eval.py [--rounds 7] [--reps 300] [--only 1d|2d|3d]

Each configuration is value and derivative (gradient in n-D) of `interp.interpolant` at one point,
built once per `search` and `strategy` the axis allows, plus the choice `"auto"` makes; and CasADi
3.8's `interpolant` (`linear`, or `bspline` for the cubics) code-generated at each `lookup_mode` it
accepts, the fastest kept and named (`docs/results/fairness.md`). Configurations:

- 1-D: `linear` and `cubic`, 32 and 1 024 sites, uniform and clustered;
- 2-D: a 64 x 64 uniform bicubic;
- 3-D: a 20 x 20 x 20 uniform trilinear (value only, as CasADi's `linear` has no cheaper mode for it);
- SP2's kinds on 1 024 clustered sites: `pchip`, `akima`, `steffen` (no CasADi counterpart) and
  `smooth_linear` (CasADi's `bspline` with `algorithm="smooth_linear"`); and `inverse()` of the
  `pchip` curve (safeguarded Newton; the forward evaluation of the same curve is its reference).

Values are checked against SciPy before timing (CasADi's bspline fit is compared at 1e-9: it misses
its own data by more than rounding, see `tests/interp/test_casadi_parity.py`).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from _harness import BUILD, ROOT, build_casadi, build_scaly, time_cell

OUT = BUILD / "eval"


def clustered(n: int) -> np.ndarray:
  u = np.linspace(0.0, 1.0, n)
  return u + 0.35 * np.sin(np.pi * u) ** 2 * (u - 0.5)


def configurations() -> list[dict]:
  rng = np.random.default_rng(0)
  out = []
  for kind in ("linear", "cubic"):
    for n in (32, 1024):
      for spacing in ("uniform", "clustered"):
        g = np.arange(n) / (n - 1.0) if spacing == "uniform" else clustered(n)
        if spacing == "uniform":
          g = np.arange(n) / 1024.0  # exactly equally spaced, which CasADi's "exact" lookup demands
        out.append({"name": f"1d_{kind}_{n}_{spacing}", "dims": 1, "grid": (g,), "values": np.sin(6 * g / g[-1]) + 0.1 * rng.normal(size=n), "kind": kind, "point": np.array([0.4137 * g[-1]])})
  g2 = np.arange(64) / 64.0
  out.append({"name": "2d_cubic_64", "dims": 2, "grid": (g2, g2), "values": np.sin(3 * g2)[:, None] * np.cos(2 * g2)[None, :] + rng.normal(scale=0.01, size=(64, 64)), "kind": "cubic", "point": np.array([0.3712, 0.6093])})
  g3 = np.arange(20) / 16.0
  out.append({"name": "3d_linear_20", "dims": 3, "grid": (g3, g3, g3), "values": rng.normal(size=(20, 20, 20)), "kind": "linear", "point": np.array([0.371, 0.911, 0.107])})
  g = clustered(1024)
  monotone = np.cumsum(rng.uniform(0.0, 1.0, 1024))
  for kind in ("pchip", "akima", "steffen", "smooth_linear"):
    out.append({"name": f"1d_{kind}_1024_clustered", "dims": 1, "grid": (g,), "values": monotone, "kind": kind, "point": np.array([0.4137])})
  out.append({"name": "1d_inverse_pchip_1024", "dims": 1, "grid": (g,), "values": monotone, "kind": "pchip", "inverse": True, "point": np.array([0.4137 * monotone[-1]])})
  return out


def scaly_variants(cfg: dict) -> dict[str, object]:
  import scaly as sc
  from scaly import interp

  grid = cfg["grid"] if cfg["dims"] > 1 else cfg["grid"][0]
  auto = interp.interpolant(grid, cfg["values"], kind=cfg["kind"])
  if cfg.get("inverse"):
    point = sc.sym("y")
    inv = auto.inverse()
    x = inv(point)
    return {"fns": {"auto": sc.Function._from_exprs(f"b_{cfg['name']}", [point], [x, sc.gradient(x, point)], ["y"], ["x", "g"])}, "auto": "newton"}
  variants = {"auto": auto}
  searches = ["count", "binary"] + (["uniform"] if all(ax.search == "uniform" or ax._uniform is not None for ax in auto.axes) else [])
  for search in searches:
    if search == "count" and max(ax.cells for ax in auto.axes) > 256:
      continue
    for strategy in ("pp", "basis"):
      variants[f"{search}/{strategy}"] = interp.interpolant(grid, cfg["values"], kind=cfg["kind"], search=search, strategy=strategy, name=f"{cfg['name']}_{search}_{strategy}")
  fns = {}
  for label, f in variants.items():
    point = sc.sym("x", () if cfg["dims"] == 1 else (cfg["dims"],))
    y = f(point)
    outs = [y] if cfg["name"].startswith("3d") else [y, sc.gradient(y, point)]
    fns[label] = sc.Function._from_exprs(f"b_{cfg['name']}_{label.replace('/', '_')}", [point], outs, ["x"], ["y", "g"][: len(outs)])
  return {"fns": fns, "auto": f"{auto.axes[0].search}/{auto.strategy}"}


def casadi_variants(cfg: dict) -> dict[str, object]:
  import casadi as ca

  if cfg["kind"] not in ("linear", "cubic", "smooth_linear") or cfg.get("inverse"):
    return {}
  method = "linear" if cfg["kind"] == "linear" else "bspline"
  modes = ["linear", "binary"] + (["exact"] if method == "linear" and "uniform" in cfg["name"] or cfg["dims"] == 3 else [])
  if cfg["dims"] == 2:
    modes = ["linear", "binary"]
  extra = {"algorithm": "smooth_linear"} if cfg["kind"] == "smooth_linear" else {}
  out = {}
  for mode in modes:
    opts = {"lookup_mode": [mode] * cfg["dims"], **extra}
    itp = ca.interpolant(f"itp_{mode}", method, [list(g) for g in cfg["grid"]], np.ravel(cfg["values"], order="F"), opts)
    x = ca.MX.sym("x", cfg["dims"])
    y = itp(x)
    outs = [y] if cfg["name"].startswith("3d") else [y, ca.jacobian(y, x)]
    out[mode] = ca.Function(f"ca_{cfg['name']}_{mode}", [x], outs)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--reps", type=int, default=300)
  parser.add_argument("--only", choices=["1d", "2d", "3d"])
  args = parser.parse_args()
  cfgs = [c for c in configurations() if args.only is None or c["name"].startswith(args.only)]
  cells: dict[str, dict[str, dict]] = {}
  autos = {}
  for cfg in cfgs:
    sc_side, ca_side = scaly_variants(cfg), casadi_variants(cfg)
    autos[cfg["name"]] = sc_side["auto"]
    cells[cfg["name"]] = {}
    for label, fn in sc_side["fns"].items():
      cells[cfg["name"]][f"scaly:{label}"] = build_scaly(fn, [cfg["point"]], OUT / cfg["name"] / f"scaly_{label.replace('/', '_')}")
    for mode, fn in ca_side.items():
      cells[cfg["name"]][f"casadi:{mode}"] = build_casadi(fn, [cfg["point"]], OUT / cfg["name"] / f"casadi_{mode}")
  best: dict[tuple[str, str], float] = {}
  for _ in range(args.rounds):
    for cfg in cfgs:
      expected = None
      for label, meta in cells[cfg["name"]].items():
        folder = OUT / cfg["name"] / label.replace(":", "_").replace("/", "_")
        t, values = time_cell(folder, meta, args.reps, 200)
        best[cfg["name"], label] = min(best.get((cfg["name"], label), np.inf), t)
        if label.startswith("scaly"):
          expected = np.load(folder / "expected.npy") if expected is None else expected
          np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12, err_msg=f"{cfg['name']} {label}")
        else:
          assert expected is not None
          np.testing.assert_allclose(values, expected, rtol=1e-9, atol=1e-9, err_msg=f"{cfg['name']} {label}")
  print("| configuration | Scaly best ns (variant) | auto ns (choice) | auto/best | CasADi ns (lookup) | best/CasADi | C lines auto / CasADi |")
  print("| --- | --- | --- | --- | --- | --- | --- |")
  rows = []
  for cfg in cfgs:
    name = cfg["name"]
    scaly = {k.split(":", 1)[1]: v for (n, k), v in best.items() if n == name and k.startswith("scaly")}
    casadi = {k.split(":", 1)[1]: v for (n, k), v in best.items() if n == name and k.startswith("casadi")}
    top = min((k for k in scaly if k != "auto"), key=scaly.__getitem__, default="auto")
    ca_top = min(casadi, key=casadi.__getitem__, default=None)
    ca_lines = cells[name][f"casadi:{ca_top}"]["c_lines"] if ca_top else "—"
    ca_cell = f"{casadi[ca_top]:.1f} ({ca_top}) | {scaly[top] / casadi[ca_top]:.2f}" if ca_top else "— | —"
    print(
      f"| {name} | {scaly[top]:.1f} ({top}) | {scaly['auto']:.1f} ({autos[name]}) | {scaly['auto'] / scaly[top]:.2f} | {ca_cell} | {cells[name]['scaly:auto']['c_lines']} / {ca_lines} |"
    )
    rows.append({"configuration": name, "scaly_ns": scaly, "casadi_ns": casadi, "auto": autos[name], "meta": cells[name]})
  (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
