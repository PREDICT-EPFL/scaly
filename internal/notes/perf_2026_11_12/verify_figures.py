# /// script
# dependencies = ["vl-convert-python"]
# ///
"""Compile and render every metric of the local Vega figures without a browser or server."""

import argparse
import copy
import csv
import json
from pathlib import Path

import vl_convert as vlc


def main(repo, out):
  assets = repo / "docs/assets/benchmarks"
  out.mkdir(parents=True, exist_ok=True)
  config = json.loads((assets / "config.json").read_text())
  dark = {
    "#2a78d6": "#3987e5",
    "#eb6834": "#d95926",
    "#0b0b0b": "#e8e8e6",
    "#52514e": "#c3c2b7",
    "#e1e0d9": "#2a2d35",
    "#c3c2b7": "#4a4d57",
    "#ffffff": "#0b0c0f",
  }
  records = []
  for name in ("compiler_run_values", "compiler_run_ratios"):
    original = json.loads((assets / f"{name}.vl.json").read_text())
    with (assets / original["data"]["url"]).open() as fp:
      rows = list(csv.DictReader(fp))
    for row in rows:
      for key in original["data"]["format"]["parse"]:
        row[key] = float(row[key]) if row[key] else None
    for metric in original["params"][0]["bind"]["options"]:
      for mode in ("light", "dark"):
        spec = copy.deepcopy(original)
        spec["data"] = {"values": rows}
        spec["params"] = [{"name": "measure", "value": metric}]
        spec["config"] = config
        spec["title"] = metric
        spec["background"] = "#ffffff" if mode == "light" else "#0b0c0f"
        if mode == "dark":
          text = json.dumps(spec)
          for a, b in dark.items():
            text = text.replace(a, b.upper())
          spec = json.loads(text)
        filename = f"{name}-{list(original['params'][0]['bind']['options']).index(metric)}-{mode}.png"
        (out / filename).write_bytes(vlc.vegalite_to_png(spec))
        records.append(dict(figure=name, metric=metric, mode=mode, path=filename))
  (out / "verification.json").write_text(json.dumps(records, indent=2) + "\n")
  print(f"Compiled and rendered {len(records)} figures")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("repo", type=Path)
  parser.add_argument("out", type=Path)
  args = parser.parse_args()
  main(args.repo, args.out)
