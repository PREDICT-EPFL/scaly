import json, sys
from collections import Counter
from pathlib import Path
S = Path(__file__).resolve().parent
od, nd = (S / sys.argv[1], S / sys.argv[2]) if len(sys.argv) > 2 else (S / "fuzz_old", S / "fuzz_new")
verbose = "-v" in sys.argv
tot = Counter()
for p in sorted(nd.glob("*.json")):
  if not (od / p.name).exists():
    continue
  o, n = json.loads((od / p.name).read_text()), json.loads(p.read_text())
  assert "/old/" in o["kkt_file"] or od.name != "fuzz_old", o["kkt_file"]
  c = Counter()
  lines = []
  for a, b in zip(o["rows"], n["rows"], strict=True):
    assert a["k"] == b["k"] and a["kind"] == b["kind"]
    c["instances"] += 1
    if (a["status"], a["iter"], a["xhash"]) == (b["status"], b["iter"], b["xhash"]):
      c["identical"] += 1
      continue
    c["differ"] += 1
    oa, ob = a["status"] == 1, b["status"] == 1
    if a["status"] != b["status"]:
      tag = "NEW LOSES STATUS" if oa and not ob else "new gains status" if ob and not oa else "status differs (neither solved)"
    elif b["iter"] > a["iter"]:
      tag = "NEW SLOWER"
    elif b["iter"] < a["iter"]:
      tag = "new faster"
    else:
      tag = "same count, other x"
    c[tag] += 1
    lines.append(f"   k {a['k']:4d} {a['kind']:9s} {tag:32s} status {a['status']:3d}/{b['status']:3d} iter {a['iter']:3d}/{b['iter']:3d} ir {a['ir']:.0f}/{b['ir']:.0f} obj {a['obj']:.9g}/{b['obj']:.9g}")
  tot.update(c)
  print(f"{o['base']:28s} n/p/m {o['n']}/{o['p']}/{o['m']}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
  if verbose or True:
    for line in lines:
      if verbose or "NEW" in line:
        print(line)
print("TOTAL: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
