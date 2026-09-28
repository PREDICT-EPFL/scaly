from pathlib import Path

import scaly as sc
from scaly.codegen import write_module


@sc.function(sc.L("x", 3), sc.L("energy", ...))
def energy(x: sc.Expr) -> sc.Expr:
  return sc.sumsqr(x)


gen_dir = Path(__file__).parent / "generated"
write_module(energy, gen_dir, lang="c")
write_module(energy, gen_dir, lang="cpp")
