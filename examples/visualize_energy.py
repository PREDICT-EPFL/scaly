import scaly as sc
from scaly.codegen import render_c_module
from scaly.viz import visualize, recordings

@sc.function(sc.arg("x", 3), outputs=sc.arg("energy"))
def energy(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x)

visualize(energy, label="Energy model")
render_c_module(energy)
print(recordings()[-1]["name"])  # Energy model
