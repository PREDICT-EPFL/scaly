import scaly as sc


@sc.function(sc.arg("x", 2), sc.arg("y", 2), outputs=sc.group(sc.arg("y", 2), sc.arg("z", 2)))
def simple(x: sc.Expr, y: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return y, 2 * x
