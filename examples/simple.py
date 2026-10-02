import scaly as sc


@sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.group(sc.arg("y", 2), sc.arg("z", 2)))
def simple(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
  x, y = inputs
  return y, 2 * x
