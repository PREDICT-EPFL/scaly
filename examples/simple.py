import scaly as sc


@sc.function(sc.G(sc.L("x", 2), sc.L("y", 2)), sc.G(sc.L("y", 2), sc.L("z", 2)))
def simple(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
  x, y = inputs
  return y, 2 * x
