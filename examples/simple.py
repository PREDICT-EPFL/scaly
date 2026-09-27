import scaly as sc


@sc.function(2, 2, output=sc.G("y", "z"))
def simple(x: sc.Expr, y: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return y, 2 * x

print(sc.codegen.render_c_module(simple).body)
