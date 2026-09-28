# How differentiation works

The [derivatives guide](../guide/derivatives.md) shows how to request
derivatives, and [Sparsity](../guide/sparsity.md) shows how to read compact
values. This page shows how Scaly builds them: which automatic differentiation
(AD) mode each request uses, how known zeros reduce the work, and how a mapped
stage function stays mapped through differentiation. The general theory is in
Griewank and Walther's book[^griewank].

## The chain rule, one node at a time

Here is the gradient of \(f(x)=\sin(x^2)\) as Scaly builds it, printed with
`sc.render_expr_assembly`:

```python
@sc.function(sc.L("x", ()), sc.L("y", ...))
def f(x: sc.Expr) -> sc.Expr:
    return (x * x).sin()

print(sc.render_expr_assembly(sc.gradient(f, "y", "x")))
```

```
expr.module {
  expr.func @f_grad_y_x(%x: tensor<float64 diff>) -> (%grad_y_x: tensor<float64 diff>) {
    %0 = expr.const {lowering="auto", value=1.} : tensor<float64>
    %1 = expr.input {lowering="auto", name="x"} : tensor<float64 diff>
    %2 = expr.mul(%1, %1) : tensor<float64 diff>
    %3 = expr.cos(%2) : tensor<float64 diff>
    %4 = expr.mul(%0, %3) : tensor<float64 diff>
    %5 = expr.mul(%4, %1) : tensor<float64 diff>
    %6 = expr.add(%5, %5) : tensor<float64 diff>
    expr.return %6
  }
}
```

A reverse pass starts from the output with weight one, `%0`, and visits the
nodes from the output back to the input. Each operation has a local rule that
adds new nodes. The sine contributes `%4`, the weight times \(\cos(x^2)\). The
product `x * x` contributes that value times `x` once for each operand, and
since both operands are the same node, the two contributions are the same node
`%5`, added to itself. [Interning](architecture.md#properties-of-the-ir) is what
makes them the same node.

The result is an ordinary expression graph, so the usual passes apply before
code generation. Simplification removes the multiplication by one and turns
`%5 + %5` into `2 * %5`, as the one statement of the lowered program shows:

```
prog.store prog.mul(2, prog.mul(prog.cos(prog.mul(%v0, %v0)), %v0)), %grad_y_x[0] : float64
```

## Which mode builds which derivative

Forward mode propagates an input direction \(v\) from the inputs to the outputs
and yields \(Jv\). Reverse mode propagates output weights \(w\) backwards and
yields \(J^Tw\). Every derivative request is built from these two passes:

| Request                                     | Construction                                          |
| ------------------------------------------- | ----------------------------------------------------- |
| `sc.forward`, `sc.jvp`                      | One forward pass                                      |
| `sc.adjoint`, `sc.vjp`, `sc.gradient`       | One reverse pass                                      |
| `sc.jacobian`                               | One batched forward pass seeded with the identity     |
| `sc.hessian`                                | Forward over reverse: the Jacobian of the gradient    |
| `sc.sparse_jacobian`                        | One batched forward pass seeded by column coloring    |
| `sc.sparse_hessian`                         | Forward over reverse, seeded by star coloring         |
| `sc.lagrangian_hessian` and its sparse form | The Hessian of the scalar \(\sigma f+\lambda^Tg\)     |

The Hessian is built *forward over reverse*: one reverse pass gives the
gradient as an expression, and forward mode then differentiates that gradient.
Each forward direction costs a small multiple of one gradient evaluation, so
the number of directions decides the cost[^griewank]. Dense Hessians use one
direction per input, and sparse Hessians use one per color. A Lagrangian
Hessian first forms the weighted sum of all outputs as one scalar, so it needs
a single reverse pass however many constraints there are.

A batched forward pass carries all its directions through the graph together,
as an extra leading axis on every tangent. Work that does not depend on the
direction is then shared. The sine rule computes one cosine for every
direction, and chain-rule steps repeated per direction become small matrix
products. An operation without a batched rule falls back to one pass per
direction, which
[`SCALY_STRICT_JVP_MANY`](../dev/contributing.md#run-the-checks)
turns into an error.

## Finding the pattern

Before building a sparse derivative, Scaly finds which entries can be nonzero.
It walks the graph once and keeps, for every node, a Boolean matrix saying which
input elements each of its elements can depend on. An elementwise operation
takes the union of its operands' rows, a sum merges all rows into one, a gather
picks rows, and a call composes the callee's pattern with its arguments'
patterns. No numbers and no derivatives are involved[^griewank].

```python
@sc.function(sc.L("x", 3), sc.L("y", ...))
def f(x: sc.Expr) -> sc.Expr:
    return sc.stack([x[0] * x[2], x[1] ** 2, 2.0 * x[0] + x[2] ** 2])

x = sc.sym("x", 3)
pattern = sc.jacobian_sparsity(f(x), x)
print(pattern.to_mask().astype(int))
# [[1 0 1]
#  [0 1 0]
#  [1 0 1]]
```

A sparse Hessian uses the same analysis on the gradient's expression graph,
then adds the transpose so that the pattern is symmetric.

## Sharing directions between columns

The Jacobian of `f` above is

\[
J(x)=\begin{bmatrix}
x_2 & 0 & x_0 \\
0 & 2x_1 & 0 \\
2 & 0 & 2x_2
\end{bmatrix}.
\]

Columns 0 and 1 have no row in common, so a single forward direction
\((1,1,0)^T\) gives both of them at once without mixing their entries. Column
coloring finds such groups[^cpr]:

```python
colors = sc.column_coloring(pattern)
print(colors)                   # (0, 0, 1)
print(sc.color_groups(colors))  # ((0, 1), (2,))
```

Each color becomes one column of a seed matrix \(S\), and forward mode computes
the compressed product \(JS\):

\[
S=\begin{bmatrix}1&0\\1&0\\0&1\end{bmatrix},
\qquad
JS=\begin{bmatrix}
x_2 & x_0 \\
2x_1 & 0 \\
2 & 2x_2
\end{bmatrix}.
\]

Evaluating those two directions at \(x=(1,2,3)\) with `sc.forward` gives the
same numbers as the compact values of `sc.sparse_jacobian`:

```python
seeds = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
fwd = sc.forward(f, "y", "x")
x_value = np.array([1.0, 2.0, 3.0])
print(np.stack([fwd((x_value, s)) for s in seeds], axis=1))
# [[3. 1.]
#  [4. 0.]
#  [2. 6.]]

sparse_jac = sc.sparse_jacobian(f, "y", "x")
print(sparse_jac(x_value))  # [3. 1. 4. 2. 6.]
```

Entry \((i,j)\) of \(J\) is entry \((i,\,\text{color}(j))\) of \(JS\). The
recovery is therefore a fixed gather from the compressed product, computed when
the derivative is built:

| Entry of \(J\) | \((0,0)\) | \((0,2)\) | \((1,1)\) | \((2,0)\) | \((2,2)\) |
| -------------- | --------- | --------- | --------- | --------- | --------- |
| Entry of \(JS\) | \((0,0)\) | \((0,1)\) | \((1,0)\) | \((2,0)\) | \((2,1)\) |
| Value          | 3         | 1         | 4         | 2         | 6         |

The coloring is greedy: each column, in order, takes the smallest color not
used by an earlier column that shares a row with it. Finding the fewest colors
is NP-hard[^color], so Scaly settles for this greedy choice. The defect
Jacobian of the
[multiple-shooting example](../guide/getting_started.md#bonus-repeated-stages-with-vmap)
needs three colors whether the horizon has 6 stages or 50. CasADi computes
sparse Jacobians and Hessians in the same way, from a propagated pattern and a
graph coloring[^casadi].

## Star coloring for Hessians

A parameter shared by every stage gives the Hessian an arrow shape, with a
diagonal and one dense row and column:

```python
N = 4

@sc.function(sc.G(sc.L("x", 1), sc.L("p", 1)), sc.L("c", ...))
def stage(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, p = inputs
    return (x * p).sin().sum()

@sc.function(sc.L("w", N + 1), sc.L("cost", ...))
def total(w: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, N, [w[:N], w[N:]]).sum()

hess = sc.sparse_hessian(total, "cost", "w")
pattern = hess.output_sparsities[0]
assert pattern is not None
print(pattern.to_mask().astype(int))
# [[1 0 0 0 1]
#  [0 1 0 0 1]
#  [0 0 1 0 1]
#  [0 0 0 1 1]
#  [1 1 1 1 1]]
print(sc.column_coloring(pattern))  # (0, 1, 2, 3, 4)
print(sc.star_coloring(pattern))    # (0, 0, 0, 0, 1)
```

Column coloring needs a color per column, because every column meets the last
row. With 50 stages it needs 51 directions. Star coloring needs two for any
number of stages. It uses symmetry: an entry the compressed product cannot
separate in one place can be read at its mirror position.

With the star coloring, \(S\) groups the four stage columns and leaves the
parameter column alone:

\[
H=\begin{bmatrix}
a_0 & & & & b_0 \\
& a_1 & & & b_1 \\
& & a_2 & & b_2 \\
& & & a_3 & b_3 \\
b_0 & b_1 & b_2 & b_3 & c
\end{bmatrix},
\qquad
HS=\begin{bmatrix}
a_0 & b_0 \\
a_1 & b_1 \\
a_2 & b_2 \\
a_3 & b_3 \\
b_0+b_1+b_2+b_3 & c
\end{bmatrix}.
\]

The last row of the first column mixes four entries, so \(b_k\) cannot be read
there. It is read from row \(k\) of the second column instead, where it stands
alone. In general, entry \((i,j)\) is read from row \(i\) of column
\(\text{color}(j)\) when no other column of that color has an entry in row
\(i\), and otherwise from row \(j\) of column \(\text{color}(i)\). Star
coloring is the condition that makes one of the two always work: the coloring
is proper, and no path of three edges in the pattern's graph uses only two
colors[^cm]. Treating \(H\) as an ordinary Jacobian instead would need a
distance-2 coloring, which is the column coloring above[^color].

## Mapped derivatives

A mapped call is one stage function applied over slices of its inputs.
Differentiating it produces a mapped call of the stage function's derivative,
over the same slices, instead of a copy per stage. Scaly caches each stage
derivative function per callee, so repeated requests reuse it.

### Forward mode

Here is the sparse Jacobian of the mapped residual from
[Sparsity](../guide/sparsity.md#sparse-derivatives-and-mapped-structure), with
`N = 4`:

```python
stage_jac = sc.sparse_jacobian(stages, "residuals", "zs")
print(sc.render_expr_assembly(stage_jac))
```

The outer function of the printed module is one mapped call and one gather
(the stage derivative function is omitted):

```
  expr.func @stages_spjac_residuals_zs(%zs: tensor<8xfloat64 diff>) -> (%spjac_residuals_zs: tensor<16xfloat64 diff>) {
    %0 = expr.input {lowering="auto", name="zs"} : tensor<8xfloat64 diff>
    %1 = expr.vmap(%0) {callee="stage_fwd2c9dba5c50e0_residual_z", length=4, output=0, slice_size=4, starts=[0], strides=[2]} : tensor<16xfloat64 diff>
    %2 = expr.gather(%1) {indices=[ 0,  2,  1, ..., 14, 13, 15]} : tensor<16xfloat64 diff>
    expr.return %2
  }
```

Scaly colors the stage's own 2 by 2 pattern rather than the 8 by 8 global
one. In the callee's name, `fwd2` counts the two seed directions and the
hexadecimal part after `c` identifies them, so derivatives of the same stage
with different seeds get different names. The
stage derivative takes only `z`, and its two seed directions are constants
inside its body, where the multiplications by zero and one fold away. The
gather puts the four values each stage returns into pattern order.

This path applies when each mapped input either is the differentiation
variable itself or does not depend on it. When the mapped inputs are slices of
it, as in the
[multiple-shooting example](../guide/getting_started.md#bonus-repeated-stages-with-vmap),
Scaly colors the global pattern instead. The derivative is still one mapped
call of a stage derivative with constant seeds, and the number of directions
is the one the global coloring needs.

### Reverse mode

Reverse mode builds one stage adjoint function and maps it over the stages.
Each stage returns the adjoint of its input slice, and the outer function adds
these into the full input. How it adds them depends on how the slices overlap.
This cost links neighbouring points of a trajectory `z` with a window of two
values and stride one, and shares a weight `p` between all stages:

```python
N = 4

@sc.function(sc.G(sc.L("zz", 2), sc.L("p", 1)), sc.L("c", ...))
def link(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    zz, p = inputs
    return p[0] * (zz[1] - zz[0].sin()) ** 2

@sc.function(sc.G(sc.L("z", N + 1), sc.L("p", 1)), sc.L("cost", ...))
def chain(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    return sc.vmap(link, N, [(z, 0, 1), p]).sum()
```

The windows over `z` overlap:

```
z index     0     1     2     3     4
stage 0   [zz0   zz1]
stage 1         [zz0   zz1]
stage 2               [zz0   zz1]
stage 3                     [zz0   zz1]
```

Scaly splits the stages into groups whose windows do not overlap, here the
even stages and the odd stages, scatters each group into a vector the size of
`z`, and adds the groups. The outer function of `sc.gradient(chain, "cost",
"z")` shows this, with the stage adjoint `link_adj0_0` omitted:

```
  expr.func @chain_grad_cost_z(%z: tensor<5xfloat64 diff>, %p: tensor<1xfloat64 diff>) -> (%grad_cost_z: tensor<5xfloat64 diff>) {
    %0 = expr.input {lowering="auto", name="z"} : tensor<5xfloat64 diff>
    %1 = expr.input {lowering="auto", name="p"} : tensor<1xfloat64 diff>
    %2 = expr.const {lowering="auto", value=1.} : tensor<float64>
    %3 = expr.const {lowering="auto", value=[1., 1., 1., 1.]} : tensor<4xfloat64>
    %4 = expr.mul(%2, %3) : tensor<4xfloat64>
    %5 = expr.vmap(%0, %1, %4) {callee="link_adj0_0", length=4, output=0, slice_size=2, starts=[0, 0, 0], strides=[1, 0, 1]} : tensor<8xfloat64 diff>
    %6 = expr.gather(%5) {indices=[0, 1, 4, 5]} : tensor<4xfloat64 diff>
    %7 = expr.scatter(%6) {indices=[0, 1, 2, 3]} : tensor<5xfloat64 diff>
    %8 = expr.gather(%5) {indices=[2, 3, 6, 7]} : tensor<4xfloat64 diff>
    %9 = expr.scatter(%8) {indices=[1, 2, 3, 4]} : tensor<5xfloat64 diff>
    %10 = expr.add(%7, %9) : tensor<5xfloat64 diff>
    expr.return %10
  }
```

`%6` holds the adjoints from stages 0 and 2, and `%8` those from stages 1 and 3.
For the shared weight, every stage reads the same slice, so the adjoint is the
sum over stages. The outer function of `sc.gradient(chain, "cost", "p")` writes
that sum as a product with a vector of ones:

```
  expr.func @chain_grad_cost_p(%z: tensor<5xfloat64 diff>, %p: tensor<1xfloat64 diff>) -> (%grad_cost_p: tensor<1xfloat64 diff>) {
    %0 = expr.const {lowering="auto", value=[1., 1., 1., 1.]} : tensor<4xfloat64>
    %1 = expr.input {lowering="auto", name="z"} : tensor<5xfloat64 diff>
    %2 = expr.const {lowering="auto", value=1.} : tensor<float64>
    %3 = expr.mul(%2, %0) : tensor<4xfloat64>
    %4 = expr.vmap(%1, %3) {callee="link_adj0_1", length=4, output=0, slice_size=1, starts=[0, 0], strides=[1, 1]} : tensor<4xfloat64 diff>
    %5 = expr.gather(%4) {indices=[0, 1, 2, 3]} : tensor<4xfloat64 diff>
    %6 = expr.reshape(%5) {shape=[4, 1]} : tensor<4x1xfloat64 diff>
    %7 = expr.matmul(%0, %6) : tensor<1xfloat64 diff>
    expr.return %7
  }
```

The three cases, for a slice of size \(n\) read with stride \(s\):

| Slices      | Stride            | Accumulation                                             |
| ----------- | ----------------- | -------------------------------------------------------- |
| Disjoint    | \(s \ge n\)       | One scatter                                              |
| Shared      | \(s = 0\)         | A sum over stages                                        |
| Overlapping | \(0 < s < n\)     | \(\lceil n/s\rceil\) groups of disjoint stages, added    |

An ordinary call is treated differently. Reverse mode differentiates the callee
and substitutes the result into the caller, so the adjoint of a call is inlined.
Only mapped calls get a separate stage adjoint function.

## Non-smooth and unsupported derivatives

The operations without derivative rules, and the zero returned through a solver
call, are listed in the guide's
[current limitations](../guide/derivatives.md#current-limitations).

[^griewank]: Andreas Griewank and Andrea Walther, *Evaluating Derivatives:
    Principles and Techniques of Algorithmic Differentiation*, 2nd edition,
    SIAM, 2008. [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)

[^cpr]: A. R. Curtis, M. J. D. Powell and J. K. Reid, "On the estimation of
    sparse Jacobian matrices", *IMA Journal of Applied Mathematics* 13(1),
    117–119, 1974. [doi:10.1093/imamat/13.1.117](https://doi.org/10.1093/imamat/13.1.117)

[^color]: Assefaw Hadish Gebremedhin, Fredrik Manne and Alex Pothen, "What color
    is your Jacobian? Graph coloring for computing derivatives", *SIAM Review*
    47(4), 629–705, 2005.
    [doi:10.1137/S0036144504444711](https://doi.org/10.1137/S0036144504444711)

[^cm]: Thomas F. Coleman and Jorge J. Moré, "Estimation of sparse Hessian
    matrices and graph coloring problems", *Mathematical Programming* 28(3),
    243–270, 1984. [doi:10.1007/BF02612334](https://doi.org/10.1007/BF02612334)

[^casadi]: Joel A. E. Andersson, Joris Gillis, Greg Horn, James B. Rawlings and
    Moritz Diehl, "CasADi: a software framework for nonlinear optimization and
    optimal control", *Mathematical Programming Computation* 11(1), 1–36, 2019.
    [doi:10.1007/s12532-018-0139-4](https://doi.org/10.1007/s12532-018-0139-4)
