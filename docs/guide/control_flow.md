# Control flow

A Python loop inside an `@sc.function` body runs when the function is traced. Each iteration adds
another copy of its expressions to the graph. `sc.scan` and `sc.while_loop` instead record a loop
whose body is a Scaly `Function`. The generated code repeats that body when the function is
evaluated, keeping its size independent of the number of steps. Storage and evaluation work still
grow with the number of steps.

## A fixed number of steps

`scan` threads a carry through a sequence of calls and stacks the outputs from each step. For
example, a double integrator has state $x_k = (p_k, v_k)$ and control $u_k$:

\[
x_{k+1} = x_k + 0.1\begin{bmatrix}v_k \\ u_k\end{bmatrix}.
\]

```python
import numpy as np
import scaly as sc

@sc.function(sc.arg("x", 2), sc.arg("u", ()))
def model(x: sc.Expr, u: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    xnext = x + 0.1 * sc.stack([x[1], u])
    return xnext, sc.sumsqr(x)

@sc.function(sc.arg("x0", 2), sc.arg("controls", 20),
             outputs=sc.group(sc.arg("final", 2), sc.arg("cost", ())))
def rollout(x0: sc.Expr, controls: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    final, (costs,) = sc.scan(model, x0, (controls,), length=20)
    return final, costs.sum()

final, cost = rollout(np.array([1.0, 0.0]), np.zeros(20))
print(final, cost)
# [1. 0.] 20.0
```

The body takes its carry first and returns the next carry first. Further results become stacked
outputs. Here `costs` has shape `(20,)`. A body may return only its carry, in which case the second
result of `scan` is an empty tuple. A zero-length scan returns its initial carry and empty stacked
outputs.

Each entry of `xs`, the third argument to `scan`, supplies one body parameter. Its leading axis
indexes the steps, so controls of shape `(20, 2)` supply a two-element control at each step.
`sc.broadcast(parameter)` passes a whole parameter to every step. `sc.window(outer, start, stride)`
reads contiguous windows of a flat array, with the window width taken from the body's parameter
declaration. These are the same argument rules as [mapped calls](functions.md#regular-repetition-vmap).

For several carries, pass a tuple as `init` and declare one body parameter for each entry. The body
returns the next carries before its per-step outputs. Each carry keeps its shape, dtype and sparse
pattern throughout the loop. Carries may contain nested tuples.

With `index=True`, the body takes an int64 scalar step number after its carries and before its
sliced inputs. `reverse=True` executes steps in descending order. Step `k` still reads input slice
`k` and writes stacked output `k`, so reversing execution does not reverse the output layout.

`trip=` accepts an int64 scalar expression for a run-time step count, clamped to `[0, length]`.
Unused stacked entries are zero. A reverse scan starts at this clamped count minus one. The shapes
stay fixed at the static `length`, regardless of the value of `trip`.

## A condition that stops the loop

`while_loop` checks its condition before each step and stops at the first false result, up to a
fixed `max_iter`. The condition and body are Functions with the same inputs. Parameters in
`params=` are passed whole to each step. For example, Newton iteration approximates $\sqrt{a}$:

```python
@sc.function(sc.arg("x", ()), sc.arg("a", ()),
             outputs=sc.arg("go", sc.TensorType((), sc.dtypes.bool_)))
def unsolved(x: sc.Expr, a: sc.Expr) -> sc.Expr:
    return (x * x - a).abs() > 1e-12

@sc.function(sc.arg("x", ()), sc.arg("a", ()))
def newton(x: sc.Expr, a: sc.Expr) -> sc.Expr:
    return 0.5 * (x + a / x)

@sc.function(sc.arg("a", ()), outputs=sc.arg("root", ()))
def root(a: sc.Expr) -> sc.Expr:
    value, count = sc.while_loop(unsolved, newton, sc.const(1.0),
                               max_iter=30, params=(a,))
    return value

print(root(np.array(2.0)))
# 1.414213562373095
```

The body returns only its next carries. `while_loop` returns the final carry and the int64 count
of executed steps. If the condition is false at entry, or `max_iter=0`, the result is the initial
carry and the count is zero. Reaching `max_iter` returns the current carry. It does not signal
convergence or raise an error. The condition can inspect carries and parameters, including the
step number with `index=True`. Intermediate quantities needed by the next condition must be part
of the carry.

## Derivatives and trajectory storage

Both loops support forward and reverse derivatives. They differentiate the calculation along the
steps taken by the primal loop. A condition and a trip count choose the executed steps and have no
derivative. The derivative holds the number of steps fixed. It does not see the jump when a small
input change adds or removes a step.

```python
control_gradient = sc.gradient(rollout, "cost", "controls")
print(control_gradient(np.array([1.0, 0.0]), np.zeros(20)).shape)
# (20,)

root_derivative = sc.gradient(root, "root", "a")
root_second_derivative = sc.hessian(root, "root", "a")
print(root_derivative(np.array(2.0)))
# 0.35355339059327373
print(root_second_derivative(np.array(2.0)))
# [[-0.08838835]]
```

Reverse mode stores the carries entering each step, then traverses exactly those steps backwards.
A function returning both the value and its gradient runs the loop forwards once and reuses the
stored values in its backward pass. Forward over reverse computes second derivatives through the
same executed steps, as in `root_second_derivative` above.

`max_trajectory=` limits the primal trajectory requested by reverse mode for one user loop. It counts
stored scalar elements across
its carries, using the static `length` or `max_iter`. For example, a four-element carry at
`length=200` needs 800 trajectory elements. Stacked outputs do not count toward this limit. Each
nested loop has its own limit, so the limit does not bound the product of their storage. The
default `None` leaves it unlimited. Exceeding a limit raises `ValueError` and names
[`sc.custom_derivative`](../api/functions.md#custom-derivative-rules), which can supply a derivative
that avoids storing the full trajectory. Tangent and backward loops created by differentiation
do not inherit this limit. Higher derivatives can therefore require additional storage.

Forward mode propagates one direction per seeded input element through every step. It does not
compress those directions into a smaller Jacobian for each step. For a Jacobian with respect to a
control sequence, seed buffers and generated constants grow with the horizon. Use reverse mode
for scalar costs differentiated with respect to per-step inputs.

## Printing inside a loop

`sc.print("x={}", x)` returns `x` and prints when generated code computes that value. A print in a
loop body runs once per executed step. A zero-trip loop does not execute its body prints. Use the
returned expression in the body's outputs. An unused print raises `ValueError` while the function
is traced, because the print does not reach its outputs.

A reverse gradient can print the step values again in reverse order when its partial derivatives
read those values:

```python
@sc.function(sc.arg("x", ()))
def printed_step(x: sc.Expr) -> sc.Expr:
    value = sc.print("x={}", x)
    return value * value + 1.0

@sc.function(sc.arg("initial", ()), outputs=sc.arg("final", ()))
def printed_rollout(initial: sc.Expr) -> sc.Expr:
    return sc.scan(printed_step, initial, length=3)[0]

printed_gradient = sc.gradient(printed_rollout, "final", "initial")
print(printed_gradient(np.array(1.0)))
# x=1
# x=2
# x=5
# x=5
# x=2
# x=1
# 80.0
```

The general [printing rules](functions.md#printing-from-generated-code) cover shared prints and
ordering. The [API reference](../api/functions.md#control-flow) lists the loop signatures.
