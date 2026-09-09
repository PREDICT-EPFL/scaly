# C-49: where the chain stage's operations go, rule by rule

Diagnosis only, 2026-09-09. Nothing in `src/` changed; every variant below was applied as an
in-memory source substitution and measured on the scalarized stage procedure at M=5.

## Summary

The chain stage kernel (`chain_eq_stage_M5_adj0_0_1_fwd24c…_adj_eq_z`: the adjoint of one RK4
stage against `lam`, then 24 unit-seed tangents of that adjoint, seeds folded) costs 28,682
arithmetic operations on the current tree, 1,142 per seed above the 1,271-operation adjoint.
CasADi SX doing the same explicit forward-over-reverse with 24 unit seeds costs 17,969.

The split, measured by building the same stage three ways and by swapping rules one at a time:

| Cause | Operations | Share of 28,682 | Fix |
|---|---:|---:|---|
| Callee boundary: link adjoint duplicated across neighbouring mass bodies | ~6,100 | 21% | inline small callees before differentiating |
| Per-formal split of the JVP through the VMAP'd adjoint callee (`_jvp` CALL/VMAP rules) | 5,734 | 20% | one joint JVP callee per active-formal set |
| Scalar rule quality (negations through `*` and `/`, `x*x` and `dot(x,x)`, DIV, SQRT) | 2,055 to 4,337 | 7 to 15% | rule edits listed at the end |
| Forward-over-reverse of an inlined graph with today's rules | 16,826 | | this is what SX does; SX needs 17,969 |

Rule quality is the smaller half. The larger half is composition: the same graph with no CALL or
VMAP nodes costs 16,826 (658 per seed), already below SX, and 14,771 (573 per seed) with the rule
edits. The "at most about 400 per seed in SX" in the todo was optimistic: SX's explicit
forward-over-reverse is 700 per seed, its one-shot `ca.hessian` 633, and the `nlp_hess_l` the harness
benchmarks, which emits only the upper triangle, about 500. Alloy computes the full 24 × 45 block.

The lean division rules from the 2026-09-08 follow-up were never landed: `git log -S"q * expr"`
finds nothing and `forward.py`/`reverse.py` still carry `(dx*y - x*dy)/y**2` and `-cot*x/y**2`. The
follow-up's 2,749 negations per stage is exactly what the lean DIV rule alone produces (reproduced
below); it is cured by the negation identity, not by touching DIV again.

## Method

Stage under test: `_eq_stage_fn(5)` from `benchmarks/problems/chain/__init__.py`, differentiated the
way `sparse_hessian` does it for the mapped stage: `R._vmap_adj_function(stage, 0, (0, 1))` for the
adjoint callee, then `F._call_jvp_many_const_function(adj_fn, 0, 0, np.eye(24))` for the folded
unit-seed tangents. Lowering the resulting `Function` with `alloy.codegen.aot._lower` and counting
unique `ProgramOp` arithmetic nodes in the scalarized procedure gives the same 28,682 as lowering the
whole `chain_nlp(5, 40).descriptor.hess` (checked; that build takes 33 s, the stage-only build 5 s).

The same stage was then built two more ways, numerically identical to the kernel above
(max abs difference 3e-17 on random inputs, checked with `check_equal.py` below):

- **inlined**: plain expressions, no `CALL` or `VMAP` (the mass body written out per mass), one
  `vjp` over the whole graph, then `_jvp_many_unrolled` with the constant eye seeds;
- **callode** / **callaccel** / **vmapaccel**: the same graph with only the ODE as a `CALL`, only the
  mass body as three `CALL`s, or only the mass body as a `VMAP` of length 3.

SX: `chain._ca_step` on `SX` symbols, `g = ca.gradient(dot(lam, step - xnext), z)`,
`ca.jtimes(g, z, SX.eye(24))`, wrapped in a `Function` with `cse`, and its instruction histogram.

Rule variants were applied by text substitution on the module sources and `exec` into the module
namespaces (`variants.py` below), so a variant is a literal diff an implementer can copy.

## Reference numbers, SX, one stage at M=5

| SX construction | Arithmetic | Histogram |
|---|---:|---|
| primal `step` | 543 | mul 149, add 185, sub 88, sq 48, twice 39, div 18, sqrt 16 |
| primal + adjoint (`gradient`) | 1,166 | mul 443, add 371, sub 145, twice 85, div 58, sq 48, sqrt 16 |
| forward-over-reverse, 24 unit seeds, `cse` | 17,969 | mul 8,735, add 4,937, sub 2,572, div 860, twice 611, neg 190, sq 48, sqrt 16 |
| same without `cse` | 20,904 | |
| `ca.hessian` one shot | 16,367 | mul 8,063, add 4,208, sub 2,395, div 836, twice 611, neg 190 |
| reverse-over-reverse, 24 seeds | 19,080 | |
| harness `nlp_hess_l`, per stage (upper triangle, includes cost) | 13,213 | mul 6,633, add 3,193, sub 1,603, div 762, twice 706, neg 252 |

So SX's explicit forward-over-reverse is (17,969 − 1,166) / 24 = 700 per seed; the todo's 400 came
from the follow-up's 10,500 figure, which this measurement does not reproduce (13,213 with the
harness's exact construction).

## Alloy, one stage at M=5, current tree

| Build | Adjoint only | Adjoint + 24 tangents | Per seed |
|---|---:|---:|---:|
| real stage (VMAP of mass body inside CALL of ODE) | 1,271 | **28,682** | 1,142 |
| vmapaccel (mass body as VMAP, ODE inlined) | 1,250 | 28,682 | 1,143 |
| callaccel (mass body as 3 CALLs, ODE inlined) | 1,250 | 22,558 | 888 |
| callode (ODE as CALL, mass body inlined) | 1,034 | 16,994 | 665 |
| inlined (no CALL, no VMAP) | 1,034 | **16,826** | 658 |
| SX forward-over-reverse | 1,166 | 17,969 | 700 |

Histogram of the real stage against SX forward-over-reverse:

| Op | Alloy | SX | Excess |
|---|---:|---:|---:|
| mul | 15,603 | 8,735 (+611 `twice`, +48 `sq`) | +6,209 |
| add | 7,388 | 4,937 | +2,451 |
| sub | 2,392 | 2,572 | −180 |
| div | 1,800 | 860 | +940 |
| neg | 1,483 | 190 | +1,293 |
| sqrt | 16 | 16 | 0 |

Two things follow from the build table. The ODE `CALL` costs 168 operations (1%): a callee with one
active formal is free. The mass body costs 11,856 whether it is a `VMAP` (28,682) or, after the
per-formal split is removed, three `CALL`s (22,558): the composition loss is about *how the mass body
is a callee at all*, plus a VMAP-only part.

## Composition, part one: the JVP through a callee is split per formal

`_jvp`'s `CALL` and `VMAP` rules (`forward.py` lines 72 to 112) take one directional derivative per
active formal, `_call_jvp_function(callee, output, formal_idx)`, and sum the results. The adjoint
callee of the mass body, `chain_mass_accel_adj0_0_1_2_3_4_5_6`, has four active formals in the
Hessian path (`left`, `pos`, `right`, `lam`), so its tangent is computed as four separate bodies
(`…_fwd_adj:accel_left` 105 ops, `_pos` 204, `_right` 99, `_lam:accel` 113, from the lowered
procedure list) where one body seeded on all four at once is 201 ops. Every shared sub-graph, the
norm and its derivative above all, is walked once per formal. `_jvp_many_structural`'s `CALL` and
`VMAP` branches (lines 366 to 511) have the same shape; the Hessian path reaches the split through
`_call_jvp_many_const_function -> _jvp_many_unrolled -> jvp -> _jvp`.

Measured with the `joint` variant (one JVP callee per `(output, active formal set)`, seeded jointly;
the implementation sketch is in `variants.py` below):

| Variant | Total | Per seed |
|---|---:|---:|
| base | 28,682 | 1,142 |
| joint | 22,948 | 903 |
| joint + all four rule edits | 19,946 | 780 |

Saving: 5,734 operations, 20%, from a change to two rules and one cache. The primal callee is not
affected (callaccel base and callaccel+joint are byte-identical after scalarization) because the
reverse `CALL` rule already grafts the adjoint inline through `_substitute`, so the split only bites
where the adjoint stays a callee, which is exactly the `VMAP` adjoint (`_vmap_vjp`).

## Composition, part two: a callee boundary blocks merging the shared link

What remains between callaccel (22,558) and inlined (16,826) is 5,732 operations, and the adjoint
alone shows it at first order (1,250 against 1,034). The CALL-mode adjoint procedure makes the
mechanism visible (variable names from `adj_callaccel.c`; `v66, v69, v72` is one link's `dist`,
`v75 = |dist|²`, `v76 = 2|dist|`):

```c
double v77 = (((((((v4 * v66) + (v44 * v69)) + (v47 * v72)) * v6) * v7) / v75) / v76);   // link adjoint, cotangent of mass i
double v78 = ((v4 * v74) + (2.0 * (v77 * v66)));
double v88 = (((((((v81 * v66) + (v84 * v69)) + (v87 * v72)) * v6) * v7) / v75) / v76); // same link, cotangent of mass i+1
double v89 = ((v81 * v74) + (2.0 * (v88 * v66)));
double v90 = ((((v5 * v42) + (2.0 * (v49 * v35))) - v78) - v89);
```

`chain_mass_accel_fn` computes both links of a mass, so every interior link is computed twice per
substep, once as the right link of mass *i* and once as the left link of mass *i+1*. The primal copies
are hash-consed away by the scalarizer (16 `sqrt` per stage, four links times four substeps, in
every build). The adjoint is not: the reverse sweep runs per call site, each copy of the link
receives its own cotangent, and the dot product, the two divisions and the multiply chain are done
twice before the two results are added. In the inlined graph `simplify_cse_fixpoint` merges the two
`link(pos_{i+1} - pos_i)` nodes first, the cotangents accumulate on the single node, and the link
adjoint runs once. Two of six link-adjoint instances per substep are duplicates, and every tangent
of a duplicated adjoint is duplicated with it, which matches the roughly 25% gap.

This is the mechanism SX gets for free from `expand`: whole-graph CSE before the reverse sweep. The
adjoints being computed with respect to the non-differentiable parameter formals is not part of it
(restricting the inner `vjp` to the state formals changes nothing: dead adjoints are already dropped
by `_schedule`'s reachability). Expression-level `simplify_cse_fixpoint` inside the JVP callee, which
`_call_jvp_function` skips, is worth 558 operations, all adds.

Two remedies. The general one is to inline small pure callees at the expression level *before*
differentiating (unroll a short `VMAP`, substitute a `CALL`), which turns the real stage into the
inlined build: 28,682 to 16,826 with no rule change, 14,771 with them. C-44 scalarizes after AD and
cannot recover this. The model-level one is to write the chain as a map over links followed by a
difference over masses, which computes each link once by construction; that is a benchmark change
and `docs/results/fairness.md` decides whether it is admissible.

## Rule quality, one rule at a time

All variants measured on the real stage (28,682 base) unless noted. "Δ" is the change in total
arithmetic; a division is worth roughly four multiplies in latency, so the DIV and SQRT rows are
latency wins that the operation count hides.

| Variant | Total | Δ | div | neg | Notes |
|---|---:|---:|---:|---:|---|
| `negfold`: `(-a)*b -> -(a*b)`, `a*(-b) -> -(a*b)`, `(-a)/b -> -(a/b)`, `a/(-b) -> -(a/b)` in `passes/arith.py::simplify_arith` | 26,827 | −1,855 | 1,764 | 904 | pushes the negation to where `a + (-b) -> a - b` already fires; also fixes the follow-up's 2,749 |
| `sq`: `x*x` and `dot(x,x)` forward `2x·dx`, reverse `(cot*x, cot*x)` | 27,254 | −1,428 | 1,800 | 1,483 | replaces `dx*x + x*dx` (two products and an add per seed) with one product; `_jvp_many_structural` MUL/MATMUL need the same |
| `div`: forward `(dx - f*dy) * (1/y)`, reverse `q = cot/y; adj_y = -(q*f)` (the `experiments.patch` rules) | 28,771 | +89 | 335 | 2,749 | −1,465 divisions, +1,266 negations; the negations are the unfolded `-(q*f)` and `-cot` shapes, gone with `negfold` |
| `div` + `negfold` | 25,758 | −2,924 | 317 | 850 | |
| `sqrt`: forward `dx * (0.5/f)` instead of `dx / (2f)` (both `_jvp` and `_jvp_many_structural`) | 28,698 | +16 | 1,579 | 1,483 | 221 divisions become multiplies |
| `sqrt_rev`: reverse `cot * (0.5/f)` | | | 74 | | adds 8 mul per stage on top of the above, 40 fewer divisions; marginal |
| `subacc`: accumulate a `NEG` cotangent by subtraction in `vjp` | 28,682 | 0 | | | never fires; `negfold` covers it |
| `div` + `sqrt` + `sq` + `negfold` | **24,345** | **−4,337** | 114 | 696 | 963 per seed |
| same on the inlined build | 14,771 | −2,055 from 16,826 | 98 | 328 | 573 per seed |
| `joint` + all four | 19,946 | −8,736 | 114 | 456 | 780 per seed |

Rules that are already fine for this stage: `MUL` with a constant or non-differentiable operand
(the zero tangent folds), `ADD`/`SUB`/`CONCAT`/`SLICE`, `POW` with exponent 2 (`x**2 -> x*x` fires
in the shared identities), `MATMUL` for the 1-D dot product apart from the `x is y` case above.

## What is inherent to forward-over-reverse here

With the graph inlined and the four rule edits applied, Alloy's explicit forward-over-reverse is
14,771 against SX's 17,969 for the identical construction, and against 16,367 for SX's one-shot
symbolic Hessian: the composition itself is not the problem, and there is nothing to gain from a
"symbolic Hessian" pass on this stage. What Alloy does not exploit is symmetry: it computes the full
24 × 45 block (24 seeds, `z` and `xnext` adjoint rows), where the harness's SX `nlp_hess_l` computes
the upper triangle and lands at 13,213 including the cost term. Star colouring cannot reduce a dense
24 × 24 block, so the remaining structural lever is C-11's: either a reverse-over-reverse per row of
the triangle, or dropping the 21 `xnext` adjoint rows whose tangents are identically zero (they fold
today, so this costs nothing but is why the output is 45 wide).

## Ranked rule and composition edits for an implementer

1. **Inline small pure callees before AD** (new: an expression-level `expand` for `CALL` with a
   body under some size and `VMAP` with a short length, applied in `sparse_hessian`/`_vmap_adj_function`
   before `vjp`). −11,856 on the stage, 41%, no rule change, and the tangent goes to 658 per seed.
   Subsumes item 2 for chain; item 2 remains for callees that stay callees.
2. **Joint JVP over all active formals** in `_jvp` `CALL`/`VMAP` and the matching
   `_jvp_many_structural` branches: `_call_jvp_function(callee, output, formals: tuple)` seeded with
   one symbol per formal, keyed by the formal set, one call/vmap per node. −5,734 (20%) on its own.
   Sketch in `variants.py` (`joint`); the `_jvp` signature change is internal to `forward.py`.
3. **Negation identities in `simplify_arith`**: `(-a)*b`, `a*(-b)`, `(-a)/b`, `a/(-b)` to an outer
   `neg`. −1,855 (6.5%), both dialects, one function. Land this before or with item 4.
4. **Lean DIV rules** from `experiments.patch` (forward `_jvp` and `_jvp_many_structural`, reverse
   `_local_vjp`). −1,465 divisions; with item 3, −1,069 operations net. The measured 2458 to 2277 µs
   came from this rule alone.
5. **`x*x` and `dot(x, x)` rules**: `args[0] is args[1]` in MUL and MATMUL, forward `(2*x)*dx` and
   `2*(x@dx)`, reverse `(cot*x, cot*x)`; same in `_jvp_many_structural`. −1,428 (5%).
6. **SQRT tangent as a shared reciprocal**: `dx * (0.5/f)` in both forward rules. 221 divisions
   become multiplies; operation count unchanged. Skip `sqrt_rev`.
7. Not worth doing: `subacc` (no effect), expression-level simplification inside `_call_jvp_function`
   (−558, all adds, moot after item 1 or 2).

Items 3 to 6 together: −4,337 (15%) on today's stage; on top of item 1: 14,771 total, 573 per seed.

## Reproduction

Everything ran from the worktree root with the venv's `casadi` 3.8.0; the scripts lived in
`/tmp/c49/` and are reproduced here in full.

```sh
uv run /tmp/c49/sx_hist.py 5          # SX histograms
uv run /tmp/c49/sx_nlp_hist.py 5      # harness nlp_hess_l per-stage count
uv run /tmp/c49/alloy_hist.py 5 base  # whole chain_nlp Hessian, per-procedure histograms (33 s build)
uv run /tmp/c49/stage_variants.py                       # real stage, base
uv run /tmp/c49/stage_variants.py div sqrt sq negfold   # real stage, rule edits
uv run /tmp/c49/stage_variants.py joint                 # real stage, joint JVP
MODE=inlined   uv run /tmp/c49/inlined_stage.py         # no CALL/VMAP; MODE=callode|callaccel|vmapaccel
MODE=callaccel PROCS=1 uv run /tmp/c49/inlined_stage.py # also list every lowered procedure
uv run /tmp/c49/check_equal.py                          # inlined tangent == stage kernel z block
```

`sx_hist.py`:

```python
import sys, os
from collections import Counter
sys.path.insert(0, os.getcwd())
import casadi as ca
from benchmarks.problems import chain

M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
nx, nz = chain.n_state(M), chain.n_state(M) + chain.NU
OPS = {getattr(ca, n): n[3:].lower() for n in dir(ca) if n.startswith('OP_') and isinstance(getattr(ca, n), int)}
SKIP = {'input', 'output', 'const', 'parameter'}

def hist(f):
  h = Counter(OPS.get(f.instruction_id(k), '?') for k in range(f.n_instructions()))
  a = {k: v for k, v in h.items() if k not in SKIP}
  return sum(a.values()), dict(sorted(a.items(), key=lambda kv: -kv[1]))

def fn(name, ins, outs, cse=True):
  return ca.Function(name, ins, outs, {'cse': cse, 'live_variables': False})

z, xnext, params, lam = ca.SX.sym('z', nz), ca.SX.sym('xnext', nx), ca.SX.sym('params', chain.N_PARAMS), ca.SX.sym('lam', nx)
step = chain._ca_step(M, z[:nx], z[nx:], params)
L = ca.dot(lam, step - xnext)
g = ca.gradient(L, z)
seeds = ca.SX.eye(nz)
print('primal step          ', hist(fn('step', [z, params], [step])))
print('primal+adjoint (grad)', hist(fn('grad', [z, params, lam], [g])))
print('fwd-over-rev 24 seeds', hist(fn('fwdrev', [z, params, lam], [ca.jtimes(g, z, seeds)])))
print('  same, no cse       ', hist(fn('fwdrev_nocse', [z, params, lam], [ca.jtimes(g, z, seeds)], cse=False)))
print('ca.hessian one-shot  ', hist(fn('hess_full', [z, params, lam], [ca.hessian(L, z)[0]])))
print('rev-over-rev 24 seeds', hist(fn('revrev', [z, params, lam], [ca.jtimes(g, z, seeds, True)])))
```

`sx_nlp_hist.py` (the harness's own construction):

```python
import sys, os
from collections import Counter
sys.path.insert(0, os.getcwd())
import casadi as ca
from benchmarks.problems import chain
from benchmarks.harness.sweep import _casadi_descriptor_kernel
OPS = {getattr(ca, n): n[3:].lower() for n in dir(ca) if n.startswith('OP_') and isinstance(getattr(ca, n), int)}
M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
z, p, cost, constraints = chain._ca_nlp_pieces(M, chain.HORIZON, ca.SX)
f = _casadi_descriptor_kernel(ca, 'sxh', z, p, cost, constraints, 'hess', expand=True, cse=True)
h = Counter(OPS.get(f.instruction_id(k), '?') for k in range(f.n_instructions()))
a = {k: v for k, v in h.items() if k not in {'input', 'output', 'const', 'parameter'}}
print(f'per stage {sum(a.values()) / chain.HORIZON:.0f}', a, 'nnz', f.sparsity_out(0).nnz())
```

`stage_variants.py` (the real stage; the shared `histogram` helper is repeated in `inlined_stage.py`):

```python
import sys, os
sys.path.insert(0, '/tmp/c49'); sys.path.insert(0, os.getcwd())
from collections import Counter
import numpy as np
from variants import F, R, A, apply
names = sys.argv[1:]
apply(names)
from benchmarks.problems.chain import _eq_stage_fn, n_state, NU
from alloy.codegen.aot import _lower
from alloy.codegen.c import render_program_c
from alloy.ir.program import ProgramOp

M = 5
nz = n_state(M) + NU
stage = _eq_stage_fn(M)
adj_fn, _ = R._vmap_adj_function(stage, 0, (0, 1))
fwd_fn, _, active = F._call_jvp_many_const_function(adj_fn, 0, 0, np.eye(nz))
ARITH = {ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.NEG, ProgramOp.SQRT, ProgramOp.POW}

def histogram(fn):
  prog = _lower(fn).prog
  for proc in prog.args[: prog.attrs['proc_count']]:
    if proc.attrs['name'] != fn.name:
      continue
    seen, hist, stack = set(), Counter(), list(proc.args[proc.attrs['param_count']:])
    while stack:
      n = stack.pop()
      if id(n) in seen:
        continue
      seen.add(id(n))
      hist[str(n.op)] += 1
      stack.extend(n.args)
    arith = {k: v for k, v in hist.items() if ProgramOp(k) in ARITH}
    return sum(arith.values()), dict(sorted(arith.items(), key=lambda kv: -kv[1])), prog
  raise KeyError(fn.name)

tag = '+'.join(names) or 'base'
adj_total, adj_hist, _ = histogram(adj_fn)
tot, hist, prog = histogram(fwd_fn)
print(f'[{tag}] adjoint only: {adj_total} {adj_hist}')
print(f'[{tag}] adjoint + 24 unit-seed tangents: {tot} {hist}  -> per seed {(tot - adj_total) / nz:.0f}')
open(f'/tmp/c49/stage_{tag}.c', 'w').write(render_program_c(prog, fwd_fn))
```

`inlined_stage.py` (same stage from plain expressions; `MODE` selects which part stays a callee):

```python
import sys, os
sys.path.insert(0, '/tmp/c49'); sys.path.insert(0, os.getcwd())
from collections import Counter
import numpy as np
from variants import F, R, A, apply
names = sys.argv[1:]
apply(names)
import alloy as al
from alloy.function import Function
from alloy.passes.expr import simplify_cse_fixpoint
from alloy.codegen.aot import _lower
from alloy.codegen.c import render_program_c
from alloy.ir.program import ProgramOp
from benchmarks.problems.chain import n_state, NU, N_PARAMS

M = 5
nx, nz = n_state(M), n_state(M) + NU
z, xnext = al.sym('z', nz), al.sym('xnext', nx)
params, lam = al.sym('params', N_PARAMS, diff=False), al.sym('lam_eq', nx)
mass, spring_d, rest_len, gravity, dt = (params[i] for i in range(N_PARAMS))
x, u = z[:nx], z[nx:]

def link(dist):
  return (spring_d / mass) * (1.0 - rest_len / al.norm_2(dist)) * dist

def rhs(state):
  positions = al.concat([al.const(np.zeros(3)), state[: 3 * (M - 1)]])
  accel = []
  for i in range(M - 2):
    left, pos, right = positions[3 * i: 3 * i + 3], positions[3 * i + 3: 3 * i + 6], positions[3 * i + 6: 3 * i + 9]
    accel.append(link(right - pos) - link(pos - left) + al.stack([0.0, 0.0, gravity]))
  return al.concat([state[3 * (M - 1):], u, *accel])

MODE = os.environ.get('MODE', 'inlined')
pslices = [params[i:i + 1] for i in range(N_PARAMS)]
if MODE == 'callode':
  xs, us = al.sym('x', nx), al.sym('u', NU)
  ps = [al.sym(n, 1, diff=False) for n in ('mass', 'spring_d', 'rest_len', 'gravity', 'dt')]
  mass, spring_d, rest_len, gravity, dt = (p_[0] for p_ in ps)
  u = us
  ode_fn = Function._from_exprs('inl_ode', [xs, us, *ps], [rhs(xs)], ['x', 'u', 'mass', 'spring_d', 'rest_len', 'gravity', 'dt'], ['xdot'])
  u = z[nx:]
  mass, spring_d, rest_len, gravity, dt = (params[i] for i in range(N_PARAMS))
  rhs = lambda state: ode_fn((state, u, *pslices))
elif MODE == 'callaccel':
  from benchmarks.problems.chain import chain_mass_accel_fn
  def rhs(state):
    positions = al.concat([al.const(np.zeros(3)), state[: 3 * (M - 1)]])
    accel = [chain_mass_accel_fn((positions[3 * i: 3 * i + 3], positions[3 * i + 3: 3 * i + 6], positions[3 * i + 6: 3 * i + 9], *pslices[:4])) for i in range(M - 2)]
    return al.concat([state[3 * (M - 1):], u, *accel])
elif MODE == 'vmapaccel':
  from benchmarks.problems.chain import chain_mass_accel_fn
  def rhs(state):
    positions = al.concat([al.const(np.zeros(3)), state[: 3 * (M - 1)]])
    accel = al.vmap(chain_mass_accel_fn, length=M - 2, inputs={"left": (positions, 0, 3), "pos": (positions, 3, 3), "right": (positions, 6, 3),
      "mass": (pslices[0], 0, 0), "spring_d": (pslices[1], 0, 0), "rest_len": (pslices[2], 0, 0), "gravity": (pslices[3], 0, 0)})
    return al.concat([state[3 * (M - 1):], u, accel])

h = dt
k1 = rhs(x); k2 = rhs(x + 0.5 * h * k1); k3 = rhs(x + 0.5 * h * k2); k4 = rhs(x + h * k3)
step = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
L = al.dot(lam, step - xnext)
adj = simplify_cse_fixpoint(R.vjp((L,), (z,), (al.const(np.ones(())),))[0])
tan = simplify_cse_fixpoint(F._jvp_many_unrolled(adj, z, al.const(np.eye(nz))))
ARITH = {ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.NEG, ProgramOp.SQRT, ProgramOp.POW}

def histogram(expr, name):   # same counting as stage_variants.py, on Function._from_exprs(name, [z, params, lam], [expr.scalar()], ...)
  fn = Function._from_exprs(name, [z, params, lam], [expr.scalar()], ['z', 'params', 'lam_eq'], ['out'])
  prog = _lower(fn).prog
  for proc in prog.args[: prog.attrs['proc_count']]:
    if proc.attrs['name'] != name:
      continue
    seen, hist, stack = set(), Counter(), list(proc.args[proc.attrs['param_count']:])
    while stack:
      n = stack.pop()
      if id(n) in seen:
        continue
      seen.add(id(n))
      hist[str(n.op)] += 1
      stack.extend(n.args)
    arith = {k: v for k, v in hist.items() if ProgramOp(k) in ARITH}
    return sum(arith.values()), dict(sorted(arith.items(), key=lambda kv: -kv[1])), prog, fn
  raise KeyError(name)

tag = MODE + '_' + ('+'.join(names) or 'base')
a_tot, a_hist, a_prog, a_fn = histogram(adj, 'inl_adj')
open(f'/tmp/c49/adj_{MODE}.c', 'w').write(render_program_c(a_prog, a_fn))
t_tot, t_hist, prog, fn = histogram(tan, 'inl_tan')
print(f'[{tag}] adjoint only: {a_tot} {a_hist}')
print(f'[{tag}] adjoint + 24 unit-seed tangents: {t_tot} {t_hist}  -> per seed {(t_tot - a_tot) / nz:.0f}')
open(f'/tmp/c49/stage_{tag}.c', 'w').write(render_program_c(prog, fn))
```

`variants.py`, the rule substitutions (each is the literal edit an implementer would make) and the
joint-JVP sketch:

```python
import alloy.ad.forward as F
import alloy.ad.reverse as R
import alloy.passes.arith as A

VARIANTS = {
  'div': [
    (F, "return save((d[0] * args[1] - args[0] * d[1]) / (args[1] ** 2))",
        "return save((d[0] - expr * d[1]) * (Expr.const(np.ones((), dtype=np.float64)) / args[1]))"),
    (F, """    y = _seed_axis(args[1], nseed, expr)
    memo[expr.id] = ret = (
      _broadcast_tangent(d[0], args[0], expr, nseed) * y - _seed_axis(args[0], nseed, expr) * _broadcast_tangent(d[1], args[1], expr, nseed)
    ) / (y**2)""",
        """    inv = _seed_axis(Expr.const(np.ones((), dtype=np.float64)) / args[1], nseed, expr)
    f = _seed_axis(expr, nseed, expr)
    memo[expr.id] = ret = (_broadcast_tangent(d[0], args[0], expr, nseed) - f * _broadcast_tangent(d[1], args[1], expr, nseed)) * inv"""),
    (R, """      _unbroadcast(cot / args[1], args[0].shape, expr.shape),
      _unbroadcast(-cot * args[0] / (args[1] ** 2), args[1].shape, expr.shape),""",
        """      _unbroadcast(cot / args[1], args[0].shape, expr.shape),
      _unbroadcast(-((cot / args[1]) * expr), args[1].shape, expr.shape),"""),
  ],
  'sqrt': [
    (F, "return save(d[0] / (2 * expr))", "return save(d[0] * (0.5 / expr))"),
    (F, "memo[expr.id] = ret = d[0] / (2 * _seed_axis(expr, nseed))", "memo[expr.id] = ret = d[0] * _seed_axis(0.5 / expr, nseed)"),
  ],
  'sqrt_rev': [(R, "return (cot / (2 * expr),)", "return (cot * (0.5 / expr),)")],
  'sq': [
    (F, "  if expr.op == ExprOp.MUL:\n    return save(d[0] * args[1] + args[0] * d[1])",
        "  if expr.op == ExprOp.MUL:\n    if args[0] is args[1]:\n      return save((2 * args[0]) * d[0])\n    return save(d[0] * args[1] + args[0] * d[1])"),
    (F, "  if expr.op == ExprOp.MATMUL:\n    return save(d[0] @ args[1] + args[0] @ d[1])",
        "  if expr.op == ExprOp.MATMUL:\n    if args[0] is args[1]:\n      return save(2 * (d[0] @ args[1]))\n    return save(d[0] @ args[1] + args[0] @ d[1])"),
    (R, "  if expr.op == ExprOp.MUL:\n    return (_unbroadcast(cot * args[1]",
        "  if expr.op == ExprOp.MUL:\n    if args[0] is args[1]:\n      return (cot * args[0], cot * args[0])\n    return (_unbroadcast(cot * args[1]"),
  ],
  'negfold': [
    (A, """    if d.kind(x) == d.kind(y) == "neg":
      return d.build("mul", (x.args[0], y.args[0]), node)""",
        """    if d.kind(x) == d.kind(y) == "neg":
      return d.build("mul", (x.args[0], y.args[0]), node)
    if d.kind(x) == "neg":
      return d.build("neg", (d.build("mul", (x.args[0], y), node),), node)
    if d.kind(y) == "neg":
      return d.build("neg", (d.build("mul", (x, y.args[0]), node),), node)"""),
    (A, """    if cy == 1 and d.fits(x, node):
      return x
  elif kind == "pow":""",
        """    if cy == 1 and d.fits(x, node):
      return x
    if d.kind(x) == "neg":
      return d.build("neg", (d.build("div", (x.args[0], y), node),), node)
    if d.kind(y) == "neg":
      return d.build("neg", (d.build("div", (x, y.args[0]), node),), node)
  elif kind == "pow":"""),
  ],
  'joint': [],  # appended below
}

def patch(mod, old, new):
  src = getattr(mod, '_patched_src', open(mod.__file__).read())
  assert src.count(old) == 1, (mod.__name__, old[:60], src.count(old))
  mod._patched_src = src.replace(old, new)

def apply(names):
  for name in names:
    for mod, old, new in VARIANTS[name]:
      patch(mod, old, new)
  if 'joint' in names:
    base = getattr(F, '_patched_src', open(F.__file__).read())
    F._patched_src = base + joint_src(base)
  for mod in (F, R, A):
    if hasattr(mod, '_patched_src'):
      exec(compile(mod._patched_src, mod.__file__, 'exec'), mod.__dict__)

# The joint variant redefines jvp/_jvp so that `wrt` may be a dict {input_id: (input, seed)}, and
# replaces the CALL/VMAP branches with one callee per (output, active formal set). The scalar rules
# are the unmodified tail of the original _jvp, extracted textually into _jvp_scalar_rules.
JOINT_HEAD = '''
_CALL_JVP_JOINT_CACHE = weakref.WeakKeyDictionary()

def jvp(expr, wrt, seed):
  return _jvp(expr, wrt, seed, {}, {})

def _jvp(expr, wrt, seed, memo, dep_memo):
  seeds = wrt if isinstance(wrt, dict) else {wrt.id: (wrt, seed)}
  return _jvpj(expr, seeds, memo, dep_memo)

def _call_jvp_joint_function(callee, output_index, formals):
  key = (output_index, formals)
  cache = _CALL_JVP_JOINT_CACHE.setdefault(callee, {})
  if key not in cache:
    seeds = {callee.inputs[i].id: (callee.inputs[i], Expr.sym(f"fwd:{callee.input_names[i]}", callee.inputs[i].shape)) for i in formals}
    deriv = _inherit_lowering(callee, _jvpj(callee.outputs[output_index], seeds, {}, {}))
    dep_memo = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    taken = tuple(i for i in formals if _depends_on(deriv, seeds[callee.inputs[i].id][1], dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + tuple(seeds[callee.inputs[i].id][1] for i in taken)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + tuple(seeds[callee.inputs[i].id][1].name for i in taken)
    name = f"{callee.name}_fwdj_{callee.output_names[output_index]}_" + "_".join(str(i) for i in formals)
    fn = Function._from_exprs(name, inputs, [deriv], input_names, [f"fwd:{callee.output_names[output_index]}"])
    cache[key] = (fn, arg_indices, taken)
  return cache[key]

def _jvpj(expr, seeds, memo, dep_memo):
  if expr.id in memo:
    return memo[expr.id]
  wrt_ids = [k for k, (w, s) in seeds.items() if not _is_zero_const(s)]
  if not wrt_ids or expr.op == ExprOp.CONST or expr.op == ExprOp.SOLVER_CALL:
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op == ExprOp.INPUT:
    memo[expr.id] = ret = seeds[expr.id][1] if expr.id in seeds and not _is_zero_const(seeds[expr.id][1]) else zeros_like(expr)
    return ret
  if not any(_depends_on(expr, seeds[i][0], dep_memo) for i in wrt_ids):
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op in (ExprOp.CALL, ExprOp.VMAP):
    callee = expr.attrs["callee"]
    tans = [_jvpj(a, seeds, memo, dep_memo) for a in expr.args]
    active = tuple(i for i, t in enumerate(tans) if not _is_zero_const(t))
    if not active:
      memo[expr.id] = ret = zeros_like(expr)
      return ret
    fn, arg_indices, taken = _call_jvp_joint_function(callee, expr.attrs["output"], active)
    if expr.op == ExprOp.CALL:
      memo[expr.id] = ret = fn._flat_symbolic_call([expr.args[i] for i in arg_indices] + [tans[i] for i in taken])[0]
    elif not taken:
      memo[expr.id] = ret = zeros_like(expr)
    else:
      starts, strides = expr.attrs["starts"], expr.attrs["strides"]
      specs = [(expr.args[i], starts[i], strides[i]) for i in arg_indices] + [(tans[i], starts[i], strides[i]) for i in taken]
      memo[expr.id] = ret = vmap(fn, expr.attrs["length"], specs)
    return ret

  def save(ret):
    memo[expr.id] = ret
    return ret

  d = [_jvpj(a, seeds, memo, dep_memo) for a in expr.args]
  return _jvp_scalar_rules(expr, expr.args, d, save)
'''

def joint_src(base):
  start = base.index('  if expr.op == ExprOp.NEG:\n    return save(-d[0])')
  end = base.index('raise NotImplementedError(f"JVP for op {expr.op!r} is not implemented")', start)
  return JOINT_HEAD + '\n\ndef _jvp_scalar_rules(expr, args, d, save):\n' + base[start:end] + 'raise NotImplementedError("JVP for op")\n'
```

`check_equal.py` builds `fwd_fn` as in `stage_variants.py` and `tan` as in `inlined_stage.py`
(`MODE=inlined`, no variants), evaluates both on `initial_state(5)` plus noise, `ChainParams()` and a
random `lam`, and compares `fwd_fn(...)` reshaped `(24, 45)[:, :24]` with `inl(...)` reshaped
`(24, 24)`: max abs difference 2.9e-17, max abs value 0.068.
