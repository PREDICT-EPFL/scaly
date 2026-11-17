# Section 4: Dtypes

Packet of section 4 of the [compiler run](../plan.md), parent #24. Built by #312 (#168), #320
(#30), #314 (#76), #360 (#55, #418), #361 (#340), #358 (#339), #405 (#363), #368 (#56), #450
(#373), #453 (#207) and #434 (#364, #414, #432). Open when this was written: #365, #372 and
#435. The section had no simplification pass (#467). AD is automatic differentiation, ABI the
application binary interface, QP a quadratic program. A select is `sc.where`, and its mask is
the condition it chose by.

## Where this section sits

- A user declares `float32`, `int32`, `int64` and `bool` leaves, writes comparisons, `sc.where`, `sc.cast`, `sc.isfinite`, `& | ~` and integer `//` and `%` with NumPy's results, and calls the generated C through pointers of the leaf types, without a detour through `double`.
- A float32 oracle with its sparse Jacobian builds freestanding for a Cortex-M4F or M7 with only libm left to link, and `frame_budget=0` moves its temporaries to caller memory. CI records flash and RAM. This completes the two Milestone 1 checks the section owns.
- `sc.where` is differentiable in both modes: the branch not chosen contributes an exact zero, across calls, maps and loop iterations, even where its partial is infinite.
- Published in `docs/guide/codegen.md` (*Float32 and integer functions*, the embedded build), `docs/guide/functions.md`, `docs/guide/derivatives.md` and `docs/dev/versioning.md`.
- Section 7 built the nonsmooth derivatives on the mask (#65), section 9 uses int64 index operands and mixed-dtype carries, section 8 owns float32 lanes (#71), and section 11 runs the oracle on the board (#208).

## Needs the maintainer's decision

1. The select-mask rule, decided by the super coordinator on 2026-11-11 ([comment on #24](https://github.com/PREDICT-EPFL/scaly/issues/24#issuecomment-6808554341)): a select's condition is an activity flag that travels with the derivative and is never derived from its value. It only transports a condition a select created. Recommend confirming. Decision 7 below has the rule and the table of boundaries as they stand today.
2. Changed by the super coordinator on 2026-11-12, to confirm or overrule: carried loops carry the flag across iterations in forward mode (#426), where the decision had them drop it. Recommend confirming, since a clipped rollout is the common case and its gradient matches the unrolled graph.
3. Still open from the rule: a reverse dense matrix product drops the flag (the select half of #435). The decision assigned it to the follow-up of #434, which took #432 only. The demo shows `[0, nan]` where forward mode is finite. Recommend fixing it as transport, with the weights #434 uses forward.
4. A new rule, not transport: should a structural zero with no select cancel an infinite partial (the rest of #435)? Recommend no general rule and IEEE behaviour. `sc.sparsify` on the constant already gives the finite gradient in #435's example, so say that beside the dense-product note in the guide.
5. Out-of-range float-to-integer casts are undefined in the generated C (#365). On x86 today, `3e9`, infinity and NaN all become `-2147483648` as int32. Recommend saturating, with NaN giving 0, as the section coordinator proposed.
6. Roadmap departure, applied by #368: the entry signature follows the leaf dtypes alone, so in an all-float64 or all-float32 Function a large integer or bool temporary stays on the stack even at `frame_budget=0` (#372). Recommend keeping the rule and reporting those bytes in the header.
7. Roadmap open question, settled by #368: a mixed-dtype workspace is caller memory without a declared type, counted in bytes and aligned to 8, with callee windows laid end to end. Recommend keeping.
8. Public interface, published: the three entry signatures, `<name>_FRAME_BYTES` and `frame_budget`, `sc.where`, `sc.cast`, `sc.equal`, `sc.not_equal`, `sc.isfinite`, the comparison operators, and signed integers that wrap as NumPy's do. `==` stays identity and `uint8` is refused. Recommend keeping.

## Expected gaps

- A reverse dense matrix product drops a select's condition, and a structural zero times an infinite partial is NaN: #435.
- A sampled sparse product rebuilds its guard from the producers it recognises: #442.
- Selected derivatives carry redundant masks, and a flagged seed table is printed as doubles (a clipped Jacobian of 18.6 kB became 69.3 kB): #443.
- A conditional call drops select activity in forward mode: #464, filed with the section 9 packet.
- Out-of-range float-to-integer casts: #365. `mean` of an integer or bool tensor raises: #430.
- Integer and bool temporaries of a uniform float Function are not spilled: #372.
- No float32 lanes and block callees: #71. Header macro names are not reserved across scopes: #326.
- The embedded build is checked for size and symbols and not run on hardware: #208.
- CasADi export and solver wrappers stay float64 and raise otherwise. Custom rules, `sc.extern` and solver calls receive numerical seeds without flags. Both are by design and have no issue.
- The section's simplification pass did not run: #467.

## Unowned

- Two sentences of the functions guide still say that Function leaves must be float64: filed as #466.
- Sections 3, 4 and 9 closed without a simplification pass: filed as #467.

# Reference

Checked on main at `cd2b6c75` on 2026-11-17. Paths are under `src/scaly/` and line numbers are on
that commit. Beside this file: `demo.py` and `demo_output.txt`.

## Decisions

Hardest to undo first.

1. **The entry signature follows the leaf dtypes** (#368; `codegen/abi.py:69-79`,
   `passes/lowering/ctx.py:1050-1056`). Every leaf float64: `const double** arg, double** res,
   int* iw, double* w, int mem`, byte for byte as before. Every leaf float32: the same with
   `float`. Anything else: `void` pointers, one cast per leaf, and `SZ_W` in bytes. `int32_t`,
   `int64_t` and `bool` cross as themselves. The roadmap had the signature depend on what the
   workspace holds. #368 replaced that line, so that neither a spill decision nor the frame budget
   can change a signature. `docs/dev/versioning.md` states it as stable.
2. **The byte workspace** (#368; `passes/program/pack_workspace.py:200-222`). Caller memory with
   no declared type, read through plain typed pointers. Each spilled slot starts at a multiple of
   64 bytes, with the padding counted in `SZ_W`. Sibling callee windows are laid end to end so
   that no byte changes type during a call, and a test compiles it under `-fstrict-aliasing`.
3. **The scalar vocabulary** (#320; `ir/expr.py`, `ad/rules.py::ELEMENTWISE`). `LT LE EQ NE AND
   OR NOT ISFINITE SELECT CAST FLOORDIV MOD` are rows of the elementwise table in both dialects.
   `==` stays identity, because interning depends on it, and `sc.equal` compares values. A Python
   scalar takes the dtype of the expression beside it, a NumPy value keeps its own, and mixed
   expression dtypes raise and name `sc.cast` (`ir/expr.py:1024-1039`, from #25).
4. **Integers behave as NumPy's** (#320, #361; `codegen/c.py:565-570`). Signed arithmetic wraps,
   done in unsigned C so that no overflow is undefined. `//` and `%` floor. A zero divisor gives
   0 and `INT64_MIN // -1` wraps. Integer sums use integer accumulators, and integer constants
   never pass through a float (#418).
5. **`uint8` is refused when the expression is built** (#405). Packed masks keep it internally.
6. **Casts** (#320; `ir/expr.py:1272-1287`). A float becomes an integer by dropping its
   fraction, a cast to bool is `x != 0`, and a cast keeps its derivative only between float
   dtypes. Out of range, the result is whatever the C cast gives (#365), and the docstring says
   it is not defined.
7. **The select-mask rule.** From the [comment on #24](https://github.com/PREDICT-EPFL/scaly/issues/24#issuecomment-6808554341),
   which this packet absorbs. One family of bugs (#339, #364, #414, #420, #432, #435) had one
   shape: a zero tangent or cotangent meets an infinite partial at a boundary and gives NaN, and
   each fix exposed the next boundary. The rule:

   - A select's condition is an activity flag that travels with the derivative. It is never
     derived from the derivative's value.
   - An inactive contribution is `where(active, contribution * partial, 0)`, exactly zero even
     when the partial is infinite or NaN.
   - A derivative that is numerically zero and active stays active, in higher derivatives too.
     Deriving activity from `cot != 0` was proposed in review and gives a wrong Hessian.
   - Nested conditions intersect and contributions combine by union. Movements and contractions
     carry nonnegative weights, so cancellation cannot switch activity off.
   - The rule transports a condition a select created. It adds no guard against structural
     zeros: without a select, IEEE `0 * inf = NaN` stays.

   What carries the flag on main today:

   | Boundary | Forward | Reverse |
   | --- | --- | --- |
   | Elementwise, casts, broadcasts, movement, sums, stack, pack and values | carries | carries |
   | Ordinary and mapped calls | carries (#434) | carries (#358, #434) |
   | Dense matrix product | carries (#434) | drops (#435) |
   | Mixed sparse and dense product | carries | carries |
   | Product of two sparse matrices | carries | rebuilds the guard from the operand's producers (#433, #442) |
   | Carried `scan` and `while_loop` | carries, in boolean carries (#426) | the conditions inside each step guard it, and nothing is passed between steps (#428) |
   | Conditional call | drops (#464) | drops |
   | Custom rules, `sc.extern`, solver calls | numerical seeds only | numerical seeds only |

8. **A flag is a select with a zero branch, not a new operation** (#320, #434;
   `ad/rules.py:391-417`). `_mask_of` recognises a top-level `where(c, inner, 0)`, also under
   `PACK` or with its branches reversed. Forward helpers take and return boolean flags only for
   the formals that have one, and the helper cache is keyed by that set, so C generated for
   derivatives without selects is byte-identical.
9. **Generated C that tests for nonfinite values refuses fast math** (#320; `codegen/c.py:78-79`).
   The source carries an `#error` under `-ffast-math` and `-ffinite-math-only`.
10. **The frame budget** (#314; `passes/program/pack_workspace.py:162-198`). Each procedure's
    estimate is its local buffers, those of the callees inlined into it and 4,096 bytes for scalar
    spills. The largest slots are spilled until the estimate fits. The default is 65,536 bytes,
    zero spills every eligible slot, and the header reports `<name>_FRAME_BYTES` beside `SZ_W`.
    It replaced the per-buffer threshold and regenerated every snapshot's recipe line.
11. **float32 through lowering** (#360, #312). Constants fold in the node's dtype and print as
    `%.9g` with an `f` suffix, libm names follow the dtype (`sinf`), and the constants of the
    structural derivative rules take the primal's dtype. Every elementwise operation runs with
    float32 leaves in block and scalar lowering, and `tests/baseline/c/float32.c` is the snapshot.
12. **The Cortex-M build** (#453; `tests/codegen/test_embedded.py`). The CI lint job builds
    `examples/embedded_oracle.py` with `zig cc -target thumb-freestanding-eabihf` for both
    cores, checks that the undefined symbols are `sinf` and `cosf`, compiles prints out, and
    writes flash and RAM to its summary: 456 bytes of flash sections and 4 bytes of static RAM,
    with 320 bytes of caller workspace. Zig rejects the issue's spelling `thumbv7em-none-eabihf`.
13. **A condition on the variables is nonlinear for the QP proof** (#320). Devrush accepted `|x|`
    as quadratic because its pattern ignored predicates.

## Demo

`uv run internal/notes/compiler_run_2026_11_10/dtypes/demo.py`. The demo of the section issue
(parts 1 and 2), then the select mask across each boundary, #435's example, and the two open
semantic questions. Output on main at `cd2b6c75`:

```text
== 1. a float32 Function with where, cast, a comparison, // and % ==
y float32 max |scaly - NumPy float32| 0.0e+00; k // 3 = [-3 -1  0  1  1  3] (int32), k % 3 = [2 2 0 1 2 2] (int32)
NumPy:                                               k // 3 = [-3 -1  0  1  1  3], k % 3 = [2 2 0 1 2 2]
float64 leaves: const double** arg, double** res, int* iw, double* w, int mem
float32 leaves: const float** arg, float** res, int* iw, float* w, int mem
mixed leaves:   const void** arg, void** res, int* iw, void* w, int mem
float32 C calls ['sinf'] and writes literals like ['0.5f']
int32 overflow wraps as NumPy: scaly -1542450945, NumPy -1542450945

== 2. the same float32 oracle, freestanding for a Cortex-M4F ==
zig cc -target thumb-freestanding-eabihf -mcpu=cortex_m4+vfp4d16sp -ffreestanding -Os -fno-unwind-tables ...
flash 248 bytes, static RAM 0 bytes, caller workspace 0 bytes
undefined symbols: cosf, sinf

== 3. a select's condition travels with the derivative ==
sqrt(where(x > 0, x, 0)) at x = [-1, 4]
  inline:                      forward d/dx0 0.0, reverse [0.   0.25]
  select in a callee:          forward d/dx0 0.0, reverse [0.   0.25]
  sqrt in a callee:            forward d/dx0 0.0, reverse [0.   0.25]
  both mapped over the entries: forward d/dx0 0.0, reverse [0.   0.25]
  through a dense product:     forward d/dx0 0.0, reverse [ 0. nan]

== 4. a structural zero with no select: #435's example ==
dense constant:     value 2.236068, forward 0.447214, reverse [nan nan]
sc.sparsify of it:  value 2.236068, forward 0.447214, reverse [0.2236068 0.2236068]
expected:           value 2.236068, forward 0.447214, reverse [0.2236068 0.2236068]

== 5. out-of-range float-to-integer casts (#365) and an integer mean (#430) ==
cast([ 3.0e+09      inf      nan -7.9e+00], int32) = [-2147483648 -2147483648 -2147483648          -7]
mean of an int64 tensor: TypeError / takes floating operands, got int64, int64; convert with sc.cast
NumPy: float64
an int64 result leaves a Function: argmax rows = [1 0]
```

Part 2 builds this demo's own small oracle, which needs no workspace. The numbers in decision 12
are those of `examples/embedded_oracle.py`, whose `.block()` hint keeps arrays on purpose. The
last line of part 5 is what #466 is about: the functions guide still says such a result must be
cast to float64.

## Reading list

About 275 lines, in reading order.

1. `ir/expr.py:1024-1039`. `_coerce` and `_operands`: weak Python scalars, strong NumPy values.
2. `ir/expr.py:1235-1269` and `1272-1287`. `where` and `cast`, with the limits their docstrings
   publish.
3. `ad/rules.py:56-65`. `Mask`: a partial that is a select.
4. `ad/rules.py:391-417`. The six-line statement of the rule, then `_mask_of`.
5. `ad/rules.py:355-388`. `carry_mask` and `accumulate`: a flag through a linear map with
   nonnegative weights, and the union of two flags.
6. `ad/rules.py:422-468`. `cotangent_mask`, `activity`, `move_mask`, `carry_masks` and `_masked`:
   nested conditions and the forward side.
7. `passes/lowering/ctx.py:1050-1056` and `codegen/abi.py:69-79`. The workspace dtype and the
   signature it gives.
8. `passes/program/pack_workspace.py:162-228`. The frame estimate, the spills, and the two
   workspace layouts.
9. `codegen/c.py:565-570`. Signed arithmetic through unsigned C.

Second: `ad/calls.py::_call_jvp_many_function` (the flags of a forward helper),
`ad/reverse.py::_masked_sampled_matmul`, the `_FLOOR_HELPERS` of `codegen/c.py`, and
`tests/codegen/test_embedded.py`.

## Source lines

Under `src/`, from `git show --numstat` of each squash commit: #312 +3 −3, #320 +577 −72, #314
+121 −43, #360 +218 −138, #361 +91 −23, #358 +100 −33, #405 +5 −1, #368 +270 −139, #450 +5 −11,
#453 none, #434 +309 −108. That is 1,699 added and 571 removed, a net 1,128. The select-mask
family accounts for 485 of the added lines when section 2's #433 (+76) is counted with #358 and
#434. After the simplification pass: not available, the pass did not run (#467).

## Least sure of

- **That the mask rule is complete.** Six bugs were found one boundary at a time, each by the
  review of the previous fix. The table above is the first statement of all boundaries, and two
  rows still drop the flag. I checked the rows the demo prints and one probe per loop case. I did
  not test a sampled sparse product, a custom rule or an extern today.
- **What the masks cost.** #434's check measured a clipped Jacobian of a 40-stage map at 81 µs
  on main and 133 µs with flags, and its Hessian at 802 µs and 625 µs, on la015 while other work
  ran. About 50 kB of each source is one seed table that is no longer bit-packed (#443). Code
  without selects is byte-identical.
- **float32 beyond elementwise code.** A float32 sparse product with its Jacobian and a float32
  `scan` of 300 steps match NumPy float32 in a probe today. Reductions with lanes, `einsum` and
  `sc.extern` in float32 were not tried, and an all-float32 Function that reaches an extern
  needing a workspace raises.
- **The embedded check proves less than its name.** It builds an object and reads its symbols
  and sections. Nothing runs on a target, stack use is an estimate checked against
  `-Wframe-larger-than` on hosts only, and the libm that `sinf` links against is the user's.
- **The out-of-range cast.** The docstring says the result is not defined. Under
  `-fsanitize=float-cast-overflow` the generated C aborts (#365), and no test compares a folded
  constant with the run-time value out of range.
- **How #434 landed.** Fable blocked it once, the fixes had a low-effort confirmation, and #432
  was folded into the same pull request on the maintainer's instruction of 2026-11-12, so the
  forward helper flags had less review than the reverse ones.
