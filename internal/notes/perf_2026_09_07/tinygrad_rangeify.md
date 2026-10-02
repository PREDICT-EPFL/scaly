# How tinygrad's rangeify loop compiler works, read 2026-09-08

Report from a read-only study of a tinygrad checkout at commit `69915d61` (2026-09-07), written to
inform C-43, C-44 and [#69]. Paths are relative to that checkout. Nothing was executed; the loop-nest
description in section 5 is derived from the code path.

## 1. The pipeline, end to end

One entry point ties it together, `tinygrad/schedule/__init__.py:130`:

```python
linear = create_schedule(get_kernel_graph(prepare_rangeify(function)))
```

**Stage A, `prepare_rangeify`** (`schedule/prepare.py:214-221`): tensor-graph hygiene before any
indexing. `pm_mops` pushes movement ops through `INDEX`/`AFTER`, `earliest_rewrites` inlines calls,
splits huge reduces, turns `COPY` into `STORE`, handles 0-size shapes.

**Stage B, `get_kernel_graph`** (`schedule/rangeify.py:377-403`), which is rangeify itself:

1. `run_rangeify` (`schedule/indexing.py:187-318`) introduces `RANGE`, converts views to index
   expressions, decides materialization.
2. `symbolic + pm_reduce_simplify + pm_const_buffer_folding + pm_remove_bufferize`
   (`rangeify.py:383`): index folding, dead-axis removal, and the cost-based un-materialization.
3. `pm_limit_bufs` (`rangeify.py:177-199`) reinserts buffers when a kernel would exceed the
   device's argument limit (`DEVICE_MAX_BUFS = {"CPU": 31, ...}`).
4. `pm_add_buffers` (`rangeify.py:262-281`): `STAGE` (formerly BUFFERIZE) becomes a real `BUFFER`
   plus `INDEX.store(...).end(ranges)`.
5. `split_kernels` (`rangeify.py:359-375`): each `STORE`/`END` with no open ranges becomes a
   `SINK(...).call(bufs)`; buffers are renumbered to `PARAM` slots and ranges from 0 so kernels dedup.

**Stage C, per-kernel codegen**, `full_rewrite_to_sink` (`codegen/__init__.py:296-415`), in order:
`pm_load_collapse`, `pm_split_ranges`, `sym`, `pm_simplify_ranges`, `apply_opts` (heuristics or
BEAM), `expander2` (UPCAST/UNROLL to shaped consts), `pm_reduce_local` (REDUCE to accumulator plus
END), `pm_add_local_buffers`, `pm_add_gpudims`, broadcast and `pm_add_loads`, `devectorizer2`,
`memory_coalescing`, symbolic and weak-dtype lowering, decomps (`codegen/decomp/*`),
`pm_add_control_flow` (linearizer), then `linearize` and render (`codegen/__init__.py:437-478`).

### What rangeify actually does

`run_rangeify` walks the tensor graph in reverse topological order and, for every node, records
`(input_ranges, output_ranges)` in `IndexingContext.range_map` (`indexing.py:10-21`). A `RANGE` is
`UOp(Ops.RANGE, src=(size,), arg=(id, AxisType))`; `new_range` returns `UOp.const(0)` for size-1
axes so degenerate axes never become loops (`indexing.py:18-21`).

Output ranges come from one of three cases (`indexing.py:223-266`):

- node is in `realize_map`: fresh ranges, and it will be materialized;
- exactly one consumer: inherit the consumer's ranges. This is fusion: producer and consumer
  share loop variables;
- two or more consumers: merge. If all consumers' index expressions per axis are identical the
  axis is shared (valids OR'd together), otherwise that axis gets a fresh range and is added to
  `realize_map` as a partial realize (`_realize_axis`).

Views become index arithmetic in `apply_movement_op` (`indexing.py:168-185`). The whole
ShapeTracker is replaced by this one function:

```python
case Ops.SHRINK:  rngs = tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
case Ops.FLIP:    rngs = tuple(((s-1)-a) if f else a for a,s,f in zip(rngs, in_shape, arg))
case Ops.EXPAND:  rngs = rngs[len(arg):]
case Ops.PAD:     rngs = ... (r-off).valid((r >= off) & (r < sh+off))
case Ops.RESHAPE: rngs = _apply_reshape(...)   # flatten to one linear index, then %, // back out
```

`_apply_reshape` (`indexing.py:152-165`) builds `sum(stride_i * r_i)` then peels it with `% s`
and `//= s` and calls `graph_rewrite(..., symbolic+pm_simplify_valid+pm_drop_and_clauses)`. The
comment is the design statement: "this simplify is doing a lot of heavy lifting. this is the
replacement for the reshape view merging code." `PAD` becomes a validity predicate, and
`convert_pad_to_where_to_keep_behavior_local` (`:107-111`) wraps the value in `valid.where(x, 0)`.
`EXPAND` drops ranges, so a broadcast producer is indexed by fewer ranges than its consumer, and
`broadcast_rngs` (`:63-66`) substitutes const 0 for broadcast axes.

Reductions: a tensor `REDUCE(op, n_axes)` (axes pre-permuted to the front by `UOp._rop`,
`ops.py:627-637`) becomes a rangeless-argument `REDUCE` whose extra sources are the reduce ranges
(`indexing.py:113-119`, `:297-299`):

```python
if x.op is Ops.REDUCE and x.arg[1]:
  rngs = tuple(rctx.new_range(s, axistype=AxisType.REDUCE) for s in x.src[0].shape[:x.arg[1]]) + out_rngs
```

So `REDUCE(src, r_k, arg=(ADD,0))` means "sum src over range r_k". Much later
`reduce_ranges_to_acc` (`codegen/__init__.py:211-221`) turns it into an explicit accumulator in
`AddrSpace.REG` plus an `END`:

```python
acc_init = acc.after(*input_ranges).store(UOp.const(identity_element(r.arg[0], r.dtype)))
acc_out  = acc_initted.store(acc_initted.alu(r.arg[0], inp)).end(*r.src[1:]).rtag("mergeable")
```

Realize decisions are made in two places. Seeds come from `pm_generate_realize_map`
(`indexing.py:45-54`): `CONTIGUOUS`, `STORE`, sources of `MSELECT/MSTACK`, custom-kernel inputs,
write-after-read hazards. Everything else is fused by default, then materialization is forced by:

- multi-consumer axis divergence (above);
- the reduce-under-broadcast rule (`indexing.py:269-275`): if a node has `ending_ranges` (ranges
  its consumers iterate but it broadcasts over) and it is elementwise or a reduce, all its axes
  get realized. That is the "do not recompute a value under an expand" rule; the comment at `:293`
  notes this is why convolutions get realized.

`create_bufferize_and_index_srcs` (`indexing.py:76-101`) then materializes: a realized source
becomes `STAGE(src, *closed_ranges)` and the consumer gets `.index(*its_ranges)`. If all the
source's ranges are closed it is `AddrSpace.GLOBAL`, otherwise `AddrSpace.LOCAL`: a partially
realized value becomes a scratch buffer inside the surviving loops.

Then rangeify argues itself back out of materialization with a cost function, `remove_bufferize`
(`rangeify.py:51-103`). Given `INDEX(STAGE(src, rngs), idxs)` it re-expresses `src`'s ranges as the
consumer's index expressions:

```python
replaced = {k:v for k,v in zip(buf.src[1:], idx.src[1:]) if ...}
return src.substitute(replaced, extra_pm=pm_gate_substitute)
```

It refuses when: the source is `CONTIGUOUS`/`NOOP`/`AFTER` or marked non-removable; more than 3
distinct buffers are read (`:86`); or any `REDUCE` inside would then read a buffer (`:88-97`),
that is, never fuse a memory-touching reduce into a re-indexed consumer. That last check plus the
3-buffer cap is essentially the whole fusion policy.

## 2. Loop order, outer versus inner, and two consumers

There is no cost model for loop order. Order falls out of `RANGE` ids:

- ids are assigned by `IndexingContext.range_idx` in reverse-toposort order, output axes left to
  right, so store axes get low ids and reduce ranges (created when the `REDUCE` node is visited,
  after its consumers) get higher ids: reduce loops end up innermost;
- `do_split_ends` (`codegen/late/linearizer.py:87-90`) splits a multi-range `END` into nested
  `END`s `sorted(..., key=arg, reverse=True)`, so the largest id closes first, innermost;
- `CFGContext` (`linearizer.py:53-85`) recovers nesting and sibling order from `END` dependencies
  and injects back-edges (`pm_add_control_flow`), and `linearize` (`:8-51`) is a priority toposort
  whose primary key is `run_count = prod(vmax+1 for r in u.ranges)`: deeper loop bodies later,
  `LOAD` early, `STORE`/`END` late.

Interchange exists only as `OptOps.SWAP`, restricted to `AxisType.GLOBAL` (`opt/postrange.py:169-178`),
that is, GPU launch dimensions. For opt purposes the axes are presented sorted by kind:
`Scheduler.rngs` sorts by `axis_to_pos[axistype]` then id, in the order `DEVICE, WEAK/LOOP, GLOBAL,
WARP, LOCAL, UPCAST, GROUP_REDUCE, REDUCE, UNROLL` (`ops.py:51-52`).

Two consumers of one producer: fused if and only if every axis's index expression agrees across
consumers (`all_same(local_rngs)` on `get_idx()`, with `get_valid()`s OR'd, `indexing.py:243-263`);
axes that disagree are materialized. On top of that, `remove_bufferize`'s cost function can undo a
materialization, and `pm_limit_bufs` can re-add one. The legality conditions are structural, not
analytic: no dependence testing, no polyhedral check, because fusion here means sharing the same
range variables, which is legal by construction.

## 3. The symbolic index simplifier and the pattern matcher

Rules as data is the whole architecture. `UPat` (`uop/ops.py:1374-1481`) is a declarative pattern
(op set, dtype set, arg, source tuple, list for permutations, `UPat` for repeat, name bindings) with
operator overloading so patterns read like expressions:
`(UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2")`. `PatternMatcher` (`:1513-1541`) is a list
of `(UPat, fn)`, indexed by root op into `pdict`, with an `early_reject` set of child ops;
`__add__` concatenates matchers and is cached. `graph_rewrite` (`:1799-1802`) plus `RewriteContext`
(`:1692-1789`) is an explicit-stack, memoized fixpoint driver with top-down (`pm`) and bottom-up
(`bpm`) modes, plus an MLIR-style single-pass `walk_rewrite`. `uop/upat.py` (175 lines) compiles
patterns to Python source for speed; `upat_interpret` is the fallback.

Two multipliers worth stealing: global hash-consing of nodes (`UOpMetaClass.__call__`,
`ops.py:192-208`, a weakref cache keyed on `(op, src, arg, tag)`), which gives structural equality,
free CSE and cheap `is` comparisons; and `recursive_property` (`:214-222`) for memoized recursive
attributes (`ranges`, `backward_slice`, `vmin/vmax`).

Index arithmetic lives in:

- `uop/symbolic.py` (485 lines), three phases: `symbolic_simple` (self, zero and const folding,
  Invalid propagation), `symbolic` (term combining, `lt_folding`, simplex canonicalization,
  two-stage ALU folding, int versus long narrowing), `sym` (plus `simplify_valid`, load/store
  folding, `reduce_mul_chain`). `uop_given_valid` (`:343-379`) simplifies an index under its guard
  by substituting range-bounded fake variables and checking all branches agree.
- `uop/divandmod.py` (109 lines), the div/mod core: `fold_divmod_general` handles nested div/mod,
  gcd factoring, congruence folding (`rem.vmin//c == rem.vmax//c` means the mod is affine on this
  range), `nest_by_factor`. Plus the recombination rule `fold_add_divmod_recombine`
  (`symbolic.py:43-59`) which reverses reshape peeling: `(x%c) + (x//c)*c -> x`.

Size of the core: symbolic 485 plus divandmod 109, about 600 lines; matcher and rewriter inside
`ops.py` about 250 lines (plus the 175-line optional compiler); rangeify 403 plus indexing 326,
729 lines; codegen driver 519 and simplify 157; opt 710 (postrange 322, heuristic 193, search 176);
late passes 411; cstyle renderer 610.

Loop-level rewrites also live as data in `codegen/simplify.py`: `simplify_merge_adjacent`
(`:23-41`) tries to fuse two adjacent loops into one (`r0,r1 -> new//s1, new%s1`) and keeps it only
if `count_divmod` did not increase; `pm_split_ranges` (`:72-76`) does the converse, splitting `r`
into `r_hi*c + r_lo` when `r % c` appears; `mark_gated` (`:43-52`) shrinks loop bounds to the
largest constant guard when a range is never used ungated. `reduce_collapse` (`:130-143`)
symbolically evaluates a whole reduce of pure index math into closed form (the `arange`
optimization): it substitutes opaque inputs for fake variables, rewrites with `pm_reduce_collapse`,
and keeps the result only if no range survives.

## 4. Small tensors, scalarization, CSE

There is no separate scalar mode; upcast and unroll axes are turned into shapes and then exploded:

- `expander2` (`codegen/__init__.py:84-90`) replaces every `RANGE` of type `UPCAST`/`UNROLL` with a
  shaped const vector `[0..n-1]` on its own axis (`build_range_map` assigns axis positions), and
  `expand_reduce` (`:49-63`) converts a reduce whose source gained such axes into a horizontal reduce.
- `pm_reduce_local` then either builds an accumulator (`reduce_ranges_to_acc`) or fully unrolls
  the horizontal part into an ALU tree: `expand_horizontal_reduce` (`:223-226`) does
  `functools.reduce(lambda x,y: x.alu(op,y), vals)`.
- `devectorizer2` / `ew_devectorizer` (`:143-169`) do the actual scalarization: `do_devectorize`
  (`:123-130`) replaces a shaped elementwise, load or store op with one node per index tuple,
  `UOp.stack(...)`.

Deciding what to unroll is entirely the OptOps layer (section 5): `SPLIT(amt, UPCAST)` on an output
axis, `SPLIT(amt, UNROLL)` on a reduce axis, with caps `amt <= 32` for UNROLL and `<= 16` for UPCAST
(`opt/postrange.py:130-131`).

CSE and constant folding on the scalarized graph are automatic: identical scalar expressions
hash-cons to the same `UOp`, and `symbolic_simple` runs interleaved with devectorization
(`codegen/__init__.py:342-358`), so `0*x`, `x*1`, `x-x`, dead lanes and `Invalid`-gated loads
(`pm_data_invalid`, `symbolic.py:79-102`: an out-of-bounds index poisons its consumers until the
LOAD folds to 0) all disappear before rendering. `pm_reduce_unparented` (`simplify.py:82-96`) turns
`sum_r(c)` into `c*len(r)` when the body does not depend on the range.

## 5. GEMM: OptOps, heuristics, and the CPU loop nest

`Scheduler` (`opt/postrange.py:17-299`) is a rewriter on the range set, not a shape tracker. The
only primitive is `shift_to` (`:90-97`):

```python
new_rng = UOp.range(amount, next(self.opt_range), new_type)
replaced_rng = rng.replace(src=(old_sz,))
sub_axis = (new_rng*old_sz + replaced_rng) if top else (replaced_rng*amount + new_rng)
self.ast = self.ast.substitute({rng: sub_axis})
```

That single substitution plus symbolic folding implements tiling, upcasting, unrolling and
localization. `OptOps` is just `TC, SPLIT, PADTO, SWAP` (`opt/__init__.py:6-7`); `SPLIT`'s target
type must be legal per `split_targets` (`postrange.py:14-15`): `UPCAST` from `{GLOBAL,LOCAL,WEAK}`,
`UNROLL` from `{REDUCE,GROUP_REDUCE}`, `LOCAL` from `{GLOBAL,WEAK}`, `GROUP_REDUCE` from `{REDUCE}`.
`PADTO` (`:153-168`) pads a loop bound and adds `valid` to affected indexes. `TC` (`:185-282`) finds
`REDUCE(ADD, MUL(a,b))`, picks (M,N,K) ranges, `PADTO`s to the tensor-core dims, applies the core's
own split recipe, and replaces the reduce with `UOp.wmma`.

`hand_coded_optimizations` (`opt/heuristic.py:8-193`) order: TC, IMAGE float4, the matvec recipe
(`MV_BLOCKSIZE/THREADS_PER_ROW/ROWS_PER_THREAD`, GPU-only, gated on `k.ren.has_local`),
`GROUP_REDUCE` 16 if the output is small, masked-axis upcasts, stride-heuristic upcasts
(`:115-138`, ranks axes by `num_strides, sum_strides` and prefers axes where some buffer has stride
0), unroll the last reduce axis (fully if at most 32, else by 4), a fallback `SPLIT(4, UPCAST)`,
locals. BEAM (`opt/search.py`) is the same `apply_opt` API driven by a fixed `actions` list
(`:14-23`) with real timing, `BEAM_UOPS_MAX`, and a disk cache.

For the C/clang backend (`ClangRenderer`, `cstyle.py:262-310`, `has_local=False`):
`convert_loop_to_global` returns immediately (`postrange.py:71-72`), so all axes stay
`AxisType.WEAK`, real `for` loops. The matvec recipe and all GROUP_REDUCE/LOCAL steps are skipped.
What remains for `C = A @ B` (say 64×64): loops `i`, `j` from the store's ranges (ids 0, 1) and `k`
from the reduce (id 2), a plain i-j-k nest with no interchange; then the heuristic unrolls `k`
(`heuristic.py:142-153`), fully if K is at most 32, else by 4, and if nothing is upcast yet,
`SPLIT(4, UPCAST)` on the innermost output axis (`:157-159`). Rendering: `RANGE` becomes
`for (int Lidx0 = 0; Lidx0 < 64; Lidx0++) {`, `END` becomes `}` (`cstyle.py:16-20`).

Multiple accumulators come for free from upcasting: `reduce_ranges_to_acc` runs after `expander2`,
so the reduce's dtype and shape already carry the upcast lanes and
`UOp.placeholder_like(r, ..., AddrSpace.REG)` (`ops.py:1146-1148`) allocates one REG slot per lane.
A 4-wide UPCAST of `j` yields 4 independent accumulators initialized before the `k` loop and stored
after it. For a matvec on CPU (no locals, no MV recipe) you get one accumulator loop with the K axis
unrolled or split by 4 and vector loads of the contiguous operand.

## 6. CPU vectorization

Two mechanisms, both explicit rather than hoped for:

1. `memory_coalescing` (`codegen/late/coalesce.py:104-171`), run after devectorization: it buckets
   every `LOAD`/`STORE` by `(op, buf, base_index, valid, arg)`, groups consecutive constant offsets,
   and replaces runs with a single wide access `UOp(Ops.SHRINK, src=(buf, offset, len))`, lengths
   `[4,2]` when `ren.supports_float4` (the base `Renderer` default is `True`, so clang qualifies;
   only WGSL disables it), `[128,64,32,16,8,4]` on DSP, and `must_divide` requires proven alignment.
   Wide values are `UOp.stack(...)`, rendered as clang ext_vector literals;
   `ClangRenderer.render_vector_prefix` (`cstyle.py:290-294`) emits
   `typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));`.
2. Autovectorization: `ClangCompiler.compile` (`runtime/support/compiler_cpu.py:19-23`) shells out
   to `clang -c -O2 -march=<cpu> -fno-math-errno -ffreestanding -nostdlib`, so remaining scalar
   loops go through LLVM's vectorizer. Accumulators being tiny `AddrSpace.REG` arrays and buffers
   being declared `restrict` (`buffer_suffix = " restrict"`, `cstyle.py:273`) is what makes that
   work. There is also a full LLVM-IR renderer (`renderer/llvmir.py`) and an x86 ISA renderer with
   its own register allocator (`codegen/late/regalloc.py`) for the same graph.

The renderer itself is simple and worth copying: `CStyleLanguage._render` (`cstyle.py:204-260`)
walks the linearized list, names each UOp (`bidx0`, `val0`, `alu0`, `Lidx3`), asks `string_rewrite`
(a `PatternMatcher` from UOp to string, `cstyle.py:11-77`) for one line, and inlines the expression
instead of emitting an SSA temporary when `child_count[u] == 1`.

## 7. Smallest-to-port ideas, and what each buys

1. Ranges as first-class values plus "views are index expressions". `apply_movement_op`
   (`indexing.py:168-185`) is about 15 lines and deletes the whole stride-view layer: reshape
   becomes `sum(stride*r)` then `%`, `//`; pad becomes a `valid` predicate; expand becomes "drop the
   range". Buys arbitrary view composition for free, and fusion becomes "share the range variable".
2. A `PatternMatcher`/`UPat`/`graph_rewrite` trio over hash-consed nodes
   (`ops.py:192-208, 1374-1541, 1692-1802`). About 300 lines gets rules as data, fixpoint rewriting,
   free CSE and structural equality. Everything else in tinygrad is a rule list on top.
3. The affine div/mod folder (`divandmod.py`, 109 lines, plus `fold_add_divmod_recombine`). This is
   what makes idea 1 viable: without congruence, gcd and nesting folding the reshape indices stay a
   pile of `%` and `//`. Buys clean strided address arithmetic.
4. Reverse-toposort range propagation with a three-case rule (fresh, inherit, merge-or-materialize)
   (`indexing.py:223-266`). About 40 lines of scheduler that gives fusion by default, and
   materialization exactly where index expressions disagree. Add the reduce-under-broadcast rule
   (`:269-275`) to avoid recompute blowups.
5. `shift_to` as the single loop transform (`postrange.py:90-97`). One substitution
   `r -> r_outer*amt + r_inner` plus the simplifier expresses tile, upcast, unroll and localize.
   Combine with an `AxisType` tag per range and an `axis_to_pos` ordering, and a tiny heuristic
   list; a search space comes for free (`search.py:14-23`).
6. Accumulator materialization plus expand-then-devectorize (`codegen/__init__.py:211-221`,
   `:84-90`, `:123-130`). Turning `REDUCE(range)` into `acc init / acc = acc op x / END(range)` and
   turning UPCAST ranges into shapes that get exploded gives multiple accumulators and full
   unrolling without special-casing.

Runner-up, cheap and high value for C: `memory_coalescing` (`coalesce.py:104-171`, about 60 lines).
Bucket loads and stores by base index and merge consecutive offsets into vector accesses; on clang
that is the difference between scalar code and SIMD you can actually see.

[#69]: https://github.com/PREDICT-EPFL/scaly/issues/69
