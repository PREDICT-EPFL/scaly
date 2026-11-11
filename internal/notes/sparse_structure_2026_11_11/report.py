"""Render the investigation note's tables from the saved evidence."""

from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def table(headers, rows):
  return "\n".join(
    [
      "| " + " | ".join(headers) + " |",
      "| " + " | ".join("---" for _ in headers) + " |",
      *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
    ]
  )


def arithmetic(row):
  return sum(v for k, v in row["program"]["executed"].items() if k not in ("index_loads", "loaded_loop_invocations", "loaded_loop_iterations"))


def main():
  kkt = list(map(json.loads, (HERE / "kkt.jsonl").read_text().splitlines()))
  bench = list(map(json.loads, (HERE / "bench.jsonl").read_text().splitlines()))
  addressing = json.loads((HERE / "addressing.json").read_text())
  maps = json.loads((HERE / "maps.json").read_text())
  coverage = {r["name"]: next(iter(r["coverage"].values())) for r in json.loads((HERE / "coverage.json").read_text())}
  rows = []
  map_rows = []
  for kind in ("kkt", "upper", "Px", "Ax", "ATy", "Agram", "Gx", "GTy", "Ggram"):
    group = [r for r in kkt if r["name"].endswith("/" + kind)]
    regions = sum((Counter(r["maps"]) for r in group), Counter())
    indices = sum((Counter(r["indices"]) for r in group), Counter())
    rows.append(
      (
        kind,
        sum(arithmetic(r) for r in group),
        sum(r["program"]["table_entries"] for r in group),
        sum(r["program"]["executed"].get("loaded_loop_invocations", 0) for r in group),
        sum(r["source_bytes"] for r in group),
        f"{sum(r['compile_seconds'] for r in group):.3f}",
      )
    )
    map_rows.append((kind, *(regions[k] for k in ("straight", "dense", "affine", "table")), indices["affine_positions"], indices["table_positions"]))
  current = table(("Kernel, 48 each", "Floating operations", "Index entries", "Loaded-bound invocations", "C bytes", "cc seconds, sum"), rows)
  regions = table(
    (
      "Kernel",
      "Straight map entries",
      "Dense map entries",
      "Affine map entries",
      "Table map entries",
      "Affine access-map positions",
      "Table access-map positions",
    ),
    map_rows,
  )
  products = []
  for kind in ("Px", "Ax", "ATy", "Gx", "GTy"):
    counts = Counter()
    for row in addressing:
      if row["name"].endswith("/" + kind):
        counts[row["baseline"]["entry_form"]] += row["baseline"]["entries"]
    products.append((kind, counts["dense"], counts["affine"], counts["table"]))
  product_regions = table(("Product", "Consecutive stored accesses", "Other affine stored accesses", "Stored accesses needing a table"), products)
  geometry = []
  for kind in ("kkt", "upper", "Px", "Ax", "Gx", "Agram", "Ggram"):
    group = [r for r in kkt if r["name"].endswith("/" + kind)]
    geometry.append(
      (
        kind,
        *(
          sum(r["pattern"][k] for r in group)
          for k in ("nnz", "block_entries", "disjoint_block_entries", "band_entries", "super_entries", "super_row_indices_saved")
        ),
      )
    )
  geometry_table = table(
    ("Pattern, 48 each", "Stored entries", "In rectangles", "In disjoint rectangles", "In diagonal runs", "In column groups", "Repeated row indices"),
    geometry,
  )
  benchmark_rows = []
  benchmark_counts = []
  for row in bench:
    p, c = row["pattern"], coverage[row["name"]]
    benchmark_rows.append(
      (row["name"], p["nnz"], p["block_entries"], p["band_entries"], p["super_entries"], c["straight"], c["dense"], c["affine"], c["table"])
    )
    benchmark_counts.append((row["name"], arithmetic(row), row["program"]["table_entries"], row["source_bytes"], f"{row['compile_seconds']:.3f}"))
  benchmark_geometry = table(
    ("Derivative", "Stored", "Rectangles", "Diagonal runs", "Column groups", "Straight writes", "Dense writes", "Affine writes", "Table writes"),
    benchmark_rows,
  )
  benchmark_code = table(("Derivative", "Floating operations", "Index entries", "C bytes", "cc seconds"), benchmark_counts)
  candidate_rows = []
  for kind in ("blocks", "bands", "supernodes"):
    for limit in (8, 100000):
      winners = [r for r in addressing if r[kind]["table_entries"] < r["actual_table_entries"] and r[kind]["regions"] <= limit]
      candidate_rows.append(
        (
          kind,
          "8" if limit == 8 else "unlimited",
          len(winners),
          sum(r["actual_table_entries"] - r[kind]["table_entries"] for r in winners),
          sum(r["baseline"]["loaded_invocations"] - r[kind]["loaded_invocations"] for r in winners),
          sum(r[kind]["regions"] for r in winners),
        )
      )
  candidate_table = table(
    (
      "Isolated split",
      "Region cap per product",
      "Winning products / 240",
      "Index entries removed",
      "Loaded-bound invocations removed",
      "Regular regions emitted",
    ),
    candidate_rows,
  )
  baseline = sum(r["actual_table_entries"] for r in addressing)
  labels = ["Current", *(f"{r[0]}, cap {r[1]}" for r in candidate_rows)]
  values = [baseline, *(baseline - r[3] for r in candidate_rows)]
  figure, axis = plt.subplots(figsize=(8, 3.8), layout="constrained")
  bars = axis.barh(labels, values, color=["#606060", *("#4477aa" if r[1] == "8" else "#cc8844" for r in candidate_rows)])
  axis.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, fontsize=9)
  axis.invert_yaxis()
  axis.set_xlim(0, baseline * 1.2)
  axis.set_xlabel("Index-table entries across 240 mixed products")
  axis.set_title("Addressing model, with eight regions or an unlimited split")
  axis.spines[["top", "right"]].set_visible(False)
  figure.savefig(HERE / "addressing.png", dpi=160)
  plt.close(figure)
  map_candidates = table(
    ("Compact-map split", "Map index entries before", "After", "Rectangles emitted"),
    [
      (
        key,
        sum(r["map_table_entries_before"] for r in maps),
        sum(r["map_table_entries_after"][key] for r in maps),
        sum(r["rectangles"].get(key, 0) for r in maps),
      )
      for key in ("eight", "all")
    ],
  )
  per_problem = []
  for name in sorted({r["name"].split("/")[0] for r in kkt}):
    group = [r for r in kkt if r["name"].startswith(name + "/")]
    main = next(r for r in group if r["name"].endswith("/kkt"))
    per_problem.append(
      (
        name,
        main["pattern"]["nnz"],
        sum(r["program"]["table_entries"] for r in group),
        sum(arithmetic(r) for r in group),
        sum(r["source_bytes"] for r in group),
        f"{sum(r['compile_seconds'] for r in group):.3f}",
      )
    )
  problem_table = table(
    ("Problem", "KKT stored entries", "Index entries, nine modules", "Floating operations, nine modules", "C bytes, nine modules", "cc seconds, sum"),
    per_problem,
  )

  note = f"""# Structure within sparse patterns, #91

- Structure detection: **build now**, limited to exact rectangles in compact maps. Propose "Split sparse compact maps into affine rectangular regions", one implementation thread. Keep bands and factorization supernodes deferred.
- Per-entry batch patterns: **keep deferred** until a real batch has different structural patterns. The two test reproductions store 8 for 4 supported entries and 4 for 2.
- Automatic dense-value compaction: **keep deferred** until after #225 and #75. The remaining 6,272 computed entries versus 1,344 in the support are seed-axis work already owned by #225.

## Scope and reproducibility

Inspected main at `ce7f8f58e88470cc9baad2c56a3b45f9e3c5572a`, before this notes-only branch.
The brief is #91 under section 2, #292. Inputs were the [compiler-run plan](../compiler_run_2026_11_10/plan.md),
the machine's suite rules, the roadmap's [sparse tensors and open question](../core_compiler_roadmap.md#sparse-tensors),
[#438's packet draft](https://github.com/PREDICT-EPFL/scaly/pull/438), and
the [sparse-kernel implementation notes](../sparse_kernels_2026_11_10/).
No source or test file changed. No devrush code was imported or built.

The 48 Maros-Meszaros archives and preprocessing are exactly those in
[test_sparse_kkt.py](../../../tests/integration/test_sparse_kkt.py). Each problem contributes nine
functions: full and upper Karush-Kuhn-Tucker (KKT) assembly, `P @ x`, `A @ x`, `A.T @ y`,
`G @ x`, `G.T @ y`, `A.T @ diag(w) @ A`, and `G.T @ diag(w) @ G`.
Here `A` holds equality rows and `G` holds non-box inequality rows, including stored zeros in
free rows. This differs from the packet demo's unsplit constraint matrix.
Assembly retains the test's shared upper-triangle values and regularization. Products take
compressed sparse column inputs with the same static patterns. Explicit stored zeros count.
All functions request block lowering, including the `.block()` marker's normal simplification.

The benchmark sample covers all four families with sparse derivatives: chain M=3, N=40,
race cars N=25, npmpc N=12 with its default decoder, and unbumpercars C=8 with its discrete model.
Both the constraint Jacobian and lower-triangle Lagrangian Hessian come from the actual solver
descriptor, the same builders the benchmark sweep uses. This is one size per family, not a sweep
over every horizon, mass count, decoder width or model.

Run from the repository root with `uv run`:

```sh
uv run internal/notes/sparse_structure_2026_11_11/study.py kkt --compile
uv run internal/notes/sparse_structure_2026_11_11/study.py bench --compile
uv run internal/notes/sparse_structure_2026_11_11/addressing.py
uv run internal/notes/sparse_structure_2026_11_11/maps.py
uv run internal/notes/sparse_structure_2026_11_11/census.py
uv run internal/notes/sparse_structure_2026_11_11/coverage.py
uv run internal/notes/sparse_structure_2026_11_11/verify.py
uv run internal/notes/sparse_structure_2026_11_11/report.py
```

`study.py` temporarily observes lowering in memory. It restores the methods afterwards.
The saved [KKT records](kkt.jsonl) and [benchmark records](bench.jsonl) contain each pattern's counts,
map regions, index-map requests, operation counts, source bytes and compile outcome.
[provenance.json](provenance.json) records the toolchain. The table below counts the standalone
generated C body, excluding the public header and its boundary-layout metadata. Rendering uses
one lane, portable C, generic CPU and no vector math library. Each generated body compiles once
with GCC 13.3 using `cc -O3 -fno-math-errno -std=c99 -c`. All 440 compilations succeeded.
The compile durations are single wall-clock observations on busy la015. They describe these
compiler invocations and support no speed or regression claim. No generation or execution time
was measured. Census oracles are rendered, not solved. Building the sparse PIQP example invokes
its existing numerical pattern probe, without timing it.

Floating operation counts expand the optimized program's constant loops and constant-table
bounds and follow procedure calls. They count each scalar arithmetic or mathematical function
evaluation once. They exclude integer addressing, copies and stores. This is a reference-program
count, before the C compiler's optimizations. The checker confirms all 336 product counts against
independent formulas, and all 240 mixed-product table models against the final generated tables.
NumPy rectangle coverage also agrees with exhaustive enumeration on twelve small masks.

## What lowering already does

`LowerCtx.index_at` factors the whole requested index sequence. A residual of length one becomes
arithmetic. Other requests keep only their residual table. It does not independently split an
irregular request at each regular subrectangle. `emit_map` first partitions by operand membership,
then emits straight statements for at most four entries or one loop for the whole region.
"Dense" below means consecutive stored accesses, not a newly detected mathematical rectangle.
Affine indexing may contain integer division for flattened coordinates.

The map-entry columns count stored-entry visits across all intermediate compact maps. An entry
copied through three intermediates counts three times. A region is a table region if any live
read or write map needs a residual table. The last two columns instead count positions in
`index_at` requests, across all access maps. One stored entry can have several such requests.
These units must not be added to the mathematical pattern's stored-entry count.

{regions}

Mixed products do not use `emit_map`. Classifying their actual stored operand addressing gives
the following counts. A whole traversal is table-based when either its source-slot permutation or
inner-coordinate map retains a residual. Loaded row bounds are separate even when both access
maps are affine. An all-dense-looking traversal can therefore still have a row-pointer table.

{product_regions}

Sparse-sparse products use one accumulator column. They have a single ordered traversal rather
than a partition into geometric regions. The access-map columns above include their operand,
output-row and output-write maps. Their output-entry counts and geometry appear below.
The number of scalar products can exceed either operand's stored-entry count.

The final program includes these tables and operations. A loaded-bound invocation is one entry
to a loop whose bounds load a constant table, expanded over all enclosing loops. It is not a
distinct loop site or the number of inner iterations. Single-row bounds can fold to constants.

{current}

The totals are {sum(r["program"]["table_entries"] for r in kkt):,} index entries and
{sum(r["source_bytes"] for r in kkt):,} generated C bytes across 432 separate modules.
Sharing is counted within a module, never between modules.
The full KKT module alone is the first kernel row, not the sum of all nine.

The benchmark derivatives on this snapshot return compact dense values plus separate sparsity
metadata, pending #48. They do not enter #46's compact sparse maps. Their final stored-output
writes nevertheless have measurable addressing. For each entry, `coverage.py` examines its
actual optimized output store and enclosing bounds. A table lookup in any address makes that
write a table write. Otherwise consecutive or broadcast accesses are dense, other table-free
accesses are affine, and a store outside loops is straight. If an entry has several writes, the
table form takes precedence, then affine, dense and straight. Every stored output entry is covered.
These counts describe final assembly, not the earlier arithmetic producing the values.

{benchmark_geometry}

All eight have zero loaded-bound loop sites. Their arithmetic and tables are:

{benchmark_code}

## What structure remains

`structure.py` uses NumPy on coordinates, with no dense array of a workload's logical shape.
Its rectangle coverage is the union of every fully occupied contiguous rectangle at least 4 by 4.
It finds them through four-row intersections, then extends matching intervals. The disjoint count
greedily takes larger rectangles first and rejects overlaps. It is a reproducible candidate cover,
not an optimal partition. "Diagonal runs" means occupied consecutive entries on one diagonal,
with length at least four. This conservative band definition does not call a mostly empty bounding
envelope a filled band. It does include single filled diagonals. Several such runs can form a band.
Column groups are maximal consecutive columns with exactly equal row sets and width at least two.
These are the question's supernode candidates, not factorization supernodes of a filled triangle.
The three coverages overlap and must not be summed. No ordering or permutation is searched.

{geometry_table}

Repeated row indices are the structural saving `(width - 1) * row_count` from storing one common
row list per column group. They are not a promise to remove that many of today's generated table
entries. Today's affine recovery may already have removed the list. For example PRIMALC1's
`Ggram` output has 52,900 stored entries, all in a dense rectangle, yet its whole kernel has zero
index-table entries and 3,179 C bytes. It already recovers its regular accesses. Detecting its
rectangle again has no table saving.

### Compact maps

`maps.py` examines the actual physical write and read arrays of all 96 assembly functions.
It splits a region into disjoint exact rectangles and a remainder, tries either loop orientation,
and applies the existing affine factorization to every resulting access map. It accepts a split
only when it reduces that region's residual-table entries. The eight-rectangle version takes at
most eight rectangles per original map region, not per function. The unlimited version is an
upper exploration of this greedy cover, not a proposed implementation.

{map_candidates}

These are unique residual-table entries used by compact maps before program optimization.
The "before" total differs from the final table total because it omits unrelated gather/layout
maps and because later passes can fold or compress tables. Splits may also change sharing with
those other users. The model does not claim these as final C savings. There are no loaded-bound
loops in either assembly function, so rectangle splits remove zero such loops.

One small case is concrete enough to justify a task. PRIMALC5's full KKT function has 4,887 final
index entries, all belonging to its two irregular compact-map regions. In the model, two disjoint
rectangles make every map arithmetic, with zero residual entries. The present function emits 30,479 C bytes,
286 additions and 8 negations. Its observed C compile duration was 0.040 seconds. A split changes
addressing and loop organization, with the same 294 floating operations. This gives a reviewable
zero-table target without building a general block-sparse representation.

### Products

`addressing.py` separately models rectangle, diagonal-run and equal-row column-group splits for
all 240 mixed products. The remaining entries keep the present row traversal. Each regular
region uses affine source, destination and vector maps wherever possible, retaining a residual
table for irregular physical source slots or common row coordinates. Row pointers remain only
for the generic remainder. Tables share identical residual arrays. One-row pointer tables fold
away, as they do in the actual program. All 240 baseline models match the generated table counts.
The table accepts only winning products. Its eight-region cap is per product. Models for the
three structures are isolated alternatives and their savings cannot be added.

![Index-table entries for the current mixed products and the alternative NumPy addressing models.](addressing.png)

Each alternative keeps a product's current form when it uses fewer entries. These are table
counts, not measured source sizes or times. The unlimited alternatives expose the cost of
allowing many more regions.

{candidate_table}

The baseline contains 196,855 index entries across these products. Rectangles remove up to 62,324,
but the largest individual reductions require many regions. DUAL3's `Px` model falls from 24,322
to 9,862 entries with 51 rectangles, while retaining all 111 loaded row-loop invocations because
the remainder still touches every row. An eight-region cap across the sample saves only 9,711
entries. QCAPRI's equal-row column groups offer 3,577 to 1,030 entries with eight regions, yet
retain all 56 loaded row-loop invocations. This is useful addressing work for a future product
task, but the compact-map target above is the smaller first change.

A dense inner kernel over exact occupied entries changes no floating operation count for these
products. `A @ x` and `A.T @ y` each do one multiplication and one accumulator addition per stored
entry. `A.T @ diag(w) @ A` does `nnz(A) + sum(row_degree(A)**2)` of each, including the first
weighted product. Across the sample the A Grams do 664,852 multiplications and 664,852 additions,
the G Grams 2,218,720 of each. Dense tiling can reuse addresses and loads or expose lanes. It
cannot remove those contributing products. Filling holes inside a candidate rectangle would
instead add operations and could evaluate an absent zero beside infinity, changing semantics.
The model only uses exact occupied regions. No alternate kernel has been implemented or compiled.

## Per-entry patterns and dense support

The repository audit read `examples/`, the benchmark builders, sparse tests, `solvers/ocp.py`,
`solvers/_stage.py`, `solvers/qp.py`, and both plugins' generated interfaces. The dynamic census
covers the eight benchmark oracles, chain's optimal-control problem (OCP) stage form through Fatrop at M=3, N=4, and the
PIQP oracle in `examples/sparse_constant_mpc.py`. It renders the latter's existing sparse mode
and, for comparison, its supported dense mode. It does not claim to instantiate every test or
every arbitrary Function a user could construct. Full records are in [census.json](census.json).

No benchmark oracle or inspected solver path builds a sparse value with per-entry patterns.
Optimal-control stage maps share one stage Function. Fatrop's stage interface deliberately
assembles dense local Jacobian and Hessian blocks. PIQP uses fixed matrices, either dense or
compact values in one static compressed sparse column layout. Neither is a heterogeneous batch.
The test named `test_sparse_arithmetic_with_different_patterns_batches_and_broadcasts` uses two
different operand patterns, with one shared pattern across the batch, not different patterns
inside that batch. Explicit numerical zeros are not a different storage pattern either.

There are small actual examples of unequal per-entry support, both reproduced by `census.py`:

| Main test | Batched shape | Stored entries | Support | Arithmetic / index tables |
| --- | --- | --- | --- | --- |
| `test_sparse_select_masks_survive_nesting_and_pattern_normalization`, stack and concat forms | 2 by 2 by 3 | 8 | 4 | 2 square roots / 0 |
| `test_sparse_constants_return_scipy_and_keep_batched_shared_patterns`, sparsify case | 2 by 3 by 4 | 4 | 2 | 0 / 0 |

The first takes the union of two different patterns. The second unions the per-entry support
when explicitly sparsifying a dense constant. They show the representation cost and correct
current behavior, not a workload whose cost warrants a new batch format. Seed-axis arrays with
different supports are real and larger, but #225 already owns their nonzero seed-entry traversal.

Dense-typed values with mostly empty support do reach lowering. `census.py` observes `Expr.support`
and allocation, then counts materialized stores after scalar expansion and folding, before
workspace packing. Support is the compiler's upper bound on entries that may be nonzero, not a
count of numerical nonzeros. A computed write below means its value is not the literal zero. Writes in
callees include their call multiplicity. It counts visits, not distinct mathematical coordinates,
and not arithmetic operations. A scatter may write zeros and then update a smaller support.

| Actual intermediate | Dense entries | Support entries | Computed writes | All writes |
| --- | --- | --- | --- | --- |
| Chain Jacobian, mapped stage derivative | 4,320 | 3,120 | 3,120 | 4,320 |
| Race-car Jacobian, mapped stage derivative | 500 | 425 | 425 | 500 |
| Unbumpercars Jacobian, one seed gather | 3,136 | 196 | 3,136 | 3,136 |
| Unbumpercars Hessian, one seed gather | 3,136 | 196 | 3,136 | 3,136 |
| Unbumpercars Hessian, final seed-axis add | 6,272 | 1,344 | 6,272 | 6,272 |
| Unbumpercars Jacobian, seed scatter | 896 | 56 | 56 | 952 |
| Fatrop initial Jacobian block | 81 | 9 | 9 | 81 |
| Fatrop padded first-stage Hessian contribution | 156 | 9 | 117 | 156 |
| PIQP example, linear term | 7 | 4 | 4 | 11 |

The first two already compute only the supported values, but still write required dense zeros.
The gathers and add show genuine remaining work over zero seed-entry pairs. The scatter already
limits arithmetic updates to its support, while retaining dense initialization. Intermediate
buffers that disappear after fusion are not counted as though they remain allocated. For example
several 6,272-entry multiplication intermediates disappear, while the final add in this table stays.
The `allocations` records retain both cases so they can be distinguished.
Fatrop also has a small non-seed copy opportunity in the padded first-stage contribution.
Its initial and terminal derivative blocks already compute only their supports, and the plugin
requires dense stage blocks at its boundary. The remaining 117 copy writes do not justify a
general compaction mechanism. They should remain in the follow-up census rather than be counted
as seed-axis arithmetic.

The PIQP dense comparison also lowers a constant 49-entry matrix with support 7 and a constant
28-entry matrix with support 10. The existing sparse option emits their compact values instead.
They are fixed constants at a deliberately dense solver boundary, not 49 or 28 floating
computations to remove. Changing that boundary automatically would violate its selected interface.
The sparse example's only partially supported allocated value is its seven-entry linear term.

## Recommendation and task cut

For structure detection, build the compact-map subset now. Proposed issue title:
**Split sparse compact maps into affine rectangular regions**.
One implementation thread should change the private region planner and its focused tests.
It should recognize exact occupied rectangles whose physical read and write maps become affine,
reuse `LowerCtx.index_at` and the existing loop emitter, and keep an irregular remainder.
It should not introduce a core operation, a block-sparse tensor type, a public option, a new
layout, reordered boundaries, band detection, supernode detection or product scheduling.
Keep a split only when it reduces addressing, and bound region growth using the emitted source
evidence rather than enabling the unlimited geometric cover.

A reviewer can check completion without an execution-time measurement:

- PRIMALC5's full assembly computes the same values and keeps its stored pattern, with zero
  kernel index-table entries, fewer than 30,479 C bytes and the same 286 additions and 8 negations.
- A focused NumPy differential checks rectangle-plus-remainder maps with unequal operand
  membership and compressed sparse column, row and coordinate layouts. Holes remain absent,
  including beside infinity. The test fails when a rectangle index or membership is perturbed.
- All 48 full and upper assemblies preserve values and patterns, do not grow their generated
  C source, and introduce no loaded-bound loops. Dense C snapshots remain byte-identical.
- The regular cases keep their table-free form. The ordinary full compiler checks pass. The
  pull request reports source bytes, reference operation counts and C-only compilation results.

Do not create a general product block planner in the same thread. Current matrix-vector kernels
have bounded stored-size tables, and extra regions can outweigh their savings. Revisit bands or
product column groups when a recurring sparse product needs them after loaded-bound lanes in
#71, or when #54 has a concrete factor/solve update. The present evidence supports the smaller
compact-map cut, not a speed claim for a block kernel.

For per-entry batch patterns, keep deferred. Revisit for a real mapped collection of different
stage models, heterogeneous meshes or matrices whose union stores substantially more than the
sum of their individual patterns. It should come with a concrete Function, stored sizes and a
consumer. The current two-entry test batches do not justify changing the shared-pattern contract.
Do not use this part of #91 to duplicate #225's seed-axis representation.

For automatic dense-value compaction, keep deferred. Finish #225's nonzero seed-entry traversal
and #75's demanded entries, then rerun these same unbumpercars cases. Revisit #91 if a non-seed
internal dense value still computes and stores a large mostly empty support after those changes.
Keep dense inputs and outputs fixed by their declared types. This requires no new issue now.
No proposed issue has been created by this investigation.

## What devrush's sparse factorization would need

Read only with `git show origin/devrush:<path>`. The relevant files are
`src/scaly/linalg/symbolic.py`, `src/scaly/linalg/sparse_factor.py` and
`src/scaly/linalg/ops/sparse_ldl.py` on that frozen branch.
The numerical algorithm factors a symmetric permuted matrix as `L D L.T`, called an LDL factorization.
Here `L` is unit lower triangular and `D` is diagonal.
Its useful structure is in the filled factor after ordering and symbolic analysis, not merely
in the original KKT pattern counted above. No factorization, ordering study or fill analysis
was built or run for this investigation.

`SymbolicLDL.chunks` groups up to eight entries in a row's update list when the corresponding
factor columns have identical suffix row lists from that row down. It supplies `ck_ptr`, `ck_q`,
`ck_width` and `ck_len`, alongside the factor's column pointers, row indices and row-to-column
positions. The numeric update computes one scale per participating column, then visits each
shared destination row once and accumulates all columns in registers. For width w and a shared
suffix of length h, this retains w*h multiplication/addition pairs but reduces destination
loads, stores and shared row-index visits from w*h to h. For w=8 that is eight to one. Unlike
a one-use assembly map, this is a repeated numeric consumer of the shared structure.

`solve_chunks` uses fundamental triangular supernodes: column j's row list starts with j+1,
and its remaining suffix equals column j+1's row list. These are not equal whole-column row
patterns. The forward solve treats the triangular rows inside the chunk in order, then updates
each common row below it once. Symbolic fill, dependency order and scratch clearing must remain
visible. A generic detector over the original pattern is insufficient.

Devrush emits one width-specific path for each width actually present, selected by a zero-or-one
loop. Main's #54 should express its algorithm as library Functions with carried loops, runtime
indexing and in-place updates, on the roadmap's existing primitives. It needs build-time symbolic
factor/chunk metadata and a measured source-size decision for width specialization. It should
reuse index arithmetic and lanes where applicable. It should not copy devrush's registered ops,
string-keyed table protocol or lowering implementation. The roadmap explicitly leaves the number
of width branches open. This is the workload that would justify revisiting supernodes.

## Limits

No run time, generation time, speedup, lane efficiency or solver iteration effect was measured.
The C compilation observations came from a busy machine. The geometric and addressing
alternatives are NumPy models, not implemented kernels. Their source size and compile time
cannot yet be measured. No dense-kernel arithmetic reduction is claimed.
The band and column-group candidates are conservative and use the existing coordinate order.
There is no search for permutations, relaxed supernodes or partly filled blocks.
Only one size per benchmark family was generated, and the audit does not dynamically execute
the whole example/test corpus. Custom derivative supports are taken as the compiler reports
them. This note leaves production correctness to the existing tests and the proposed change's
required differential tests. It does not modify the roadmap or close #91.

## Per-problem evidence

Each row sums nine separately generated modules. [kkt.jsonl](kkt.jsonl) retains the per-function
and per-pattern details, [addressing.json](addressing.json) the 240 product models,
[maps.json](maps.json) the 96 assembly models, and [coverage.json](coverage.json) the eight
benchmark output classifications. The counts are reproducible without a performance run.

{problem_table}
"""
  (HERE / "README.md").write_text(note)


if __name__ == "__main__":
  main()
